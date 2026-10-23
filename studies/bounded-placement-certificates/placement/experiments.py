"""Frozen illustrative grids. No devices, models, networks, or measurement calls."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import csv
import hashlib
import json

from placement.solver import solve, validate_scenario, stage_accounting, build_certificate
from placement.checker import check_certificate, check_rejection_bound

GIB=2**30
MIB=2**20


def unique_bytes(s):
    endpoints = s['embedding_bytes'] if s['tied_embeddings'] else s['embedding_bytes']+s['head_bytes']
    return sum(l['weight_bytes']+l['kv_bytes'] for l in s['layers'])+endpoints+s['controller_reserve_bytes']


def _toy(weights,budgets):
    n=len(weights)
    ids=[f'd{i}' for i in range(len(budgets))]
    return dict(version=1,layers=[dict(weight_bytes=w,kv_bytes=0,workspace_bytes=0,compute_ns=0) for w in weights],
        devices=[dict(id=name,budget_bytes=b,runtime_reserve_bytes=0,eligible=True,
                      layer_support=[True]*n,embedding_support=True,head_support=True) for name,b in zip(ids,budgets)],
        embedding_bytes=0,head_bytes=0,tied_embeddings=True,embedding_workspace_bytes=0,
        head_workspace_bytes=0,embedding_compute_ns=0,head_compute_ns=0,
        controller_reserve_bytes=0,activation_reserve_bytes=0,activation_payload_bytes=1,
        feedback_payload_bytes=1,links=[dict(source=a,target=b,available=True,
                 latency_ns=1,payload_bps=8_000_000_000) for a in ids for b in ids if a!=b])


def make_surrogate(homes,reserve_mib=0,endpoint_bytes=0,workspace_mib=0,all_wan=False):
    if type(homes) is not int or not 1<=homes<=6: raise ValueError('homes must be 1..6')
    s=_toy([0]*80,[b*GIB for _ in range(homes) for b in (8,2,2)])
    remaining=42_000_000_000-endpoint_bytes
    if remaining<0: raise ValueError('endpoint exceeds stipulated unique weights')
    each,extra=divmod(remaining,80)
    for i,l in enumerate(s['layers']):
        l.update(weight_bytes=each+(i<extra),kv_bytes=2*8*128*8192*1*2,
                 workspace_bytes=workspace_mib*MIB,compute_ns=None)
    s.update(embedding_bytes=endpoint_bytes,head_bytes=endpoint_bytes,
             embedding_compute_ns=None,head_compute_ns=None,
             controller_reserve_bytes=2*GIB,activation_reserve_bytes=reserve_mib*MIB,
             activation_payload_bytes=16_384,feedback_payload_bytes=4,
             evidence_kind='analytical_assumptions_not_measurement',
             surrogate_warning='80-block envelope-matched arithmetic; not an actual checkpoint tensor decomposition')
    for i,d in enumerate(s['devices']):
        d['home']=i//3
        d['label']=('laptop_allocation','phone_allocation_1','phone_allocation_2')[i%3]
    homes_by_id={d['id']:d['home'] for d in s['devices']}
    for e in s['links']:
        local=homes_by_id[e['source']]==homes_by_id[e['target']] and not all_wan
        e.update(latency_ns=1_000_000 if local else 20_000_000,
                 payload_bps=1_000_000_000 if local else 20_000_000)
    validate_scenario(s)
    return s


def _known_fit(s,d,a,b):
    device=s['devices'][d]
    return (device['eligible'] is True and device['budget_bytes'] is not None and
            all(v is True for v in device['layer_support'][a:b]) and
            (a!=0 or device['embedding_support'] is True) and
            (b!=len(s['layers']) or device['head_support'] is True) and
            stage_accounting(s,d,a,b)['total_bytes']<=device['budget_bytes'])


def _greedy(s,order):
    links={(e['source'],e['target']) for e in s['links'] if e['available'] is True}
    placement=[]; pos=0; total=len(s['layers'])
    for d in order:
        name=s['devices'][d]['id']
        if placement and (s['devices'][placement[-1][0]]['id'],name) not in links: continue
        for end in range(total,pos,-1):
            if end==total and placement and (name,s['devices'][placement[0][0]]['id']) not in links: continue
            if _known_fit(s,d,pos,end):
                placement.append((d,pos,end)); pos=end; break
        if pos==total:
            c=build_certificate(s,placement)
            checked=check_certificate(s,c)
            if not checked['valid']: raise AssertionError(checked)
            return dict(status='feasible',certificate=c)
    return dict(status='not_found')


def run_baselines(s):
    validate_scenario(s)
    known=all(d['eligible'] is not None and (d['eligible'] is False or d['budget_bytes'] is not None) for d in s['devices'])
    cap=sum(d['budget_bytes'] for d in s['devices'] if d['eligible'] is True and d['budget_bytes'] is not None)
    aggregate=dict(unique_required_bytes=unique_bytes(s),known_eligible_allocation_bytes=cap,
                   passes_known_necessary_condition=unique_bytes(s)<=cap if known else None,
                   warning='Necessary only; omits placement, extra tied copies and per-used-stage reserves/workspace')
    best=dict(status='not_found')
    for d in range(len(s['devices'])):
        if _known_fit(s,d,0,len(s['layers'])):
            cert=build_certificate(s,[(d,0,len(s['layers']))])
            assert check_certificate(s,cert)['valid']
            best=dict(status='feasible',certificate=cert); break
    order=list(range(len(s['devices'])))
    descending=sorted(order,key=lambda i:-(s['devices'][i]['budget_bytes'] or 0))
    return dict(aggregate=aggregate,best_single=best,input_order_greedy=_greedy(s,order),
                descending_budget_greedy=_greedy(s,descending))


def frozen_cases():
    cases=[]
    def add(name,group,s,**attrs): cases.append(dict(id=name,group=group,scenario=s,**attrs))
    for h in range(3,7):
        for reserve in (0,64,128,192,256,384,512):
            add(f'envelope_h{h}_r{reserve}','reserve_sensitivity',make_surrogate(h,reserve_mib=reserve),
                homes=h,reserve_mib=reserve,endpoint_bytes=0,workspace_mib=0)
    for h in (4,5):
        for e in (0,GIB//2,GIB,2*GIB,3*GIB,4*GIB,8*GIB):
            for workspace in (0,64,256):
                add(f'endpoint_h{h}_e{e}_w{workspace}','endpoint_sensitivity',
                    make_surrogate(h,endpoint_bytes=e,workspace_mib=workspace),
                    homes=h,reserve_mib=0,endpoint_bytes=e,workspace_mib=workspace)
    add('indivisible_3gib','counterexample',_toy([3*GIB]*4,[8*GIB,2*GIB,2*GIB]))
    add('contiguity_2_6_2','counterexample',_toy([2*GIB,6*GIB,2*GIB],[4*GIB,6*GIB]))
    q=_toy([3,3],[5,4]); q['devices'][1]['head_support']=False
    add('greedy_endpoint_trap','baseline_counterexample',q)
    q=_toy([2,2],[3,3]); q['activation_reserve_bytes']=2
    add('stage_reserve','counterexample',q)
    q=_toy([2,2],[4,4]); q.update(embedding_bytes=3,head_bytes=3)
    add('tied_duplication','counterexample',q)
    q=_toy([2],[100]); q['devices'][0]['eligible']=False
    add('explicit_ineligible','eligibility',q)
    q=_toy([2],[None])
    add('unknown_allocation','unknown',q)
    q=_toy([2],[2]); q['devices'][0]['eligible']=None
    add('unknown_eligibility','unknown',q)
    q=_toy([2,2],[3,3]); q['links']=q['links'][:1]
    add('missing_feedback_link','link',q)
    q=_toy([2,2],[3,3]); q['links'][1]['available']=None
    add('unknown_feedback_link','unknown',q)
    q=_toy([2,2],[3,3]); q['links'][1]['payload_bps']=None; q['layers'][0]['compute_ns']=None
    add('unknown_timing','unknown',q)
    add('all_wan_h4','link',make_surrogate(4,all_wan=True),homes=4,reserve_mib=0,endpoint_bytes=0,workspace_mib=0)
    q=make_surrogate(4)
    for d in q['devices']:
        if d['label'].startswith('phone'): d['eligible']=False
    add('phones_ineligible_h4','eligibility',q,homes=4,reserve_mib=0,endpoint_bytes=0,workspace_mib=0)
    add('deliberate_search_cap','cap',_toy([2,2],[3,3]),max_states=0)
    return cases


def _as_fraction(d):
    return None if d is None else Fraction(d['numerator'],d['denominator'])


def _summary(outcome):
    result=dict(status=outcome['status'],stages=None,resident_bytes=None,communication_ms=None,
                critical_path_known=False,extra_endpoint_bytes=None)
    if 'certificate' in outcome:
        c=outcome['certificate']
        result.update(stages=len(c['stages']),resident_bytes=sum(x['memory']['total_bytes'] for x in c['stages']),
                      communication_ms=None if c['timing']['communication_ns'] is None else
                          float(_as_fraction(c['timing']['communication_ns'])/10**6),
                      critical_path_known=c['timing']['critical_path_ns'] is not None)
    return result


def reproduce(output_dir):
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    cases=frozen_cases()
    assert len(cases)<=150
    bundle=[]; rows=[]; checks=[]
    for case in cases:
        s=case['scenario']
        result=solve(s,max_states=case.get('max_states',250_000))
        b=run_baselines(s)
        entries=[('solver',result)]+[(k,v) for k,v in b.items() if k!='aggregate']
        for name,outcome in entries:
            if 'certificate' in outcome:
                checked=check_certificate(s,outcome['certificate'])
                assert checked['valid'],(case['id'],name,checked)
                checks.append(dict(case_id=case['id'],kind=name,check=checked))
        if result.get('rejection_bound') is not None:
            checked=check_rejection_bound(s,result['rejection_bound'])
            assert checked['valid'],(case['id'],checked)
            checks.append(dict(case_id=case['id'],kind='negative_bound',check=checked))
        packed=dict(case,solver=result,baselines=b)
        bundle.append(packed)
        summary=_summary(result)
        row=dict(id=case['id'],group=case['group'],homes=case.get('homes',''),
                 reserve_mib=case.get('reserve_mib',''),endpoint_gib='' if 'endpoint_bytes' not in case else case['endpoint_bytes']/GIB,
                 workspace_mib=case.get('workspace_mib',''),unique_required_gib=unique_bytes(s)/GIB,
                 known_allocation_gib=b['aggregate']['known_eligible_allocation_bytes']/GIB,
                 aggregate_pass=b['aggregate']['passes_known_necessary_condition'],
                 solver_status=result['status'],solver_stages=summary['stages'],
                 solver_resident_gib=None if summary['resident_bytes'] is None else summary['resident_bytes']/GIB,
                 solver_communication_ms=summary['communication_ms'],
                 critical_path_known=summary['critical_path_known'],
                 maximum_coverable_layers=(result.get('rejection_bound') or {}).get('maximum_coverable_layers'))
        if 'certificate' in result:
            endpoints=sum(x['memory']['endpoint_bytes'] for x in result['certificate']['stages'])
            unique_endpoint=s['embedding_bytes'] if s['tied_embeddings'] else s['embedding_bytes']+s['head_bytes']
            row['extra_endpoint_gib']=(endpoints-unique_endpoint)/GIB
        else: row['extra_endpoint_gib']=None
        for name in ('best_single','input_order_greedy','descending_budget_greedy'):
            short=_summary(b[name])
            row[name+'_status']=short['status']; row[name+'_stages']=short['stages']
            row[name+'_communication_ms']=short['communication_ms']
        rows.append(row)
    package=dict(evidence_kind='analytical_not_measurement',model_version=1,cases=bundle)
    (root/'all_cases.json').write_text(json.dumps(package,indent=2,sort_keys=True)+'\n')
    (root/'independent_checks.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')
    with (root/'sensitivity.csv').open('w',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    counts={key:sum(r['solver_status']==key for r in rows) for key in ('feasible','infeasible','unknown','search_incomplete')}
    summary=dict(case_count=len(rows),statuses=counts,independently_checked_witnesses_and_bounds=len(checks),
                 aggregate_positive_solver_negative=[r['id'] for r in rows if r['aggregate_pass'] is True and r['solver_status']=='infeasible'],
                 greedy_misses=[r['id'] for r in rows if r['solver_status']=='feasible' and r['input_order_greedy_status']!='feasible'],
                 all_outputs_are_assumption_conditional=True)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

if __name__=='__main__':
    import sys
    directory=sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parents[1]/'results')
    print(json.dumps(reproduce(directory),indent=2,sort_keys=True))
