"""Independent witness verification. Deliberately imports no solver code.

Recalculates directly from the JSON scenario and supplied intervals; does not
certify optimality, actual memory telemetry, device support, quality or consent.
"""
import hashlib
import json
from fractions import Fraction

def _source(s):
    """Fail closed on malformed model input, independently of solver validation."""
    if not (type(s) is dict and type(s['version']) is int and (s['version'] == 1)):
        raise ValueError('invalid scenario or certificate')
    if not (type(s['layers']) is list and 1 <= len(s['layers']) <= 128):
        raise ValueError('invalid scenario or certificate')
    if not (type(s['devices']) is list and 1 <= len(s['devices']) <= 18):
        raise ValueError('invalid scenario or certificate')
    integer = lambda x: type(x) is int and x >= 0
    tri = lambda x: x is None or type(x) is bool
    for layer in s['layers']:
        if not all((integer(layer[k]) for k in ('weight_bytes', 'kv_bytes', 'workspace_bytes'))):
            raise ValueError('invalid scenario or certificate')
        if not (layer['compute_ns'] is None or integer(layer['compute_ns'])):
            raise ValueError('invalid scenario or certificate')
    ids = []
    for d in s['devices']:
        if not (type(d['id']) is str and d['id'] and (d['id'] not in ids)):
            raise ValueError('invalid scenario or certificate')
        ids.append(d['id'])
        if not (d['budget_bytes'] is None or integer(d['budget_bytes'])):
            raise ValueError('invalid scenario or certificate')
        if not integer(d['runtime_reserve_bytes']):
            raise ValueError('invalid scenario or certificate')
        if not all((tri(d[k]) for k in ('eligible', 'embedding_support', 'head_support'))):
            raise ValueError('invalid scenario or certificate')
        if not (type(d['layer_support']) is list and len(d['layer_support']) == len(s['layers'])):
            raise ValueError('invalid scenario or certificate')
        if not all((tri(x) for x in d['layer_support'])):
            raise ValueError('invalid scenario or certificate')
    for k in ('embedding_bytes', 'head_bytes', 'embedding_workspace_bytes', 'head_workspace_bytes', 'controller_reserve_bytes', 'activation_reserve_bytes', 'activation_payload_bytes', 'feedback_payload_bytes'):
        if not integer(s[k]):
            raise ValueError('invalid scenario or certificate')
    for k in ('embedding_compute_ns', 'head_compute_ns'):
        if not (s[k] is None or integer(s[k])):
            raise ValueError('invalid scenario or certificate')
    if not type(s['tied_embeddings']) is bool:
        raise ValueError('invalid scenario or certificate')
    if not (not s['tied_embeddings'] or s['embedding_bytes'] == s['head_bytes']):
        raise ValueError('invalid scenario or certificate')
    if type(s['links']) is not list:
        raise ValueError('links must be a list')
    pairs = []
    for e in s['links']:
        pair = (e['source'], e['target'])
        if not (pair not in pairs and pair[0] in ids and (pair[1] in ids) and (pair[0] != pair[1])):
            raise ValueError('invalid scenario or certificate')
        pairs.append(pair)
        if not tri(e['available']):
            raise ValueError('invalid scenario or certificate')
        if not (e['latency_ns'] is None or integer(e['latency_ns'])):
            raise ValueError('invalid scenario or certificate')
        if not (e['payload_bps'] is None or (integer(e['payload_bps']) and e['payload_bps'] > 0)):
            raise ValueError('invalid scenario or certificate')
    return hashlib.sha256(json.dumps(s, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _identical(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _q(x):
    return {'numerator': x.numerator, 'denominator': x.denominator}

def check_certificate(s, c):
    errors = []
    try:
        digest = _source(s)
        if not (type(c) is dict and type(c['version']) is int and (c['version'] == 1)):
            raise ValueError('certificate version')
        if not c['scenario_sha256'] == digest:
            raise ValueError('scenario hash')
        if not (type(c['stages']) is list and c['stages']):
            raise ValueError('missing stages')
        count = len(s['layers'])
        devices = {d['id']: d for d in s['devices']}
        seen, cursor, memory_totals = (set(), 0, [])
        for stage in c['stages']:
            name, a, b = (stage['device_id'], stage['start'], stage['end'])
            if not (type(a) is int and type(b) is int and (a == cursor) and (a < b <= count)):
                raise ValueError('invalid contiguous coverage')
            if not (name in devices and name not in seen):
                raise ValueError('missing or repeated device')
            seen.add(name)
            cursor = b
            d = devices[name]
            if not (d['eligible'] is True and d['budget_bytes'] is not None):
                raise ValueError('unknown/ineligible allocation')
            if not all((d['layer_support'][i] is True for i in range(a, b))):
                raise ValueError('unknown/unsupported layer')
            if not (a != 0 or d['embedding_support'] is True):
                raise ValueError('unknown/unsupported embedding')
            if not (b != count or d['head_support'] is True):
                raise ValueError('unknown/unsupported head')
            weights = sum((s['layers'][i]['weight_bytes'] for i in range(a, b)))
            cache = sum((s['layers'][i]['kv_bytes'] for i in range(a, b)))
            extras = []
            if a == 0:
                extras.append(s['embedding_bytes'])
            if b == count:
                extras.append(s['head_bytes'])
            endpoints = sum(extras)
            if a == 0 and b == count and s['tied_embeddings']:
                endpoints = s['embedding_bytes']
            peaks = [s['layers'][i]['workspace_bytes'] for i in range(a, b)]
            if a == 0:
                peaks.append(s['embedding_workspace_bytes'])
            if b == count:
                peaks.append(s['head_workspace_bytes'])
            reserve = s['controller_reserve_bytes'] if a == 0 else 0
            expected = {'weight_bytes': weights, 'kv_bytes': cache, 'endpoint_bytes': endpoints, 'runtime_reserve_bytes': d['runtime_reserve_bytes'], 'activation_reserve_bytes': s['activation_reserve_bytes'], 'workspace_bytes': max(peaks), 'controller_reserve_bytes': reserve}
            expected['total_bytes'] = weights + cache + endpoints + d['runtime_reserve_bytes'] + s['activation_reserve_bytes'] + max(peaks) + reserve
            if not _identical(stage['memory'], expected):
                raise ValueError('memory arithmetic mismatch')
            if not expected['total_bytes'] <= d['budget_bytes']:
                raise ValueError('allocation exceeded')
            memory_totals.append(expected['total_bytes'])
        if not cursor == count:
            raise ValueError('incomplete final coverage')
        expected_edges = []
        names = [x['device_id'] for x in c['stages']]
        for i in range(len(names) - 1):
            expected_edges.append((names[i], names[i + 1], 'activation', s['activation_payload_bytes']))
        if len(names) > 1:
            expected_edges.append((names[-1], names[0], 'feedback', s['feedback_payload_bytes']))
        if type(c['transfers']) is not list:
            raise ValueError('transfers must be a list')
        if not len(c['transfers']) == len(expected_edges):
            raise ValueError('missing or extra transfer')
        links = {(e['source'], e['target']): e for e in s['links']}
        comm_sum, missing = (Fraction(0), [])
        for transfer, (a, b, kind, size) in zip(c['transfers'], expected_edges):
            if not ((a, b) in links and links[a, b]['available'] is True):
                raise ValueError('unavailable/unknown directed link')
            link = links[a, b]
            partial = Fraction(0)
            before = len(missing)
            if link['latency_ns'] is None:
                missing.append(f'{kind}:{a}->{b}:latency_ns')
            else:
                partial += Fraction(link['latency_ns'])
            if link['payload_bps'] is None:
                missing.append(f'{kind}:{a}->{b}:payload_bps')
            else:
                partial += Fraction(8 * size, link['payload_bps']) * 1000000000
            expected = {'source': a, 'target': b, 'kind': kind, 'payload_bytes': size, 'duration_ns': None if len(missing) != before else _q(partial), 'known_lower_bound_ns': _q(partial)}
            if not _identical(transfer, expected):
                raise ValueError('transfer arithmetic mismatch')
            comm_sum += partial
        comm_missing = bool(missing)
        compute_sum, compute_missing = (Fraction(0), False)
        for i in range(count):
            value = s['layers'][i]['compute_ns']
            if value is None:
                compute_missing = True
                missing.append(f'layer:{i}:compute_ns')
            else:
                compute_sum += value
        for key in ['embedding_compute_ns', 'head_compute_ns']:
            if s[key] is None:
                compute_missing = True
                missing.append(key)
            else:
                compute_sum += s[key]
        expected = {'communication_ns': None if comm_missing else _q(comm_sum), 'compute_ns': None if compute_missing else _q(compute_sum), 'critical_path_ns': None if missing else _q(comm_sum + compute_sum), 'communication_lower_bound_ns': _q(comm_sum), 'critical_path_lower_bound_ns': _q(comm_sum + compute_sum), 'unknown_terms': missing}
        if not _identical(c['timing'], expected):
            raise ValueError('timing arithmetic/unknown mismatch')
        return {'valid': True, 'errors': [], 'stage_count': len(names), 'resident_bytes': sum(memory_totals), 'scenario_sha256': digest}
    except (AssertionError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        errors.append(str(exc) or 'malformed input or certificate')
    return {'valid': False, 'errors': errors}

def check_rejection_bound(s, bound):
    """Independently verify optimistic first-prefix or endpoint-pair count bounds."""
    try:
        digest = _source(s)
        if type(bound) is not dict:
            raise ValueError('no bound supplied')
        total = len(s['layers'])
        regular, firsts, lasts, monolithic = [], [], [], False
        for device in s['devices']:
            cap = device['budget_bytes']
            counts, first, last = [], 0, 0
            if device['eligible'] is not False:
                for a in range(total):
                    for b in range(a+1,total+1):
                        support = device['layer_support'][a:b]
                        if False in support: continue
                        block = s['layers'][a:b]
                        mem = sum(v['weight_bytes'] + v['kv_bytes'] for v in block)
                        mem += device['runtime_reserve_bytes'] + s['activation_reserve_bytes']
                        peak = max(v['workspace_bytes'] for v in block)
                        if cap is None or mem + peak <= cap: counts.append(b-a)
                        if a == 0 and device['embedding_support'] is not False:
                            prefix_peak = max(peak,s['embedding_workspace_bytes'])
                            prefix = mem + prefix_peak + s['embedding_bytes'] + s['controller_reserve_bytes']
                            if cap is None or prefix <= cap: first = max(first,b)
                        if b == total and device['head_support'] is not False:
                            suffix = mem + max(peak,s['head_workspace_bytes']) + s['head_bytes']
                            if cap is None or suffix <= cap: last = max(last,total-a)
                        if a == 0 and b == total and device['embedding_support'] is not False and device['head_support'] is not False:
                            endpoint_cost = s['embedding_bytes'] if s['tied_embeddings'] else s['embedding_bytes']+s['head_bytes']
                            full = mem + max(peak,s['embedding_workspace_bytes'],s['head_workspace_bytes']) + endpoint_cost + s['controller_reserve_bytes']
                            if cap is None or full <= cap: monolithic=True
            regular.append(max(counts,default=0)); firsts.append(first); lasts.append(last)
        if bound.get('kind') == 'first_prefix_plus_individual_maximum_contiguous_counts':
            maximum = max([firsts[i]+sum(regular[j] for j in range(len(regular)) if j != i)
                           for i in range(len(regular)) if firsts[i] > 0], default=0)
            expected = {'kind':'first_prefix_plus_individual_maximum_contiguous_counts',
                        'per_device_max_layers':regular,'per_device_first_prefix_layers':firsts,
                        'maximum_coverable_layers':maximum,'required_layers':total,
                        'endpoint_head_omitted_optimistically':True}
        elif bound.get('kind') == 'endpoint_pair_plus_individual_maximum_contiguous_counts':
            possible_counts = [total if monolithic else 0]
            for i in range(len(regular)):
                for j in range(len(regular)):
                    if i != j and firsts[i] > 0 and lasts[j] > 0:
                        possible_counts.append(firsts[i]+lasts[j]+sum(regular[k] for k in range(len(regular)) if k != i and k != j))
            maximum=max(possible_counts)
            expected = {'kind':'endpoint_pair_plus_individual_maximum_contiguous_counts',
                        'per_device_max_layers':regular,'per_device_first_prefix_layers':firsts,
                        'per_device_last_suffix_layers':lasts,'monolithic_possible':monolithic,
                        'maximum_coverable_layers':maximum,'required_layers':total,
                        'endpoint_devices_distinct':True}
        else:
            raise ValueError('unsupported bound kind')
        if not _identical(bound,expected): raise ValueError('bound arithmetic mismatch')
        if maximum >= total: raise ValueError('bound does not reject the scenario')
        return {'valid':True,'errors':[], 'maximum_coverable_layers':maximum,
                'required_layers':total,'scenario_sha256':digest}
    except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError) as exc:
        return {'valid':False,'errors':[str(exc) or 'malformed input or bound']}


def verify_bundle(bundle):
    """Validate every supplied witness/bound; do not certify search-only negatives."""
    errors, witnesses, bounds, search_only = [], 0, 0, []
    try:
        if type(bundle) is not dict:
            raise ValueError('bundle must be a dictionary')
        if (bundle['evidence_kind'] != 'analytical_not_measurement' or
                type(bundle['model_version']) is not int or bundle['model_version'] != 1):
            raise ValueError('unsupported bundle model/evidence label')
        if type(bundle['cases']) is not list:
            raise ValueError('cases must be a list')
        for case in bundle['cases']:
            if type(case) is not dict or type(case.get('baselines',{})) is not dict:
                raise ValueError('case and baselines must be dictionaries')
            source=case['scenario']; _source(source)
            outcomes=[('solver',case['solver'])]+[(k,v) for k,v in case.get('baselines',{}).items() if k!='aggregate']
            for outcome_index,(name,outcome) in enumerate(outcomes):
                if type(outcome) is not dict:
                    raise ValueError('outcome must be a dictionary')
                allowed_statuses = ({'feasible','infeasible','unknown','search_incomplete'}
                                    if outcome_index == 0 else {'feasible','not_found'})
                if type(outcome.get('status')) is not str or outcome['status'] not in allowed_statuses:
                    raise ValueError(f'{name}: unrecognized outcome status')
                if outcome.get('status')=='feasible' and 'certificate' not in outcome:
                    errors.append(f"{case['id']}:{name}: feasible claim lacks certificate")
                if 'certificate' in outcome:
                    checked=check_certificate(source,outcome['certificate'])
                    if not checked['valid']: errors.append(f"{case['id']}:{name}: {checked['errors']}")
                    witnesses+=1
            result=case['solver']
            if result.get('rejection_bound') is not None:
                checked=check_rejection_bound(source,result['rejection_bound'])
                if not checked['valid']: errors.append(f"{case['id']}:bound: {checked['errors']}")
                bounds+=1
            elif result['status']=='infeasible':
                search_only.append(case['id'])
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        errors.append(str(exc))
    return {'valid':not errors,'errors':errors,'witnesses_checked':witnesses,
            'rejection_bounds_checked':bounds,'search_only_negatives_not_certified':search_only}


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as source:
        report=verify_bundle(json.load(source))
    print(json.dumps(report,indent=2,sort_keys=True))
    raise SystemExit(0 if report['valid'] else 1)
