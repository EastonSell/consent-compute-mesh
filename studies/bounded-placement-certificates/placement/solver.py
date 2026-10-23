"""Exact bounded resident contiguous-chain arithmetic, Python stdlib only.

A feasible status certifies supplied memory/connectivity assumptions, not deployment.
Unknown inputs are never silently credited. Search limits yield incomplete, not UNSAT.
"""
from fractions import Fraction
from hashlib import sha256
import json

MAX_DEVICES = 18
MAX_LAYERS = 128


def _integer(x, name, *, positive=False, nullable=False):
    if nullable and x is None:
        return
    if type(x) is not int or x < (1 if positive else 0):
        raise ValueError(f'{name} must be an exact {"positive" if positive else "nonnegative"} integer')


def _tri(x, name):
    if x is not None and type(x) is not bool:
        raise ValueError(f'{name} must be true, false or null')


def _validate_scenario(s):
    if not isinstance(s, dict) or type(s.get('version')) is not int or s['version'] != 1:
        raise ValueError('scenario version must be 1')
    layers, devices = s.get('layers'), s.get('devices')
    if not isinstance(layers, list) or not 1 <= len(layers) <= MAX_LAYERS:
        raise ValueError('require 1..128 layers')
    if not isinstance(devices, list) or not 1 <= len(devices) <= MAX_DEVICES:
        raise ValueError('require 1..18 devices')
    for l in layers:
        for k in ('weight_bytes', 'kv_bytes', 'workspace_bytes'):
            _integer(l[k], k)
        _integer(l['compute_ns'], 'compute_ns', nullable=True)
    ids = []
    for d in devices:
        if not isinstance(d['id'], str) or not d['id'] or d['id'] in ids:
            raise ValueError('device IDs must be nonempty and unique')
        ids.append(d['id'])
        _integer(d['budget_bytes'], 'budget_bytes', nullable=True)
        _integer(d['runtime_reserve_bytes'], 'runtime_reserve_bytes')
        for k in ('eligible', 'embedding_support', 'head_support'):
            _tri(d[k], k)
        if not isinstance(d['layer_support'], list) or len(d['layer_support']) != len(layers):
            raise ValueError('one support flag per layer required')
        for x in d['layer_support']:
            _tri(x, 'layer_support')
    for k in ('embedding_bytes', 'head_bytes', 'embedding_workspace_bytes',
              'head_workspace_bytes', 'controller_reserve_bytes', 'activation_reserve_bytes',
              'activation_payload_bytes', 'feedback_payload_bytes'):
        _integer(s[k], k)
    for k in ('embedding_compute_ns', 'head_compute_ns'):
        _integer(s[k], k, nullable=True)
    if type(s['tied_embeddings']) is not bool:
        raise ValueError('tied_embeddings must be boolean')
    if s['tied_embeddings'] and s['embedding_bytes'] != s['head_bytes']:
        raise ValueError('tied embedding/head byte quantities must agree')
    seen = set()
    for e in s['links']:
        pair = (e['source'], e['target'])
        if pair in seen or pair[0] not in ids or pair[1] not in ids or pair[0] == pair[1]:
            raise ValueError('links require distinct known IDs and unique directed pairs')
        seen.add(pair)
        _tri(e['available'], 'link available')
        _integer(e['latency_ns'], 'latency_ns', nullable=True)
        _integer(e['payload_bps'], 'payload_bps', positive=True, nullable=True)
    # Also reject noncanonical/unserializable annotations, NaN, etc.
    try:
        json.dumps(s, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('scenario must be canonical JSON data') from exc


def validate_scenario(s):
    try:
        _validate_scenario(s)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('malformed or incomplete scenario schema') from exc


def scenario_hash(s):
    return sha256(json.dumps(s, sort_keys=True, separators=(',', ':'),
                            allow_nan=False).encode()).hexdigest()


def stage_accounting(s, device, start, end):
    d, layers = s['devices'][device], s['layers'][start:end]
    first, last = start == 0, end == len(s['layers'])
    endpoint = (s['embedding_bytes'] if first else 0) + (s['head_bytes'] if last else 0)
    if first and last and s['tied_embeddings']:
        endpoint = s['embedding_bytes']
    workspace = max([l['workspace_bytes'] for l in layers] +
                    ([s['embedding_workspace_bytes']] if first else []) +
                    ([s['head_workspace_bytes']] if last else []))
    out = dict(weight_bytes=sum(l['weight_bytes'] for l in layers),
               kv_bytes=sum(l['kv_bytes'] for l in layers), endpoint_bytes=endpoint,
               runtime_reserve_bytes=d['runtime_reserve_bytes'],
               activation_reserve_bytes=s['activation_reserve_bytes'],
               workspace_bytes=workspace,
               controller_reserve_bytes=s['controller_reserve_bytes'] if first else 0)
    out['total_bytes'] = sum(out.values())
    return out


def _ratio(x):
    return dict(numerator=x.numerator, denominator=x.denominator)


def build_certificate(s, placement):
    """Produce claimed accounting; independent checker must accept it before use."""
    validate_scenario(s)
    stages = [dict(device_id=s['devices'][d]['id'], start=a, end=b,
                   memory=stage_accounting(s, d, a, b)) for d, a, b in placement]
    links = {(e['source'], e['target']): e for e in s['links']}
    transfers, unknown, comm_lb = [], [], Fraction(0)
    edges = [(x['device_id'], y['device_id'], 'activation', s['activation_payload_bytes'])
             for x, y in zip(stages, stages[1:])]
    if len(stages) > 1:
        edges.append((stages[-1]['device_id'], stages[0]['device_id'],
                      'feedback', s['feedback_payload_bytes']))
    for a, b, kind, size in edges:
        e = links.get((a, b))
        lb, missing = Fraction(0), []
        if e is None or e['available'] is not True:
            missing.append(f'{kind}:{a}->{b}:availability')
        if e is None or e['latency_ns'] is None:
            missing.append(f'{kind}:{a}->{b}:latency_ns')
        else:
            lb += e['latency_ns']
        if e is None or e['payload_bps'] is None:
            missing.append(f'{kind}:{a}->{b}:payload_bps')
        else:
            lb += Fraction(size * 8 * 10**9, e['payload_bps'])
        comm_lb += lb
        unknown.extend(missing)
        transfers.append(dict(source=a, target=b, kind=kind, payload_bytes=size,
                              duration_ns=None if missing else _ratio(lb),
                              known_lower_bound_ns=_ratio(lb)))
    comm_unknown = bool(unknown)
    compute_lb = Fraction(0)
    compute_unknown = False
    for i, l in enumerate(s['layers']):
        if l['compute_ns'] is None:
            compute_unknown = True
            unknown.append(f'layer:{i}:compute_ns')
        else:
            compute_lb += l['compute_ns']
    for name in ('embedding_compute_ns', 'head_compute_ns'):
        if s[name] is None:
            compute_unknown = True
            unknown.append(name)
        else:
            compute_lb += s[name]
    return dict(version=1, scenario_sha256=scenario_hash(s), stages=stages,
                transfers=transfers, timing=dict(
                    communication_ns=None if comm_unknown else _ratio(comm_lb),
                    compute_ns=None if compute_unknown else _ratio(compute_lb),
                    critical_path_ns=None if unknown else _ratio(comm_lb + compute_lb),
                    communication_lower_bound_ns=_ratio(comm_lb),
                    critical_path_lower_bound_ns=_ratio(comm_lb + compute_lb),
                    unknown_terms=unknown))


class _Capped(Exception):
    pass


def _search(s, relaxed, max_states, max_transitions):
    ds, layers = s['devices'], s['layers']
    n, total = len(ds), len(layers)
    edges = {(e['source'], e['target']): e for e in s['links']}
    allowed = lambda x: x is True or (relaxed and x is None)
    def route(a, b):
        e = edges.get((ds[a]['id'], ds[b]['id']))
        return e is not None and allowed(e['available'])
    def fit(d, a, b):
        x = ds[d]
        return (allowed(x['eligible']) and
                (x['budget_bytes'] is not None or relaxed) and
                all(allowed(v) for v in x['layer_support'][a:b]) and
                (a != 0 or allowed(x['embedding_support'])) and
                (b != total or allowed(x['head_support'])) and
                (x['budget_bytes'] is None or
                 stage_accounting(s, d, a, b)['total_bytes'] <= x['budget_bytes']))
    # An optimistic per-device bound: omit endpoint/controller bytes/support.
    max_counts = []
    for d, x in enumerate(ds):
        best = 0
        if allowed(x['eligible']) and (x['budget_bytes'] is not None or relaxed):
            for a in range(total):
                resident, workspace = 0, 0
                for b in range(a, total):
                    if not allowed(x['layer_support'][b]):
                        break
                    resident += layers[b]['weight_bytes'] + layers[b]['kv_bytes']
                    workspace = max(workspace, layers[b]['workspace_bytes'])
                    cost = resident + workspace + x['runtime_reserve_bytes'] + s['activation_reserve_bytes']
                    if x['budget_bytes'] is not None and cost > x['budget_bytes']:
                        break
                    best = max(best, b - a + 1)
        max_counts.append(best)
    # Any plan has a first stage. Bound its true prefix, omit output-head costs
    # on all stages to remain optimistic (including a putative full first stage).
    first_counts = []
    for d, x in enumerate(ds):
        best = 0
        if allowed(x['eligible']) and allowed(x['embedding_support']) and (x['budget_bytes'] is not None or relaxed):
            resident, workspace = 0, s['embedding_workspace_bytes']
            for b in range(total):
                if not allowed(x['layer_support'][b]): break
                resident += layers[b]['weight_bytes'] + layers[b]['kv_bytes']
                workspace = max(workspace, layers[b]['workspace_bytes'])
                cost = resident + workspace + x['runtime_reserve_bytes'] + s['activation_reserve_bytes'] + s['controller_reserve_bytes'] + s['embedding_bytes']
                if x['budget_bytes'] is not None and cost > x['budget_bytes']: break
                best = b + 1
        first_counts.append(best)
    max_cover = max([first_counts[d] + sum(max_counts) - max_counts[d]
                     for d in range(n) if first_counts[d]] or [0])
    bound = dict(kind='first_prefix_plus_individual_maximum_contiguous_counts',
                 per_device_max_layers=max_counts, per_device_first_prefix_layers=first_counts,
                 maximum_coverable_layers=max_cover, required_layers=total,
                 endpoint_head_omitted_optimistically=True)
    receipt = dict(states=0, transitions=0, state_limit=max_states, transition_limit=max_transitions,
                   relaxation=relaxed, complete=False)
    if max_cover < total:
        receipt['complete'] = True
        return None, receipt, bound
    # Stronger post-result proof refinement: first and last are distinct devices
    # unless the entire model has a valid single-device placement. Counts on all
    # other devices remain optimistic and may refer to overlapping intervals.
    last_counts = []
    for d, x in enumerate(ds):
        best = 0
        if allowed(x['eligible']) and allowed(x['head_support']) and (x['budget_bytes'] is not None or relaxed):
            resident, workspace = 0, s['head_workspace_bytes']
            for a in range(total-1, -1, -1):
                if not allowed(x['layer_support'][a]): break
                resident += layers[a]['weight_bytes'] + layers[a]['kv_bytes']
                workspace = max(workspace, layers[a]['workspace_bytes'])
                cost = resident + workspace + x['runtime_reserve_bytes'] + s['activation_reserve_bytes'] + s['head_bytes']
                if x['budget_bytes'] is not None and cost > x['budget_bytes']: break
                best = total-a
        last_counts.append(best)
    monolithic = any(fit(d, 0, total) for d in range(n))
    pair_max = max([first_counts[a] + last_counts[b] + sum(max_counts[d] for d in range(n) if d not in (a,b))
                    for a in range(n) for b in range(n) if a != b and first_counts[a] and last_counts[b]] or [0])
    endpoint_cover = max(total if monolithic else 0, pair_max)
    endpoint_bound = dict(kind='endpoint_pair_plus_individual_maximum_contiguous_counts',
                          per_device_max_layers=max_counts, per_device_first_prefix_layers=first_counts,
                          per_device_last_suffix_layers=last_counts, monolithic_possible=monolithic,
                          maximum_coverable_layers=endpoint_cover, required_layers=total,
                          endpoint_devices_distinct=True)
    if endpoint_cover < total:
        receipt['complete'] = True
        return None, receipt, endpoint_bound
    # On a complete availability graph, identical capability/memory devices are
    # interchangeable for stage count. Timing is deliberately not the objective.
    complete_graph = all(route(a,b) for a in range(n) for b in range(n) if a != b)
    signatures = [json.dumps({k:v for k,v in x.items() if k not in ('id', 'home', 'label')}, sort_keys=True)
                  for x in ds]
    found = None
    try:
        for target in range(1, min(n,total) + 1):
            failed = set()
            def dfs(pos, mask, prev, first, left):
                key = (pos, mask, prev, first, left)
                if key in failed: return None
                if receipt['states'] >= max_states: raise _Capped
                receipt['states'] += 1
                available = [d for d in range(n) if not (mask >> d) & 1]
                if len(available) < left or total-pos < left or sum(sorted((max_counts[d] for d in available), reverse=True)[:left]) < total-pos:
                    failed.add(key); return None
                seen_signatures = set()
                for d in available:
                    if not max_counts[d]: continue
                    if complete_graph:
                        if signatures[d] in seen_signatures: continue
                        seen_signatures.add(signatures[d])
                    if prev != -1 and not route(prev,d): continue
                    if left == 1 and first != -1 and not route(d,first): continue
                    ends = [total] if left == 1 else range(total-left+1, pos, -1)
                    for end in ends:
                        if receipt['transitions'] >= max_transitions: raise _Capped
                        receipt['transitions'] += 1
                        if end <= pos or not fit(d,pos,end): continue
                        if left == 1:
                            return [(d,pos,end)]
                        result = dfs(end, mask | (1 << d), d, d if first == -1 else first, left-1)
                        if result is not None: return [(d,pos,end)] + result
                failed.add(key)
                return None
            found = dfs(0,0,-1,-1,target)
            if found is not None:
                receipt['complete'] = True
                receipt['minimum_stage_count'] = target
                return found, receipt, bound
        receipt['complete'] = True
    except _Capped:
        pass
    return None, receipt, bound


def solve(s, max_states=250_000, max_transitions=5_000_000):
    validate_scenario(s)
    _integer(max_states, 'max_states'); _integer(max_transitions, 'max_transitions')
    if max_states > 250_000 or max_transitions > 5_000_000:
        raise ValueError('limits may be lowered, not raised beyond frozen resource caps')
    known, receipt, bound = _search(s, False, max_states, max_transitions)
    if known is not None:
        return dict(status='feasible', certificate=build_certificate(s, known),
                    minimum_stage_count_proven=True, search=[receipt])
    if not receipt['complete']:
        return dict(status='search_incomplete', minimum_stage_count_proven=False, search=[receipt])
    relaxed, second, relaxed_bound = _search(s, True, max_states, max_transitions)
    if relaxed is not None:
        return dict(status='unknown', optimistic_witness=[dict(device_id=s['devices'][d]['id'], start=a, end=b)
                                                        for d,a,b in relaxed],
                    minimum_stage_count_proven=False, search=[receipt,second])
    if not second['complete']:
        return dict(status='search_incomplete', minimum_stage_count_proven=False, search=[receipt,second])
    return dict(status='infeasible', minimum_stage_count_proven=False, search=[receipt,second],
                rejection_bound=relaxed_bound if relaxed_bound['maximum_coverable_layers'] < len(s['layers']) else None,
                reason='necessary_bound' if relaxed_bound['maximum_coverable_layers'] < len(s['layers']) else 'complete_finite_search')
