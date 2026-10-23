"""Independent exhaustive permutation/cut oracle, for small fully known cases.

Uses no solver, checker, fit routine, pruning, symmetry reduction or memoization.
"""
from itertools import permutations, combinations


def brute_min_stages(s):
    length = len(s['layers'])
    graph = {(e['source'],e['target']) for e in s['links'] if e['available'] is True}
    for stage_count in range(1,min(length,len(s['devices']))+1):
        for order in permutations(s['devices'],stage_count):
            names = [d['id'] for d in order]
            if stage_count > 1 and any((names[i],names[(i+1)%stage_count]) not in graph for i in range(stage_count)):
                continue
            for cuts in combinations(range(1,length),stage_count-1):
                boundaries = (0,)+cuts+(length,)
                good = True
                for j, d in enumerate(order):
                    a,b = boundaries[j],boundaries[j+1]
                    if d['eligible'] is not True or d['budget_bytes'] is None:
                        good=False; break
                    if any(x is not True for x in d['layer_support'][a:b]):
                        good=False; break
                    if (j == 0 and d['embedding_support'] is not True) or (j == stage_count-1 and d['head_support'] is not True):
                        good=False; break
                    footprint = d['runtime_reserve_bytes'] + s['activation_reserve_bytes']
                    temporary = 0
                    for i in range(a,b):
                        footprint += s['layers'][i]['weight_bytes']
                        footprint += s['layers'][i]['kv_bytes']
                        temporary = max(temporary,s['layers'][i]['workspace_bytes'])
                    if j == 0:
                        footprint += s['embedding_bytes'] + s['controller_reserve_bytes']
                        temporary = max(temporary,s['embedding_workspace_bytes'])
                    if j == stage_count-1:
                        if stage_count > 1 or not s['tied_embeddings']:
                            footprint += s['head_bytes']
                        temporary = max(temporary,s['head_workspace_bytes'])
                    if footprint+temporary > d['budget_bytes']:
                        good=False; break
                if good: return stage_count
    return None
