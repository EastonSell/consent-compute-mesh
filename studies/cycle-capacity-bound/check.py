"""Exact synthetic graph-bound checks; standard library only; no hardware calls."""
from itertools import combinations, permutations
from fractions import Fraction
import json
from pathlib import Path


def connected(vertices, arcs):
    if not vertices:
        return True
    seen = {next(iter(vertices))}
    while True:
        nxt = seen | {v for u in seen for v in vertices if (u,v) in arcs or (v,u) in arcs}
        if nxt == seen:
            return seen == set(vertices)
        seen = nxt


def blocks(n, arcs):
    # Tiny independent exhaustive block oracle, not a production algorithm.
    good = []
    for k in range(2,n+1):
        for sub in combinations(range(n),k):
            s = frozenset(sub)
            if connected(s,arcs) and all(connected(s-{v},arcs) for v in s):
                good.append(s)
    return [s for s in good if not any(s < t for t in good)]


def strong_components(n, arcs):
    reach = [[i == j or (i,j) in arcs for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] |= reach[i][k] and reach[k][j]
    return set(frozenset(j for j in range(n) if reach[i][j] and reach[j][i]) for i in range(n))


def capacity_bound(q, arcs):
    regions = [b & s for b in blocks(len(q),arcs) for s in strong_components(len(q),arcs)]
    return max([max(q, default=0)] + [sum(q[i] for i in r) for r in regions])


def cycles(n, arcs):
    for k in range(2,n+1):
        for p in permutations(range(n),k):
            if p[0] != min(p):
                continue
            if all((p[i],p[(i+1)%k]) in arcs for i in range(k)):
                yield p


def exact_best(q, costs, layers, compute):
    if max(q) >= layers:
        return compute
    candidates = [compute + sum(costs[p[i],p[(i+1)%len(p)]] for i in range(len(p)))
                  for p in cycles(len(q),costs) if sum(q[i] for i in p) >= layers and len(p) <= layers]
    return min(candidates,default=None)


def deadline_bound(q, costs, layers, compute, deadline):
    total = 0
    kmin = None
    for k, capacity in enumerate(sorted(q,reverse=True),1):
        total += capacity
        if total >= layers:
            kmin = k
            break
    if kmin is None:
        return max(q), None
    if kmin == 1:  # Monolith handled separately; no mandatory transfer.
        return sum(q), None
    emin = min(costs.values(), default=Fraction(0))
    limit = deadline - compute - (kmin-1)*emin
    retained = {arc for arc,cost in costs.items() if cost <= limit}
    return capacity_bound(q,retained), limit


def run():
    n=4
    possible=[(i,j) for i in range(n) for j in range(n) if i!=j]
    vectors=[(1,1,1,1),(4,1,2,3),(0,1,2,0)]
    checks=0
    for mask in range(1<<len(possible)):
        arcs={e for j,e in enumerate(possible) if mask>>j&1}
        cs=list(cycles(n,arcs))
        for q in vectors:
            exact=max([max(q)]+[sum(q[i] for i in c) for c in cs])
            assert exact <= capacity_bound(q,arcs), (mask,q,exact)
            checks+=1
    q=(4,1,2,3); layers=8; compute=Fraction(8)
    star={(i,j):Fraction(1) for i,j in possible if i==0 or j==0}
    complete={e:star.get(e,Fraction(100)) for e in possible}
    routed={e:star.get(e,Fraction(2)) for e in possible}
    assert sum(q)==10 and len(strong_components(4,star))==1
    assert capacity_bound(q,star)==7
    assert exact_best(q,star,layers,compute) is None
    assert exact_best(q,complete,layers,compute)==110
    assert exact_best(q,routed,layers,compute)==12
    rows=[]
    for deadline in range(8,121):
        bound,limit=deadline_bound(q,complete,layers,compute,Fraction(deadline))
        best=exact_best(q,complete,layers,compute)
        if bound<layers:
            assert best>deadline
        rows.append(dict(deadline=deadline,capacity_bound=bound,edge_limit=str(limit)))
    assert rows[109-8]['capacity_bound']==7
    assert rows[110-8]['capacity_bound']==10
    # Extra leaves with <=3 capacity cannot improve the star's bound of 7.
    for count in range(2,7):
        qq=(4,)+(3,)*count
        aa={(i,j) for i in range(count+1) for j in range(count+1) if i!=j and (i==0 or j==0)}
        assert capacity_bound(qq,aa)==7
    return dict(evidence_kind='synthetic_exact_arithmetic_not_measurement',
                directed_graphs=4096,capacity_vectors=3,exhaustive_bound_checks=checks,
                star=dict(capacities=list(q),aggregate=10,scc_count=1,cycle_capacity_bound=7,required_layers=8,feasible=False),
                complete_graph=dict(exact_optimal_token_time=110,units='synthetic time units',
                                    deadline_109_bound=7,deadline_110_bound=10),
                relay_closure=dict(exact_optimal_token_time=12,feasible=True),deadline_sweep=rows)

if __name__=='__main__':
    result=run()
    Path(__file__).with_name('RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='deadline_sweep'},indent=2))
