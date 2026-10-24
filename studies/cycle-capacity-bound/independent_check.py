"""Independent reviewer: nonempty interval placement oracle; no author cycle oracle."""
import importlib.util
from pathlib import Path
from itertools import permutations, combinations
import json
P=Path(__file__).parent
s=importlib.util.spec_from_file_location('subject',P/'check.py');subject=importlib.util.module_from_spec(s);s.loader.exec_module(subject)

def placements(q, costs, layers):
    for n in range(1,min(len(q),layers)+1):
        for order in permutations(range(len(q)),n):
            if n>1 and any((order[i-1],order[i]) not in costs for i in range(n)):
                continue
            for cuts in combinations(range(1,layers),n-1):
                bounds=(0,)+cuts+(layers,)
                if all(bounds[i+1]-bounds[i]<=q[d] for i,d in enumerate(order)):
                    yield order,bounds,0 if n==1 else sum(costs[order[i-1],order[i]] for i in range(n))

def run():
    q=(4,1,2,3);arcs=[(i,j) for i in range(4) for j in range(4) if i!=j]
    optima={}
    witnesses={}
    for name,leafcost in [('restricted',None),('expensive',100),('routed',2)]:
        costs={(i,j):1 if 0 in (i,j) else leafcost for i,j in arcs if 0 in (i,j) or leafcost is not None}
        rows=list(placements(q,costs,8))
        optimum=min((8+r[2] for r in rows),default=None)
        optima[name]=optimum
        witnesses[name]=next((r for r in rows if 8+r[2]==optimum),None)
    assert optima=={'restricted':None,'expensive':110,'routed':12}
    # Every subset of self-loops on every four-node digraph leaves the bound
    # unchanged. Monolithic placement needs no remote feedback in source model.
    loop_checks=0
    for mask in range(4096):
        aa={e for i,e in enumerate(arcs) if mask>>i&1}
        original=subject.capacity_bound(q,aa)
        for loops in range(16):
            assert subject.capacity_bound(q,aa|{(i,i) for i in range(4) if loops>>i&1})==original
            loop_checks+=1
    # One-way dyad cannot be a two-stage placement; reverse arc makes it valid.
    assert not list(placements((1,1),{(0,1):1},2))
    assert list(placements((1,1),{(0,1):1,(1,0):1},2))
    assert subject.capacity_bound((1,1),{(0,1)})==1
    assert subject.capacity_bound((1,1),{(0,1),(1,0)})==2
    # Bidirected K2,3 is strongly connected and one undirected block but has no
    # 5-cycle. All capacities 1: the region bound passes while placement fails.
    bip={(i,j):1 for i in range(5) for j in range(5) if (i<2)!=(j<2)}
    assert subject.capacity_bound((1,)*5,bip)==5
    assert not list(placements((1,)*5,bip,5))
    # Broad tiny deadline falsification across all digraphs, varied capacities,
    # heterogeneous nonnegative edge costs, and certified compute lower bound.
    deadline_checks=0
    for mask in range(4096):
        costs={e:(3*e[0]+e[1])%5 for i,e in enumerate(arcs) if mask>>i&1}
        for caps in [(1,1,1,1),(0,1,2,0),(2,1,2,1)]:
            layers=3
            rows=list(placements(caps,costs,layers))
            actual=min((6+r[2] for r in rows),default=None)
            for deadline in range(16):
                bound,_=subject.deadline_bound(caps,costs,layers,2,deadline)
                if bound<layers:
                    assert actual is None or actual>deadline,(mask,caps,deadline,actual)
                deadline_checks+=1
    return {'optima':optima,'interval_witnesses':witnesses,'self_loop_invariance_checks':loop_checks,'additional_deadline_checks':deadline_checks,'passing_not_sufficient':'bidirected K2,3, five unit capacities, L=5; bound=5, no placement'}
if __name__=='__main__':
    result=run();(P/'INDEPENDENT_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
