# Deadline-aware cycle-capacity bound for household inference

Local analytical review, 2026-10-24. No publication, device use, model execution, API purchase, credentials, or changes to earlier latency-publication artifacts.

## Finding

Aggregate memory, even aggregate memory inside a strongly connected logical network, can overstate usable layer capacity by an arbitrarily large factor. In the published one-contiguous-stage-per-device model, interstage transfers plus final-to-first feedback form a directed simple cycle. Every such cycle lies within one strongly connected component and one vertex-biconnected block of the underlying undirected logical transfer graph. Memory scattered across separate blocks cannot all contribute to a single permitted placement.

A target token deadline strengthens the result: prune transfers which individually cannot fit after reserving a lower bound for compute and the other mandatory transfers, then apply the block-capacity bound. This provides a joint memory/connectivity/serial-communication rejection test, not just a RAM test.

This is a new repository-specific analytical derivation in this review, not a claim to invent graph blocks, heterogeneous inference, or latency-aware partitioning. It is not a calibrated claim about household hardware.

## Source and assumptions

Inspected MODEL.md at immutable repository commit cf6f52f44ce70514cd1e9cdb011b3aed4f53b4c5:
https://github.com/EastonSell/household-compute/blob/cf6f52f44ce70514cd1e9cdb011b3aed4f53b4c5/studies/bounded-placement-certificates/docs/MODEL.md

The retrieved Git blob is e5fb1022639b7db3d2e1f182d7e2cc6427c753e5; SOURCE_MODEL.md preserves its content. The published model states contiguous layer intervals, each device at most once, a directed link for each interstage and last-to-first transfer, and nonnegative serial compute/communication costs. It already has memory count bounds ignoring topology, and correctly distinguishes stage-count from latency optimality. This review extends the former; it does not reopen earlier publication work.

All graphs here describe ALLOWED END-TO-END LOGICAL TRANSFERS, not physical cable/Wi-Fi/router edges. A router or relay can forward messages without becoming a compute stage. If leaf-to-leaf routed traffic is allowed, represent it as a logical leaf-to-leaf edge with justified end-to-end cost; do not apply a physical star's graph directly. Consent is an external prerequisite, not established by arithmetic or connectivity. Only explicitly authorized devices/data-transfer relationships may be used operationally. To prove impossibility under unresolved facts, admit unknown eligibility/availability optimistically, and use zero for unknown nonnegative timing lower bounds, never as a performance estimate.

The theorem does not cover a runtime that revisits a device for multiple disjoint intervals, a different controller topology, absent feedback, dynamic offloading, replication, tensor parallelism, or speculative decoding without a new model. Changing any of those can remove the obstruction.

## Theorem 1: cycle-local memory capacity

Let L be the number of layers. Let q_d be the published optimistic maximum count of consecutive layers fitting device d, after its local and per-stage reserves and workspace, but ignoring endpoint/controller costs. These are counts, not aggregate bytes: heterogeneous sizes, unsupported layers, and indivisibility remain relevant.

Construct G from all logically possible directed transfers, ignoring payload-role distinctions if needed as an optimistic union. Let S range over G's strongly connected components. Let B range over vertex-biconnected blocks of G's undirected shadow, including an ordinary edge as a two-vertex block. Define R = S intersect B.

For a multi-device placement, covered layers are at most:

    U = max_R sum_{d in R} q_d.

Define a maximum over no multi-device regions as zero. Self-loops do not count as multi-stage cycles. Handle each possible monolithic placement separately. A convenient looser all-placement bound is max(U, max_d q_d). If it is strictly below L, no permitted placement exists.

Proof: the distinct stage devices and feedback form a directed cycle. For two devices the undirected shadow is a dyad. For three or more it contains an undirected simple cycle, which is biconnected because deleting any vertex leaves a connected path. Hence its vertices lie in a single maximal block; their directed cycle also places all in one SCC. Each device contributes at most q_d, so their sum is at most the region sum. Endpoints, interval compatibility, and edge directions discarded by the bound can only increase this upper bound. The rejection is sound but passing is not sufficient. For example, a bidirected K2,3 graph with five unit-capacity devices has region bound 5 but no five-device simple cycle: bipartite cycles have even length, so five unit-sized layers cannot be placed in this model.

The published endpoint-pair refinement can be restricted to each region:

    U_endpoint = max_{R, d != e in R} [ f_d + h_e + sum_{j in R minus {d,e}} q_j ].

It is safe to exclude pairs whose e-to-d feedback is explicitly unavailable. Check monoliths separately with full endpoint accounting. This refined formula is proved by the same argument but is NOT implemented or exhaustively tested by this review's small checker.

## Theorem 2: deadline-filtered graph

Assume no monolith can fit. Sort q in descending order, and let k_min be the smallest prefix count whose sum reaches L. If no such prefix exists, memory already rejects. Every placement uses at least k_min devices and therefore exactly k >= k_min transfers including feedback.

Let C0 be a certified nonnegative lower bound on the sum of compute costs for one token. Let e0 be a common nonnegative lower bound on every possible activation OR feedback transfer. Both may be zero when unknown. For a candidate token deadline T, retain a logical arc if at least one permitted payload role has transfer lower bound at most:

    theta(T) = T - C0 - (k_min - 1) e0.

Use an optimistic union of payload roles; keep unknown timing with lower bound zero. If theta is negative, no multi-stage placement can meet the deadline. Recompute the SCC/block bound U(T) on this filtered graph. If U(T) < L, no placement in the declared model can meet T.

Proof: in a placement meeting T, any one transfer leaves k-1 other transfers, each costing at least e0, and total compute at least C0. Its cost is therefore at most theta(T). Every actual cycle arc survives the filter. Apply Theorem 1. Using lower bounds or an optimistic union only admits extra arcs, so cannot create an erroneous rejection.

Do not mistake one-way propagation latency for pipeline resource service time. This theorem concerns one autoregressive stream's serial token deadline. It does not prove aggregate batched/pipelined throughput. Unknown compute in the published 70B surrogate prevents a complete numerical inference-speed estimate.

## Exact synthetic examples

All layer units and time units below are illustrative, not bytes, milliseconds, or a measured model. Eight identical unit-sized layers; device capacities [4,1,2,3]; zero reserves/endpoints; compute 1 unit/layer, hence C0=8. Devices are all consented and supported by stipulation only.

1. Restricted logical star: device 0 exchanges directly with every other device, but no leaf-to-leaf logical transfer is permitted. Each transfer costs 1. Total layer capacity is 10 and the whole graph is strongly connected. Each block is a hub-leaf pair, so usable cycle capacity is only 4+3=7. Eight layers cannot be placed. Adding arbitrarily many leaves each of capacity at most 3 increases aggregate memory without raising this bound. It does not make the optimum worse: it simply fails to improve it.

2. Complete logical graph with expensive leaf-to-leaf transfers: spoke transfers cost 1; all leaf-to-leaf transfers cost 100. At least three devices are required, because the two largest capacities total 7. Any cycle with at least three vertices must have a leaf-to-leaf transfer. The exact optimum is 8+1+100+1=110, attained with capacities 4,3,1. For T=109, theta=109-8-2=99, so all expensive edges are pruned; the block bound is 7 and rejects. At T=110, theta=100 and the full graph returns. The independent exact oracle finds a feasible optimum of 110. Thus the deadline filter is sharp on this example, while aggregate memory and unfiltered connectivity do not reject.

3. Relay-capable physical star: if leaf-to-leaf communication may route through the hub at modeled cost 2, add those logical edges. The exact optimum becomes 8+1+2+1=12. The graph obstruction disappears. This is why the result must not be represented as 'a home router cannot combine more than two computers.' The hub's forwarding role is distinct from ownership of a compute interval; shared-link contention would require additional resource accounting.

## Verification

Run python check.py. It uses only itertools, fractions, json, and pathlib. It independently enumerates all directed graphs on four labeled vertices (4096), all simple directed cycles, and maximal undirected blocks by subset/removal checks. For each graph and three heterogeneous capacity vectors, it checks that the cycle-local capacity upper bound dominates every cycle's capacity and the monolithic alternative: 12,288 exact bound checks. It also checks 113 integer deadlines (8 through 120), the exact 109/110 boundary, the relay alternative, and five growing-star instances. RESULTS.json contains reproducible exact outputs.

These are finite tests of the small implementation, not a machine-checked universal theorem, a production certificate integration, or a rerun of the repository's full test suite. The mathematical proofs above carry the general claim. The endpoint-pair extension is not in the implemented test scope. No existing solver, result grid, or publication branch was modified.

## Prior art and claim boundary

- EdgePipe, Hu et al., 2021, already optimizes heterogeneous inference partitioning with compute, memory and network bandwidth: https://arxiv.org/abs/2110.14895 . This review does not claim that combining those constraints is new.
- Petals, Borzunov et al., 2023, studies uneven hardware, dynamic participation, fault tolerance and distributed LLM throughput: https://arxiv.org/abs/2312.08361 . This review does not establish novelty over its routing or load-balancing algorithms.
- Biconnected blocks and articulation-point algorithms are classical. NetworkX's official documentation defines the relevant blocks, includes the dyad convention, and cites Hopcroft and Tarjan (1973): https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.biconnected_components.html . The cycle-containment argument is elementary graph theory, not a new theorem of general graph theory.

A defensible contribution is a small, auditable rejection certificate specialized to this repository's one-use-per-device/feedback-cycle model, with a sharp synthetic memory-plus-deadline counterexample and an explicit routed-network control. No exhaustive literature novelty search has been performed. A next review could implement this as an optional topology-aware negative certificate and independently reconstruct its graph, capacities and bounds. That is proposed work, not completed or authorized publication.
