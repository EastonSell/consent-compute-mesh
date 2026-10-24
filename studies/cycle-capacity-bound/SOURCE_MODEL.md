# Model and proof scope

## Exact supplied arithmetic

For layer i, let resident weights be w_i, KV be k_i and sequential workspace be p_i. A stage on device d owns the interval [a,b), with one device appearing at most once in the chain. Let r_d be runtime reserve, A be per-stage activation-buffer reserve, C be first-stage controller reserve, and E/H be embedding/head tensors.

The checked memory is:

    sum_{a <= i < b}(w_i+k_i) + endpoint_copies(a,b)
      + r_d + A + max(p_a,...,p_{b-1}, applicable endpoint workspace)
      + (C if a=0 else 0).

It must fit the supplied simultaneously addressable allocation m_d, and all layer/endpoint eligibility flags must be explicitly true. For tied E=H, a monolithic stage charges one copy; distinct first/last stages each charge a copy. Workspace max assumes sequential nonoverlapping kernels. Buffer simultaneity must be included explicitly in A; no measured peak-memory guarantee is asserted.

Each interstage activation and final-to-first feedback transfer uses its own directed link and:

    latency_ns + payload_bytes * 8 * 10^9 / payload_bps.

These serial costs plus supplied compute_ns form only the declared one-token graph's modeled critical path. An unknown term remains unknown; summing just known nonnegative terms is an explicit lower bound. No missing compute, packet overhead, thermal limit, availability schedule or runtime measurement becomes a zero estimate. Monolithic communication is zero because this model has no remote controller; that assumption is not a general architecture fact.

## Positive witness

A witness is bound to canonical input JSON by SHA256 and lists each interval, device, local component byte count, transfer and timing term. The standalone checker reconstructs coverage, uniqueness, support, budgets, endpoint copies, peak workspace, transfers and exact rational timing directly from the source scenario. It imports no solver or shared accounting implementation. A hash prevents accidental mismatches, not malicious authenticated changes; this is not a cryptographic trust/security protocol.

Accepting a witness establishes the arithmetic constraints for that supplied model. It does not establish the truth of the device budgets, runtime support, consent, model tensor decomposition, quality, performance or other physical assumptions. The checker does not prove minimum stage count or general solver correctness.

## Independently checked negative count bounds

Let q_d be the largest number of consecutive layers fitting device d after local reserve, activation reserve and peak layer workspace, while optimistically ignoring endpoint and controller costs/support. Let f_d be the largest supported prefix fitting with embedding/controller state, optimistically ignoring head state. Let h_d be the largest suffix fitting with head state, optimistically ignoring embedding/controller state. Unknown eligibility/support/budgets are optimistically admitted for these rejection proofs; explicit false remains excluded.

For any actual first device d, its contribution cannot exceed f_d and every other device's contribution cannot exceed q_j. Thus:

    total covered layers <= max_d [ f_d + sum_{j != d} q_j ].

A stronger bound handles every distinct endpoint pair d,e:

    total covered layers <= max_{d != e}
        [ f_d + h_e + sum_{j not in {d,e}} q_j ],

unless a valid monolithic placement exists, which is checked separately with tied/untied endpoint rules. These are optimistic upper bounds: maxima may refer to overlapping or mutually incompatible intervals, and topology is ignored. Those relaxations can prevent rejection, but cannot invalidate a strict shortfall. If the resulting bound is below L, no placement in the declared model exists. A zero candidate-endpoint maximum means no full chain is possible, not that the devices physically hold zero tensors.

The checker recomputes q/f/h independently by exhaustive interval examination. It rejects mismatched/forged bound fields and never treats an inconclusive bound as a proof. The endpoint-pair refinement was added after the first grid, as declared in AMENDMENTS.md. It is a proof-strengthening step, not a prospectively confirmed empirical hypothesis.

## Solver search and its limits

The solver examines target stage counts 1..min(device count,layer count). For a fixed target it recursively enumerates unused devices in input order and longest-to-shortest feasible next intervals. A completed prefix fixes all state relevant to the remaining search: next layer, used-device set, previous device, first device and remaining stage count. Failed states are memoized. An optimistic sum of the largest remaining q_d values prunes only branches unable to cover the remaining layers; nonempty-stage and unused-device count constraints add valid pruning.

On a complete directed availability graph, devices with identical memory/support/runtime/eligibility signatures may be exchanged for stage-count feasibility. The symmetry reduction keeps the first unused representative. Home labels and link metrics do not affect this objective; they do affect the selected path's subsequent timing. If connectivity is not complete, this symmetry reduction is disabled. No latency-optimality claim follows from it.

Without resource exhaustion, the enumeration contains every permitted interval/device assignment up to safe symmetry and pruning. Returning the first result after all lower stage counts fail establishes minimum stage count within this model. A state/transition cap interrupts that argument and returns `search_incomplete`, never `infeasible`. Receipt `complete` means the decision/minimum-count search completed, not that every feasible witness was enumerated.

Known-input search runs first. If it fails completely, the relaxed search grants only unknown eligibility/support/connectivity and unbounded unknown allocations. If even that optimistic problem fails, no completion of those unknowns can fit the model; otherwise status is `unknown`, with a clearly provisional witness. Unknown link metrics/compute do not invalidate an already known memory/connectivity witness, but prevent full timing certification.

Finite oracle tests enumerate permutations and cut combinations independently, without solver fit routines, pruning, symmetry or memoization. Agreement on 1,170 tiny zero-extra-state cases and 96 endpoint/reserve/connectivity fixtures supports implementation correctness on those cases only; it is not a machine-checked universal theorem.

## Optional-device monotonicity and objective mismatch

If constraints, existing devices and links remain unchanged, adding optional devices enlarges the feasible set. Therefore the true optimum of the **same** latency objective cannot worsen: the old plan remains available. The tests explicitly rebind/check the four-home witness against five-/six-home inventories and retain identical timing.

The solver instead minimizes stage count, not latency. Changing the objective's preferred plan can select more WAN transfers, so the reported communication of that chosen path can rise. Simple greedy can therefore outperform that selected path in communication while using more stages. Do not call that an inherent penalty of extra households, a nonmonotonic capacity frontier, or proof that the best planner becomes slower. Latency-aware/Pareto search is the next-loop optimization target.
