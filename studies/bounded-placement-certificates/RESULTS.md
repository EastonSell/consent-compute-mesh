> Publication derivative: scientific values and contracts are unchanged. Internal publication-control prose and release-relative links were edited; this is not the exact executed/frozen document. See [PUBLICATION_TRANSFORMATIONS.json](PUBLICATION_TRANSFORMATIONS.json) for original and public hashes and the exact edit map.

# Results: certificates replace aggregate intuition, not deployment validation

Revision 2 note: every numerical result below and all four result files are unchanged from sealed v1. The original 42-test count below describes v1; nine schema-hardening regressions raise the current suite to 51. This revision is correctness polish, not a new research result.

Evidence class: deterministic analytical scenarios only. No actual device, model, physical link, timing, free-memory, thermal, energy, consent-enforcement or model-quality measurement was collected.

## 1. What is new and what remains open

The paper's aggregate memory scenario is now connected to an explicit bounded layer/device placement model. Supplied inventories can receive independently checked positive memory/connectivity witnesses or optimistic layer-count rejection bounds. The package also exposes the inadequacy of a stage-count objective for link-sensitive latency. This is a transparent formal/computational artifact, not a claim of a new state-of-the-art placement algorithm, a functioning household mesh, general model capability or dissertation completion.

The model permits one resident nonempty contiguous layer interval per used device. Embeddings/controller live on the first stage, the head on the last; tied weights require endpoint copies if split. Every stage charges KV, runtime, peak sequential workspace and activation-buffer reserve. Direct activation and feedback links are checked separately. See PROTOCOL.md, AMENDMENTS.md and docs/MODEL.md for the exact scope.

## 2. The 43.616 GiB / 12 GiB calculation does not specify a placement

The paper computes an illustrative unique envelope:

    1.20 * (70*10^9 parameters * 4/8 bytes)
      + 80 * (2*8*128*8192*1*2 bytes of KV)
      + 2 GiB
    = 43.6155481338501 GiB.

This uses a generic 70B nominal-4-bit parameter budget and a documented 80-layer KV shape, not an actual measured checkpoint tensor/format layout. The 20% and 2 GiB terms are analyst choices. Dividing by an assumed 12 GiB bundle gives four homes, but each bundle is [8,2,2] GiB, not one 12 GiB address space.

For an envelope-matched surrogate, assign 42,000,000,000 weight-plus-format bytes equally across 80 blocks, 32 MiB KV per block, and the whole 2 GiB reserve to a first-stage controller. Embedding/head bytes, workspace and extra local reserve are zero in this row, expressly stylized rather than realistic. All allocations and support are assumed true. Each layer therefore consumes exactly 558,554,432 bytes, about 0.520194352 GiB.

In four homes a non-first laptop can hold 15 layers, the controller-bearing first laptop 11, and each phone allocation 3. Thus:

    11 + 3*15 + 8*3 = 80 layers, across all 12 devices.

With only 11 stages the optimistic maximum is 11+3*15+7*3=77. The solver certifies a 12-stage placement with intervals [0,11), [11,14), [14,17), [17,32), [32,35), [35,38), [38,53), [53,56), [56,59), [59,74), [74,77), [77,80). Every local byte count is independently checked. **This is a certificate of that surrogate, not a valid real-70B checkpoint placement.** Actual layer/endpoint tensors, quantization metadata, operator support and safe allocations remain unknown.

## 3. Aggregate-positive but impossible after small local reserves

The frozen local reserve grid treats the added bytes as a per-used-stage activation-buffer reserve. This is a sensitivity parameter, not measured runtime memory.

| Homes | Reserve per used stage | Exact modeled result | Minimum stages if feasible |
|---:|---:|---|---:|
| 3 | all 0..512 MiB grid values | Infeasible | — |
| 4 | 0 / 64 / 128 / 192 MiB | Feasible | 12 |
| 4 | 256 / 384 / 512 MiB | Infeasible | — |
| 5 | 0 / 64 / 128 / 192 MiB | Feasible | 8 |
| 5 | 256 / 384 MiB | Feasible | 10 |
| 5 | 512 MiB | Feasible | 12 |
| 6 | all grid values | Feasible | 6 |

At four homes and 256 MiB per stage, a normal laptop holds at most 14 layers, the first laptop 11, and every phone still 3. Even optimistically ignoring output-head costs and connectivity:

    11 + 3*14 + 8*3 = 77 < 80.

No permitted placement exists. Yet the maximum 12-stage aggregate state is only 43.615548+12*0.25 = 46.615548 GiB, below the 48 GiB allocation. The aggregate calculation has 1.384452 GiB of slack, yet no valid per-device assignment exists. The independently checked count bound rejects the case without enumerating placements. Aggregate total capacity is demonstrably insufficient even after charging these reserves globally.

## 4. Embedding/head decomposition and tied-weight duplication

To expose underdetermination without changing the unique envelope, move one tied embedding/head copy out of the layer-weight total. Split endpoint stages each retain a copy; this increases aggregate residency by one endpoint tensor. Integer quotient/remainder distribution preserves exact bytes, as prospectively corrected before the grid run.

Four homes, zero workspace/local activation reserve:

| Tied endpoint copy | Result | Stages | Split resident GiB if feasible |
|---:|---|---:|---:|
| 0 GiB | Feasible | 12 | 43.615548 |
| 0.5 GiB | Infeasible | — | — |
| 1 GiB | Infeasible | — | — |
| 2 GiB | Feasible | 11 | 45.615548 |
| 3 GiB | Feasible | 12 | 46.615548 |
| 4 GiB | Infeasible | — | — |
| 8 GiB | Infeasible | — | — |

This sensitivity is intentionally nonmonotonic: moving bytes from every layer into endpoint tensors changes packing granularity and which devices can host endpoints. These are different synthetic decompositions of the same unique total, not a claim that changing an actual model's embedding size improves quality or memory.

At 4 GiB endpoints, split residency would be 47.615548 GiB, below 48 GiB. But no device can host the whole model; endpoints must occupy two distinct laptop allocations. The first holds at most 4 layers, the last 8, each of the other two laptops 17, and each phone 4:

    4 + 8 + 2*17 + 8*4 = 78 < 80.

This stronger endpoint-pair no-placement certificate was added **after** the first grid revealed the initial first-prefix bound was inconclusive. The amendment and first-run artifacts are retained; the grid and outcomes were not changed. At five homes, the same 4 GiB/zero-workspace scenario does fit, with a checked 10-stage witness. The 64/256 MiB workspace sensitivities and all failures remain in sensitivity.csv.

## 5. Device count, stage count and communication are not interchangeable

Assume, not measure: 16,384-byte activations; intrahome 1000 Mbps / 1 ms; interhome 20 Mbps / 20 ms; four-byte final-to-first feedback. All main-surrogate compute times are unknown.

For the four-home, 12-stage grouped path:

- Eight intrahome activations: 8*1.131072 ms.
- Three interhome activations: 3*26.5536 ms.
- Interhome feedback: 20.0016 ms.
- Communication total: **108.710976 ms**, carrying 180,228 directed payload bytes.

The activation-only component is 88.709376 ms. The paper's separate four-stage all-WAN activation-only illustration is 79.6608 ms and is not a placement for these 12 allocations. Making every link WAN in the checked 12-stage path yields 312.0912 ms. These are supplied-graph communication components/lower bounds, not predicted observed latency or measured tokens/s. Compute, protocol/packet work, queueing, contention, thermal behavior and tails remain uncredited unknowns.

### The retained adverse baseline comparison

| Zero-reserve inventory | Solver's minimum stages | Communication of chosen solver path | Input-order greedy stages / communication |
|---|---:|---:|---:|
| 4 homes | 12 | 108.710976 ms | 12 / 108.710976 ms |
| 5 homes | 8 | 129.609216 ms | 12 / 108.710976 ms |
| 6 homes | 6 | 152.769600 ms | 12 / 108.710976 ms |

**Adding optional devices does not worsen the true latency optimum under unchanged constraints.** The four-home witness remains valid with identical timing in the five-/six-home inventories; an explicit test checks that. The worsening above is solely the selected minimum-stage objective/path, which is not latency-aware. It must not be presented as inherent harm from more households.

Simple baselines also have limits. In four grid cases, input-order greedy fails while the exact solver finds a valid placement, including tied 2/3 GiB endpoint cases. For example, the 2 GiB endpoint row has an 11-stage witness but both greedy orderings fail. Conversely, at five homes / 4 GiB endpoints / zero workspace, input-order greedy finds 11 stages, descending-budget greedy fails, and exact search finds 10. A greedy failure is not infeasibility; fewer stages is not a service-speed claim. All known feasible baseline witnesses are independently checked.

The next optimization increment is therefore latency-aware/Pareto placement under the same input/quality assumptions, not a claim that this stage-count-first solver is a superior real planner.

## 6. Unknowns, caps and validation

The frozen grid has **84 rows: 44 modeled feasible, 36 infeasible, 3 unknown, 1 deliberately search-incomplete.** Some rows repeat equivalent arithmetic inputs across sensitivities. These are grid descriptors, not independent experiments, probabilities, population rates or quality trials.

- Unknown allocation, eligibility or feedback-link availability receives only an optimistic provisional witness, not a certified placement.
- Known connectivity with an unknown rate/compute term can have a valid memory/connectivity certificate, while communication/total critical path remains unknown as applicable.
- A missing directed return link is unavailable, not a free/unknown transfer. Its tiny case is a complete finite-search negative, explicitly **not** independently certified by a count bound.
- A zero visited-state limit is retained as `search_incomplete`; no resource exhaustion becomes infeasibility.

The final standalone checker accepts **117 positive solver/baseline witnesses and 35 negative count bounds**. Baseline duplication is not extra scientific evidence. The remaining negative is the missing-feedback toy above. The suite has 42 named tests, including all **1,170** predeclared tiny zero-extra-state inventories and **96** selected endpoint/reserve/connectivity fixtures against an algorithmically independent exhaustive permutation/cut oracle. Tampering, gaps, reused devices, tied accounting, boolean-as-integer fields, unknown timing, asymmetric feedback and Python optimized-mode checks are covered. Test-first failures and corrections are retained, including a mistaken fixture-count expectation corrected from 192 to 96, with no solver outcome change.

Maximum observed final-grid search per pass: 2,508 visited states and 66,377 transitions, below fixed 250,000 / 5,000,000 caps. Input size is at most 18 devices / 80 layers in the grid, with a 128-layer implementation limit. Source/tests and output checks use only the standard library. Final regenerated result files are byte-identical in a second output directory; see checks/determinism.json. Verification receipts and source hashes bind the exact artifact reviewed.

## 7. Scientific stopping boundary

This increment closes a concrete arithmetic gap: totals now have explicit witnesses, falsifiers and uncertainty labels. It does **not** close empirical eligibility, runtime/format/kernel feasibility, safe concurrent duties, task/model quality, actual topology, sustained service, privacy/security/consent enforcement, energy, thermal or recovery questions. No average household is synthesized from vendor RAM or household-presence data. All eligibility true values in surrogate rows are assumptions.

A real inventory/checkpoint/runtime study remains necessary for deployment claims. This analytical package does not establish deployment readiness.
