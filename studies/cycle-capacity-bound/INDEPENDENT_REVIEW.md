# Independent falsification receipt

Reviewed 2026-10-24 UTC, independently of the derivation author. Scope: the mathematical claims and executable synthetic checks in this directory, not production integration, hardware measurements, repository-wide tests, or a literature novelty search. No publication or external-state change was performed.

## Verdict

No substantive counterexample found to the cycle-capacity or deadline-pruning claims under their stated assumptions. Reproduced the saved RESULTS.json exactly by importing check.py and calling run(), without overwriting the saved result. All 4,096 loopless four-node digraphs, 12,288 capacity checks, and 113 integer deadlines were reproduced. Exact synthetic optima remain 110 and 12.

Two minor presentation/provenance issues were sent to the author:

1. The inspected SOURCE_MODEL.md has two trailing newlines, so its actual Git blob is c810f8094ea108b78ed5a290c5060a3a4f6bec06 rather than the claimed source blob e5fb1022639b7db3d2e1f182d7e2cc6427c753e5. Removing exactly one final newline reproduces the claimed blob exactly. This is a non-substantive preservation discrepancy, not conflicting model text. Restore original bytes and refresh hashes or disclose normalization.
2. Define the maximum over an empty family of multi-device regions as zero. The implementation already safely handles edgeless graphs by including the separate monolithic bound. The written formula alone should state the convention.

## Premise and proof review

The preserved source explicitly imposes one nonempty contiguous interval per selected device, each device used at most once, a directed link for each interstage activation, and directed final-to-first feedback. For k >= 2 these distinct device vertices and k transfers form a directed simple cycle. This conclusion does not require intermediate physical forwarding routers to own compute stages. A monolith has no remote communication in the source model and is a separate case, not a required self-loop.

Each cycle is inside one SCC. For k >= 3 its underlying undirected cycle is biconnected and lies within one maximal vertex-biconnected block. For k = 2 the dyad convention is needed and is implemented. Intersecting SCCs with undirected blocks is a safe region relaxation. Capacity sums may overcount compatible intervals or directed cycle membership; that weakens rejection only. Unknown facts must be optimistically admitted, not silently excluded, when the intended conclusion is impossibility under unresolved inputs.

The k_min construction is a valid lower bound because no chosen k devices can have greater aggregate q than the largest k capacities. C0 must lower-bound the full actual compute sum; e0 must lower-bound every possible transfer in either payload role. For any actual arc, the remaining k-1 transfers cost at least (k-1)e0 >= (k_min-1)e0 because e0 is nonnegative. Thus a deadline-feasible actual arc survives an optimistic role-union graph filtered with T-C0-(k_min-1)e0. Loose or zero lower bounds are safe, not speed estimates. This needs nonnegative serial costs, the stated feedback topology, and separately excluded monoliths; it does not establish pipelined throughput. If q permits k_min=1 despite true endpoint constraints excluding monoliths, the written threshold remains safe but weaker; the implementation conservatively gives no deadline rejection in that case.

The optional endpoint-pair formula is mathematically safe because the actual first/last pair occurs in a region and the actual contributions are bounded by f/h/q. It is not exercised by the supplied checker or my supplemental checker. Excluding only explicitly impossible feedback pairs is safe; unproven/unknown pairs cannot be excluded.

## Independent executable falsification

independent_check.py implements a separate device-permutation and nonempty layer-cut oracle, rather than calling the author's cycle enumerator or exact_best helper. For the uniform unit-layer fixtures it obtains:

- Restricted logical star: no placement.
- Expensive leaf-to-leaf complete logical graph: optimum 110.
- Routed leaf-to-leaf cost-2 closure: optimum 12.

An attaining witness in both feasible cases uses devices (0,1,3), with cuts (0,4,5,8), hence 4/1/3 layers. Compute is 8 and serial transfer costs are respectively 102 and 4. This independently checks real contiguous nonempty assignments, not just sufficient aggregate capacity on a cycle.

Additional checks:

- 65,536 checks: all 16 self-loop subsets for each of 4,096 four-node directed off-diagonal graphs leave the implemented capacity bound unchanged. This checks invariance, not self-loop-dependent monolithic semantics.
- Explicit asymmetric dyad test: a single directed arc cannot support two stages; adding its reverse permits the two-stage feedback cycle.
- 196,608 deadline checks: all 4,096 four-node digraphs, three capacity vectors including zero entries, and 16 deadlines. Edge costs vary deterministically over 0..4. Actual compute is 6 while the supplied certified compute lower bound is only 2. Every rejection is compared with the independent nonempty interval oracle. No erroneous rejection occurred.
- A concrete false-positive of the bound confirms necessity is not sufficiency: bidirected K2,3 with five unit-capacity devices has one SCC and one undirected block, bound 5, but cannot place five unit layers because a bipartite graph has no 5-cycle. The independent placement oracle confirms infeasibility.

The author's exact_best helper admits cycles solely by aggregate capacity and stage count; outside its positive-capacity, unit-layer synthetic fixtures it is not a general supported-interval placement oracle. In particular, a selected zero-capacity stage could make that helper optimistic. None of the reported exact numerical examples use such inputs. Neither checker is a machine-checked universal proof.

## Routing and claim limits

A physical hub-and-spoke network does not imply a restricted logical star. Permitted routed leaf-to-leaf transfers create logical edges even if the forwarding hub is not selected for a compute stage. If forwarding is excluded, aggregate memory can grow without improving the single-cycle bound; if forwarding is allowed with cost 2, the supplied example becomes feasible at 12. Contention, concurrency, runtime feasibility, consent, and hardware calibration are not established by this arithmetic. Optional devices cannot worsen the true optimum with unchanged old constraints.

No novelty assertion was independently established. The graph facts are classical; at most this is a repository-specific rejection derivation and synthetic falsification exercise under the preserved source model.

## Hash-pinned inspected scope

SHA256 of files at initial review (before any author correction):

- REVIEW.md: ca7e41f20ef85d9f2afc890273f4dcd8ce4f7fe93962b318a57e5dbd3f69f04f
- check.py: a8926d4bd4492e11558836a3c2d57322dd9abf09bd97fcca82635580e789dde2
- RESULTS.json: 238dc840c182f01b699a3b3d5df067fad0b98646e9ddfea4ea86797dd17d97e1
- SOURCE_MODEL.md: d8b9f5236ce69482e65a708a6c2440b357fe32895c2a28de4b1ce0f6ff003950

Supplemental reviewer artifacts:

- independent_check.py: ab05119c0c4e4389e150acf2e1c423a23b80e4edd4710b4c265c8d8040cc8974
- INDEPENDENT_RESULTS.json: 795865f7607f4d4bc87994992964a814c9d524dff8b158854a13e0fb6ad603ab

This receipt authenticates only the locally inspected bytes and tests; it does not independently attest remote repository hosting or physical measurements. The supplied immutable source URL and restored Git-blob hash are the recorded provenance anchors.

## Final frozen-artifact acceptance, 2026-10-24 05:48 UTC

Both minor issues are now resolved in the reviewed final bytes. SOURCE_MODEL.md now reproduces Git blob e5fb1022639b7db3d2e1f182d7e2cc6427c753e5 directly, without normalization. REVIEW.md explicitly defines an empty region maximum as zero, separates self-loops from multi-stage cycles, and includes the bidirected K2,3 insufficiency example. All five entries in the author's SHA256SUMS verify. check.py remains unchanged. I reran run() and the entire supplemental independent checker against these final bytes; both reproduced their saved JSON results exactly (accounting only for JSON tuple-to-array serialization in the supplemental result).

Final accepted SHA256 values:

- SOURCE_MODEL.md: 1503eb45e99feed5ea770b3ef08b3514c18f51816a654d013961231447b094f9
- check.py: a8926d4bd4492e11558836a3c2d57322dd9abf09bd97fcca82635580e789dde2
- REVIEW.md: 4ccac2ab2c483541b29ab1ebbc5699757ff09ddd5499d7f2ac28830a442c34d6
- RESULTS.json: 238dc840c182f01b699a3b3d5df067fad0b98646e9ddfea4ea86797dd17d97e1
- CHECK_RECEIPT.txt: 6aa640da845676b2dfa776bc842da8918be1feaff78ad73165d500e73a7a0e91

Disposition: accepted as a scoped local analytical review and synthetic test package, with no remaining substantive mathematical or reproducibility blocker identified. All limitations above still apply. Acceptance is technical review, not authorization to publish; no public writes were performed by this reviewer.
