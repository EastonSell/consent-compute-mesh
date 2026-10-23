# Bounded placement certificates for household compute

An analytical counterexample package, not a household deployment demonstration. No real device, model, network, safe-allocation, thermal, quality or service-rate measurements were collected.

The accepted paper’s **43.615548 GiB aggregate envelope does not specify a per-device placement**. Under an expressly synthetic 80-block decomposition and four assumed [8, 2, 2] GiB homes, the baseline needs all **12 devices/stages**, not four stages. Adding only 256 MiB reserve per used stage makes the same surrogate impossible despite aggregate slack. Tied endpoint copies create another aggregate-positive but impossible case.

**Stage-count minimization is not latency minimization.** The stage-first solver sometimes selects a slower communication path than simple greedy. Optional devices cannot worsen the true optimum when the old placement remains allowed. These calculations do not establish actual checkpoint/runtime feasibility.

Read [RESULTS.md](RESULTS.md), [the mathematical model](docs/MODEL.md), [protocol](PROTOCOL.md), [amendments](AMENDMENTS.md), and [schema-hardening history](SCHEMA_REVISION.md).

## What is retained

- All 84 rows: 44 modeled feasible, 36 infeasible, three unknown, one deliberately search-incomplete.
- All 117 positive solver/baseline witnesses and 35 independently checkable negative bounds; the missing-feedback finite-search negative is not a count-bound certificate.
- All greedy misses, unknown/capped cases, endpoint/reserve sensitivities, and the stage-count/latency adverse comparison.
- The complete first grid and amendment snapshots, including the post-result proof refinement.
- Exact accepted Python source/tests. Revision 2 changes only checker interfaces; all four numerical result files remain identical to v1.

## Verify saved evidence without a new experiment

Python 3.10+ and the standard library suffice. From this study directory:

    python reconstruct_evidence.py --out /tmp/placement-evidence-new
    python -m placement.checker /tmp/placement-evidence-new/results/all_cases.json
    python -O -m placement.checker /tmp/placement-evidence-new/results/all_cases.json

The output directory must not exist and its parent must exist. Reconstruction verifies both compressed and restored hashes and recovers the exact original JSON bytes. It does not invoke the solver or modify saved evidence. [results/summary.json](results/summary.json), [results/sensitivity.csv](results/sensitivity.csv), and [results/independent_checks.json](results/independent_checks.json) are directly readable.

The bounded checker certifies witness arithmetic and the specified negative bounds, not general solver completeness, latency optimality or deployment safety. Exhaustive tiny-oracle agreement is limited to its enumerated domain.

## Optional independent reproduction

These commands start a new computation and were not run to create this public package:

    python -m unittest discover -v
    python -m placement.experiments /tmp/placement-grid-new

Choose a fresh directory for every run. The original experiment CLI can overwrite files in an existing directory; do not point it at retained evidence. The solver caps remain 18 devices, 128 layers, 250,000 visited states and 5,000,000 candidate transitions per pass. A cap is search-incomplete, never infeasibility.

## Publication and provenance boundary

Scientific Python files and the three small saved result files are byte-identical to the accepted source. The two large scientific JSONs use deterministic lossless gzip and recover exact bytes. Public documentation is a labeled derivative removing internal publication controls and fixing release-relative links. Original executed/frozen documents and full internal receipts remain private; a public derivative is not relabeled as an executed freeze. [PUBLICATION_TRANSFORMATIONS.json](PUBLICATION_TRANSFORMATIONS.json) binds every source/output and lists every documentation edit. Chronological protocol records are not independently timestamped preregistration. [SOURCES.json](SOURCES.json) pins the already-public antecedents. Repository licensing applies.
