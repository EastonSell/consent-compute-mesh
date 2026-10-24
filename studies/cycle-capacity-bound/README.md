# Cycle-capacity and deadline rejection supplement

This additive analytical supplement studies the repository's existing one-contiguous-stage-per-device, final-to-first-feedback model. It is synthetic exact arithmetic, not a household hardware benchmark, production solver integration, or claim of research novelty.

## Read first

- [Derivation and limitations](REVIEW.md)
- [Preserved source model](SOURCE_MODEL.md)
- [Independent falsification receipt](INDEPENDENT_REVIEW.md)
- [Author results](RESULTS.json) and [independent results](INDEPENDENT_RESULTS.json)

The frozen reviews record what was done before publication; their statements about no publication refer to those review stages. The independent reviewer accepted the final hashes, including the two corrected presentation/provenance details documented in that receipt. The reviewed source/code/results are preserved byte-for-byte here.

## What the example establishes

A restricted logical star has aggregate layer capacity 10 but cycle capacity 7, so it cannot place eight unit layers. Adding costly logical leaf-to-leaf edges makes the exact synthetic optimum 110 time units, with a sharp 109/110 deadline rejection boundary. Allowing routed leaf-to-leaf transfers at cost 2 changes the optimum to 12.

Crucially, physical links are not logical transfer edges. A physical home-router star can allow routed leaf-to-leaf traffic; the forwarding hub need not own a compute stage. The restricted-star result must not be applied to physical topology without establishing the allowed end-to-end logical transfers. Passing the bound does not prove feasibility. There is no measured speed, consent, runtime, or throughput claim.

## Reproduce

Python 3.10+ with the standard library only; no installation, account, model download, or hardware access is needed. From this directory:

```sh
python check.py
python independent_check.py
sha256sum -c SHA256SUMS
sha256sum -c PACKAGE_SHA256SUMS
```

The scripts deterministically regenerate their respective saved JSON results. The author check covers 4,096 digraphs, 12,288 capacity checks, and 113 deadlines. The separate nonempty-interval placement oracle adds 65,536 self-loop-invariance checks and 196,608 deadline checks. It confirms exact optima 110 and 12 and a bound-passing but infeasible K2,3 example.

These are finite implementation checks, not a machine-checked universal proof or repository-wide test run. The optional endpoint-pair refinement is not implemented. The author's exact_best helper is scoped to its stated fixtures, not a general supported-interval placement solver. PACKAGE_SHA256SUMS covers all other files in this package; SHA256SUMS is the original author manifest retained unchanged.
