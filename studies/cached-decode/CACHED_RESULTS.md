# Cached decode across two local worker processes

## Result

The FP32 two-worker split supports prefill followed by cached single-token autoregressive decoding. Each worker holds only its own 15-layer KV cache. Both retained runs reproduced every full-vocabulary reference logit **byte for byte**, with exact greedy tokens and exact reset/replay output.

This is a correctness and accounting milestone, not a speedup, distributed deployment, representative language benchmark, or memory-saving result.

## Fixed experiment and evidence

- Frozen v4 source manifest: `cached_checkpoints/v4/source_hashes.json`.
- Pinned SmolLM2-135M-Instruct revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; local verified weights/config, Torch 2.5.1, Transformers 4.46.3, eager CPU FP32, one thread per process.
- Three fixed synthetic token prefixes (lengths 4, 8, 16), each followed by eight greedy cached decode steps; reset/replay of the first prefix and its eight decode steps. Four prefills + 32 decodes = 36 paired checks/run.
- Two separately allocated retained directories: `cached_runs/retained-v4-001` and `cached_runs/retained-v4-002`. All 72 rows across both runs passed. Each run retains 72 complete 49,152-element FP32 logit files, not only aggregate errors. These are timing repetitions and a reset test, not 72 independent quality examples.
- Exact cross-run output reproduction verified by `mesh.cached_verify`; row coverage, finite values, hashes, greedy argmax, autoregressive token chaining and reset replay checked. Sources and actual cleanup were checked separately, because the artifact verifier alone does not prove provenance/cleanup.
- Both runs: worker exitcodes `[0, 0]`, no forced termination, workers and tracked I/O threads cleaned up. Maximum observed logit error was zero.

## Separate phase accounting

Descriptive medians only; no warmup excluded, no inferential statistics, and different prefix lengths mixed within prefill. First prefill includes cold effects; replay prefill is warm. Same-host scheduling and correctness audit overhead are included.

| Run | Phase | Pairs | Split median | Reference median | Logical boundary frame |
| --- | --- | ---: | ---: | ---: | ---: |
| 001 | Prefill | 4 | 98.97 ms | 98.55 ms | 9,320–36,970 B |
| 001 | Decode | 32 | 74.99 ms | 67.94 ms | 2,407–2,408 B |
| 002 | Prefill | 4 | 90.45 ms | 84.81 ms | 9,320–36,970 B |
| 002 | Decode | 32 | 71.31 ms | 65.62 ms | 2,407–2,408 B |

The split is slower on median decode in these runs. This is not evidence against multi-device value; equally, the short boundary frame is not evidence of practical network throughput.

Topology is controller → stage0 → controller → stage1 → controller. The controller relays activations, so one logical stage boundary entails **two actual IPC transfers**. Both request and response application-frame sizes are recorded per stage. The logical-boundary metric is the stage1 request only, not total network traffic.

A single-token activation contains 2,304 payload bytes. Full-vocabulary audit logits contain **196,608 payload bytes per step**, dominating the actual summed IPC: approximately 201,760–201,770 bytes per decode across both runs. Prefill summed IPC is approximately 215,596–270,954 bytes. A future deployment could sample at stage1 and return a token ID; that alternative is not implemented or measured here. Application-frame sizes include our metadata but exclude multiprocessing outer framing and OS overhead; no network packets or actual link behavior were measured.

## Resources and ownership

Actual stage-local cache tensors use 23,040 bytes per cached token per worker, increasing with context. They never cross the boundary. Decoder weights are partitioned, but the tied embedding/output-head matrix is loaded once per worker (113,246,208 bytes each), and the controller holds a complete monolithic reference for comparison. No total-memory saving is claimed.

Sampled aggregate controller + direct-child RSS peaks were 2,192,982,016 and 2,169,843,712 bytes (about 2.04 and 2.02 GiB). RSS sums may count shared pages more than once and are not unique physical memory. The 3 GiB guard is a 50 ms sampled/cooperative check, not an OS-enforced cap. Timing deadlines bound IPC waits; they do not forcibly bound reference model loading/inference.

## Tests, review and failure retention

81 tests passed in the default runtime, with six optional tests skipped. The earlier pinned-runtime suite passed with three real-model tests skipped; these retained runs are additional real-weight evidence. Protocol tests reject stale epochs/sequence positions, invalid phase/length, malformed metadata and duplicate keys. Real spawned-child tests cover partial-frame timeout/cancellation and blocked sends. Failed exchanges poison the session; reset cannot revive it.

Independent review required prospective fixes before model execution: bounded partial-frame receives, tracked I/O cleanup, readiness typing, resource startup guards and matching reference output-head shape. Source snapshots v1–v4 remain available.

The first development smoke failed before inference because this container lacks the `/proc/.../children` accounting interface. It was retained unchanged with zero rows, not dropped or relabeled successful. The reviewed resource-only v4 amendment enumerates supported status/PPid fields; the same predeclared development smoke then passed all four checks. No prompt, step count or numerical threshold was changed based on outcomes.

Evidence-component failure/SIGINT tests verify incremental row retention before and after a row; they do not establish full inference-runner interruption recovery, SIGKILL finalization or power-loss durability. See `CACHED_EXPERIMENT.md`, `CACHED_STAGE_HISTORY.md` and `CACHED_LIMITATIONS.md` for exact scope.

## Reproduction

Use the existing documented optional runtime and pinned local model. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Run:

```sh
python -m mesh.cached_experiment --output NEW_OUTPUT_DIR \
  --model-dir PINNED_LOCAL_MODEL \
  --source-hashes cached_checkpoints/v4/source_hashes.json --opt-in
```

Repeat into another new directory, then:

```sh
python -m mesh.cached_verify FIRST_OUTPUT_DIR SECOND_OUTPUT_DIR
```

Existing output paths are rejected. `rows.jsonl` is flushed/fsynced after every case before correctness assertions; full logits and failure/terminal records are retained. Source hashes must match the frozen manifest. Verify terminal cleanup and source provenance separately. Descriptive results are in `cached_checkpoints/v4/retained_summary.json`; all result hashes are in `retained_artifact_hashes.txt` beside it.
