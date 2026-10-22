# Optional real pretrained-model partition

This extension runs real pretrained SmolLM2-135M-Instruct weights in two persistent local processes. It is separate from the standard-library synthetic experiments. It does not download weights or install dependencies, and opens no network listener.

## Pinned assets and runtime

- Model: HuggingFaceTB/SmolLM2-135M-Instruct.
- Revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Model card declares Apache-2.0; the model weights are not included in this repository and retain their own license.
- Optional tested runtime: Python 3.12, torch 2.5.1 CPU, transformers 4.46.3, safetensors 0.4.5, NumPy 2.3.5. See `requirements-pretrained.txt`.

The local config and safetensors must match the SHA-256 values in `mesh/pretrained_model.py`. The loader verifies them, uses only local files, disables remote-code trust and requires safetensors. The CLI sets Hugging Face offline flags. No credentials are required. Existing synthetic commands still require only Python's standard library.

With the optional runtime and pinned model files already available:

```sh
HF_HOME=/path/to/writable/cache python -m mesh.pretrained_experiment --opt-in --model-dir /path/to/pinned/model --repeats 3
HF_HOME=/path/to/writable/cache MESH_MODEL_DIR=/path/to/pinned/model python -m unittest discover -s tests -p test_pretrained_optional.py -v
```

Ordinary test discovery skips real-weight integration tests unless `MESH_MODEL_DIR` is set. A skip is not evidence that the pretrained adapter passed. The separate optional run is required.

## Exact computation contract

This adapter is deliberately narrow:

- FP32 CPU, eager attention, one intra-op thread per process.
- Batch size 1, 1–64 unpadded token IDs, no attention-mask input or padding support.
- Full causal forward pass with positions 0 through sequence length minus 1 on every request.
- No KV cache, past state, incremental decoding, beam search, generation loop or cache transport.
- Stage 0: token embedding and decoder layers 0–14.
- Stage 1: decoder layers 15–29, final RMS norm and tied output head.
- Output: all vocabulary logits for the final sequence position only.

Each worker loads only its assigned layers. The token embedding weights are also needed as the tied output-head weights, so that tensor exists in both workers. The monolithic reference remains in the controller during paired timing. This has greater total resident memory than one ordinary model process; it is not a memory-saving claim.

The explicit upper-triangular causal mask and rotary positions are rebuilt consistently in each partition. Cache use is disabled in the reference and split paths. The reference requests only the final-token logits (`num_logits_to_keep=1`) to avoid unfairly charging it for output projections at every token.

## Pipe protocol

Control and token messages are bounded JSON. Activation/logit frames contain a bounded JSON shape/dtype header and contiguous little-endian float32 bytes. The frame cap is 2 MiB. Payload sizes must match dimensions exactly. There is no pickle message payload and no user-supplied executable function. Each stage checks its expected tensor shape; stage 1 rejects non-finite activations. Python spawn still starts trusted local worker functions using its normal bootstrap mechanism.

This is a trusted same-host experiment, not a network security protocol or sandbox. It does not implement peer authentication, remote enrollment, encryption or general remote execution. A failed request invalidates the session. Large-library process teardown gets a three-second grace period before termination fallback; startup and cleanup are excluded from request timings.

## Correctness before timing

Two public text prefixes are represented by fixed token IDs from the pinned tokenizer, without a chat template. The runner compares every vocabulary logit at the final position to the monolithic reference, using `rtol=1e-4`, `atol=1e-4`, and matching next-token argmax. Both preflight cases must pass before any timing sample is retained. Every retained pair is checked again.

The optional integration tests also cover sequence lengths 1, 4 and 8, plus the actual cross-process binary transport and normal worker exits. Passing these validates the implemented split for these inputs, not every sequence, model architecture or numerical environment. No language-model quality benchmark is claimed.

## Modest paired experiment

The runner compares two input lengths and two link assumptions: no deliberate delay, and an emulated 10 ms one-way latency plus 1,000,000 bytes/second. The delay is charged once for the actual encoded intermediate activation frame, not input tokens, final logits or control messages. It is a controller wait, not a measured or shaped network.

Each case excludes one warmup pair and retains three paired samples by default. Reference-first and split-first alternate, giving a 2/1 order imbalance for three samples. Cases remain in fixed order. Summaries are descriptive minimum/median/mean/maximum, not confidence intervals or evidence of population speedup. Concurrent work and shared-host scheduling can materially affect timings.

The Linux runner checks observed `MemAvailable` before loading and requires at least 3 GiB. That observation is not a cgroup limit or memory reservation. It reports load/startup times and per-process peak RSS separately; peak RSS values are not simultaneous memory measurements and should not simply be summed as exact system usage.

No physical network, VM, ESP32, Android/adb or heterogeneous machine is involved. This is forward-pass latency, not autoregressive tokens/second, energy efficiency or deployment readiness.


## Execution chronology and interpretation

A preliminary exploratory timing execution completed before publication of this
source checkpoint and before independent review was complete. Its output is
preserved separately as `results/pretrained_exploratory_prepublication.json` for
the results checkpoint. This protocol was not independently preregistered.
The source checkpoint precedes the planned post-publication repeat, not every
experiment. The later repeat will use the same protocol unless a review-driven
change is recorded explicitly. Neither execution is an independent hardware
trial or a statistically powered confirmatory study.

### Review amendments after the preliminary run

Before the post-publication repeat, independent review found that malformed worker
reports could escape without invalidating a session. The amended protocol checks
report fields and numeric types, stage/request order, expected activation/logit
shape and finite values. Protocol failures now return `protocol_error` and prevent
session reuse. Seven regression tests cover malformed/missing/out-of-order/stale
reports and bad frames. These checks add real validation overhead, so preliminary
and amended timings must not be treated as an unchanged-protocol replication.

A second amendment preserves the failure type, phase, partial case evidence and
worker-cleanup report when a correctness gate or measured request fails. Such
failed experiments return JSON and nonzero status rather than dropping evidence
through a CLI argument error. Invalid arguments and incompatible model/runtime
setup remain ordinary setup errors. The optional integration test deliberately
injects a failed response to verify retention; it is a failure-path test, not a
performance sample. No tolerances, model weights, cases or repeat counts were
changed in response to preliminary numerical results.
