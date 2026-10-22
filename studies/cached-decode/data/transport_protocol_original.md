# Activation transport: source/protocol checkpoint

## Status and question

This protocol is fixed **before any model forward execution in this new experiment**. Only synthetic codec tests, failure/cleanup tests and local tokenizer preprocessing have run at the source checkpoint. Parent publication/review precedes the retained experiment. This is not independent preregistration. All previous household files remain unchanged in their publication workspace; this development copy is isolated.

Question: how do FP16 and symmetric per-tensor INT8 boundary activations change bytes, serialization time, same-host latency and numerical drift relative to FP32 transport? This is a communication-bottleneck experiment, not a quality or multi-device throughput benchmark.

## Fixed model and workload

- Same pinned SmolLM2-135M-Instruct revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, verified local safetensors/config hashes via existing adapter.
- CPU FP32 model arithmetic, eager attention, one thread per process, batch one, no padding, no KV cache, no generated-token loop, same layer14/15 split. Two persistent spawned workers, private pipes only. No listeners or physical devices.
- 12 original public diagnostic prefixes in `data/transport_prefixes.json`, chosen before outcomes. They cover factual completion, arithmetic, Python, JSON, Spanish, French, Chinese, instructions, narrative, science, comparison and a longer46-token context. Prefixes span7–46tokens. Exact IDs and local tokenizer hashes are retained, no chat template. These are not a representative benchmark or held-out quality dataset.
- Formats: FP32, IEEE FP16, and symmetric signed INT8 per **entire** `[1, sequence_length, 576]` activation tensor. Quantization does not change model weights, activations inside either stage, KV or final FP32 logits.
- INT8: scale=`max(abs(x))/127`, nearest-integer ties-to-even via NumPy `rint`, clip to[-127,127], encode signed bytes. Zero tensors use scale1.0. Scale is serialized as a round-trippable JSON double; dequantization multiplies in float64 then casts to FP32. Record scale, original maxabs, zero flag, rounded endpoint count and actual clipped count separately. Endpoint occupancy is not automatically clipping. FP16 conversion uses NumPy IEEE rounding; overflow/nonfinite frames fail rather than silently saturate.

## Fixed schedule and gate

1. Check observed host memory and, when available, remaining cgroup budget. Require minimum3GiB before loading; observations are not allocation guarantees. Verify shared assets, load one reference and sequentially load two partition workers. Do not download or install.
2. For all12prefixes, compare FP32 transport against monolithic FP32 reference with `atol=rtol=1e-4` and exact next-token agreement. Stop if any fails. Retain the gate evidence; do not begin timing until all gates pass.
3. Two link conditions: no injected delay; **assumed**10ms one-way plus frame_bytes/1,000,000bytes/s. Delay applies only to the single stage boundary, not input-token or returned-logit traffic. It is an emulated wait, not an actual network.
4. For each prefix/link, exclude one warmup pair per codec:72warmup pairs total. Then3repeats×3codecs×12prefixes×2links = **216retained paired runs**, with one monolithic reference and one split call per run. Twelve distinct prompts, not216independent quality examples.
5. Codec order rotates over three repeats, so each codec occupies each position once per prefix/link. Monolithic-first versus split-first alternates by fixed index parity. Prefix and link order remain fixed, not randomized. Gate and warmup results must not be used to change prefixes, formats, repetitions or schedule.

## Measurements and interpretation

Per run retain full frame bytes (header and scale included), raw FP32 tensor size, both worker compute times, both worker encode/decode times, controller frame-validation time, total wall time, requested/observed emulated wait and wall time minus observed wait. The latter remains a same-host timing that includes pipe transfers, checks, scheduling and unrelated overhead; it is not pure compute time or predicted WAN latency. Stage0 encode and stage1 decode isolate the activation codec; stage0 decode concerns token JSON, stage1 encode concerns final FP32 logits. Final-logit frame validation and transfer are intentionally still present.

Compare final logits to the paired monolithic reference using max absolute error, RMS error, relative L2 error, **KL(reference || codec)** via float64 stable `log_softmax`, reference top1–top2 margin and exact next-token ID/agreement. KL can have a negligible negative floating-point residue; retain raw value rather than treating it as a meaningful negative divergence. Numerical drift is not interchangeable with token agreement, and agreement on12prefixes does not establish preserved generation quality. No pass threshold is imposed on lossy arms; report every outcome. FP32 gates remain enforced in retained pairs.

Report descriptive medians/min/max of retained samples, per-prefix accuracy and matched byte ratios against the FP32 frame for the same prefix. Aggregate timings mix several prefix lengths and should not imply a population speedup or calibrated confidence interval. All real compute occurs on one shared host. Compression may save emulated transmission time yet cost local serialization time. Three repeated deterministic outputs are timing repetitions, not additional independent accuracy evidence.

## Safety and failure evidence

Frames have fixed schema, bounded shapes/lengths, codec/scale validation and no pickle or executable payload. Reject unknown codecs, NaN/Inf, wrong sizes, invalid symmetric INT8 bytes, malformed/out-of-order metrics, or FP16 overflow. A request timeout/cancellation/protocol failure invalidates the session. Context exit records exitcodes, forced terminations and cleanup status; failed runs retain partial samples and failure reasons. Argparse handles missing explicit opt-in. No device discovery, model installation, credential use or OS network changes occur.

## Reproduce after review/publication

Reuse the previously documented optional CPU environment and pinned local model. No new dependency is introduced beyond its NumPy/PyTorch/Transformers stack. From this source tree:

```sh
python -m unittest discover -s tests -v
HF_HOME=/workspace/shared/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/workspace/shared/adaptive-pretrained-venv/bin/python -m mesh.transport_experiment \
  --model-dir /workspace/shared/pretrained-models/SmolLM2-135M-Instruct-12fd25f \
  --opt-in > results/transport_comparison.json
```

Shared paths above are the research environment example; other users substitute their local documented runtime/model paths. No retained result exists at this checkpoint. Tests use synthetic arrays/fake workers, not model outcome selection. Once published, record source commit, exact command, start/end time, environment and result hash alongside the retained report. Preserve failures and any amendments before reruns.
