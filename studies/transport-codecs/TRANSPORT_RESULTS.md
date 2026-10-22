# Activation transport result: smaller bytes can destroy the signal

**Primary run:** 216 paired measurements on one shared CPU host, 12 fixed original prefixes, three codecs, two link conditions and three timing repetitions. All 12 FP32 correctness gates and retained FP32 samples exactly matched monolithic logits. Both workers exited normally with code zero; 72 warmup pairs were excluded. This is not a physical-network or model-quality benchmark.

![Calculated compression and output-preservation summary](figures/transport_codec_summary.svg)

## The result worth keeping

| Boundary format | Median full-frame fraction of matched FP32 | Same reference next token, distinct prefixes | Worst absolute logit drift | Worst KL(reference ‖ codec), nats |
|---|---:|---:|---:|---:|
| FP32 | 100% | 12 / 12 | 0 | 0 |
| FP16 | 50.10% | 12 / 12 | 0.0042515 | 4.5686 × 10⁻⁸ |
| Whole-tensor symmetric INT8 | 25.20% | **0 / 12** | **33.1948** | **17.4605** |

The payload alone is exactly half/quarter size; headers and the INT8 scale make full-frame ratios slightly larger and prefix-dependent. “Agreement” means agreement with this model's monolithic next-token choice, **not factual correctness or task accuracy**. The three repeats and two links produced identical numerical outcomes per prefix/codec, so they do not create 72 independent quality examples per codec.

**FP16 is a promising boundary encoding for this narrow setup. The tested whole-tensor INT8 scheme fails this numerical-fidelity check despite saving more bytes.** This does not condemn INT8 generally, other scaling granularity, other layers or other models. We did not tune quantization after seeing the result or replace any samples.

## Timings: real local work, hypothetical transmission waits

| Boundary format | No-delay split median | Emulated-link split median | Observed injected wait median |
|---|---:|---:|---:|
| FP32 | 111.22 ms | 154.31 ms | 41.29 ms |
| FP16 | 110.02 ms | 140.73 ms | 25.82 ms |
| INT8 | 109.49 ms | 134.49 ms | 18.04 ms |

Each cell contains 36 paired runs spanning 12 different prefix lengths. The emulated boundary is **assumed 10 ms one-way plus frame bytes / 1 MB/s**, not a measured link. The close no-delay medians are not reliable evidence of a speed advantage; scheduling, shared-host load and differing paired reference medians remain visible in the raw data. Faster delivery of incorrect INT8 results is not useful acceleration.

Actual no-delay median activation encode/decode times were approximately FP32 0.187/0.136 ms, FP16 0.217/0.151 ms and INT8 0.248/0.200 ms. These measurements include codec validation/copy costs in the corresponding functions. Parent validation, local pipes and final FP32-logit transport remain inside total wall time; startup, warmup and numerical metric calculation are excluded. No generation loop or KV-cache optimization was evaluated.

## Why INT8 failed: primary observations versus post-hoc inspection

The primary run logged activation maximum magnitudes of approximately **23,704–27,317**, producing whole-tensor steps of **186.64–215.09**. Thus magnitudes below roughly 93.32–107.55 round to zero. There was no clipping before casting; one element per tensor reached a quantization endpoint. This established a plausible coarse-grid mechanism, but the primary run did not log the activation distribution or archive tensors.

A **separate, explicitly post-hoc diagnostic** was then authorized to inspect all 12 unchanged prefixes without tuning any model, split or codec. It archives original FP32 activations, exact encoded frames and final FP32 logits for every codec. A second small diagnostic archives monolithic baseline logits. These are not replacements for the primary 216-run evidence and make no timing claims. Independent signed-byte parsing, Python ties-to-even rounding and scalar scale reconstruction check the actual frames for sign/scale/endianness mistakes. The diagnostic found **99.256–99.879% of activation entries became zero**. Original median absolute values were only 1.96–2.54; global relative L2 reconstruction error was 3.04–4.65%, demonstrating why a whole-tensor norm dominated by large outliers can hide loss of small features. All 36post-hoc logit/argmax/KL outcomes exactly reproduced the primary numerical evidence against separately archived monolithic logits. These observations support the coarse-grid explanation in this setup, not a general statement about INT8. See the separate diagnostic records for exact per-prefix values and archive hashes.

## Primary-source context: outliers are a known quantization concern

[LLM.int8() (Dettmers et al., NeurIPS 2022)](https://arxiv.org/abs/2208.07339v2) uses vector-wise normalization for most features and a higher-precision path for outlier feature dimensions in matrix multiplication. [SmoothQuant (Xiao et al., ICML 2023)](https://arxiv.org/abs/2211.10438v7) transforms activation/weight scaling offline to reduce activation quantization difficulty for W8A8 execution. Their original abstracts and version histories were checked October 22,2026.

These papers provide context for why scale granularity and outliers matter, not validation of this experiment or ready-made fixes. Our experiment quantizes one residual-stream boundary tensor **for transport**, then resumes FP32 computation; it neither implements nor reproduces either paper's quantized-GEMM method, calibration or quality evaluation. The observed failure is a specific local result consistent with the general concern, not a newly discovered universal law.

## Provenance and reproducibility

- [Prospective protocol](TRANSPORT_EXPERIMENT.md), [original wording](data/transport_protocol_original.md), and [prospective amendment record](data/transport_protocol_amendment.json). The amendment removed GitHub upload completion as a prerequisite, while preserving independent local review and frozen source before execution. Float64 metric arithmetic was corrected before any model forward. This was not independently preregistered.
- [Frozen source hashes](results/transport_source_checkpoint.json), [execution record](results/transport_execution_record.json), and [fixed texts/token IDs/tokenizer hashes](data/transport_prefixes.json).
- [Unchanged primary JSON](results/transport_comparison.json), [six-condition summary CSV](results/transport_summary.csv), [36 prefix/codec numerical rows](results/transport_prefix_metrics.csv).
- [Post-hoc activation diagnostic](results/transport_posthoc_diagnostic.json), [monolithic-reference verification](results/transport_posthoc_reference.json). Raw archives are listed separately in `TRANSPORT_BINARY_FILES.txt`; absence from a text-only publication must not be mistaken for absence of recorded arrays.

Runtime: Python 3.12.14, PyTorch 2.5.1+cpu, Transformers 4.46.3, NumPy 2.3.5, x86_64; one CPU thread per model process. Host available memory before the primary load was about 8.92 GB; no finite cgroup limit was observed. Exact model and configuration hashes are checked by the existing adapter. External start/end observations are bounds, not precise process timestamps.

Rebuild descriptive tables without rerunning the model:

```sh
python -m mesh.transport_report
python -m mesh.transport_figure  # optional matplotlib, for SVG/PNG
```

Run the separately labeled post-hoc diagnostics with the same documented optional runtime and local weights:

```sh
python -m mesh.transport_diagnostic --model-dir LOCAL_PINNED_MODEL --opt-in
python -m mesh.transport_reference_diagnostic --model-dir LOCAL_PINNED_MODEL --opt-in
```

All prompts were fixed before primary outcomes. No cases were removed, thresholds relaxed or quantizer settings changed. Caught failures would retain partial results, but the primary CLI has no incremental durability on abrupt termination; this limitation was disclosed before execution. The next scientific question is whether a separately specified finer-grained or outlier-aware codec preserves these signals, not whether the present negative result can be relabeled a success.
