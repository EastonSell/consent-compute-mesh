# Start here: enough memory is not the same as fast inference

**The question:** What useful model work could a household's devices perform together, assuming every owner has given full, informed consent?

The research question is practical capability, not whether to withhold consent. A phone, laptop, router, or refrigerator is not automatically an interchangeable model worker. Accessible memory, supported computation, sustained speed, and communication all matter. This repository is research, not a working household inference service.

## First separate capacity from speed

**Weights** are the model's stored numerical parameters. More aggregate memory can help store more weights, but it does not show that each assigned part fits a particular device or that a request finishes faster.

Memory also holds the runtime, temporary work, and a **KV cache**: saved attention keys and values used while generating later tokens. That cache grows with context length and batch size. Device operating systems and other apps need their own space too. Use an allocatable budget after reservations, not advertised total RAM.

![Two separate checks: a hypothetical 70B model needs 32.596 GiB of ideal 4-bit weights or a 43.616 GiB illustrative runtime envelope; at assumed 12 GiB pools the aggregate lower bounds are three or four. Below, two serial compute stages plus a 26.5536 ms hypothetical transfer are slower than the same total compute without transfer.](figures/capacity-versus-speed.svg)

## A transparent memory calculation

Take a rounded **70 billion parameters** at an ideal **4 bits per weight**:

`70,000,000,000 × 4 / 8 = 35,000,000,000 bytes`

That is **35.00 GB**, but **32.596 GiB**. One GB is 1,000,000,000 bytes; one GiB is 1,073,741,824 bytes. Ideal packing excludes scales, metadata, mixed-precision tensors, and actual runtime behavior.

Using the [cited household capability atlas's](../HOUSEHOLD_ATLAS.md#reproducible-capacity-arithmetic) explicit 70B dimensions, an FP16 KV cache at 8,192 tokens and batch one is **2.5 GiB**. Add an *illustrative*, not measured, 20% weight overhead and 2 GiB aggregate workspace:

`1.20 × weights + KV + 2 GiB = 43.616 GiB`

At an **assumed 12 GiB allocatable per pool**, the aggregate lower bound changes from **3 pools** for weights alone to **4** for that envelope. This is not a household average, measured free RAM, sufficient shard-placement proof, or upper bound on required memory. The atlas separates cited device exemplars from analyst-chosen baskets; unknown spare capacity remains unknown, not silently credited or declared physically zero.

## Why adding devices need not make one request faster

**Bandwidth** is data delivered per second. **Latency** is the fixed delay before that delivery is useful. A layer split sends intermediate numerical results, called **activations**, across a boundary.

Hypothetically send **16 KiB = 16,384 bytes** over **20 Mbps** payload bandwidth with **20 ms one-way latency**:

`20 ms + 16,384 / (20,000,000 / 8) seconds = 26.5536 ms`

Mbps means megabits, not megabytes, per second. With serial stage compute of 0.30 s + 0.30 s, one such crossing makes **0.6265536 s**, versus **0.60 s** without the crossing, assuming the same total compute. A slower second device could add more delay. This is an example, not a measured device comparison.

For 15 serial boundaries, communication alone is 0.398304 s/token, giving a roughly **2.511 token/s communication-only ceiling** under this topology. It excludes compute, serialization, return traffic, contention, and failures. It is not a predicted achieved rate. Pipeline overlap across independent requests can change throughput; it does not erase dependencies in one serial request.

Try the [offline calculator](examples/capacity_latency.py), using Python 3.10+ from the repository root:

```sh
python docs/examples/capacity_latency.py
python -m unittest discover -s docs/examples -p 'test_*.py' -v
```

All link rates, delays, budgets, and overheads in this teaching example are explicit assumptions. It performs no device discovery, network access, package installation, or model execution.

## What the real comparisons establish

- **Monolithic versus split:** keep model/input fixed and compare outputs as well as time. The [pretrained split study](../PRETRAINED_RESULTS.md) matches reference logits on its few fixed inputs over local private pipes. Those are single-host measurements with emulated delays, not Wi-Fi or household measurements. Small noisy timing differences are not established speedups.
- **Encoding control:** [FP32/FP16/INT8 transport](../studies/transport-codecs/TRANSPORT_RESULTS.md) compares bytes and numerical preservation. FP16 kept the reference next-token choice on 12 fixed prefixes; the tested whole-tensor INT8 scheme kept none. That is not general language quality or a verdict on every INT8 method.
- **Cached versus full-forward scope:** [cached decode](../studies/cached-decode/CACHED_RESULTS.md) retains stage-local caches and matches reference logits/tokens in two 36-row runs. Split median decode was slower. Duplicated components and the controller's reference model prevent a total-memory-saving claim.
- **Framing versus networking:** the [TCP framing codec](../studies/tcp-framing-codec/README.md) tests source-only message framing and byte accounting. It implements no sockets or model execution. Forecast bytes are not measured TCP performance.

## Inspect and reproduce

The [root commands](../README.md#run) run the standard-library analytical model and local kernel tests. `python -m unittest discover -s tests -v` runs the root suite; its real-model integration tests require separate explicit setup. The [study guide](../studies/README.md) separates retained-evidence checks from optional pinned-model reproduction. The household atlas links original sources and machine-readable calculations.

**Useful next questions:** Does each shard plus local cache actually fit? Which operators can each exact device run? What is sustained prefill/decode speed under thermal and household load? What happens on real links or disconnects? Compare quality, per-request latency, multi-request throughput, and energy separately. Full consent supplies permission, not these missing measurements.
