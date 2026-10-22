# consent-compute-mesh

Opt-in local inference mesh research: reproducible bandwidth, latency, memory, and device feasibility experiments.

## Current status

This is a newly implemented, offline research baseline, not a recovered previous implementation or a working distributed inference service. It contains:

- A serial, single-request placement/latency model with explicit consent, memory and directed-link checks.
- A real, local CPU microbenchmark using a deterministic pure-Python matrix-vector kernel.
- Reproducible illustrative scenarios and automated tests.
- A real two-process, private-pipe synthetic neural forward experiment with explicitly emulated link delays and failure handling. See [Process experiments](PROCESS_EXPERIMENT.md).

The optional process experiment starts only fixed-kernel local workers after explicit opt-in. It does **not** discover devices, listen on a port, connect to hardware, install packages, load an AI model, or execute remote jobs. Python 3.10+ and its standard library are sufficient.

## Run

From the repository root:

```sh
python -m unittest discover -s tests -v
python -m mesh simulate examples/illustrative.json
python -m mesh benchmark --size 64 --iterations 20 --repeats 5
```

Both commands print JSON. To save your own measurements, redirect stdout to a new file. Benchmark runtime depends on your Python interpreter, hardware and competing workloads. Do not compare its nominal operation rate directly to advertised hardware FLOPS or use it as an LLM token-rate estimate.

## What is measured versus assumed?

`results/illustrative_simulation.json` is deterministic analytical output from `examples/illustrative.json`. Every device speed, memory budget, stage workload and link property in that example is invented for demonstration, not a specification or measured performance claim. The device names are illustrative labels.

`results/local_cpu_benchmark.json` is a real run in the development cloud container using CPython 3.12.14 on x86_64. The file contains raw wall-clock samples, workload dimensions, operation-count convention and checksum. It is a small Python-interpreter benchmark, **not measured neural-network inference**. It does not calibrate the simulated device rates. Run it yourself for a fresh local measurement.

## Example finding, not a hardware claim

For the supplied assumptions:

- Laptop-only placement: 0.60 seconds estimated compute, no inter-device transfer.
- Split placement: 1.50 seconds estimated compute plus 1.01 seconds transfer, totaling 2.51 seconds.
- ESP32-class placeholder: rejected because the assigned stage exceeds its assumed memory budget.
- Non-opted-in device: rejected even though its illustrative speed is highest.

This demonstrates why adding devices is not automatically faster: intermediate activations and slower workers can dominate. It does not prove that all distributed placements are slower.

## Model and units

Each stage has `operations`, `weights_bytes`, `workspace_bytes` and `output_bytes`. Each device supplies available `memory_bytes`, effective workload-specific `ops_per_second` and explicit boolean `consent`. Each directed link supplies effective payload `bytes_per_second` and fixed one-way `latency_ms` per crossing. Bytes are bytes, not bits; seconds are used in results.

For a given explicit placement:

```text
compute_seconds = sum(stage.operations / assigned_device.ops_per_second)
transfer_seconds = sum(boundary.output_bytes / link.bytes_per_second + link.latency_ms / 1000)
total_seconds = compute_seconds + transfer_seconds
```

Only boundaries crossing devices incur a modeled transfer. There is no automatic network routing. Reverse links must be provided separately.

All assigned stage weights stay resident. Required memory on each device is the sum of its assigned weights plus the maximum of `(workspace + incoming activation + output activation)` across its stages. This conservative accounting assumes separate activation buffers, no in-place reuse, and no overlapped requests. Initial input is zero bytes in this baseline, and weights are already loaded. Set device memory to the budget available **after** OS/runtime reservations, not advertised total RAM.

The command compares only the named placements in the scenario, sorts feasible candidates by estimated latency, and reports rejected candidates separately. It is not an exhaustive placement optimizer.

## ESP32 and heterogeneous-device limits

The ESP32-class example is deliberately a small-memory placeholder. No ESP32 board, firmware or wireless link was benchmarked. Boards differ in RAM, PSRAM, kernels, clocks and radio conditions. Do not assume a desktop model shard fits or that Wi-Fi bandwidth implies adequate inference throughput. A real experiment must measure available memory, supported kernels, workload latency, thermal/power limits and effective link transfer independently. Tiny sensor preprocessing or control may be a more realistic role than a large-model shard, but that remains an untested design direction here.

## Safety and consent boundary

This program is offline. A boolean in a local JSON file is **not** authentication, proof of ownership, or enforceable consent. It only exercises fail-closed placement logic: a selected device must have `consent: true`, exactly. It grants no access to anyone's devices.

Before adding any real worker protocol, require explicit owner enrollment, authenticated peers, encrypted transport, bounded work/resource quotas, cancellation and consent revocation, and a clear policy for whether prompts or activations may leave the originating device. Do not add arbitrary remote-code execution or expose a listener by default. There is no real network protocol in this checkpoint.

## Limitations and next experiments

The model omits batching, pipeline overlap, concurrent requests, KV caches, serialization, weight loading, packet contention, reliability, tensor parallelism and energy. Supply those memory costs conservatively through workspace where applicable, or extend and validate the model before drawing conclusions. Effective operations/sec is workload-specific, not a portable device constant.

Next useful work is to measure a small real inference workload on one explicitly opted-in host, then validate a two-device latency prediction against actual transfers under a separately reviewed consent/authentication design. Hardware and distributed validation have not been performed.
