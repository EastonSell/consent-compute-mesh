# Same-host worker experiment

This additive milestone runs **two actual Python worker processes**, spawned separately, over private OS pipes. It is a safe first step toward studying the idea of running cooperating workers in separate VMs, but **does not create VMs or emulate their CPU/memory isolation**. It needs no installations, network listeners, LAN access, adb, administrative privileges or OS network changes. The original analytical model and CPU microbenchmark remain available unchanged.

## Run

```sh
python -m mesh.process_cli --opt-in
python -m mesh.process_cli --opt-in --latency-ms 10 --bytes-per-second 100000
python -m mesh.process_cli --opt-in --fault disconnect
python -m mesh.process_cli --opt-in --fault stall --timeout-seconds 0.1
python -m unittest discover -s tests -v
```

Without `--opt-in`, the command refuses to start workers. Fault examples intentionally return exit code 1 and a JSON failure report. Successful runs return 0; invalid arguments return 2. The opt-in is only for these two fixed-kernel local workers, not other machines or future network access.

## What actually executes

A deterministic width-64 vector passes through two dense layers with generated synthetic weights, biases and ReLU activations. Worker 0 executes layer 0; worker 1 executes layer 1. The controller relays the intermediate activation through JSON-encoded pipe messages. The same two layers also run in the controller as the single-process reference. The report includes maximum absolute output difference and checksum. This validates partitioning and transport consistency, not predictive accuracy: there is no trained model, dataset, text generation, or quality evaluation.

Each run creates two fresh workers, waits for readiness, runs one forward pass, then closes them. Width is bounded to 1–256. Workers execute only the fixed function, never supplied code or commands. This is not a security sandbox, scheduler, multi-user runtime, or persistent mesh.

## Real timing versus emulated timing

Measured values are:

- `reference_seconds`: one actual forward pass in the controller.
- `startup_seconds`: actual process launch and readiness time, excluded from inference timing.
- `worker_compute_seconds`: actual fixed-layer execution time in each worker, excluding JSON decoding/encoding.
- `inference_wall_seconds`: actual controller wall time for the partitioned request, including IPC, serialization, scheduling and any emulated waiting; excludes startup and cleanup.
- `emulated_wait_observed_seconds`: actual wall time spent in the deliberate delay wait, including scheduler overshoot.
- `cleanup_seconds`: actual worker shutdown/termination time.

Assumptions are:

- `latency_ms_assumption`: one-way latency charged once at the interstage activation boundary.
- `bytes_per_second_assumption`: effective payload bandwidth, not measured link bandwidth.
- `emulated_wait_requested_seconds`: latency / 1000 plus encoded activation bytes / bandwidth. Omitted bandwidth means no bandwidth delay, not an infinite-bandwidth hardware claim.

The actual pipe message bytes determine `interstage_payload_bytes`. Initial input, final output, readiness and control traffic are **not** assigned emulated link delays. There is no packet model, queueing, contention, retransmission, jitter or bandwidth sharing. The delay runs in the controller before it sends the activation to worker 1. It is not a network traffic shaper.

`results/local_process_no_delay.json` is a real same-host run without deliberate delay. `results/local_process_emulated_link.json` is a separate run with 10 ms latency and 100,000 B/s assumed bandwidth. For width 64, the interstage activation encodes to 707 bytes, so the requested delay is 17.07 ms. Both recorded runs exactly match their reference output. `results/local_process_disconnect.json` records an intentionally disconnected worker and successful cleanup.

These are **single samples** on a shared cloud CPU, not statistically robust performance comparisons. Startup costs dominate small workloads; worker processes can be slower than same-process execution. Do not claim speedup from these examples, infer physical hardware throughput, or equate private pipes with Wi-Fi, Ethernet, ESP32, Android/adb or VM networking.

## Timeouts, cancellation and cleanup

Startup has a separate 10-second deadline. `--timeout-seconds` bounds the request after readiness. Polling and deliberate waits check cancellation/deadline at roughly 10 ms intervals; OS scheduling and cleanup mean this is not a hard real-time guarantee. Workers normally receive a stop message and exit. A deliberately stalled worker is given a 0.2-second grace period, then terminated; a kill fallback is available if termination does not complete. Exit codes, forced-termination flags and whether all workers exited are reported.

For programmatic cancellation, pass a `threading.Event` as `cancel_event` to `run_experiment` and set it from another thread. Use the usual `if __name__ == '__main__':` guard in scripts because workers use Python's `spawn` start method. Tests exercise cancellation during an emulated wait, request timeout, a disconnected worker, a stalled worker and worker cleanup. A forcibly terminated worker can have a nonzero exit code; this is expected in the stall fault test.

This experiment does not establish authentication, owner enrollment, revocable network consent or resource isolation. Those remain prerequisites before any real cross-host protocol. A future VM experiment would need separately authorized setup and independent measurement of its runtime and virtual-network behavior.
