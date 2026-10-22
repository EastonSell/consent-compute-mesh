# Recorded real-model partition results

The amended-protocol repeat used source commit `e1fa478d91d878c8480ed24956106e0d8bbc8254` after publication and review. The earlier exploratory execution remains available separately; see `results/pretrained_execution_record.json` for chronology and output hashes. Neither protocol was independently preregistered.

Both runs passed their two logit-correctness gates and all 12 retained paired samples with maximum absolute difference 0.0. Next-token argmax matched. Each case excluded one warmup pair and retained only three samples; both workers exited normally without forced termination.

## Post-publication repeat

| Input tokens | Emulated link | Reference median (ms) | Split median (ms) | Split min–max (ms) |
|---:|---|---:|---:|---:|
| 5 | None | 76.94 | 76.85 | 76.32–81.50 |
| 5 | 10 ms + 1 MB/s | 74.34 | 103.65 | 103.15–105.01 |
| 18 | None | 101.37 | 111.84 | 106.07–118.34 |
| 18 | 10 ms + 1 MB/s | 107.17 | 167.10 | 166.29–172.77 |

The activation frames were 11,557 bytes for 5 tokens and 41,510 bytes for 18 tokens. The emulated waits requested 21.557 ms and 51.510 ms respectively. Actual validation, serialization and IPC costs remain in split timings. Startup took 3.40 seconds and reference model loading took 0.75 seconds; neither is included in the table.

## What this establishes

- A real pinned 135M-parameter model can be split at this layer boundary, transported losslessly over the tested private-pipe protocol, and match the monolithic forward pass on these inputs.
- The 18-token no-delay split was slower in the recorded sample medians. The 5-token no-delay medians were essentially tied; a difference of about a tenth of a millisecond in three noisy samples does not establish speedup.
- Added emulated communication delays increased end-to-end time as expected. This is not a measurement of Ethernet, Wi-Fi, VMs, ESP32 or any physical network.

## Limits and preliminary execution

Only two prefixes and three retained samples per case were tested, with fixed case order and 2/1 within-pair order imbalance. The host was shared and other research work occurred around these executions; changing contention is a plausible contributor to timing differences. Reported ranges describe observed samples, not confidence intervals.

The exploratory run is preserved as `results/pretrained_exploratory_prepublication.json`; the amended repeat is `results/pretrained_postpublication_repeat.json`. The preliminary run happened before publication. Independent review then added report/frame validation and failure-evidence preservation. Those changes add overhead, so the two outputs must not be represented as identical-protocol replication or cherry-picked to claim a performance improvement.

Observed available host memory before the amended repeat was 8525.3 MiB. Controller peak RSS was 1065.5 MiB. Worker peak-RSS readings and load times are in the raw output, but per-process peaks are not simultaneous usage and can include process-startup high-water effects. No memory-saving claim is made.

This is full-forward final-token-logit latency, not autoregressive generation throughput, language-task quality, training benefit, energy efficiency or deployment readiness. KV cache, padding, batching and cross-host operation remain unsupported.
