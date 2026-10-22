# Repeated paired measurements with persistent workers

The first process experiment uses one request per fresh worker pair. This milestone adds reusable workers and preserves the previous command unchanged.

```sh
python -m mesh.repeated --opt-in --widths 8 32 64 --repeats 7 --warmup 2
python -m unittest discover -s tests -v
```

## Method

Only two worker processes are launched for the entire suite. Each serves every width/link case sequentially over private pipes. We measure the same fixed synthetic dense/ReLU workload as the previous milestone, not a pretrained model.

For each case, both execution modes run during two excluded warmup pairs, followed by seven retained pairs. Retained pairs alternate reference-first and split-first order to reduce systematic ordering bias. Both modes use identical deterministic inputs and layers; every retained pair must match exactly. A mismatch or failed request stops the suite and records the failure. Timed-out sessions cannot be reused, preventing a late response from contaminating the next request.

Startup/cleanup are reported separately. `split_seconds` includes real serialization, IPC, scheduling, worker execution and any deliberately emulated wait. `reference_seconds` measures the same forward computation in the controller without serialization/IPC. Worker compute times, payload bytes, requested delay and observed wait are retained for each pair. Warmup samples are excluded from summaries and not stored.

The CLI compares no deliberate delay against an emulated 10 ms one-way interstage latency and 100,000 bytes/second payload bandwidth. The Python API accepts a bounded list of explicit link assumptions. These are controller sleeps, not actual network traffic shaping. Links apply only to the intermediate activation, as described in [Process experiments](PROCESS_EXPERIMENT.md).

## Recorded result

`results/repeated_process_suite.json` contains 42 measured pairs across widths 8, 32 and 64, and the two link cases. Twelve warmup pairs were excluded. All measured outputs matched exactly, and both workers exited normally.

For width 64 without deliberate delay, the recorded median was 0.913 ms for the same-process reference and 1.548 ms for the split execution. The split minimum/maximum were 1.283/1.845 ms. With the emulated link, the split median was 19.039 ms. These values describe this recorded run only and will change on rerun.

The results demonstrate overhead rather than speedup for this tiny workload. Reusing workers removes repeated startup cost, but does not eliminate serialization, IPC, scheduling or communication cost.

## What confidence means here

Reported count, median, mean, minimum and maximum are **descriptive sample statistics**, not confidence intervals, population percentiles or guarantees. There are only seven retained samples per case, from one shared cloud host in one session. Repeated measurements are not independent hardware trials. Adjacent cases share workers and may be affected by cache warmth, CPU scheduling, thermal state and background load. Cases run in fixed order, even though the order within pairs alternates.

We have strong direct evidence that the tested partitioning preserves output and that these workers are reused and cleaned up. We do not yet have evidence of real-world latency, heterogeneous hardware scaling, useful model quality, energy savings or robust speedup. No VM, ESP32, Android, adb, LAN or physical network was used.

## Lifecycle and limits

Explicit opt-in remains required. Widths are 1–256, at most eight widths and eight links are accepted, repeats are 1–100, warmups are 0–10, and suites are limited to 1,000 total pairs. Derived delays are checked for finite values before worker startup. Each request has the same five-second default deadline; startup has a separate ten-second budget. Cancellation and cleanup retain the previous private-pipe behavior.

This is an experiment tool, not a service API or resource sandbox. The worker accepts only the fixed local kernel. Python callers must use a `__main__` guard with the spawn process method.

## Next: a real pretrained workload, not more synthetic features

The persistent-session lifecycle and paired measurement format are the intended base for a separately tested fixed pretrained-model adapter. That extension must use the same pinned model and input tokens in both reference and split execution, retain layers across requests, specify dtype and CPU threading, verify output/logit tolerances, and separately record model-load time and actual encoded activation sizes. Model-loading dependencies remain optional: this repository's current commands require only the standard library.

A candidate runtime/model is being evaluated separately; no pretrained-model inference or partition validation is claimed in this checkpoint. Introducing an adapter does not authorize downloading arbitrary code, enabling remote-code trust, contacting other devices or opening network listeners.
