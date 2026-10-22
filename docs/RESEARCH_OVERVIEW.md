# Household Compute
## Resource feasibility and correctness of consent-based model partitioning

Evidence snapshot: 22 October 2026. Read the companion [household inference feasibility paper](HOUSEHOLD_INFERENCE_FEASIBILITY.md) for detailed bounds, scaling calculations and their assumptions.

## Abstract

This project asks when consenting household devices could support inference that is too large for one device, and whether partitioning would be useful at an explicit service target. An analytical baseline checks device memory, stage costs and directed-link transfers. Single-host process studies test synthetic and pinned pretrained-model partitions; transport and cached-decoding extensions retain numerical evidence and resource ledgers. Cached FP32 decoding matches monolithic logits exactly on the fixed diagnostic sequences, while split median decode is slower. Household memory scenarios and transport forecasts remain calculations rather than observations of a deployed household. Model size is treated as a parameter-storage requirement. Capability, response quality and practical usefulness require separate measurements.

## Question and motivation

Memory that fits somewhere in a home is not necessarily memory available to an inference process. Operating systems, runtime libraries, activations, caches, duplicated weights and unsupported kernels constrain placement. A new device can add capacity while lengthening the path every generated token must traverse.

The central questions are:

1. Can every required stage fit on an eligible, consenting device after operational reservations?
2. Can a split preserve the chosen model's outputs under its actual numerical and transport implementation?
3. At a fixed workload, does it meet latency, throughput, quality and resource targets better than a measured alternative?

Consent is assumed for the feasibility question. Deployment must still implement enrollment, authentication, quotas and revocation. An analytical consent field does not implement those mechanisms.

## Related primary work

- [Distributed Inference and Fine-tuning of Large Language Models Over the Internet](https://arxiv.org/html/2312.08361v1), the Petals work, studies pipeline-sharded inference, load balancing and fault recovery. Its network measurements establish concrete precedents, not expected performance for unspecified household devices.
- [Model-Distributed Inference for Large Language Models at the Edge](https://arxiv.org/html/2505.18164v1) studies partitioned edge inference and recurrent pipeline parallelism across multiple sequences. Its measured device/network configuration must remain distinct from a generic home and from single-request latency.
- [The Llama 3 Herd of Models](https://arxiv.org/html/2407.21783v3#S3.T3) documents model architectures useful for weight and KV-cache arithmetic. Architecture and parameter count do not substitute for capability evaluation.

The [companion paper](HOUSEHOLD_INFERENCE_FEASIBILITY.md) develops these relationships. [The device-source record](../HOUSEHOLD_ATLAS.md) retains the household scenario inputs and uncertainty.

## Hypotheses and controls

**Feasibility hypothesis:** partitioning can make a particular model placement fit when the same model exceeds an individual device's usable budget. Evaluation must account for all resident weights and working state, including duplication. Aggregate advertised RAM alone cannot establish this condition.

**Correctness hypothesis:** a declared lossless split should match its monolithic reference under the same model revision, numerical mode and inputs. Identity/reset checks, full-vocabulary logits and complete row coverage test this narrow claim.

**Performance hypothesis:** some device/link configurations may satisfy a service target that a single device cannot. A future test must compare the best measured eligible single-device baseline, with equal model, precision, prompt/output lengths and quality criterion. Capacity benefit and latency benefit are separate outcomes.

**Coordination boundary:** even perfect placement decisions remain subject to required tensor sizes, supported computation and communication on the critical path. “More capable orchestration” cannot be inferred from fitting more parameters, nor can it remove those physical constraints.

## Methods and controls

The root [analytical baseline](../README.md#run) estimates serial compute plus crossing-boundary latency and payload transfer. Its memory rule keeps assigned weights resident and conservatively counts working buffers. It compares supplied placements, not every possible optimizer output. Illustrative rates and budgets are deliberately assumed.

[Process](../PROCESS_EXPERIMENT.md), [repeated-timing](../REPEATED_EXPERIMENT.md) and [pretrained](../PRETRAINED_EXPERIMENT.md) studies use local workers and private pipes. Delays are emulated where declared. The [transport-codec study](../studies/transport-codecs/TRANSPORT_EXPERIMENT.md) varies numerical transport on fixed prefixes. The [cached study](../studies/cached-decode/CACHED_EXPERIMENT.md) keeps each worker's KV cache local and compares every retained output against a monolithic FP32 reference. The [TCP codec](../studies/tcp-framing-codec/README.md) tests framing and accounting without sockets.

## Measured findings versus calculations

**Measured on one host:** real pretrained partition outputs matched the reference on the retained diagnostic inputs. The transport study retained 216 paired measurements; FP16 preserved argmax on the 12 fixed prefixes, while its tested global-scale INT8 method preserved none. [Transport results](../studies/transport-codecs/TRANSPORT_RESULTS.md).

**Measured cached decoding:** two retained runs each completed four prefills and 32 decode checks. Full logits, greedy tokens and reset/replay output matched exactly. Split/reference median decode was 74.99/67.94 ms and 71.31/65.62 ms. The controller retained a full reference; tied embedding/head weights were duplicated. Audit logits dominated IPC, and summed RSS can double-count shared pages. These measurements establish neither total-memory savings nor network throughput. [Cached results](../studies/cached-decode/CACHED_RESULTS.md).

**Calculated:** the 14 device exemplars, memory baskets, 27 capacity cases, link sensitivities and codec byte forecasts are reproducible scenario arithmetic. They are not a representative household survey or measured Wi-Fi/TCP deployment. [Scenario sources](../HOUSEHOLD_ATLAS.md).

**Proposed:** physical-device inference, authenticated cross-host transport, production consent enforcement, broad language-quality evaluation and energy comparisons remain unrun here. No corpus or harness candidate adds completed evidence by its existence.

## Reproducibility and limitations

[Start here](START_HERE.md) offers a standard-library illustration. [Root commands](../README.md#run) exercise the baseline; the [study guide](../studies/README.md) separates retained-evidence verification from optional model setup and new inference. Existing failure records and source amendments remain intact.

The current evidence uses small models, fixed diagnostic inputs and a shared host. Emulated delays omit contention, retransmission, radio conditions and real peer failure. The analytical baseline omits or simplifies overlap, batching, concurrency and energy. A pure-Python operation benchmark does not calibrate token generation. Privacy and correctness threats from actual peers require separate treatment.

## Falsifiable next steps

1. Predeclare usable memory, supported kernels, input/output lengths and service targets for a small consenting device set. Reject placements that fail full-state accounting before model execution.
2. Under a reviewed authenticated transport design, measure the same split locally and across hosts. Retain wire bytes, device memory, latency distributions, throughput and quality. Reject a claimed speedup if it vanishes against the best eligible single-device baseline or depends on unequal quality.
3. Test cancellation, bounded resource use and revocation independently of numerical correctness. A correct token sequence alone cannot establish safe operation.

The companion paper supplies a theoretical framework for choosing these tests. Its calculated bounds and proposed experiments must retain their own evidence labels.
