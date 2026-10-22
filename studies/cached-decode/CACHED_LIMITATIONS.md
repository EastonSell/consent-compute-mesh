# Scope and accounting notes

This prototype uses informed, explicit local opt-in and private pipes between two processes on one host. It is not a VM, Android/ADB deployment, phone cluster, real network, or framework implementing all future consent/security/revocation requirements.

Controller sends token IDs to stage0, receives a stage0 activation response, forwards those activations in a new request to stage1, and receives the full vocabulary logits. Thus one logical stage boundary crosses two actual IPC paths via the controller. request_bytes and response_bytes each contain stage0 and stage1 frame lengths. activation_frame_bytes describes the stage1 request only, NOT all IPC bytes or network traffic. Our application headers are included; multiprocessing outer framing and OS overhead are excluded. There are no real network packets here.

Full vocabulary FP32 logits add 196608 payload bytes per step, far larger than a single-token boundary activation's 2304 payload bytes. This is deliberate correctness-audit overhead. A deployed implementation could sample at the final stage and return a token ID, but that alternative is not implemented, measured or used to project savings here.

Worker times include decoding and packing. Controller times include validation, IPC and scheduling. Prefill and decode are separate phases; none of these same-host measurements estimates household throughput. Worker processes load assigned layers only, but tied embedding/output-head weights are duplicated across workers and a complete controller reference is also loaded. Cache tensor bytes are measured; RSS guards are sampled and may overshoot, not OS-enforced memory caps. No memory-saving claim is supported.

Evidence-component SIGINT tests establish that component's incremental row retention. They do not prove full inference-runner interruption recovery, SIGKILL finalization, or power-loss durability. Every failure remains in a fresh output directory. Earlier source checkpoints and the failed pre-forward v3 resource smoke are retained rather than overwritten.
