from copy import deepcopy

def scenario(weights=(2, 2), budgets=(3, 3), **overrides):
    n = len(weights)
    devices = [dict(id=f'd{i}', budget_bytes=b, runtime_reserve_bytes=0,
                    eligible=True, layer_support=[True]*n,
                    embedding_support=True, head_support=True)
               for i, b in enumerate(budgets)]
    s = dict(version=1, layers=[dict(weight_bytes=w, kv_bytes=0, workspace_bytes=0,
                                   compute_ns=0) for w in weights],
             devices=devices, embedding_bytes=0, head_bytes=0, tied_embeddings=True,
             embedding_workspace_bytes=0, head_workspace_bytes=0,
             embedding_compute_ns=0, head_compute_ns=0,
             controller_reserve_bytes=0, activation_reserve_bytes=0,
             activation_payload_bytes=1, feedback_payload_bytes=1,
             links=[dict(source=a['id'], target=b['id'], available=True,
                         latency_ns=1, payload_bps=8_000_000_000)
                    for a in devices for b in devices if a != b])
    s.update(overrides)
    return deepcopy(s)
