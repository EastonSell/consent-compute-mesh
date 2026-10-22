"""Hypothetical arithmetic only; no device, network, or model is executed.

Run from the repository root: python docs/examples/capacity_latency.py
"""
import math

GB = 1_000_000_000
GIB = 1_073_741_824


def _positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Expected a positive integer")
    return value


def weight_bytes(parameters, bits):
    """Ideal packed weights only, rounded up to a whole byte; no metadata."""
    return (_positive_integer(parameters) * _positive_integer(bits) + 7) // 8


def kv_bytes(layers, kv_heads, head_dimension, context_tokens, batch=1, element_bytes=2):
    """Ideal dense keys + values under explicitly supplied dimensions."""
    dimensions = (layers, kv_heads, head_dimension, context_tokens, batch, element_bytes)
    return 2 * math.prod(_positive_integer(value) for value in dimensions)


def pool_count(required_bytes, allocatable_bytes):
    """Aggregate capacity lower bound, not proof that a placement exists."""
    required = _positive_integer(required_bytes)
    budget = _positive_integer(allocatable_bytes)
    return (required + budget - 1) // budget


def crossing_seconds(payload_bytes, bandwidth_mbps, latency_ms):
    """Hypothetical one-way serial transfer: latency + bytes / payload byte rate.

    Mbps means decimal megabits/second; divide by eight to obtain bytes/second.
    """
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise ValueError("Payload must be a nonnegative whole-byte count")
    for value in (bandwidth_mbps, latency_ms):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Link assumptions must be finite numbers")
    if bandwidth_mbps <= 0 or latency_ms < 0:
        raise ValueError("Bandwidth must be positive and latency nonnegative")
    return latency_ms / 1000 + payload_bytes / (bandwidth_mbps * 1_000_000 / 8)


def main():
    print("HYPOTHETICAL CALCULATIONS, NOT DEVICE OR NETWORK MEASUREMENTS")
    weights = weight_bytes(70_000_000_000, 4)
    kv = kv_bytes(80, 8, 128, 8192)
    # Explicit illustrative 20% weight overhead + 2 GiB aggregate workspace.
    envelope = (weights * 6 + 4) // 5 + kv + 2 * GIB
    budget = 12 * GIB
    print(f"Ideal 70B 4-bit weights: {weights:,} B = {weights / GB:.2f} GB = {weights / GIB:.3f} GiB")
    print(f"Assumed FP16 KV at 8192 tokens, batch 1: {kv / GIB:.3f} GiB")
    print(f"Illustrative envelope: {envelope:,} B = {envelope / GIB:.3f} GiB")
    print(f"At assumed 12 GiB per pool: weights-only {pool_count(weights, budget)}; envelope {pool_count(envelope, budget)}")
    crossing = crossing_seconds(16_384, 20, 20)
    print(f"Assumed 16 KiB crossing at 20 Mbps + 20 ms one-way: {1000 * crossing:.4f} ms")
    print(f"Assumed serial 0.30 s + 0.30 s compute with one crossing: {0.6 + crossing:.7f} s, versus 0.6000000 s unsplit")
    serial = 15 * crossing
    print(f"15 serial crossings: {serial:.6f} s/token; communication-only ceiling {1 / serial:.3f} token/s")
    print("No compute, serialization, return traffic, contention, or actual placement is established by that ceiling.")


if __name__ == "__main__":
    main()
