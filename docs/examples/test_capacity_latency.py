"""Unit and arithmetic checks for hypothetical scenarios, not device benchmarks."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name("capacity_latency.py")


class CapacityLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = None
        if PATH.exists():
            spec = importlib.util.spec_from_file_location("capacity_latency", PATH)
            cls.demo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.demo)

    def setUp(self):
        self.assertIsNotNone(self.demo, "Teaching module is not implemented yet")

    def test_nominal_four_bit_weights_use_decimal_parameter_count(self):
        self.assertEqual(self.demo.weight_bytes(70_000_000_000, 4), 35_000_000_000)

    def test_fractional_bytes_round_up(self):
        self.assertEqual(self.demo.weight_bytes(3, 4), 2)

    def test_kv_includes_keys_and_values(self):
        self.assertEqual(self.demo.kv_bytes(32, 8, 128, 8192), self.demo.GIB)

    def test_seventy_billion_capacity_matches_explicit_assumptions(self):
        weights = self.demo.weight_bytes(70_000_000_000, 4)
        kv = self.demo.kv_bytes(80, 8, 128, 8192)
        budget = 12 * self.demo.GIB
        self.assertEqual(self.demo.pool_count(weights, budget), 3)
        envelope = (weights * 6 + 4) // 5 + kv + 2 * self.demo.GIB
        self.assertEqual(self.demo.pool_count(envelope, budget), 4)

    def test_megabits_are_not_megabytes(self):
        self.assertAlmostEqual(self.demo.crossing_seconds(16384, 20, 20), 0.0265536)

    def test_fifteen_serial_boundaries_add(self):
        seconds = 15 * self.demo.crossing_seconds(16384, 20, 20)
        self.assertAlmostEqual(seconds, 0.398304)

    def test_zero_payload_still_pays_latency(self):
        self.assertEqual(self.demo.crossing_seconds(0, 20, 20), 0.02)

    def test_memory_boundary_is_inclusive(self):
        self.assertEqual(self.demo.pool_count(24, 12), 2)
        self.assertEqual(self.demo.pool_count(25, 12), 3)

    def test_invalid_inputs_are_rejected(self):
        for call in (lambda: self.demo.weight_bytes(True, 4),
                     lambda: self.demo.weight_bytes(8, 0),
                     lambda: self.demo.kv_bytes(0, 8, 128, 8192),
                     lambda: self.demo.pool_count(2, 0),
                     lambda: self.demo.crossing_seconds(-1, 20, 20),
                     lambda: self.demo.crossing_seconds(1, 0, 20),
                     lambda: self.demo.crossing_seconds(1, float("nan"), 20)):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
