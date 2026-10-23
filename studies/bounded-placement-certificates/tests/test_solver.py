import copy
from fractions import Fraction
import unittest
from placement.solver import solve, validate_scenario, build_certificate, stage_accounting
from tests.helpers import scenario

class SolverTests(unittest.TestCase):
    def test_split_certificate_covers_exact_layers(self):
        s = scenario()
        r = solve(s)
        self.assertEqual(r['status'], 'feasible')
        self.assertEqual([(x['device_id'], x['start'], x['end']) for x in r['certificate']['stages']],
                         [('d0', 0, 1), ('d1', 1, 2)])
        self.assertTrue(r['minimum_stage_count_proven'])
        self.assertEqual(r['certificate']['timing']['communication_ns'], {'numerator':4, 'denominator':1})
    def test_monolithic_uses_one_tied_copy(self):
        s = scenario((2,), (7,), embedding_bytes=3, head_bytes=3)
        r = solve(s)
        self.assertEqual(r['status'], 'feasible')
        self.assertEqual(r['certificate']['stages'][0]['memory']['endpoint_bytes'], 3)
        self.assertEqual(r['certificate']['stages'][0]['memory']['total_bytes'], 5)
    def test_tied_split_duplicates_and_can_fail_aggregate(self):
        s = scenario((2,2), (4,4), embedding_bytes=3, head_bytes=3)
        self.assertEqual(solve(s)['status'], 'infeasible')
    def test_workspace_peak_kv_reserve_controller(self):
        s = scenario((2,3), (100,), controller_reserve_bytes=7, activation_reserve_bytes=5)
        s['layers'][0].update(kv_bytes=1, workspace_bytes=9)
        s['layers'][1].update(kv_bytes=2, workspace_bytes=4)
        s['devices'][0]['runtime_reserve_bytes'] = 6
        m = stage_accounting(s, 0, 0, 2)
        self.assertEqual(m, dict(weight_bytes=5, kv_bytes=3, endpoint_bytes=0,
            runtime_reserve_bytes=6, activation_reserve_bytes=5, workspace_bytes=9,
            controller_reserve_bytes=7, total_bytes=35))
    def test_fragmentation_rejects_equal_aggregate(self):
        r = solve(scenario((3,3,3,3), (8,2,2)))
        self.assertEqual(r['status'], 'infeasible')
        self.assertEqual(r['rejection_bound']['maximum_coverable_layers'], 2)
    def test_contiguity_not_bin_packing(self):
        self.assertEqual(solve(scenario((2,6,2), (4,6)))['status'], 'infeasible')
    def test_unknown_budget_is_not_zero_or_certified(self):
        s = scenario((2,), (None,))
        r = solve(s)
        self.assertEqual(r['status'], 'unknown')
        self.assertNotIn('certificate', r)
        self.assertTrue(r['optimistic_witness'])
    def test_explicit_ineligible_is_excluded_even_optimistically(self):
        s = scenario((2,), (100,))
        s['devices'][0]['eligible'] = False
        self.assertEqual(solve(s)['status'], 'infeasible')
    def test_unknown_support_not_credited(self):
        s = scenario((2,), (2,))
        s['devices'][0]['layer_support'][0] = None
        self.assertEqual(solve(s)['status'], 'unknown')
    def test_unknown_timing_stays_unknown(self):
        s = scenario()
        s['layers'][0]['compute_ns'] = None
        s['links'][0]['latency_ns'] = None
        t = solve(s)['certificate']['timing']
        self.assertIsNone(t['critical_path_ns'])
        self.assertIsNone(t['communication_ns'])
        self.assertTrue(t['unknown_terms'])
    def test_asymmetric_feedback_is_charged(self):
        s = scenario()
        s['links'][1]['latency_ns'] = 10
        t = solve(s)['certificate']['timing']
        self.assertEqual(t['communication_ns'], {'numerator':13, 'denominator':1})
    def test_missing_return_link_rejects_chain(self):
        s = scenario()
        s['links'] = s['links'][:1]
        self.assertEqual(solve(s)['status'], 'infeasible')
    def test_unknown_return_link_only_relaxed_witness(self):
        s = scenario()
        s['links'][1]['available'] = None
        self.assertEqual(solve(s)['status'], 'unknown')
    def test_cap_does_not_become_infeasible(self):
        r = solve(scenario(), max_states=0)
        self.assertEqual(r['status'], 'search_incomplete')
    def test_limits_cannot_be_raised_beyond_frozen_caps(self):
        with self.assertRaises(ValueError): solve(scenario(),max_states=250_001)
        with self.assertRaises(ValueError): solve(scenario(),max_transitions=5_000_001)
    def test_deterministic(self):
        s = scenario((1,2,1), (3,3,3))
        self.assertEqual(solve(s), solve(copy.deepcopy(s)))
    def test_malformed_schema_raises_value_error(self):
        examples=[]
        s=scenario(); s['layers'][0]=None; examples.append(s)
        s=scenario(); del s['head_bytes']; examples.append(s)
        s=scenario(); s['links']=None; examples.append(s)
        s=scenario(); s['devices'][0]['layer_support']='yes'; examples.append(s)
        for s in examples:
            with self.assertRaises(ValueError): validate_scenario(s)
    def test_strict_invalid_inputs(self):
        for key, value in [('activation_reserve_bytes', -1), ('embedding_bytes', True),
                           ('tied_embeddings', 1)]:
            s = scenario(); s[key] = value
            with self.assertRaises(ValueError): validate_scenario(s)
        s = scenario(); s['devices'][0]['budget_bytes'] = 1.5
        with self.assertRaises(ValueError): validate_scenario(s)
        s = scenario(); s['head_bytes'] = 1
        with self.assertRaises(ValueError): validate_scenario(s)
        s = scenario(); s['links'][0]['payload_bps'] = 0
        with self.assertRaises(ValueError): validate_scenario(s)
        s = scenario(); s['devices'][0]['id'] = s['devices'][1]['id']
        with self.assertRaises(ValueError): validate_scenario(s)

if __name__ == '__main__': unittest.main()
