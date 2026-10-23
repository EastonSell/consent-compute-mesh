import unittest
from placement.experiments import make_surrogate, unique_bytes, run_baselines, frozen_cases
from placement.solver import solve
from placement.checker import check_certificate
from tests.helpers import scenario

GIB=2**30
class ExperimentTests(unittest.TestCase):
    def test_generator_preserves_exact_envelope(self):
        for endpoint in (0,GIB//2,GIB,2*GIB,3*GIB,4*GIB,8*GIB):
            s=make_surrogate(4,endpoint_bytes=endpoint)
            self.assertEqual(unique_bytes(s),42_000_000_000+int(2.5*GIB)+2*GIB)
            self.assertEqual(sum(l['kv_bytes'] for l in s['layers']),int(2.5*GIB))
            self.assertEqual(len(s['devices']),12)
            self.assertEqual(sum(d['budget_bytes'] for d in s['devices']),48*GIB)
            weights=[l['weight_bytes'] for l in s['layers']]
            self.assertLessEqual(max(weights)-min(weights),1)
            self.assertTrue(all(l['compute_ns'] is None for l in s['layers']))
    def test_grid_stays_bounded_and_ids_unique(self):
        cases=frozen_cases()
        self.assertGreaterEqual(len(cases),70)
        self.assertLessEqual(len(cases),150)
        self.assertEqual(len({x['id'] for x in cases}),len(cases))
    def test_greedy_failure_is_not_infeasibility(self):
        s=scenario((3,3),(5,4))
        s['devices'][1]['head_support']=False
        b=run_baselines(s)
        self.assertEqual(b['input_order_greedy']['status'],'not_found')
        self.assertEqual(solve(s)['status'],'feasible')
    def test_baseline_witnesses_independently_checked(self):
        s=scenario((2,2),(3,3))
        b=run_baselines(s)
        self.assertTrue(b['aggregate']['passes_known_necessary_condition'])
        self.assertEqual(b['best_single']['status'],'not_found')
        for k in ('input_order_greedy','descending_budget_greedy'):
            self.assertEqual(b[k]['status'],'feasible')
            self.assertTrue(check_certificate(s,b[k]['certificate'])['valid'])
    def test_endpoint_pair_bound_independently_rejects_four_gib(self):
        from placement.checker import check_rejection_bound
        s=make_surrogate(4,endpoint_bytes=4*GIB)
        r=solve(s)
        self.assertEqual(r['status'],'infeasible')
        self.assertIsNotNone(r['rejection_bound'])
        self.assertEqual(r['rejection_bound']['maximum_coverable_layers'],78)
        self.assertTrue(check_rejection_bound(s,r['rejection_bound'])['valid'])
    def test_optional_devices_preserve_existing_feasible_placement(self):
        from placement.solver import build_certificate
        old=solve(make_surrogate(4))['certificate']
        intervals=[(int(x['device_id'][1:]),x['start'],x['end']) for x in old['stages']]
        for homes in (5,6):
            expanded=make_surrogate(homes)
            inherited=build_certificate(expanded,intervals)
            self.assertTrue(check_certificate(expanded,inherited)['valid'])
            self.assertEqual(inherited['timing'],old['timing'])
            chosen=solve(expanded)['certificate']
            self.assertLess(len(chosen['stages']),len(inherited['stages']))
            self.assertGreater(chosen['timing']['communication_ns']['numerator'],
                               inherited['timing']['communication_ns']['numerator'])
    def test_unknown_aggregate_not_credited(self):
        b=run_baselines(scenario((2,),(None,)))
        self.assertIsNone(b['aggregate']['passes_known_necessary_condition'])

if __name__ == '__main__': unittest.main()
