import itertools
import unittest
from placement.solver import solve
from placement.checker import check_certificate, check_rejection_bound
from tests.helpers import scenario
from tests.oracle import brute_min_stages

class ExhaustiveOracleTests(unittest.TestCase):
    def test_all_1170_frozen_tiny_cases(self):
        count = 0
        for nd in range(1,4):
            for budgets in itertools.product((1,2,3),repeat=nd):
                for nl in range(1,5):
                    for weights in itertools.product((1,2),repeat=nl):
                        s = scenario(weights,budgets)
                        expected = brute_min_stages(s)
                        r = solve(s)
                        actual = len(r['certificate']['stages']) if r['status']=='feasible' else None
                        self.assertEqual(actual,expected,(weights,budgets,r))
                        self.assertIn(r['status'],('feasible','infeasible'))
                        if actual is not None:
                            self.assertTrue(check_certificate(s,r['certificate'])['valid'])
                        elif r['rejection_bound'] is not None:
                            self.assertTrue(check_rejection_bound(s,r['rejection_bound'])['valid'])
                        count += 1
        self.assertEqual(count,1170)
    def test_endpoint_reserve_and_connectivity_fixtures(self):
        count = 0
        for tied, endpoint, reserve, controller in itertools.product((True,False),(0,1,2),(0,1),(0,1)):
            s = scenario((1,2,1),(3,4,5),tied_embeddings=tied,
                         embedding_bytes=endpoint,head_bytes=endpoint,
                         activation_reserve_bytes=reserve,controller_reserve_bytes=controller)
            s['layers'][1]['kv_bytes'] = 1
            s['layers'][2]['workspace_bytes'] = 1
            for variant in range(4):
                import copy
                q=copy.deepcopy(s)
                if variant == 1: q['devices'][0]['head_support']=False
                if variant == 2: q['links'][0]['available']=False
                if variant == 3: q['devices'][1]['layer_support'][1]=False
                expected=brute_min_stages(q); r=solve(q)
                actual=len(r['certificate']['stages']) if r['status']=='feasible' else None
                self.assertEqual(actual,expected,(tied,endpoint,reserve,controller,variant,r))
                if actual is not None: self.assertTrue(check_certificate(q,r['certificate'])['valid'])
                count += 1
        self.assertEqual(count,96)

if __name__ == '__main__': unittest.main()
