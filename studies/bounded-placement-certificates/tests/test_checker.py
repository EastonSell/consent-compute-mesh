import copy
import unittest
from placement.solver import solve
from placement.checker import check_certificate, check_rejection_bound
from tests.helpers import scenario

class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.s = scenario()
        self.c = solve(self.s)['certificate']
    def test_accepts_independently_recomputed_valid_witness(self):
        self.assertTrue(check_certificate(self.s, self.c)['valid'])
    def test_forged_memory_rejected(self):
        self.c['stages'][0]['memory']['total_bytes'] = 0
        self.assertFalse(check_certificate(self.s, self.c)['valid'])
    def test_changed_scenario_rejected(self):
        self.s['devices'][0]['budget_bytes'] = 1
        self.assertFalse(check_certificate(self.s, self.c)['valid'])
    def test_gap_reuse_and_out_of_range_rejected(self):
        for key, value in [('start',1), ('end',3), ('device_id','d1'), ('start',True)]:
            c = copy.deepcopy(self.c); c['stages'][0][key] = value
            self.assertFalse(check_certificate(self.s, c)['valid'])
    def test_endpoint_tie_copy_accounting_rejected_if_forged(self):
        s = scenario((1,), (5,), embedding_bytes=2, head_bytes=2)
        c = solve(s)['certificate']; c['stages'][0]['memory']['endpoint_bytes'] = 4
        self.assertFalse(check_certificate(s,c)['valid'])
    def test_missing_or_forged_transfer_rejected(self):
        self.c['transfers'].pop()
        self.assertFalse(check_certificate(self.s,self.c)['valid'])
    def test_forged_timing_rejected(self):
        self.c['timing']['critical_path_ns']['numerator'] = 0
        self.assertFalse(check_certificate(self.s,self.c)['valid'])
    def test_unknown_compute_cannot_claim_zero(self):
        self.s['layers'][0]['compute_ns'] = None
        self.c = solve(self.s)['certificate']
        self.assertTrue(check_certificate(self.s,self.c)['valid'])
        self.c['timing']['critical_path_ns'] = dict(numerator=4,denominator=1)
        self.assertFalse(check_certificate(self.s,self.c)['valid'])
    def test_false_eligibility_not_overridden_by_hash(self):
        from hashlib import sha256
        import json
        self.s['devices'][0]['eligible'] = False
        self.c['scenario_sha256'] = sha256(json.dumps(self.s,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertFalse(check_certificate(self.s,self.c)['valid'])
    def test_malformed_never_crashes(self):
        for c in ({}, None, [], {'version':1,'stages':None}):
            self.assertFalse(check_certificate(self.s,c)['valid'])
    def test_independent_no_placement_bound(self):
        s = scenario((3,3,3,3),(8,2,2))
        b = solve(s)['rejection_bound']
        self.assertTrue(check_rejection_bound(s,b)['valid'])
        b['maximum_coverable_layers'] = 0
        self.assertFalse(check_rejection_bound(s,b)['valid'])
    def test_checker_works_when_python_assertions_are_disabled(self):
        import subprocess, sys
        command = ("from tests.helpers import scenario; from placement.solver import solve; "
                   "from placement.checker import check_certificate; s=scenario(); "
                   "c=solve(s)['certificate']; c['stages'][0]['memory']['total_bytes']=0; "
                   "print(check_certificate(s,c)['valid'])")
        result=subprocess.run([sys.executable,'-O','-c',command],capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'False')
    def test_boolean_is_not_an_integer_in_certificate(self):
        self.c['stages'][0]['memory']['kv_bytes']=False
        self.assertFalse(check_certificate(self.s,self.c)['valid'])
        self.c=solve(self.s)['certificate']; self.c['version']=True
        self.assertFalse(check_certificate(self.s,self.c)['valid'])
    def test_bundle_verification_is_standalone_and_tamper_evident(self):
        from placement.checker import verify_bundle
        bundle={'evidence_kind':'analytical_not_measurement','model_version':1,'cases':[
            {'id':'one','scenario':self.s,'solver':{'status':'feasible','certificate':self.c},'baselines':{}}]}
        self.assertTrue(verify_bundle(bundle)['valid'])
        bundle['cases'][0]['solver']['certificate']['stages'][0]['memory']['total_bytes']=0
        self.assertFalse(verify_bundle(bundle)['valid'])
    def test_not_a_negative_bound_when_it_can_fit(self):
        self.assertFalse(check_rejection_bound(self.s, None)['valid'])

if __name__ == '__main__': unittest.main()
