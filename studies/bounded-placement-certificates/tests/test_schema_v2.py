"""Reviewer-reported malformed interfaces must fail without changing arithmetic."""
from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys
import unittest
from placement.solver import solve
from placement.checker import check_certificate, verify_bundle
from tests.helpers import scenario


def mono_bundle():
    s=scenario((1,),(2,))
    c=solve(s)['certificate']
    return {'evidence_kind':'analytical_not_measurement','model_version':1,
            'cases':[{'id':'mono','scenario':s,'solver':{'status':'feasible','certificate':c},'baselines':{}}]}


def invalid_bundles():
    variants=[]
    b=mono_bundle(); b['model_version']=True; variants.append(('boolean_model_version',b))
    b=mono_bundle(); b['cases'][0]['solver']['status']='not_a_solver_status'; variants.append(('unrecognized_solver_status',b))
    b=mono_bundle(); b['cases'][0]['baselines']['input_order_greedy']={'status':'not_a_baseline_status'}; variants.append(('unrecognized_baseline_status',b))
    b=mono_bundle(); s=b['cases'][0]['scenario']; s['links']={}
    b['cases'][0]['solver']['certificate']['scenario_sha256']=sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    variants.append(('dict_instead_of_links_list',b))
    b=mono_bundle(); b['cases'][0]['solver']['certificate']['transfers']={}; variants.append(('dict_instead_of_transfers_list',b))
    b=mono_bundle(); b['cases']={}; variants.append(('dict_instead_of_cases_list',b))
    return variants


class SchemaV2Tests(unittest.TestCase):
    def test_valid_monolithic_lists_still_accepted(self):
        self.assertTrue(verify_bundle(mono_bundle())['valid'])
    def test_boolean_model_version_rejected(self):
        self.assertFalse(verify_bundle(invalid_bundles()[0][1])['valid'])
    def test_unrecognized_solver_status_rejected(self):
        self.assertFalse(verify_bundle(invalid_bundles()[1][1])['valid'])
    def test_unrecognized_baseline_status_rejected(self):
        self.assertFalse(verify_bundle(invalid_bundles()[2][1])['valid'])
    def test_links_must_be_list_even_without_transfers(self):
        b=invalid_bundles()[3][1]; c=b['cases'][0]
        self.assertFalse(check_certificate(c['scenario'],c['solver']['certificate'])['valid'])
        self.assertFalse(verify_bundle(b)['valid'])
    def test_transfers_must_be_list_even_when_empty(self):
        b=invalid_bundles()[4][1]; c=b['cases'][0]
        self.assertFalse(check_certificate(c['scenario'],c['solver']['certificate'])['valid'])
        self.assertFalse(verify_bundle(b)['valid'])
    def test_cases_must_be_list_even_when_empty(self):
        self.assertFalse(verify_bundle(invalid_bundles()[5][1])['valid'])
    def test_baseline_named_solver_cannot_use_solver_only_status(self):
        b=mono_bundle(); b['cases'][0]['baselines']['solver']={'status':'unknown'}
        self.assertFalse(verify_bundle(b)['valid'])
    def test_all_reported_gaps_rejected_in_optimized_python(self):
        command=("from tests.test_schema_v2 import invalid_bundles; "
                 "from placement.checker import verify_bundle; "
                 "print([verify_bundle(b)['valid'] for _,b in invalid_bundles()])")
        completed=subprocess.run([sys.executable,'-O','-c',command],check=True,capture_output=True,text=True)
        self.assertEqual(completed.stdout.strip(),'[False, False, False, False, False, False]')

if __name__=='__main__': unittest.main()
