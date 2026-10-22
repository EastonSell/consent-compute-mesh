import importlib.util
import math
import unittest
from mesh.transport_experiment import numerical_metrics

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch runtime unavailable')
class MetricTests(unittest.TestCase):
    def test_identity_is_zero(self):
        import torch
        x=torch.tensor([[1.,2.,3.]])
        metrics=numerical_metrics(x,x)
        self.assertEqual(metrics['logit_max_abs_error'],0.)
        self.assertEqual(metrics['kl_reference_to_codec'],0.)
        self.assertTrue(metrics['next_token_agreement'])

    def test_kl_direction_matches_independent_probability_formula(self):
        import torch
        p=[.8,.2];q=[.3,.7]
        reference=torch.tensor([[math.log(v) for v in p]],dtype=torch.float64)
        actual=torch.tensor([[math.log(v) for v in q]],dtype=torch.float64)
        expected=sum(a*math.log(a/b) for a,b in zip(p,q))
        metrics=numerical_metrics(actual,reference)
        self.assertAlmostEqual(metrics['kl_reference_to_codec'],expected,places=12)
        self.assertFalse(metrics['next_token_agreement'])
        self.assertNotAlmostEqual(metrics['kl_reference_to_codec'],numerical_metrics(reference,actual)['kl_reference_to_codec'])

    def test_large_finite_logits_do_not_overflow_subtraction_or_margin(self):
        import torch
        reference=torch.tensor([[3e38,-3e38]],dtype=torch.float32)
        actual=-reference
        metrics=numerical_metrics(actual,reference)
        for key in ['logit_max_abs_error','logit_rms_error','kl_reference_to_codec','reference_top1_top2_logit_margin']:
            self.assertTrue(math.isfinite(metrics[key]),key)
        self.assertGreater(metrics['logit_max_abs_error'],5e38)

if __name__=='__main__':unittest.main()
