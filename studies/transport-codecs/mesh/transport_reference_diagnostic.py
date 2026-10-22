"""Post-hoc monolithic-logit archive supplement; no timing claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .pretrained_model import load_reference
from .transport_experiment import numerical_metrics,memory_observation

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-dir',required=True);parser.add_argument('--opt-in',action='store_true');args=parser.parse_args()
    if not args.opt_in:parser.error('explicit opt-in required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    record={'kind':'post_hoc_monolithic_reference_archive','timing_claims':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'memory_before_load':memory_observation(),'comparisons':[]}
    original=(ROOT/'results/transport_comparison.json').read_bytes();record['original_result_sha256']=hashlib.sha256(original).hexdigest()
    result=json.loads(original);diagnostic=json.loads((ROOT/'results/transport_posthoc_diagnostic.json').read_text())
    archive=ROOT/diagnostic['archive_path']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=diagnostic['archive_sha256']:raise ValueError('diagnostic archive hash mismatch')
    arrays=np.load(archive,allow_pickle=False);record['codec_archive_sha256']=diagnostic['archive_sha256']
    cases=json.loads((ROOT/'data/transport_prefixes.json').read_text())['cases'];model=load_reference(args.model_dir);references={}
    with torch.inference_mode():
        for case in cases:
            ids=torch.tensor([case['tokens']],dtype=torch.int64)
            expected=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,num_logits_to_keep=1).logits[:,-1,:]
            references[case['id']+'__monolithic_logits_fp32']=expected.numpy().copy()
            for codec in ['fp32','fp16','int8']:
                actual=torch.from_numpy(arrays[case['id']+'__'+codec+'__logits_fp32'].copy())
                metrics=numerical_metrics(actual,expected)
                primary=next(s for s in result['samples'] if s['case']==case['id'] and s['codec']==codec)
                matches=all(metrics[key]==primary[key] for key in metrics)
                record['comparisons'].append({'case':case['id'],'codec':codec,'metrics_exactly_match_primary':matches,**metrics})
                if codec=='fp32' and not torch.equal(actual,expected):raise ValueError('post-hoc FP32 logits differ from monolithic')
                if not matches:raise ValueError('post-hoc numerical metrics differ from primary evidence')
    path=ROOT/'results/transport_posthoc_reference.npz';np.savez_compressed(path,**references)
    record['reference_archive_path']='results/transport_posthoc_reference.npz';record['reference_archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();record['reference_archive_bytes']=path.stat().st_size
    record['original_result_unchanged']=hashlib.sha256((ROOT/'results/transport_comparison.json').read_bytes()).hexdigest()==record['original_result_sha256']
    (ROOT/'results/transport_posthoc_reference.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('Saved monolithic reference logits; all36post-hoc numerical outcomes exactly match primary evidence.')

if __name__=='__main__':main()
