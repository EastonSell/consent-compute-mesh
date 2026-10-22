"""Explicit post-hoc inspection, no timing or quality claims; unchanged codecs."""
import hashlib
import json
import os
from pathlib import Path
import struct

from .transport_codec import encode_activation,decode_activation
from .pretrained_model import Partition
from .transport_experiment import numerical_metrics,memory_observation

ROOT=Path(__file__).resolve().parents[1]


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-dir',required=True);parser.add_argument('--opt-in',action='store_true');args=parser.parse_args()
    if not args.opt_in:parser.error('explicit opt-in required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=(ROOT/'results/transport_comparison.json').read_bytes();original_sha=hashlib.sha256(original).hexdigest()
    record={'kind':'post_hoc_activation_diagnostic','purpose':'Investigate previously observed per-tensor INT8 failure, not a new tuned comparison or independent quality sample.','original_result_sha256':original_sha,'prefix_sha256':hashlib.sha256((ROOT/'data/transport_prefixes.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'memory_before_load':memory_observation(),'timing_claims':False,'execution':'Single-process stage0 once per prefix; stage1 once per codec, unchanged model and codec. No KV cache.','rows':[]}
    cases=json.loads((ROOT/'data/transport_prefixes.json').read_text())['cases'];arrays={}
    stage0=Partition(args.model_dir,0);stage1=Partition(args.model_dir,1)
    with torch.inference_mode():
        for case in cases:
            hidden=stage0(torch.tensor([case['tokens']],dtype=torch.int64)).numpy()
            arrays[case['id']+'__activation_fp32']=hidden.copy();reference=None
            for codec in ['fp32','fp16','int8']:
                frame,meta=encode_activation(hidden,codec);decoded,header=decode_activation(frame)
                name=case['id']+'__'+codec
                arrays[name+'__frame']=np.frombuffer(frame,dtype=np.uint8).copy()
                output=stage1(torch.from_numpy(decoded));arrays[name+'__logits_fp32']=output.numpy().copy()
                if codec=='fp32':reference=output.clone()
                delta=decoded.astype(np.float64)-hidden.astype(np.float64)
                independent=True
                if codec=='int8':
                    length=struct.unpack('<I',frame[:4])[0];payload=frame[4+length:]
                    signed=[v[0] for v in struct.iter_unpack('<b',payload)]
                    expected=[max(-127,min(127,round(float(v)/meta['scale']))) for v in hidden.flat]
                    if signed!=expected:raise ValueError('independent signed-byte/rounding mismatch')
                    manual=np.asarray([v*meta['scale'] for v in signed],dtype=np.float32).reshape(hidden.shape)
                    if not np.array_equal(manual,decoded):raise ValueError('independent scale reconstruction mismatch')
                else:
                    length=struct.unpack('<I',frame[:4])[0];dtype='<f4' if codec=='fp32' else '<f2'
                    manual=np.frombuffer(frame[4+length:],dtype=dtype).astype(np.float32).reshape(hidden.shape)
                    if not np.array_equal(manual,decoded):raise ValueError('independent float payload mismatch')
                nonzero=hidden!=0
                record['rows'].append({'case':case['id'],'codec':codec,'token_count':len(case['tokens']),**meta,
                    'frame_sha256':hashlib.sha256(frame).hexdigest(),'zero_fraction_after_decode':float(np.mean(decoded==0)),
                    'original_nonzero_fraction_rounded_to_zero':float(np.count_nonzero(nonzero&(decoded==0))/max(1,np.count_nonzero(nonzero))),
                    'activation_max_abs_error':float(np.max(np.abs(delta))),
                    'activation_relative_l2_error':float(np.linalg.norm(delta)/max(1e-30,np.linalg.norm(hidden.astype(np.float64)))),
                    'original_abs_median':float(np.median(np.abs(hidden))),
                    'original_abs_p99':float(np.percentile(np.abs(hidden),99)),
                    'independent_payload_check_passed':independent,**numerical_metrics(output,reference)})
    archive=ROOT/'results/transport_posthoc_arrays.npz';np.savez_compressed(archive,**arrays)
    record['archive_path']='results/transport_posthoc_arrays.npz';record['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest();record['archive_bytes']=archive.stat().st_size
    if hashlib.sha256((ROOT/'results/transport_comparison.json').read_bytes()).hexdigest()!=original_sha:raise ValueError('original result changed')
    record['original_result_unchanged']=True
    (ROOT/'results/transport_posthoc_diagnostic.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('Saved separately labeled post-hoc raw activations, encoded frames and final logits; original result unchanged.')


if __name__=='__main__':main()
