"""Predeclared codec comparison: real same-host compute, explicitly emulated link."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .pretrained_model import load_reference, verify_model, MODEL_ID, REVISION
from .transport_session import TransportSession
from .repeated import summarize

ROOT=Path(__file__).resolve().parents[1]
CODECS=('fp32','fp16','int8')
LINKS=({'latency_ms':0,'bytes_per_second':None},{'latency_ms':10,'bytes_per_second':1_000_000})
REPEATS=3


def numerical_metrics(actual,reference):
    import torch
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():raise ValueError('nonfinite logits')
    delta=actual.double()-reference.double()
    logp=torch.log_softmax(reference.double(),dim=-1)
    logq=torch.log_softmax(actual.double(),dim=-1)
    top=reference.double().topk(2,dim=-1).values
    return {'logit_max_abs_error':delta.abs().max().item(),
            'logit_rms_error':delta.square().mean().sqrt().item(),
            'logit_relative_l2_error':(delta.norm()/reference.double().norm().clamp_min(1e-30)).item(),
            'kl_reference_to_codec':(logp.exp()*(logp-logq)).sum().item(),
            'reference_next_token':reference.argmax(-1).item(),'codec_next_token':actual.argmax(-1).item(),
            'next_token_agreement':actual.argmax(-1).item()==reference.argmax(-1).item(),
            'reference_top1_top2_logit_margin':(top[0,0]-top[0,1]).item()}


def memory_observation():
    available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
    if available is None:raise ValueError('cannot establish available memory')
    observation={'host_available_bytes':available,'cgroup_remaining_bytes':None}
    limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip()!='max':
        observation['cgroup_remaining_bytes']=max(0,int(limit.read_text())-int(current.read_text()))
        available=min(available,observation['cgroup_remaining_bytes'])
    if available<3*1024**3:raise ValueError('at least 3 GiB observed effective available memory required')
    return observation


def run(directory,*,opt_in=False):
    if opt_in is not True:raise ValueError('explicit opt-in required')
    if sys.platform!='linux':raise ValueError('timing runner currently requires Linux')
    result={'kind':'activation_transport_comparison','status':'running','model_id':MODEL_ID,'revision':REVISION,
            'model_dtype':'float32','activation_codecs':list(CODECS),'kv_cache':False,'batch_size':1,
            'split_after_layer':14,'threads_per_process':1,'repeats_per_condition':REPEATS,
            'prefix_sha256':hashlib.sha256((ROOT/'data/transport_prefixes.json').read_bytes()).hexdigest(),
            'correctness_gates':[],'samples':[],'warmup_count':0,'cleanup':None,
            'scope':'One same-host local-pipe experiment; no physical-network throughput or model-quality claim.'}
    session=None
    try:
        result['memory_before_load']=memory_observation()
        result['model_sha256']=verify_model(directory)
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        start=time.perf_counter();model=load_reference(directory)
        result['reference_load_seconds']=time.perf_counter()-start
        cases=json.loads((ROOT/'data/transport_prefixes.json').read_text())['cases']
        def reference(tokens):
            ids=torch.tensor([tokens],dtype=torch.int64);mask=torch.ones_like(ids)
            start=time.perf_counter()
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False,num_logits_to_keep=1).logits[:,-1,:]
            return logits,time.perf_counter()-start
        def split_call(tokens,codec,link):
            split=session.forward(tokens,codec=codec,timeout_seconds=15,**link)
            if split['status']!='ok':
                result['failed_request']=split
                raise RuntimeError('transport request failed: '+split['status'])
            return split
        session=TransportSession(directory,opt_in=True)
        with torch.inference_mode(),session:
            result['worker_load_reports']=session.load_reports
            result['worker_startup_seconds']=session.startup_seconds
            result['phase']='fp32_equivalence_gate'
            for case in cases:
                expected,_=reference(case['tokens']);split=split_call(case['tokens'],'fp32',LINKS[0])
                actual=split['logits'];metrics=numerical_metrics(actual,expected)
                result['correctness_gates'].append({'case':case['id'],**metrics})
                torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
                if not metrics['next_token_agreement']:raise ValueError('FP32 gate next-token disagreement')
            result['phase']='paired_measurements'
            for case_index,case in enumerate(cases):
                for link_index,link in enumerate(LINKS):
                    # One excluded warmup for each codec/case/link condition.
                    for codec in CODECS:
                        reference(case['tokens']);split_call(case['tokens'],codec,link);result['warmup_count']+=1
                    for repeat in range(REPEATS):
                        shift=(case_index+link_index+repeat)%3
                        order=CODECS[shift:]+CODECS[:shift]
                        for position,codec in enumerate(order):
                            reference_first=(case_index+link_index+repeat+position)%2==0
                            if reference_first:expected,reference_seconds=reference(case['tokens'])
                            split=split_call(case['tokens'],codec,link)
                            if not reference_first:expected,reference_seconds=reference(case['tokens'])
                            actual=split.pop('logits')
                            sample={'case':case['id'],'token_count':len(case['tokens']),'codec':codec,
                                    'link_index':link_index,'link_assumptions':link,'repeat':repeat,
                                    'codec_order':list(order),'position_in_codec_order':position,
                                    'reference_first':reference_first,'reference_seconds':reference_seconds,
                                    **split,**numerical_metrics(actual,expected)}
                            sample['raw_fp32_activation_bytes']=len(case['tokens'])*576*4
                            sample['local_seconds_excluding_observed_emulated_wait']=split['split_seconds']-split['emulated_wait_observed_seconds']
                            result['samples'].append(sample)
                            if codec=='fp32':
                                torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
                                if not sample['next_token_agreement']:raise ValueError('FP32 paired next-token disagreement')
            result['summaries']=[]
            for codec in CODECS:
                for link_index in range(len(LINKS)):
                    samples=[s for s in result['samples'] if s['codec']==codec and s['link_index']==link_index]
                    result['summaries'].append({'codec':codec,'link_index':link_index,'sample_count':len(samples),
                        'next_token_agreement_count':sum(s['next_token_agreement'] for s in samples),
                        **{field:summarize([s[field] for s in samples]) for field in ['split_seconds','reference_seconds','local_seconds_excluding_observed_emulated_wait','logit_max_abs_error','logit_rms_error','kl_reference_to_codec','activation_frame_bytes']}})
            result['status']='ok';result['phase']='complete'
    except Exception as error:
        result['status']='failed';result['failure']={'type':type(error).__name__,'reason':str(error)[:1500]}
    finally:
        if session is not None:
            session.close();result['cleanup']=session.cleanup
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--opt-in',action='store_true')
    args=parser.parse_args()
    if not args.opt_in:parser.error('explicit --opt-in required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    result=run(args.model_dir,opt_in=True)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['status']=='ok' and result['cleanup']['workers_cleaned_up'] else 1


if __name__=='__main__':raise SystemExit(main())
