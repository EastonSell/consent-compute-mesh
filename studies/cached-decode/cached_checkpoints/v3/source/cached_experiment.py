"""Retained exact cached-decode checks. Run only after independent source review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from .cached_evidence import Evidence, source_hashes

PREFIXES = [[1,2,3,4],[25,100,200,300,400,500,600,700],[42]*16]
BUDGET = 3 * 1024**3


def available_bytes():
    fields = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0])*1024
    maximum=Path('/sys/fs/cgroup/memory.max')
    current=Path('/sys/fs/cgroup/memory.current')
    if maximum.exists() and current.exists() and maximum.read_text().strip() != 'max':
        available=min(available,int(maximum.read_text())-int(current.read_text()))
    return available


def live_rss():
    # Spawned workers and multiprocessing resource tracker are direct children.
    pids = [os.getpid()]
    children=Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children')
    pids += [int(x) for x in children.read_text().split()]
    total=0
    for pid in pids:
        try: total+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError: pass
    return total


def run(output, directory, expected_hashes):
    evidence=Evidence(output)
    session=None
    stopped=threading.Event()
    resource_exceeded=threading.Event()
    peak=[0]
    watcher=None
    monitor_errors=[]
    def monitor():
        try:
            while not stopped.wait(.05):
                peak[0]=max(peak[0],live_rss())
                if peak[0]>BUDGET:
                    raise RuntimeError('RSS resource budget exceeded')
        except BaseException as error:
            monitor_errors.append(str(error))
            resource_exceeded.set()
            if session is not None:
                session.cancel_event.set()
                for process in session.processes:
                    try:
                        if process.is_alive(): process.terminate()
                    except ValueError: pass
    try:
        root=Path(__file__).resolve().parents[1]
        actual=source_hashes(root)
        evidence.write('source_hashes.json',actual)
        if actual != json.loads(Path(expected_hashes).read_text()):
            raise ValueError('frozen source hash mismatch')
        available=available_bytes()
        evidence.write('environment.json',{'available_bytes':available,'budget_bytes':BUDGET,
                         'start_unix':time.time(),'prefixes':PREFIXES,'decode_steps':8,
                         'thread_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}})
        if available < BUDGET: raise RuntimeError('less than 3 GiB available')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            if os.environ.get(key) != '1':raise RuntimeError('thread environment must be explicitly one')
        import torch
        import transformers
        from .pretrained_model import load_reference, MODEL_ID, REVISION, HASHES
        from .cached_session import CachedSession
        evidence.write('runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__,'model_id':MODEL_ID,'revision':REVISION,'model_hashes':HASHES})
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        watcher=threading.Thread(target=monitor, daemon=True); watcher.start()
        reference=load_reference(directory)
        if resource_exceeded.is_set():raise RuntimeError('resource monitor failed: '+str(monitor_errors))
        session=CachedSession(directory,opt_in=True)
        if resource_exceeded.is_set():raise RuntimeError('resource monitor failed: '+str(monitor_errors))
        with session, torch.inference_mode():
            replay=[]
            coverage=[]
            for case_index,prefix in enumerate(PREFIXES+[PREFIXES[0]]):
                session.reset()
                cache=None
                tokens=prefix
                for step in range(9):
                    if resource_exceeded.is_set():raise RuntimeError('RSS resource budget exceeded')
                    started=time.perf_counter()
                    result=reference(torch.tensor([tokens]),past_key_values=cache,use_cache=True,num_logits_to_keep=1)
                    ref_seconds=time.perf_counter()-started
                    cache=result.past_key_values
                    expected=result.logits[:,-1,:].contiguous()
                    observed=session.forward(tokens)
                    actual_logits=observed.pop('logits').contiguous()
                    exact=torch.equal(expected,actual_logits)
                    ref_token=int(expected.argmax(-1).item()); split_token=int(actual_logits.argmax(-1).item())
                    names=[]
                    hashes=[]
                    for label,tensor in [('reference',expected),('split',actual_logits)]:
                        name=f'case-{case_index}-step-{step}-{label}.f32'
                        data=tensor.numpy().astype('<f4',copy=False).tobytes()
                        with (evidence.path/name).open('xb') as f:
                            f.write(data); f.flush(); os.fsync(f.fileno())
                        names.append(name); hashes.append(hashlib.sha256(data).hexdigest())
                    replay_ok=case_index != 3 or hashes == replay[step]
                    if case_index == 0: replay.append(hashes)
                    row=dict(case=case_index,step=step,input_tokens=tokens,exact=exact,
                        allclose=bool(torch.allclose(expected,actual_logits,atol=1e-4,rtol=1e-4)),
                        max_abs_error=float((expected-actual_logits).abs().max()),reference_token=ref_token,
                        split_token=split_token,reference_seconds=ref_seconds,files=names,sha256=hashes,
                        reset_replay_exact=replay_ok,live_rss_bytes=live_rss(),**observed)
                    evidence.append(row)
                    coverage.append((case_index,step))
                    if not exact or hashes[0] != hashes[1] or ref_token != split_token or not replay_ok:
                        raise AssertionError('exact correctness/reset gate failed; evidence retained')
                    if resource_exceeded.is_set():raise RuntimeError('RSS resource budget exceeded')
                    tokens=[ref_token]
        if resource_exceeded.is_set():raise RuntimeError('RSS resource budget exceeded')
        if not session.cleanup['workers_cleaned_up'] or not session.cleanup['io_threads_cleaned_up'] or any(session.cleanup['worker_exitcodes']):
            raise RuntimeError('worker cleanup failed')
        if coverage != [(case,step) for case in range(4) for step in range(9)]:
            raise AssertionError('case coverage failed')
        evidence.finish('complete',end_unix=time.time(),rows=36,cleanup=session.cleanup,peak_sampled_aggregate_rss_bytes=peak[0])
    except BaseException as error:
        if session is not None: session.close()
        evidence.finish('failed',error_type=type(error).__name__,error=str(error),
                        cleanup=session.cleanup if session else None,monitor_errors=monitor_errors,peak_sampled_aggregate_rss_bytes=peak[0])
        raise
    finally:
        stopped.set()
        if watcher:watcher.join(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--model-dir',required=True)
    parser.add_argument('--source-hashes',required=True)
    parser.add_argument('--opt-in',action='store_true',required=True)
    args=parser.parse_args()
    run(args.output,args.model_dir,args.source_hashes)

if __name__=='__main__':main()
