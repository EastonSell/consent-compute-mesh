"""Development-only four-pair smoke; never part of retained timing schedule."""
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
from mesh.cached_evidence import Evidence, source_hashes
from mesh.cached_experiment import available_bytes, live_rss, BUDGET


def main():
    evidence=Evidence(sys.argv[1]); session=None; stop=threading.Event(); exceeded=threading.Event();peak=[0]
    def monitor():
        try:
            while not stop.wait(.05):
                peak[0]=max(peak[0],live_rss())
                if peak[0]>BUDGET:raise RuntimeError('RSS budget exceeded')
        except BaseException:
            exceeded.set()
            if session is not None:session.cancel_event.set()
    watcher=threading.Thread(target=monitor,daemon=True)
    try:
        evidence.write('purpose.json',{'purpose':'development equivalence smoke, not retained benchmark',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':source_hashes(Path(__file__).resolve().parents[1])})
        if available_bytes()<BUDGET:raise RuntimeError('less than 3GiB available')
        import torch
        from mesh.pretrained_model import load_reference
        from mesh.cached_session import CachedSession
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        watcher.start()
        reference=load_reference(sys.argv[2])
        if exceeded.is_set():raise RuntimeError('resource budget exceeded')
        session=CachedSession(sys.argv[2],opt_in=True)
        with session,torch.inference_mode():
            cache=None;tokens=[1,42,314,7]
            first_hash=None
            for step in range(4):
                if exceeded.is_set():raise RuntimeError('resource budget exceeded')
                if step==3:
                    session.reset();cache=None;tokens=[1,42,314,7]
                output=reference(torch.tensor([tokens]),past_key_values=cache,use_cache=True,num_logits_to_keep=1)
                cache=output.past_key_values
                reference_logits=output.logits[:,-1,:].contiguous()
                result=session.forward(tokens)
                split=result.pop('logits').contiguous()
                data=[]
                for label,tensor in [('reference',reference_logits),('split',split)]:
                    blob=tensor.numpy().astype('<f4',copy=False).tobytes();data.append(blob)
                    with (evidence.path/f'{step}-{label}.f32').open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
                digest=hashlib.sha256(data[0]).hexdigest()
                if step==0:first_hash=digest
                exact=data[0]==data[1];replay=step!=3 or digest==first_hash
                evidence.append(dict(step=step,exact_bytes=exact,reset_replay=replay,
                    max_abs_error=float((reference_logits-split).abs().max()),**result))
                if not exact or not replay:raise AssertionError('development equivalence failed')
                tokens=[int(reference_logits.argmax(-1).item())]
        if exceeded.is_set():raise RuntimeError('resource budget exceeded')
        evidence.finish('complete',cleanup=session.cleanup,peak_sampled_aggregate_rss_bytes=peak[0])
    except BaseException as error:
        if session:session.close()
        evidence.finish('failed',error=repr(error),cleanup=session.cleanup if session else None,peak_sampled_aggregate_rss_bytes=peak[0]);raise
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(1)

if __name__=='__main__':main()
