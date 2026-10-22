"""Two local spawned workers. Strict per-session epochs; no network listeners."""
import json
import multiprocessing
import time
import math
import threading
from .process_mesh import _check
from .cached_protocol import State, encode, decode
from .pretrained_protocol import MAX_FRAME_BYTES, validate_tokens
from .transport_session import _recv, pack_tensor, unpack_tensor, _peak_rss_kib
from .session import WorkerSession


def send_bounded(connection, payload, deadline, cancel_event):
    done = threading.Event()
    errors = []
    def send():
        try:
            connection.send_bytes(payload)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    while not done.wait(.01):
        _check(deadline, cancel_event)
    _check(deadline, cancel_event)
    if errors:
        raise errors[0]


def worker(connection, directory, stage):
    try:
        import torch
        from .cached_model import CachedPartition
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        partition = CachedPartition(directory, stage)
        state = State()
        connection.send_bytes(encode({'ready': stage}))
        with torch.inference_mode():
            while True:
                frame = connection.recv_bytes(MAX_FRAME_BYTES)
                if not frame:
                    return
                command, payload = decode(frame)
                state.accept(command)
                started = time.perf_counter()
                if command['op'] == 'reset':
                    if payload:
                        raise ValueError('reset payload forbidden')
                    partition.reset()
                    output = b''
                else:
                    if stage == 0:
                        tokens = json.loads(payload)
                        validate_tokens(tokens)
                        if len(tokens) != command['length']:
                            raise ValueError('token length mismatch')
                        value = torch.tensor([tokens], dtype=torch.int64)
                    else:
                        value = unpack_tensor(payload)
                        if list(value.shape) != [1, command['length'], 576]:
                            raise ValueError('activation length mismatch')
                    output = pack_tensor(partition(value))
                report = dict(command, stage=stage, seconds=time.perf_counter()-started,
                              cache_bytes=partition.cache_bytes(), peak_rss_kib=_peak_rss_kib())
                connection.send_bytes(encode(report, output))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class CachedSession(WorkerSession):
    def __init__(self, directory, *, opt_in=False):
        super().__init__(opt_in=opt_in)
        self.directory = str(directory)
        self.state = State()
        self.shutdown_grace_seconds = 3

    def __enter__(self):
        if self.processes or self.cleanup is not None:
            raise RuntimeError('cannot restart session')
        ctx = multiprocessing.get_context('spawn')
        try:
            for stage in range(2):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker, args=(child, self.directory, stage))
                try:
                    process.start()
                except BaseException:
                    parent.close(); child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
                report, payload = decode(_recv(parent, time.perf_counter()+120, self.cancel_event))
                if report != {'ready': stage} or payload:
                    raise ValueError('invalid readiness')
            self.reset()
            return self
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _exchange(self, command, payload, timeout_seconds):
        if self.failed or self.cleanup is not None or len(self.connections) != 2:
            raise RuntimeError('inactive or failed session')
        started = time.perf_counter()
        reports, sizes = [], []
        try:
            if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError('invalid timeout')
            self.state.accept(command)
            for stage, connection in enumerate(self.connections):
                request = encode(command, payload)
                sizes.append(len(request))
                send_bounded(connection, request, started + timeout_seconds, self.cancel_event)
                report, payload = decode(_recv(connection, started + timeout_seconds, self.cancel_event))
                if not isinstance(report, dict) or set(report) != set(command) | {'stage', 'seconds', 'cache_bytes', 'peak_rss_kib'}:
                    raise ValueError('invalid report schema')
                if any(type(report[k]) is not type(v) or report[k] != v for k,v in dict(command, stage=stage).items()):
                    raise ValueError('out-of-order response')
                if type(report['seconds']) not in (int,float) or not math.isfinite(report['seconds']) or report['seconds'] < 0:
                    raise ValueError('invalid timing')
                # Pinned model: 15 layers * K/V * 3 KV heads * 64 head dim * FP32.
                expected = 15 * 2 * 3 * 64 * self.state.position * 4
                if type(report['cache_bytes']) is not int or report['cache_bytes'] != expected:
                    raise ValueError('invalid owned cache size')
                if report['peak_rss_kib'] is not None and (type(report['peak_rss_kib']) is not int or report['peak_rss_kib'] <= 0):
                    raise ValueError('invalid RSS')
                if command['op'] == 'reset':
                    if payload: raise ValueError('reset output forbidden')
                else:
                    import torch
                    tensor = unpack_tensor(payload)
                    shape = [1,command['length'],576] if stage == 0 else [1,49152]
                    if list(tensor.shape) != shape or not torch.isfinite(tensor).all():
                        raise ValueError('invalid output tensor')
                reports.append(report)
            return {'phase':command['op'], 'position':command['position'], 'request_bytes':sizes,
                    'activation_frame_bytes': sizes[1] if command['op'] != 'reset' else 0,
                    'response_bytes':len(encode(reports[-1], payload)), 'worker_reports':reports,
                    'split_seconds':time.perf_counter()-started,
                    'logits':unpack_tensor(payload) if payload else None}
        except BaseException:
            self.failed = True
            self.state.failed = True
            raise

    def reset(self):
        return self._exchange({'op':'reset','epoch':self.state.epoch+1,'sequence':0,'position':0,'length':0}, b'', 30)

    def forward(self, tokens, timeout_seconds=30):
        try:
            validate_tokens(tokens)
            command = {'op':'prefill' if self.state.position == 0 else 'decode', 'epoch':self.state.epoch,
                       'sequence':self.state.sequence+1, 'position':self.state.position, 'length':len(tokens)}
            return self._exchange(command, json.dumps(tokens).encode(), timeout_seconds)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.cleanup is not None:
            return
        if self.failed or self.cancel_event.is_set():
            # Do not enqueue shutdown into a potentially blocked pipe.
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for connection in self.connections:
                connection.close()
            for process in self.processes:
                process.join(1)
                if process.is_alive():
                    process.kill(); process.join(1)
            self.cleanup = {'workers_cleaned_up':all(not p.is_alive() for p in self.processes),
                            'worker_exitcodes':[p.exitcode for p in self.processes],
                            'worker_forced_termination':[True for p in self.processes]}
            for process in self.processes:
                if not process.is_alive():process.close()
        else:
            super().close()
