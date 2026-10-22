"""Optional persistent model partitions over private pipes, no network service."""
import json
import math
import struct
import multiprocessing
try:
    import resource
except ImportError:  # Windows and other platforms without Unix resource accounting.
    resource = None
import time

from .transport_codec import encode_activation, decode_activation, CODECS
from .pretrained_protocol import MAX_FRAME_BYTES, decode_frame, encode_frame, validate_tokens
from .process_mesh import _Stopped, _check, _wait
from .session import WorkerSession


def _peak_rss_kib():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _validate_metrics(report, timing_key):
    value = report[timing_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid worker timing')
    if report['peak_rss_kib'] is not None and (type(report['peak_rss_kib']) is not int or report['peak_rss_kib'] <= 0):
        raise ValueError('invalid worker memory metric')



def _validate_transport_metrics(meta, header, element_count, payload_bytes):
    fields={'codec','scale','elements','payload_bytes','max_abs_activation','zero_tensor','clipped_before_cast_count','rounded_to_endpoint_count'}
    if not isinstance(meta,dict) or set(meta)!=fields or meta['codec']!=header['codec'] or meta['scale']!=header['scale']:
        raise ValueError('invalid codec metrics schema')
    for key,expected in [('elements',element_count),('payload_bytes',payload_bytes)]:
        if type(meta[key]) is not int or meta[key]!=expected:raise ValueError('invalid codec sizes')
    for key in ['clipped_before_cast_count','rounded_to_endpoint_count']:
        if type(meta[key]) is not int or not 0<=meta[key]<=element_count:raise ValueError('invalid codec count')
    maximum=meta['max_abs_activation']
    if type(maximum) not in (int,float) or not math.isfinite(maximum) or maximum<0 or type(meta['zero_tensor']) is not bool or meta['zero_tensor']!=(maximum==0):
        raise ValueError('invalid codec range')


def pack_tensor(tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype('<f4', copy=False)
    return encode_frame(list(array.shape), array.tobytes())


def unpack_tensor(frame):
    import numpy as np
    import torch
    shape, payload = decode_frame(frame)
    return torch.from_numpy(np.frombuffer(payload, dtype='<f4').copy().reshape(shape))


def _recv(connection, deadline, cancel_event):
    while True:
        _check(deadline, cancel_event)
        if connection.poll(min(.01, max(0, deadline - time.perf_counter()))):
            try:
                return connection.recv_bytes(MAX_FRAME_BYTES)
            except (EOFError, OSError):
                raise _Stopped('disconnected') from None


def _transport_worker(connection, directory, stage):
    try:
        import torch
        from .pretrained_model import Partition
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        started = time.perf_counter()
        partition = Partition(directory, stage)
        connection.send_bytes(json.dumps({'ready': True, 'stage': stage, 'load_seconds': time.perf_counter() - started,
                                          'peak_rss_kib': _peak_rss_kib()}).encode())
        request_id = 0
        with torch.inference_mode():
            while True:
                payload = connection.recv_bytes(MAX_FRAME_BYTES)
                if not payload:
                    return
                request_id += 1
                decode_started = time.perf_counter()
                if stage == 0:
                    command=json.loads(payload)
                    if not isinstance(command,dict) or set(command)!={'tokens','codec'} or command['codec'] not in CODECS:
                        raise ValueError('invalid transport request')
                    codec=command['codec']
                    value=torch.tensor([validate_tokens(command['tokens'])],dtype=torch.int64)
                else:
                    array,metadata=decode_activation(payload)
                    value=torch.from_numpy(array)
                decode_seconds=time.perf_counter()-decode_started
                started = time.perf_counter()
                output = partition(value)
                elapsed = time.perf_counter() - started
                encode_started=time.perf_counter()
                if stage==0:
                    frame,transport_meta=encode_activation(output.detach().cpu().numpy(),codec)
                else:
                    frame=pack_tensor(output)
                    transport_meta=None
                encode_seconds=time.perf_counter()-encode_started
                connection.send_bytes(json.dumps({'stage': stage, 'request_id': request_id, 'compute_seconds': elapsed, 'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds, 'transport_meta': transport_meta,
                                                  'peak_rss_kib': _peak_rss_kib()}).encode())
                connection.send_bytes(frame)
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class TransportSession(WorkerSession):
    def __init__(self, directory, *, opt_in=False, cancel_event=None):
        super().__init__(opt_in=opt_in, cancel_event=cancel_event)
        self.directory = str(directory)
        self.load_reports = []
        self.shutdown_grace_seconds = 3.0
        self.request_id = 0

    def __enter__(self):
        if self.processes or self.cleanup is not None:
            raise RuntimeError('session cannot be restarted')
        started = time.perf_counter()
        context = multiprocessing.get_context('spawn')
        try:
            for stage in range(2):
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_transport_worker, args=(child, self.directory, stage))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
                # Sequential readiness avoids simultaneous model-load transients.
                report = json.loads(_recv(parent, started + 120, self.cancel_event))
                if not isinstance(report, dict) or set(report) != {'ready', 'stage', 'load_seconds', 'peak_rss_kib'} or report['ready'] is not True or type(report['stage']) is not int or report['stage'] != stage:
                    raise ValueError('invalid readiness report')
                _validate_metrics(report, 'load_seconds')
                self.load_reports.append(report)
            self.startup_seconds = time.perf_counter() - started
            return self
        except BaseException:
            self.close()
            raise

    def forward(self, tokens, *, codec='fp32', latency_ms=0, bytes_per_second=None, timeout_seconds=10):
        import math
        if self.failed or self.cleanup is not None or len(self.connections) != 2:
            raise RuntimeError('session is not active or has failed')
        validate_tokens(tokens)
        if codec not in CODECS:raise ValueError('unsupported codec')
        for name, value, positive in [('latency_ms', latency_ms, False), ('bytes_per_second', bytes_per_second, True), ('timeout_seconds', timeout_seconds, True)]:
            if name == 'bytes_per_second' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f'invalid {name}')
        if bytes_per_second is not None and not math.isfinite(MAX_FRAME_BYTES / bytes_per_second + latency_ms / 1000):
            raise ValueError('derived delay must be finite')
        self.request_id += 1
        payload = json.dumps({'tokens':tokens,'codec':codec}).encode()
        result = {'codec':codec,'worker_encode_seconds':[],'worker_decode_seconds':[],'parent_validation_seconds':0.0,'transport_meta':None,'worker_compute_seconds': [], 'worker_peak_rss_kib': [], 'emulated_wait_requested_seconds': 0.0,
                  'emulated_wait_observed_seconds': 0.0, 'activation_frame_bytes': 0}
        started = time.perf_counter()
        deadline = started + timeout_seconds
        try:
            for stage, connection in enumerate(self.connections):
                _check(deadline, self.cancel_event)
                if stage == 1:
                    result['activation_frame_bytes'] = len(payload)
                    delay = latency_ms / 1000 + (len(payload) / bytes_per_second if bytes_per_second is not None else 0)
                    result['emulated_wait_requested_seconds'] = delay
                    wait_started = time.perf_counter()
                    try:
                        if delay:
                            _wait(delay, deadline, self.cancel_event)
                    finally:
                        result['emulated_wait_observed_seconds'] = time.perf_counter() - wait_started if delay else 0.0
                connection.send_bytes(payload)
                report = json.loads(_recv(connection, deadline, self.cancel_event))
                if not isinstance(report, dict) or set(report) != {'stage', 'request_id', 'compute_seconds', 'peak_rss_kib','encode_seconds','decode_seconds','transport_meta'}:
                    raise ValueError('invalid response report')
                if type(report['stage']) is not int or report['stage'] != stage or type(report['request_id']) is not int or report['request_id'] != self.request_id:
                    raise ValueError('out-of-order response report')
                _validate_metrics(report, 'compute_seconds')
                _validate_metrics(report,'encode_seconds')
                _validate_metrics(report,'decode_seconds')
                payload = _recv(connection, deadline, self.cancel_event)
                validation_started=time.perf_counter()
                if stage==0:
                    array,metadata=decode_activation(payload)
                    if list(array.shape)!=[1,len(tokens),576] or metadata['codec']!=codec:
                        raise ValueError('invalid activation tensor')
                    _validate_transport_metrics(report['transport_meta'],metadata,array.size,array.size*{'fp32':4,'fp16':2,'int8':1}[codec])
                    result['transport_meta']=report['transport_meta']
                else:
                    shape,values=decode_frame(payload)
                    if report['transport_meta'] is not None or shape != [1,49152] or not all(math.isfinite(value[0]) for value in struct.iter_unpack('<f',values)):
                        raise ValueError('invalid response tensor')
                result['parent_validation_seconds']+=time.perf_counter()-validation_started
                result['worker_encode_seconds'].append(report['encode_seconds'])
                result['worker_decode_seconds'].append(report['decode_seconds'])
                result['worker_compute_seconds'].append(report['compute_seconds'])
                result['worker_peak_rss_kib'].append(report['peak_rss_kib'])
            result['logits'] = unpack_tensor(payload)
            result['status'] = 'ok'
        except (ValueError, KeyError, TypeError, UnicodeDecodeError, struct.error) as error:
            self.failed = True
            result['status'] = 'protocol_error'
        except (_Stopped, BrokenPipeError, EOFError, OSError) as error:
            self.failed = True
            result['status'] = str(error) if isinstance(error, _Stopped) else 'disconnected'
        result['split_seconds'] = time.perf_counter() - started
        return result
