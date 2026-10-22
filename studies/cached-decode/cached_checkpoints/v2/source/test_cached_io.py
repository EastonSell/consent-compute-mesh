import multiprocessing
import os
import struct
import threading
import time
import unittest
from mesh.cached_session import CachedSession, io_bounded
from mesh.process_mesh import _Stopped


def partial_frame(connection):
    os.write(connection.fileno(), struct.pack('!i',100)+b'abc')
    time.sleep(20)


class CachedIOTests(unittest.TestCase):
    def test_partial_receive_timeout_and_cancel_cleanup(self):
        for cancel in (False,True):
            session=CachedSession('unused',opt_in=True)
            ctx=multiprocessing.get_context('spawn')
            parent,child=ctx.Pipe()
            process=ctx.Process(target=partial_frame,args=(child,))
            process.start();child.close()
            session.processes.append(process);session.connections.append(parent)
            timer=None
            if cancel:
                timer=threading.Timer(.15,session.cancel_event.set);timer.start()
            started=time.perf_counter()
            try:
                with self.assertRaises(_Stopped):
                    io_bounded(lambda:parent.recv_bytes(1024),started+.3,session.cancel_event,session.io_threads)
            finally:
                session.failed=True;session.close()
                if timer:timer.join()
            self.assertLess(time.perf_counter()-started,3)
            self.assertTrue(session.cleanup['workers_cleaned_up'])
            self.assertTrue(session.cleanup['io_threads_cleaned_up'])

    def test_blocked_send_cleanup(self):
        session=CachedSession('unused',opt_in=True)
        ctx=multiprocessing.get_context('spawn')
        parent,child=ctx.Pipe()
        process=ctx.Process(target=partial_frame,args=(child,))
        process.start();child.close()
        session.processes.append(process);session.connections.append(parent)
        try:
            with self.assertRaises(_Stopped):
                io_bounded(lambda:parent.send_bytes(b'x'*2000000),time.perf_counter()+.2,session.cancel_event,session.io_threads)
        finally:
            session.failed=True;session.close()
        self.assertTrue(session.cleanup['io_threads_cleaned_up'])

if __name__=='__main__':unittest.main()

class FakeConnection:
    def __init__(self, report):self.report=report;self.closed=False
    def send_bytes(self,payload):pass
    def recv_bytes(self,maximum):return self.report
    def close(self):self.closed=True


class CachedResponseTests(unittest.TestCase):
    def test_reset_partial_failure_poisoned(self):
        from mesh.cached_protocol import encode
        command=dict(op='reset',epoch=1,sequence=0,position=0,length=0)
        for change in ({'epoch':0},{'stage':False},{'cache_bytes':1},{'seconds':float('inf')},{'extra':1}):
            session=CachedSession('unused',opt_in=True)
            good=dict(command,stage=0,seconds=0,cache_bytes=0,peak_rss_kib=1)
            bad=dict(command,stage=1,seconds=0,cache_bytes=0,peak_rss_kib=1)
            bad.update(change)
            # Infinity cannot pass the encoder either; emulate a malicious peer.
            if change.get('seconds') == float('inf'):
                import json
                header=json.dumps(bad).encode();frame=struct.pack('<I',len(header))+header
            else:frame=encode(bad)
            session.connections=[FakeConnection(encode(good)),FakeConnection(frame)]
            with self.assertRaises(ValueError):session.reset()
            self.assertTrue(session.failed)
            with self.assertRaises(RuntimeError):session.reset()
            session.close()
            self.assertTrue(session.cleanup['io_threads_cleaned_up'])
