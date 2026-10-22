import json
import threading
import time
import unittest
from unittest.mock import patch
from mesh.transport_session import TransportSession


def _stalling_worker(connection,directory,stage):
    try:
        connection.send_bytes(json.dumps({'ready':True,'stage':stage,'load_seconds':0.,'peak_rss_kib':None}).encode())
        while connection.recv_bytes():time.sleep(5)
    except (EOFError,BrokenPipeError):pass
    finally:connection.close()


class TransportFailureTests(unittest.TestCase):
    def test_timeout_invalidates_and_cleans_actual_workers(self):
        with patch('mesh.transport_session._transport_worker',_stalling_worker):
            session=TransportSession('unused',opt_in=True)
            with session:
                session.shutdown_grace_seconds=.1
                result=session.forward([1,2],codec='int8',timeout_seconds=.03)
                self.assertEqual(result['status'],'timeout')
                self.assertTrue(session.failed)
                with self.assertRaises(RuntimeError):session.forward([1],codec='fp16')
            self.assertTrue(session.cleanup['workers_cleaned_up'])
            self.assertTrue(any(session.cleanup['worker_forced_termination']))

    def test_cancel_before_startup_cleans_actual_workers(self):
        event=threading.Event();event.set()
        with patch('mesh.transport_session._transport_worker',_stalling_worker):
            session=TransportSession('unused',opt_in=True,cancel_event=event)
            session.shutdown_grace_seconds=.1
            with self.assertRaises(Exception):session.__enter__()
            self.assertTrue(session.cleanup['workers_cleaned_up'])

    def test_corrupt_report_invalidates_session(self):
        session=TransportSession('unused',opt_in=True)
        class Fake:
            def send_bytes(self,payload):pass
        session.connections=[Fake(),Fake()]
        with patch('mesh.transport_session._recv',return_value=b'{}'):
            result=session.forward([1],codec='fp32')
        self.assertEqual(result['status'],'protocol_error')
        self.assertTrue(session.failed)

    def test_normal_exit_is_clean(self):
        with patch('mesh.transport_session._transport_worker',_stalling_worker):
            session=TransportSession('unused',opt_in=True)
            with session:pass
            self.assertTrue(session.cleanup['workers_cleaned_up'])
            self.assertEqual(session.cleanup['worker_exitcodes'],[0,0])

if __name__=='__main__':unittest.main()
