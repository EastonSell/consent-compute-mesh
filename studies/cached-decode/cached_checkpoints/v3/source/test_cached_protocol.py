import unittest
from mesh.cached_protocol import State, encode, decode


def cmd(op='reset', epoch=1, sequence=0, position=0, length=0):
    return dict(op=op, epoch=epoch, sequence=sequence, position=position, length=length)


class CachedProtocolTests(unittest.TestCase):
    def test_roundtrip(self):
        self.assertEqual(decode(encode(cmd(), b'abc')), (cmd(), b'abc'))

    def test_reset_and_decode(self):
        state = State(); state.accept(cmd())
        state.accept(cmd('prefill',1,1,0,4)); state.accept(cmd('decode',1,2,4,1))
        self.assertEqual(state.position,5)
        state.accept(cmd(epoch=2)); self.assertEqual(state.position,0)
        state.accept(cmd('prefill',2,1,0,64))
        with self.assertRaises(ValueError): state.accept(cmd('decode',2,2,64,1))

    def test_fail_closed(self):
        bad = [cmd('decode',1,1,0,1),cmd('prefill',1,2,0,2),cmd('prefill',0,1,0,2),
               cmd('prefill',1,1,1,2),cmd('prefill',1,1,0,0),cmd('prefill',True,1,0,2),
               cmd('reset',1),dict(cmd(), extra=1),[],cmd('unknown',1,1,0,2)]
        for command in bad:
            with self.subTest(command=command):
                state=State(); state.accept(cmd())
                with self.assertRaises(ValueError):state.accept(command)
                with self.assertRaises(ValueError):state.accept(cmd(epoch=2))

    def test_stale_session(self):
        state=State(); state.accept(cmd()); state.accept(cmd(epoch=2))
        with self.assertRaises(ValueError):state.accept(cmd('prefill',1,1,0,2))

    def test_malformed_frames(self):
        import struct
        for frame in [b'',b'abcd',struct.pack('<I',5000),struct.pack('<I',13)+b'{"a":1,"a":2}']:
            with self.assertRaises((ValueError,UnicodeDecodeError)):decode(frame)

if __name__ == '__main__':unittest.main()

class TensorHeaderTests(unittest.TestCase):
    def test_duplicate_tensor_key_rejected(self):
        from mesh.cached_session import unpack_tensor
        import struct
        header=b'{"shape":[1],"shape":[1],"dtype":"<f4"}'
        with self.assertRaises(ValueError):unpack_tensor(struct.pack('<I',len(header))+header+b'\x00'*4)
