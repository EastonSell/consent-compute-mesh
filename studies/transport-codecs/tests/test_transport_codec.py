import unittest
import struct
import json
try:
    import numpy as np
except ImportError:
    np=None
from mesh.transport_codec import encode_activation, decode_activation

@unittest.skipIf(np is None,'optional NumPy runtime unavailable')
class CodecTests(unittest.TestCase):
    def test_fp32_exact_and_wire_sizes(self):
        x=np.linspace(-3,3,576,dtype=np.float32).reshape(1,1,576)
        sizes=[]
        for codec in ['fp32','fp16','int8']:
            frame,meta=encode_activation(x,codec)
            y,decoded=decode_activation(frame)
            self.assertEqual(decoded['codec'],codec)
            self.assertEqual(y.dtype,np.float32)
            self.assertEqual(y.shape,x.shape)
            if codec=='fp32':np.testing.assert_array_equal(x,y)
            if codec=='fp16':np.testing.assert_allclose(x,y,atol=.001,rtol=.001)
            if codec=='int8':self.assertLessEqual(float(np.max(np.abs(x-y))),meta['scale']/2+1e-6)
            sizes.append(len(frame))
        self.assertGreater(sizes[0],sizes[1]);self.assertGreater(sizes[1],sizes[2])

    def test_zero_tensor_and_constant_are_finite(self):
        for number in [0.,-2.,1e-40]:
            x=np.full((1,2,576),number,dtype=np.float32)
            frame,meta=encode_activation(x,'int8');y,_=decode_activation(frame)
            self.assertTrue(np.isfinite(y).all())
            np.testing.assert_allclose(x,y,atol=1e-44,rtol=1e-6)
            if number==0:self.assertEqual(meta['scale'],1.)

    def test_reject_nonfinite_shape_and_half_overflow(self):
        for number in [np.inf,np.nan]:
            with self.assertRaises(ValueError):encode_activation(np.full((1,1,576),number,dtype=np.float32),'int8')
        with self.assertRaises(ValueError):encode_activation(np.zeros((2,1,576),dtype=np.float32),'fp32')
        with self.assertRaises(ValueError):encode_activation(np.full((1,1,576),1e10,dtype=np.float32),'fp16')

    def test_malformed_frames(self):
        frame,_=encode_activation(np.ones((1,1,576),dtype=np.float32),'int8')
        for bad in [b'',frame[:4],frame[:-1],frame+b'x']:
            with self.assertRaises(ValueError):decode_activation(bad)
        size=struct.unpack('<I',frame[:4])[0];header=json.loads(frame[4:4+size]);payload=frame[4+size:]
        for key,value in [('codec','pickle'),('scale',0),('scale',float('nan')),('shape',[1,65,576])]:
            h=dict(header);h[key]=value;raw=json.dumps(h).encode()
            with self.assertRaises(ValueError):decode_activation(struct.pack('<I',len(raw))+raw+payload)
        with self.assertRaises(ValueError):decode_activation(frame[:-1]+b'\x80')

if __name__=='__main__':unittest.main()
