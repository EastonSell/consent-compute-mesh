"""Bounded activation-only codecs; FP32 compute resumes after decoding."""
import json
import math
import struct

MAX_BYTES=200000
CODECS={'fp32': '<f4','fp16': '<f2','int8': '|i1'}


def _shape(shape):
    if not isinstance(shape,list) or len(shape)!=3 or any(type(n) is not int for n in shape) or shape[0]!=1 or not 1<=shape[1]<=64 or shape[2]!=576:
        raise ValueError('activation shape must be [1,1..64,576]')
    return math.prod(shape)


def encode_activation(value,codec):
    import numpy as np
    if codec not in CODECS:raise ValueError('unsupported codec')
    x=np.asarray(value)
    count=_shape(list(x.shape))
    if x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('finite float32 activations required')
    maximum=float(np.max(np.abs(x)))
    scale=None;clipped=0;endpoints=0
    if codec=='int8':
        scale=maximum/127 if maximum else 1.0
        rounded=np.rint(x.astype(np.float64)/scale)  # nearest, ties to even
        clipped=int(np.count_nonzero((rounded < -127)|(rounded > 127)))
        encoded=np.clip(rounded,-127,127).astype(np.int8)
        endpoints=int(np.count_nonzero(np.abs(encoded)==127))
    else:
        with np.errstate(over='ignore'):
            encoded=x.astype(CODECS[codec])
        if not np.isfinite(encoded).all():raise ValueError('codec would overflow finite range')
    header=json.dumps({'version':1,'shape':list(x.shape),'codec':codec,'scale':scale},separators=(',',':'),allow_nan=False).encode('ascii')
    payload=encoded.tobytes(order='C')
    frame=struct.pack('<I',len(header))+header+payload
    if len(frame)>MAX_BYTES:raise ValueError('activation frame exceeds budget')
    return frame,{'codec':codec,'scale':scale,'elements':count,'payload_bytes':len(payload),
                  'max_abs_activation':maximum,'zero_tensor':maximum==0,
                  'clipped_before_cast_count':clipped,'rounded_to_endpoint_count':endpoints}


def decode_activation(frame):
    import numpy as np
    if not isinstance(frame,bytes) or not 4<=len(frame)<=MAX_BYTES:raise ValueError('invalid frame length')
    size=struct.unpack('<I',frame[:4])[0]
    if not 1<=size<=512 or len(frame)<4+size:raise ValueError('invalid header length')
    try:
        header=json.loads(frame[4:4+size])
        if not isinstance(header,dict) or set(header)!={'version','shape','codec','scale'} or type(header['version']) is not int or header['version']!=1:raise ValueError('invalid activation header')
        codec=header['codec']
        if codec not in CODECS:raise ValueError('unsupported codec')
        count=_shape(header['shape']);dtype=np.dtype(CODECS[codec])
        payload=frame[4+size:]
        if len(payload)!=count*dtype.itemsize:raise ValueError('payload length mismatch')
        encoded=np.frombuffer(payload,dtype=dtype)
        scale=header['scale']
        if codec=='int8':
            if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0 or np.any(encoded==-128):raise ValueError('invalid symmetric int8 scale or payload')
            with np.errstate(over='ignore'):
                decoded=(encoded.astype(np.float64)*scale).astype(np.float32)
        else:
            if scale is not None:raise ValueError('unexpected scale')
            decoded=encoded.astype(np.float32)
        if not np.isfinite(decoded).all():raise ValueError('nonfinite decoded activation')
        return decoded.reshape(header['shape']).copy(),header
    except (TypeError,KeyError,UnicodeDecodeError,json.JSONDecodeError,OverflowError) as error:
        raise ValueError('invalid activation frame') from error
