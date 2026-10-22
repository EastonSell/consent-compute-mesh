"""Strict state transitions, validated before any cache mutation."""
import json
import struct
from .pretrained_protocol import MAX_FRAME_BYTES


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def encode(command, payload=b''):
    header = json.dumps(command, separators=(',', ':'), allow_nan=False).encode()
    if len(header) > 4096 or len(payload) + len(header) + 4 > MAX_FRAME_BYTES:
        raise ValueError('frame exceeds bounds')
    return struct.pack('<I', len(header)) + header + payload


def decode(frame):
    if not 4 <= len(frame) <= MAX_FRAME_BYTES:
        raise ValueError('invalid frame length')
    size, = struct.unpack('<I', frame[:4])
    if not 1 <= size <= 4096 or len(frame) < 4 + size:
        raise ValueError('invalid header length')
    return json.loads(frame[4:4+size], object_pairs_hook=no_duplicates), frame[4+size:]


class State:
    def __init__(self):
        self.epoch = 0
        self.sequence = 0
        self.position = 0
        self.failed = False

    def accept(self, command):
        try:
            if self.failed or not isinstance(command, dict) or set(command) != {'op', 'epoch', 'sequence', 'position', 'length'}:
                raise ValueError('invalid command schema/state')
            for name in ('epoch', 'sequence', 'position', 'length'):
                if type(command[name]) is not int:
                    raise ValueError('noninteger state')
            op, epoch, seq, pos, length = (command[k] for k in ('op', 'epoch', 'sequence', 'position', 'length'))
            if op == 'reset':
                if (epoch, seq, pos, length) != (self.epoch + 1, 0, 0, 0):
                    raise ValueError('invalid reset')
                self.epoch, self.sequence, self.position = epoch, 0, 0
            else:
                if self.epoch < 1 or (epoch, seq, pos) != (self.epoch, self.sequence + 1, self.position):
                    raise ValueError('out-of-order or stale command')
                if op != ('prefill' if pos == 0 else 'decode') or not 1 <= length <= 64 or pos + length > 64 or (pos and length != 1):
                    raise ValueError('invalid phase or length')
                self.sequence, self.position = seq, pos + length
        except BaseException:
            self.failed = True
            raise
