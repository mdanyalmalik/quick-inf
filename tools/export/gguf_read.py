"""A minimal GGUF reader (metadata and tensor data, memory-mapped), enough for PrismML's ternary Bonsai packs, whose
PQ2_0 tensors are a type the gguf package doesn't know. Tensors come back as raw bytes plus their type id; PQ2_0 is
recognised by its size (34 bytes per 128 weights: a float16 scale, then 32 bytes of 2-bit codes)."""

import struct
from pathlib import Path

import numpy as np

SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
F32, F16, BF16 = 0, 1, 30
PQ2_BLOCK_BYTES = 34


class Gguf:
    def __init__(self, path: Path):
        self.data = np.memmap(path, dtype=np.uint8, mode='r')
        self.pos = 0
        if bytes(self.data[:4]) != b'GGUF':
            raise ValueError(f'{path} is not a GGUF file')
        self.pos = 4
        self.version = self._read('I')
        tensors, kvs = self._read('Q'), self._read('Q')
        self.meta = {}
        for _ in range(kvs):
            key = self._string()
            self.meta[key] = self._value(self._read('I'))
        self.tensors = {}
        for _ in range(tensors):
            name = self._string()
            dims = [self._read('Q') for _ in range(self._read('I'))]
            kind, offset = self._read('I'), self._read('Q')
            self.tensors[name] = (dims, kind, offset)
        align = self.meta.get('general.alignment', 32)
        self.base = (self.pos + align - 1) // align * align

    def _read(self, fmt: str):
        size = struct.calcsize('<' + fmt)
        value = struct.unpack_from('<' + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def _string(self) -> str:
        n = self._read('Q')
        s = bytes(self.data[self.pos:self.pos + n]).decode('utf-8', 'replace')
        self.pos += n
        return s

    def _value(self, kind: int):
        if kind == 8:
            return self._string()
        if kind == 9:
            item, n = self._read('I'), self._read('Q')
            return [self._value(item) for _ in range(n)]
        return self._read(SCALAR[kind])

    def shape(self, name: str) -> tuple[int, ...]:
        """Row-major shape (GGUF lists dimensions innermost first)."""
        return tuple(reversed(self.tensors[name][0]))

    def raw(self, name: str, nbytes: int) -> np.ndarray:
        _, _, offset = self.tensors[name]
        start = self.base + offset
        return self.data[start:start + nbytes]

    def array(self, name: str) -> np.ndarray:
        """A float tensor (F32, F16 or BF16) as float32, row-major."""
        dims, kind, _ = self.tensors[name]
        n = int(np.prod(dims))
        if kind == F32:
            return np.frombuffer(self.raw(name, 4 * n), np.float32).reshape(self.shape(name)).copy()
        if kind == F16:
            return np.frombuffer(self.raw(name, 2 * n), np.float16).astype(np.float32).reshape(self.shape(name))
        if kind == BF16:
            # bfloat16 is float32's upper half.
            bits = np.frombuffer(self.raw(name, 2 * n), '<u2').astype(np.uint32) << 16
            return bits.view(np.float32).reshape(self.shape(name))
        raise ValueError(f'{name}: not a float tensor (type {kind})')

    def is_pq2(self, name: str) -> bool:
        dims, kind, _ = self.tensors[name]
        return kind not in (F32, F16, BF16) and len(dims) == 2 and dims[0] % 128 == 0

    def ternary(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """A PQ2_0 matrix [rows, width]: codes q ∈ {0, 1, 2} (uint8, the weight is (q − 1) × scale) and float16
        scales [rows, width / 128]. Each 128-weight block: a float16 scale, then 8 little-endian uint32 words, weight
        16w + l at bits 2l of word w (PrismML's codec.py)."""
        rows, width = self.shape(name)
        blocks = rows * width // 128
        data = np.asarray(self.raw(name, blocks * PQ2_BLOCK_BYTES)).reshape(blocks, PQ2_BLOCK_BYTES)
        scales = data[:, :2].copy().view('<f2').reshape(rows, width // 128)
        words = data[:, 2:].copy().view('<u4').reshape(rows, width // 16)
        codes = np.empty((rows, width), np.uint8)
        for lane in range(16):
            codes[:, lane::16] = (words >> (2 * lane)) & 3
        if codes.max() > 2:
            raise ValueError(f'{name}: code 3 in a ternary tensor')
        return codes, scales
