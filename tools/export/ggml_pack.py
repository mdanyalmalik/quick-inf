"""llama.cpp's calibrated quantization types (IQ2_S, IQ3_S, IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K) re-laid out for
Quick Inference's WebGPU kernels (com.microsoft MatMulGgml and GatherGgml in our ONNX Runtime build), without changing
a single weight.

GGUF blocks of 256 weights pack their fields at byte offsets that aren't multiples of 4 (IQ2_S is 82 bytes), and the
K types interleave sub-blocks within a byte (Q4_K's first 32 weights are the low nibbles of 32 bytes, the next 32 the
high ones). Here each row is

    [ one record of R words per superblock ][ A float16 values per superblock (d, and dmin or IQ4_XS's high scale bits) ]

and inside a record, each sub-block of 32 weights has its codes in words of its own, weight i at bit (i * bits) % 32 of
word (i * bits) // 32; the per-sub-block scales stay in llama.cpp's own packing. Rows come out the same size as in
the GGUF (Q3_K and Q6_K gain 2 bytes per row when a row has an odd number of superblocks). `dequantize` decodes the
layout the way the shaders do; `check` compares it with gguf-py's dequantization, which must match exactly.
"""

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants

QK = 256
# ggml type id -> (words per record, float16 values per superblock)
LAYOUTS = {
    T.Q3_K: (27, 1),
    T.Q4_K: (35, 2),
    T.Q5_K: (43, 2),
    T.Q6_K: (52, 1),
    T.IQ2_S: (20, 1),
    T.IQ3_S: (27, 1),
    T.IQ4_XS: (33, 2),
}
BLOCK_BYTES = {kind: quants.GGML_QUANT_SIZES[kind][1] for kind in LAYOUTS}
IQ4_VALUES = np.array(quants.IQ4_NL.kvalues, np.float32)


def row_words(kind: int, k: int) -> int:
    record, aux = LAYOUTS[kind]
    superblocks = k // QK
    return superblocks * record + (superblocks * aux + 1) // 2


def grid_values(kind: int) -> np.ndarray:
    """IQ2_S's 1024 vectors of 8 magnitudes or IQ3_S's 512 vectors of 4 (float32, [entries, length])."""
    cls = {T.IQ2_S: quants.IQ2_S, T.IQ3_S: quants.IQ3_S}[kind]
    cls.init_grid()
    return cls.grid.reshape(-1, cls.grid_shape[1]).astype(np.float32)


def grid(kind: int) -> np.ndarray | None:
    """The op's grid input (uint32 words), or None for types without one: IQ2_S's vectors of 8 magnitudes and IQ3_S's
    of 4 as float16 (two per word), IQ4_XS's 16 values as float32 bits."""
    if kind == T.IQ4_XS:
        return IQ4_VALUES.view('<u4').copy()
    if kind not in (T.IQ2_S, T.IQ3_S):
        return None
    return grid_values(kind).astype(np.float16).view('<u4').reshape(-1).copy()


def _bits(codes: np.ndarray, bits: int) -> np.ndarray:
    """codes [..., 32] of `bits` bits each -> [..., bits] words, weight i at bit (i * bits) % 32 of word (i * bits) // 32."""
    per_word = 32 // bits
    c = codes.astype(np.uint32).reshape(*codes.shape[:-1], bits, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(*([1] * (c.ndim - 1)), per_word)
    return np.bitwise_or.reduce(c << shifts, axis=-1)


def _unbits(words: np.ndarray, bits: int) -> np.ndarray:
    per_word = 32 // bits
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).reshape(*([1] * words.ndim), per_word)
    return ((words[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*words.shape[:-1], 32)


def _words(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view('<u4')


def pack(kind: int, data: np.ndarray, rows: int, k: int) -> np.ndarray:
    """GGUF tensor bytes (rows of k weights) -> uint32 [rows, row_words(kind, k)]."""
    kind = T(kind)
    nsb = k // QK
    blocks = np.asarray(data, np.uint8).reshape(rows * nsb, BLOCK_BYTES[kind])
    n = blocks.shape[0]
    f16 = lambda off: blocks[:, off:off + 2].copy().view('<u2').reshape(n)
    if kind == T.Q4_K or kind == T.Q5_K:
        d, dmin, scales = f16(0), f16(2), blocks[:, 4:16]
        if kind == T.Q4_K:
            qs = blocks[:, 16:144]
        else:
            qh, qs = blocks[:, 16:48], blocks[:, 48:176]
        low = (qs.reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        subs = [_bits(low.reshape(n, 8, 32), 4)]
        if kind == T.Q5_K:
            high = (qh.reshape(n, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1
            subs.append(_bits(high, 1))
        record = np.concatenate([_words(scales), np.concatenate(subs, axis=-1).reshape(n, -1)], axis=1)
        aux = np.stack([d, dmin], axis=1)
    elif kind == T.Q6_K:
        ql, qh, scales, d = blocks[:, :128], blocks[:, 128:192], blocks[:, 192:208], f16(208)
        low = ((ql.reshape(n, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(n, 8, 32)
        high = ((qh.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, 8, 32)
        record = np.concatenate([np.concatenate([_bits(low, 4), _bits(high, 2)], axis=-1).reshape(n, -1), _words(scales)], axis=1)
        aux = d.reshape(n, 1)
    elif kind == T.Q3_K:
        hmask, qs, scales, d = blocks[:, :32], blocks[:, 32:96], blocks[:, 96:108], f16(108)
        low = ((qs.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(n, 8, 32)
        high = ((hmask.reshape(n, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1).reshape(n, 8, 32)
        record = np.concatenate([np.concatenate([_bits(low, 2), _bits(high, 1)], axis=-1).reshape(n, -1), _words(scales)], axis=1)
        aux = d.reshape(n, 1)
    elif kind == T.IQ4_XS:
        d, scales_h, scales_l, qs = f16(0), f16(2), blocks[:, 4:8], blocks[:, 8:136]
        codes = ((qs.reshape(n, 8, 1, 16) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(n, 8, 32)
        record = np.concatenate([_words(scales_l), _bits(codes, 4).reshape(n, -1)], axis=1)
        aux = np.stack([d, scales_h], axis=1)
    elif kind == T.IQ2_S:
        d, qs, signs, qh, scales = f16(0), blocks[:, 2:34], blocks[:, 34:66], blocks[:, 66:74], blocks[:, 74:82]
        record = np.concatenate([np.stack([_words(qs), _words(signs)], axis=-1).reshape(n, 16), _words(qh), _words(scales)], axis=1)
        aux = d.reshape(n, 1)
    elif kind == T.IQ3_S:
        d, qs, qh, signs, scales = f16(0), blocks[:, 2:66], blocks[:, 66:74], blocks[:, 74:106], blocks[:, 106:110]
        sub = np.concatenate([_words(qs).reshape(n, 8, 2), _words(signs).reshape(n, 8, 1)], axis=-1)
        record = np.concatenate([sub.reshape(n, 24), _words(qh), _words(scales)], axis=1)
        aux = d.reshape(n, 1)
    else:
        raise ValueError(f'unsupported ggml type {kind.name}')
    words, halves = LAYOUTS[kind]
    assert record.shape[1] == words, (kind.name, record.shape)
    halves_row = aux.reshape(rows, nsb * halves)
    if halves_row.shape[1] % 2:
        halves_row = np.concatenate([halves_row, np.zeros((rows, 1), np.uint16)], axis=1)
    out = np.concatenate([record.reshape(rows, nsb * words).astype(np.uint32), np.ascontiguousarray(halves_row).view('<u4')], axis=1)
    assert out.shape[1] == row_words(kind, k)
    return out


def _scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q4_K/Q5_K's 12 scale bytes -> 6-bit scales and mins [n, 8] (llama.cpp's get_scale_min_k4)."""
    q = scales.astype(np.uint32)
    sc, m = np.empty((q.shape[0], 8), np.uint32), np.empty((q.shape[0], 8), np.uint32)
    for j in range(8):
        if j < 4:
            sc[:, j], m[:, j] = q[:, j] & 63, q[:, j + 4] & 63
        else:
            sc[:, j] = (q[:, j + 4] & 15) | ((q[:, j - 4] >> 6) << 4)
            m[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return sc, m


def dequantize(kind: int, packed: np.ndarray, k: int) -> np.ndarray:
    """uint32 [rows, row_words] -> float32 [rows, k], decoding the layout the way the shaders do."""
    kind = T(kind)
    rows, nsb = packed.shape[0], k // QK
    words, halves = LAYOUTS[kind]
    n = rows * nsb
    rec = packed[:, :nsb * words].reshape(n, words)
    aux = np.ascontiguousarray(packed[:, nsb * words:]).view('<f2')[:, :nsb * halves].reshape(n, halves)
    d = aux[:, 0].astype(np.float32).reshape(n, 1)
    as_bytes = lambda w: np.ascontiguousarray(w).view(np.uint8)
    if kind in (T.Q4_K, T.Q5_K):
        sc, m = _scale_min_k4(as_bytes(rec[:, :3]))
        dmin = aux[:, 1].astype(np.float32).reshape(n, 1)
        per = 4 if kind == T.Q4_K else 5
        sub = rec[:, 3:].reshape(n, 8, per)
        q = _unbits(sub[..., :4], 4)
        if kind == T.Q5_K:
            q = q | (_unbits(sub[..., 4:5], 1) << 4)
        w = (d * sc.astype(np.float32))[..., None] * q.astype(np.float32) - (dmin * m.astype(np.float32))[..., None]
    elif kind == T.Q6_K:
        sub = rec[:, :48].reshape(n, 8, 6)
        q = (_unbits(sub[..., :4], 4) | (_unbits(sub[..., 4:6], 2) << 4)).astype(np.float32) - 32
        scales = as_bytes(rec[:, 48:52]).view(np.int8).astype(np.float32).reshape(n, 8, 2, 1)
        w = (d.reshape(n, 1, 1, 1) * scales) * q.reshape(n, 8, 2, 16)
    elif kind == T.Q3_K:
        sub = rec[:, :24].reshape(n, 8, 3)
        q = (_unbits(sub[..., :2], 2) | (_unbits(sub[..., 2:3], 1) << 2)).astype(np.float32) - 4
        b = as_bytes(rec[:, 24:27]).astype(np.int32)
        g = np.arange(16)
        scales = (((b[:, g % 8] >> (4 * (g // 8))) & 15) | (((b[:, 8 + g % 4] >> (2 * (g // 4))) & 3) << 4)) - 32
        w = (d.reshape(n, 1, 1, 1) * scales.astype(np.float32).reshape(n, 8, 2, 1)) * q.reshape(n, 8, 2, 16)
    elif kind == T.IQ4_XS:
        j = np.arange(8)
        low = (as_bytes(rec[:, :1]).astype(np.int32)[:, j // 2] >> (4 * (j % 2))) & 15
        high = (aux[:, 1:2].copy().view('<u2').astype(np.int32) >> (2 * j)) & 3
        scales = (low | (high << 4)) - 32
        q = IQ4_VALUES[_unbits(rec[:, 1:].reshape(n, 8, 4), 4)]
        w = (d * scales.astype(np.float32))[..., None] * q
    elif kind == T.IQ2_S:
        table = grid_values(kind)
        sub = rec[:, :16].reshape(n, 8, 2)
        qs, signs = as_bytes(sub[..., 0:1]).reshape(n, 8, 4), as_bytes(sub[..., 1:2]).reshape(n, 8, 4)
        qh, sc = as_bytes(rec[:, 16:18]).reshape(n, 8, 1), as_bytes(rec[:, 18:20]).reshape(n, 8)
        index = qs.astype(np.int32) | (((qh.astype(np.int32) >> (2 * np.arange(4))) & 3) << 8)
        sign = np.where((signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1, -1.0, 1.0).astype(np.float32)
        scale = d.reshape(n, 1, 1) * (np.float32(0.5) + ((sc[..., None] >> np.array([0, 4], np.uint8)) & 15).astype(np.float32)) * np.float32(0.25)
        w = scale.reshape(n, 8, 2, 1) * (table[index] * sign).reshape(n, 8, 2, 16)
    elif kind == T.IQ3_S:
        table = grid_values(kind)
        sub = rec[:, :24].reshape(n, 8, 3)
        qs, signs = as_bytes(sub[..., 0:2]).reshape(n, 8, 8), as_bytes(sub[..., 2:3]).reshape(n, 8, 4)
        qh, sc = as_bytes(rec[:, 24:26]).reshape(n, 8, 1), as_bytes(rec[:, 26:27]).reshape(n, 4)
        index = qs.astype(np.int32) | (((qh.astype(np.int32) >> np.arange(8)) & 1) << 8)
        sign = np.where((signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1, -1.0, 1.0).astype(np.float32)
        scale = d * (1 + 2 * ((sc[:, np.arange(8) // 2] >> (4 * (np.arange(8) % 2))) & 15)).astype(np.float32)
        w = scale.reshape(n, 8, 1) * (table[index].reshape(n, 8, 32) * sign.reshape(n, 8, 32))
    else:
        raise ValueError(f'unsupported ggml type {kind.name}')
    return np.asarray(w, np.float32).reshape(rows, k)


def check(kind: int, data: np.ndarray, rows: int, k: int, packed: np.ndarray) -> None:
    """Raises unless the packed layout decodes to exactly gguf-py's dequantization of the original bytes."""
    want = quants.dequantize(np.asarray(data, np.uint8).reshape(rows, k // QK * BLOCK_BYTES[T(kind)]), T(kind)).reshape(rows, k)
    got = dequantize(kind, packed, k)
    if not np.array_equal(got, want.astype(np.float32)):
        bad = np.argwhere(got != want)[0]
        raise ValueError(f'{T(kind).name}: packed layout differs at {tuple(bad)}: {got[tuple(bad)]} vs {want[tuple(bad)]}')
