"""Q2_K block decoding for TileLang kernels.

GGUF Q2_K (ggml `block_q2_K`): 256 weights in 84 bytes,
    uint8 scales[16]; uint8 qs[64]; half d; half dmin;
Viewed as uint32 [21]: words 0..3 are the 16 scale bytes, 4..19 the 64
quant bytes, 20 is d (low half) and dmin (high half).

Weight w (0..255) lives in 16-weight sub-block is = w // 16 with scale
byte scales[is] (low nibble: scale, high nibble: min). Its 2-bit code sits
in byte qs[(w // 128) * 32 + (w % 32)] at shift 2 * ((w % 128) // 32):
    value = d * (scales[is] & 0xF) * q  -  dmin * (scales[is] >> 4)
For sub-block is: the 16 codes are the bytes qs[b0 .. b0 + 16) with
b0 = (is // 8) * 32 + (is % 2) * 16, all at shift 2 * ((is % 8) // 2).
"""

from __future__ import annotations

import tilelang.language as T

QK_K = 256
U32_PER_BLOCK = 21
SUB_BLOCKS = 16          # per block, 16 weights each


def sub_block_qs_word(blk, is_, k):
    """uint32 index of code bytes 4k..4k+3 of sub-block is_ (k in 0..3)."""
    b0 = (is_ // 8) * 32 + (is_ % 2) * 16
    return blk * U32_PER_BLOCK + 4 + (b0 + 4 * k) // 4


def sub_block_shift(is_):
    return 2 * ((is_ % 8) // 2)


def scale_byte(w_at, blk, is_):
    """scales[is_] of block blk as int32 (low nibble scale, high nibble min)."""
    word = w_at(blk * U32_PER_BLOCK + is_ // 4)
    return T.Cast(T.int32, (word >> (8 * (is_ % 4))) & 0xFF)


def d_dmin(w_at, blk):
    word = w_at(blk * U32_PER_BLOCK + 20)
    d = T.Cast(T.float32, T.reinterpret(T.float16, T.Cast(T.uint16, word & 0xFFFF)))
    dmin = T.Cast(T.float32, T.reinterpret(T.float16, T.Cast(T.uint16, word >> 16)))
    return d, dmin
