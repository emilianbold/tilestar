"""IQ2_XXS block decoding for TileLang kernels.

GGUF IQ2_XXS (ggml `block_iq2_xxs`): 256 weights in 66 bytes,
    half d; uint16 qs[32];
Viewed as uint16 [33]: u16 0 is d, u16 1 + 4*ib + {0,1,2,3} describe the
32-weight sub-block ib (0..7):
    aux8  = bytes of (qs[0] | qs[1] << 16)   -> 4 indices into the 256-entry
                                                grid of 8 magnitudes each
    aux32 = qs[2] | qs[3] << 16              -> bits 7l..7l+6: sign index of
                                                group l; bits 28..31: scale
    weight[l*8 + j] = grid[aux8[l]][j] * (ksigns[sign_l] & (1 << j) ? -1 : 1)
    sub-block scale = d * (0.5 + (aux32 >> 28)) * 0.25
The grid (256 x 8 bytes) and ksigns (128 bytes) tables are passed to the
kernels as buffers (ds4_iq2_tables.h on the host), not embedded, so the
same source serves every target. They are typed uint32 (little-endian
byte packing) because Metal has no 8/16-lane byte vectors and TileLang's
copy vectoriser would otherwise emit uchar16 for the shared-memory staging.
"""

from __future__ import annotations

import tilelang.language as T

QK_K = 256
U16_PER_BLOCK = 33
GRID_BYTES = 256 * 8
KSIGNS_BYTES = 128
GRID_WORDS = GRID_BYTES // 4
KSIGNS_WORDS = KSIGNS_BYTES // 4
GRID_FLOATS = 256 * 8   # the grid pre-converted to f32 (ds4_tl_iq2_tables_buffer)


def signed_f32(mag_f32, signs, j):
    """mag (a non-negative f32 grid value) with the sign of bit j of `signs`
    applied through the float sign bit: one shift, one and, one xor."""
    sbit = T.Cast(T.uint32, (signs >> j) & 1) << 31
    return T.reinterpret(T.float32, T.reinterpret(T.uint32, mag_f32) ^ sbit)


def byte_of(words, byte_index):
    """Byte `byte_index` of a little-endian uint32 buffer, as int32."""
    return T.Cast(T.int32, (words[byte_index // 4] >> (8 * (byte_index % 4))) & 0xFF)


def sub_block_words(w_at, blk, ib):
    """The four uint16 of sub-block ib of block blk: (q0, q1, q2, q3)."""
    base = blk * U16_PER_BLOCK + 1 + ib * 4
    return w_at(base), w_at(base + 1), w_at(base + 2), w_at(base + 3)


def grid_index(q0, q1, l):
    """aux8[l] for l in 0..3 (l may be a loop variable)."""
    word = T.if_then_else(l < 2, q0, q1)
    return T.Cast(T.int32, (word >> (8 * (l & 1))) & 0xFF)


def group_scale(d_u16, q2, q3):
    """d * (0.5 + (aux32 >> 28)); the trailing * 0.25 is applied once per row."""
    aux32 = T.Cast(T.uint32, q2) | (T.Cast(T.uint32, q3) << 16)
    return T.Cast(T.float32, T.reinterpret(T.float16, d_u16)) * (0.5 + T.Cast(T.float32, aux32 >> 28))


def sign_index(q2, q3, l):
    aux32 = T.Cast(T.uint32, q2) | (T.Cast(T.uint32, q3) << 16)
    return T.Cast(T.int32, (aux32 >> (7 * l)) & 127)


def signed_value(grid, ksigns, gidx, sidx, j):
    """grid[gidx][j] with the sign of bit j of ksigns[sidx], as float."""
    mag = T.Cast(T.float32, grid[gidx * 8 + j])
    neg = (T.Cast(T.int32, ksigns[sidx]) >> j) & 1
    return mag * (1.0 - 2.0 * T.Cast(T.float32, neg))
