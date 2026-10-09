"""DeepSeek V4 compressor / indexer small kernels.

References in ds4:
  Metal kernel_dsv4_compressor_exact_pool_ratio4_decode_ggml  metal/dsv4_kv.metal:439
        kernel_dsv4_comp_row_finalize_f32                     metal/dsv4_rope.metal:647
        kernel_dsv4_indexer_hadamard_fp4_f32                  metal/dsv4_kv.metal:163
        kernel_dsv4_sort_i32_rows_asc                         metal/dsv4_misc.metal:5481
  CPU   compressor_pool_decode_state, dsv4_hadamard128_inplace_cpu,
        dsv4_fp4_act_quantize_row_inplace_cpu                 ds4.c:14599, 3764

The ratio-4 compressor keeps an 8-row state per layer ([8][2*head_dim]:
rows 0..3 the previous token group, 4..7 the current one; plane 0 / plane 1
halves of each row). Every 4th token it pools the 8 rows per column with
a softmax over the scores ("exact ... ggml" = bit-identical to ds4's
reference softmax/mul/sum_rows chain), then finalises the pooled rows:
RMS norm, RoPE tail, FP8 (attention row, 512) or Hadamard + FP4 (indexer
row, 128), and shifts rows 4..7 down to 0..3.

Rows are staged in threadgroup memory between the steps: TileLang orders
shared-memory dependencies between threads, not global-memory ones.

No `from __future__ import annotations` (see utility.py).
"""

import tilelang.language as T

from tlk import KernelSpec
from kernels.dsv4 import _yarn_corr_dims, _yarn_cos_sin, _e4m3fn_round

HADAMARD_NORM_128 = 0.08838834764831845  # 1 / sqrt(128)


# ---------------------------------------------------------------------------
# Ratio-4 decode pool: dst[col] = sum_j softmax_j(score[j, col]) * kv[j, col]
# ---------------------------------------------------------------------------

def compressor_pool_ratio4_f32(head_dim: int, threads: int = 128):
    """One thread per column. idx(j) = j*2*head_dim + (j >= 4 ? head_dim : 0) + col.
    Summation orders follow the hand-written kernel: S = (((e0+e1)+e2)+e3) +
    (((e4+e5)+e6)+e7), p = e * (1/S), and the final 8-term sum in the order
    of an 8-lane simd_sum, which on Apple GPUs is the xor butterfly with
    offsets 1, 2, 4: ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)) (measured on the
    M5 Max). The softmax weights, the products and each
    level of the final tree go through threadgroup memory with barriers in
    between, like ds4's device store/reload boundaries: ds4's shader
    library is compiled with fast-math, under which the compiler
    reassociates register-only sums. Grid: (ceildiv(head_dim, threads),)."""
    assert head_dim % threads == 0
    state_len = 8 * 2 * head_dim

    @T.prim_func
    def kernel(
        dst: T.Tensor((head_dim,), T.float32),
        state_kv: T.Tensor((state_len,), T.float32),
        state_score: T.Tensor((state_len,), T.float32),
    ):
        with T.Kernel(head_dim // threads, threads=threads) as bx:
            p_s = T.alloc_shared((threads, 8), T.float32)
            prod_s = T.alloc_shared((threads, 8), T.float32)
            tid = T.get_thread_binding()
            col = bx * threads + tid
            s = T.alloc_local((8,), T.float32)
            for j in T.unroll(8):
                s[j] = state_score[j * 2 * head_dim + (j // 4) * head_dim + col]
            m = T.max(T.max(T.max(s[0], s[1]), T.max(s[2], s[3])), T.max(T.max(s[4], s[5]), T.max(s[6], s[7])))
            for j in T.unroll(8):
                s[j] = T.exp(s[j] - m)
            # The two lane sums of ds4's softmax (A = lane 0, B = lane 1) are
            # combined by a simd_sum; a register-only A + B expression would
            # be reassociated with the chains by the fast-math compiler.
            prod_s[tid, 0] = ((s[0] + s[1]) + s[2]) + s[3]
            prod_s[tid, 1] = ((s[4] + s[5]) + s[6]) + s[7]
            T.sync_threads()
            inv = 1.0 / (prod_s[tid, 0] + prod_s[tid, 1])
            for j in T.unroll(8):
                p_s[tid, j] = s[j] * inv   # registers: ds4's `softmax4 *= inv_sum` compiles the same way
            T.sync_threads()
            for j in T.unroll(8):
                prod_s[tid, j] = state_kv[j * 2 * head_dim + (j // 4) * head_dim + col] * p_s[tid, j]
            T.sync_threads()
            # Each butterfly level goes through threadgroup memory: with
            # fast-math on, the Metal compiler reassociates a register-only
            # 8-term sum, and the simd_sum it must match cannot be reassociated.
            for j in T.unroll(4):
                p_s[tid, j] = prod_s[tid, 2 * j] + prod_s[tid, 2 * j + 1]
            T.sync_threads()
            for j in T.unroll(2):
                prod_s[tid, j] = p_s[tid, 2 * j] + p_s[tid, 2 * j + 1]
            T.sync_threads()
            dst[col] = prod_s[tid, 0] + prod_s[tid, 1]

    return kernel


# ---------------------------------------------------------------------------
# Hadamard (Sylvester order, 1/sqrt(128)) + E2M1 fake quantisation per 32.
# ---------------------------------------------------------------------------

def _e2m1_round(y):
    """Nearest E2M1 value of y (|y| <= 6), ties to even code: for |y| >= 1
    round-to-nearest-even on the single mantissa bit through the float32
    bit pattern; below 1 the grid is uniform 0.5, so rint(2y)/2."""
    bits = T.reinterpret(T.uint32, y)
    rounded = (bits + T.uint32(0x1FFFFF) + ((bits >> 22) & T.uint32(1))) & T.uint32(0xFFC00000)
    normal = T.reinterpret(T.float32, rounded)
    sub = T.round(y * 2.0) * 0.5
    return T.if_then_else(T.abs(y) < 1.0, sub, normal)


@T.macro
def _hadamard_fp4_128(vals, absbuf, amax_s, tid, n_threads, out, out_base):
    """vals[128] holds the row (shared). In place: Hadamard butterflies with
    ascending stride (ds4's order), scale by 1/sqrt(128), per-32 amax ->
    power-of-two scale -> E2M1 round trip; result written to out[out_base + tid].
    Threads >= 128 (when n_threads > 128) only take part in the barriers."""
    for st in T.serial(7):
        stride = 1 << st
        if tid < 128 and (tid & stride) == 0:
            base = (tid & ~(2 * stride - 1)) + (tid & (stride - 1))
            a = vals[base]
            b = vals[base + stride]
            vals[base] = a + b
            vals[base + stride] = a - b
        T.sync_threads()
    if tid < 128:
        absbuf[tid] = T.abs(vals[tid] * HADAMARD_NORM_128)
    T.sync_threads()
    if tid < 4:
        mx = T.alloc_var(T.float32)
        mx = absbuf[tid * 32]
        for k in T.serial(31):
            mx = T.max(mx, absbuf[tid * 32 + 1 + k])
        amax_s[tid] = mx
    T.sync_threads()
    if tid < 128:
        v = vals[tid] * HADAMARD_NORM_128
        amax = T.max(amax_s[tid >> 5], 7.052966104933725e-38)
        scale = T.exp2(T.ceil(T.log2(amax / 6.0)))
        out[out_base + tid] = _e2m1_round(T.max(T.min(v / scale, 6.0), -6.0)) * scale


def indexer_hadamard_fp4_f32(head_dim: int = 128):
    """kernel_dsv4_indexer_hadamard_fp4_f32, in place on [rows][128] f32
    rows with an element stride. Grid: (rows,), 128 threads."""
    assert head_dim == 128
    rows = T.dynamic("rows")
    x_len = T.dynamic("x_len")

    @T.prim_func
    def kernel(x: T.Tensor((x_len,), T.float32), rows: T.int32, row_stride: T.int32):
        with T.Kernel(rows, threads=128) as row:
            vals = T.alloc_shared((128,), T.float32)
            absbuf = T.alloc_shared((128,), T.float32)
            amax_s = T.alloc_shared((4,), T.float32)
            tid = T.get_thread_binding()
            vals[tid] = x[row * row_stride + tid]
            T.sync_threads()
            _hadamard_fp4_128(vals, absbuf, amax_s, tid, 128, x, row * row_stride)

    return kernel


# ---------------------------------------------------------------------------
# Ascending bitonic sort of int32 rows (unique ids), kernel_dsv4_sort_i32_rows_asc.
# ---------------------------------------------------------------------------

def sort_i32_rows_asc(width: int = 512):
    """dst[t, :] = sorted(src[t, :]) ascending. One threadgroup of `width`
    threads per row; width a power of two. Grid: (rows,)."""
    assert (width & (width - 1)) == 0
    n_log2 = width.bit_length() - 1
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(dst: T.Tensor((rows, width), T.int32), src: T.Tensor((rows, width), T.int32)):
        with T.Kernel(rows, threads=width) as t:
            buf = T.alloc_shared((width,), T.int32)
            tid = T.get_thread_binding()
            buf[tid] = src[t, tid]
            T.sync_threads()
            for ks in T.serial(n_log2):
                k = 2 << ks
                for js in T.serial(ks + 1):
                    j = k >> (js + 1)
                    other = tid ^ j
                    if other > tid:
                        a = buf[tid]
                        b = buf[other]
                        swap = T.if_then_else((tid & k) == 0, a > b, a < b)
                        if swap:
                            buf[tid] = b
                            buf[other] = a
            dst[t, tid] = buf[tid]

    return kernel


# ---------------------------------------------------------------------------
# Compressor row finalize (decode, ratio 4): three threadgroup roles.
# ---------------------------------------------------------------------------

def comp_row_finalize_f32(attn_dim: int = 512, index_dim: int = 128, n_rot: int = 64, threads: int = 256):
    """kernel_dsv4_comp_row_finalize_f32. Threadgroup 0: attention row
    [512]: RMS norm (scale = 1/sqrt(mean + eps), * w), YaRN RoPE on the last
    64 dims at position pos0, E4M3FN round trip of the first 448 per 64-block,
    f32 row written back and f16 copy into the compressed cache row.
    Threadgroup 1: indexer row [128]: RMS norm, RoPE tail (n_nope 64),
    Hadamard + E2M1 round trip. Threadgroups 2..: shift state rows 4..7 to
    0..3 for both compressors ([8][1024] and [8][256] f32).
    Grid: (2 + ceildiv(4*1024 + 4*256, threads),)."""
    assert attn_dim == 512 and index_dim == 128 and n_rot == 64 and threads == 256
    attn_state_half = 4 * 2 * attn_dim    # 4 rows x 1024
    index_state_half = 4 * 2 * index_dim  # 4 rows x 256
    shift_blocks = (attn_state_half + index_state_half + threads - 1) // threads
    attn_nope = attn_dim - n_rot
    pairs = n_rot // 2

    @T.prim_func
    def kernel(
        attn_row: T.Tensor((attn_dim,), T.float32),
        attn_norm_w: T.Tensor((attn_dim,), T.float32),
        attn_cache: T.Tensor((attn_dim,), T.float16),
        index_row: T.Tensor((index_dim,), T.float32),
        index_norm_w: T.Tensor((index_dim,), T.float32),
        attn_state_kv: T.Tensor((2 * attn_state_half,), T.float32),
        attn_state_score: T.Tensor((2 * attn_state_half,), T.float32),
        index_state_kv: T.Tensor((2 * index_state_half,), T.float32),
        index_state_score: T.Tensor((2 * index_state_half,), T.float32),
        rms_eps: T.float32,
        pos0: T.int32,
        n_ctx_orig: T.int32,
        freq_base: T.float32,
        freq_scale: T.float32,
        ext_factor: T.float32,
        attn_factor: T.float32,
        beta_fast: T.float32,
        beta_slow: T.float32,
    ):
        with T.Kernel(2 + shift_blocks, threads=threads) as bx:
            tid = T.get_thread_binding()
            if bx == 0:
                row = T.alloc_shared((attn_dim,), T.float32)
                absbuf = T.alloc_shared((attn_dim,), T.float32)
                amax_s = T.alloc_shared((8,), T.float32)
                sq = T.alloc_fragment((1, attn_dim), T.float32)
                ss = T.alloc_fragment((1,), T.float32)
                T.copy(attn_row[0:attn_dim], sq[0, :])
                for i, j in T.Parallel(1, attn_dim):
                    sq[i, j] = sq[i, j] * sq[i, j]
                T.reduce_sum(sq, ss, dim=1)
                scale = 1.0 / T.sqrt(ss[0] / attn_dim + rms_eps)
                for j in T.serial(2):
                    e = tid + j * threads
                    row[e] = (attn_row[e] * scale) * attn_norm_w[e]
                T.sync_threads()
                if tid < pairs:
                    low, high = _yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
                    r = 2 * tid
                    theta = T.Cast(T.float32, pos0) * T.pow(freq_base, T.Cast(T.float32, -r) / n_rot)
                    c, s = _yarn_cos_sin(theta, r, freq_scale, ext_factor, attn_factor, low, high, 0)
                    x0 = row[attn_nope + r]
                    x1 = row[attn_nope + r + 1]
                    row[attn_nope + r] = x0 * c - x1 * s
                    row[attn_nope + r + 1] = x0 * s + x1 * c
                T.sync_threads()
                for j in T.serial(2):
                    e = tid + j * threads
                    absbuf[e] = T.abs(row[e])
                T.sync_threads()
                if tid < attn_nope // 64:
                    mx = T.alloc_var(T.float32)
                    mx = absbuf[tid * 64]
                    for k in T.serial(63):
                        mx = T.max(mx, absbuf[tid * 64 + 1 + k])
                    amax_s[tid] = mx
                T.sync_threads()
                for j in T.serial(2):
                    e = tid + j * threads
                    if e < attn_nope:
                        amax = T.max(amax_s[e >> 6], 1.0e-4)
                        qscale = T.exp2(T.ceil(T.log2(amax / 448.0)))
                        y = T.max(T.min(row[e] / qscale, 448.0), -448.0)
                        row[e] = _e4m3fn_round(y) * qscale
                T.sync_threads()
                for j in T.serial(2):
                    e = tid + j * threads
                    attn_row[e] = row[e]
                    attn_cache[e] = T.Cast(T.float16, row[e])
            elif bx == 1:
                vals = T.alloc_shared((index_dim,), T.float32)
                iabs = T.alloc_shared((index_dim,), T.float32)
                iamax = T.alloc_shared((4,), T.float32)
                isq = T.alloc_shared((index_dim,), T.float32)
                itot = T.alloc_shared((1,), T.float32)
                if tid < index_dim:
                    isq[tid] = index_row[tid] * index_row[tid]
                T.sync_threads()
                if tid == 0:
                    acc = T.alloc_var(T.float32)
                    acc = 0.0
                    for k in T.serial(index_dim):
                        acc += isq[k]
                    itot[0] = acc
                T.sync_threads()
                iscale = 1.0 / T.sqrt(itot[0] / index_dim + rms_eps)
                if tid < index_dim:
                    vals[tid] = (index_row[tid] * iscale) * index_norm_w[tid]
                T.sync_threads()
                if tid < pairs:
                    low, high = _yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
                    r = 2 * tid
                    theta = T.Cast(T.float32, pos0) * T.pow(freq_base, T.Cast(T.float32, -r) / n_rot)
                    c, s = _yarn_cos_sin(theta, r, freq_scale, ext_factor, attn_factor, low, high, 0)
                    x0 = vals[index_dim - n_rot + r]
                    x1 = vals[index_dim - n_rot + r + 1]
                    vals[index_dim - n_rot + r] = x0 * c - x1 * s
                    vals[index_dim - n_rot + r + 1] = x0 * s + x1 * c
                T.sync_threads()
                _hadamard_fp4_128(vals, iabs, iamax, tid, threads, index_row, 0)
            else:
                gid = (bx - 2) * threads + tid
                if gid < attn_state_half:
                    attn_state_kv[gid] = attn_state_kv[attn_state_half + gid]
                    attn_state_score[gid] = attn_state_score[attn_state_half + gid]
                elif gid - attn_state_half < index_state_half:
                    g2 = gid - attn_state_half
                    index_state_kv[g2] = index_state_kv[index_state_half + g2]
                    index_state_score[g2] = index_state_score[index_state_half + g2]

    return kernel


KERNELS = [
    KernelSpec(name="tl_compressor_pool_ratio4_f32_512", build=lambda: compressor_pool_ratio4_f32(512), group="compressor", targets=("metal",),
               notes="dst[col] = sum_j softmax_j(score) * kv over the 8-row ratio-4 state, head_dim 512; grid=(4)"),
    KernelSpec(name="tl_compressor_pool_ratio4_f32_128", build=lambda: compressor_pool_ratio4_f32(128), group="compressor", targets=("metal",),
               notes="same, head_dim 128 (indexer); grid=(1)"),
    KernelSpec(name="tl_indexer_hadamard_fp4_f32_128", build=indexer_hadamard_fp4_f32, group="compressor", targets=("metal",),
               notes="in place: Hadamard_128 / sqrt(128), per-32 E2M1 round trip; grid=(rows)"),
    KernelSpec(name="tl_sort_i32_rows_asc_512", build=sort_i32_rows_asc, group="compressor", targets=("metal",),
               notes="ascending bitonic sort of int32 rows of 512; grid=(rows)"),
    KernelSpec(name="tl_comp_row_finalize_f32_512_128", build=comp_row_finalize_f32, group="compressor", targets=("metal",),
               notes="attn row norm+rope+fp8+f16 commit | indexer row norm+rope+hadamard+fp4 | state shift; grid=(22)"),
]


# ---------------------------------------------------------------------------
# Decode QKV + compressor projections with the state store
# (kernel_dsv4_qkv_pair_quad_compressor_store_q8_0): one dispatch, roles by
# threadgroup range. Threadgroups [0, q_tiles) : q_a Q8_0 rows; then kv Q8_0
# rows; then the attention compressor (kv + gate f16 matrices, 4 rows each,
# written to the cur outputs and appended to the state); then the indexer
# compressor likewise (ratio 4 only). All against the same x[4096].
# ---------------------------------------------------------------------------

from kernels.q8 import _q8_0_f32_accumulate, _q8_0_f32_tile, F32_ROWS, f32_threads  # noqa: E402


def qkv_pair_quad_compressor_store(in_dim: int = 4096, rows: int = F32_ROWS, ape_f16: bool = True):
    """grid = (q_dim/rows + kv_dim/rows + width0/rows + width1/rows,), threads as
    the Q8_0 f32 GEMV. For the compressor rows the state append is
    state_kv[dst_row][c] = kv_cur[c], state_score[dst_row][c] = sc_cur[c] + ape[pos % ratio][c]
    with dst_row = ratio == 4 ? 4 + pos % 4 : pos % ratio."""
    assert in_dim % 32 == 0
    threads = f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, 4)
    chunks = in_dim // 4
    fk = min(chunks, max(1, threads * 32 // rows))
    while chunks % fk:
        fk -= 1
    fcper = chunks // fk
    q_dim = T.dynamic("q_dim")
    kv_dim = T.dynamic("kv_dim")
    width0 = T.dynamic("width0")
    width1 = T.dynamic("width1")
    ape_dtype = T.float16 if ape_f16 else T.float32
    ape0_len = T.dynamic("ape0_len")
    ape1_len = T.dynamic("ape1_len")
    st0_len = T.dynamic("st0_len")
    st1_len = T.dynamic("st1_len")

    # Passing the dynamic width vars as macro arguments renamed both after the
    # macro parameter (two kernel args called "width"), so the macro selects
    # width0 / width1 by closure from a Python-level `which`.
    @T.macro
    def f16_pair_tile(which, r0, w_a, w_b, out_a, out_b, ape, state_kv, state_sc, dst_row, pos_mod, x):
        # (no local alias of the width var: the eager builder renames an
        #  assigned Var after the Python name)
        acc_a = T.alloc_fragment((rows, fk), T.float32)
        acc_b = T.alloc_fragment((rows, fk), T.float32)
        sum_a = T.alloc_fragment((rows,), T.float32)
        sum_b = T.alloc_fragment((rows,), T.float32)
        T.clear(acc_a)
        T.clear(acc_b)
        for r, ks in T.Parallel(rows, fk):
            wl = T.alloc_local((4,), T.float16)
            wm = T.alloc_local((4,), T.float16)
            xl = T.alloc_local((4,), T.float32)
            for i in T.serial(fcper):
                c = (i * fk + ks) * 4
                for j in T.vectorized(4):
                    wl[j] = w_a[r0 + r, c + j]
                for j in T.vectorized(4):
                    wm[j] = w_b[r0 + r, c + j]
                for j in T.vectorized(4):
                    xl[j] = x[c + j]
                acc_a[r, ks] += (T.Cast(T.float32, wl[0]) * xl[0] + T.Cast(T.float32, wl[1]) * xl[1]
                                 + T.Cast(T.float32, wl[2]) * xl[2] + T.Cast(T.float32, wl[3]) * xl[3])
                acc_b[r, ks] += (T.Cast(T.float32, wm[0]) * xl[0] + T.Cast(T.float32, wm[1]) * xl[1]
                                 + T.Cast(T.float32, wm[2]) * xl[2] + T.Cast(T.float32, wm[3]) * xl[3])
        T.reduce_sum(acc_a, sum_a, dim=1)
        T.reduce_sum(acc_b, sum_b, dim=1)
        for r in T.Parallel(rows):
            col = r0 + r
            out_a[col] = sum_a[r]
            out_b[col] = sum_b[r]
            state_kv[dst_row * (width0 if which == 0 else width1) + col] = sum_a[r]
            state_sc[dst_row * (width0 if which == 0 else width1) + col] = sum_b[r] + T.Cast(
                T.float32, ape[pos_mod * (width0 if which == 0 else width1) + col])

    @T.prim_func
    def kernel(
        qr: T.Tensor((q_dim,), T.float32),
        kv_raw: T.Tensor((kv_dim,), T.float32),
        qw0: T.Tensor((q_dim, blocks * 17), T.uint16),
        qw1: T.Tensor((kv_dim, blocks * 17), T.uint16),
        cw0a: T.Tensor((width0, in_dim), T.float16),
        cw0b: T.Tensor((width0, in_dim), T.float16),
        cw1a: T.Tensor((width1, in_dim), T.float16),
        cw1b: T.Tensor((width1, in_dim), T.float16),
        x: T.Tensor((in_dim,), T.float32),
        cdst_a0: T.Tensor((width0,), T.float32),
        cdst_b0: T.Tensor((width0,), T.float32),
        cdst_a1: T.Tensor((width1,), T.float32),
        cdst_b1: T.Tensor((width1,), T.float32),
        ape0: T.Tensor((ape0_len,), ape_dtype),
        ape1: T.Tensor((ape1_len,), ape_dtype),
        state0_kv: T.Tensor((st0_len,), T.float32),
        state0_sc: T.Tensor((st0_len,), T.float32),
        state1_kv: T.Tensor((st1_len,), T.float32),
        state1_sc: T.Tensor((st1_len,), T.float32),
        ratio: T.int32,
        pos: T.int32,
    ):
        with T.Kernel((q_dim + kv_dim + width0 + width1) // rows, threads=threads) as bx:
            blocks_ = blocks  # noqa: F841  (annotation-only names)
            ape0_len_ = ape0_len  # noqa: F841
            ape1_len_ = ape1_len  # noqa: F841
            st0_len_ = st0_len  # noqa: F841
            st1_len_ = st1_len  # noqa: F841
            q_tiles = q_dim // rows
            kv_tiles = kv_dim // rows
            c0_tiles = width0 // rows
            pos_mod = pos % ratio
            dst_row = T.if_then_else(ratio == 4, 4 + pos_mod, pos_mod)
            if bx < q_tiles:
                acc = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
                rowsum = T.alloc_fragment((rows,), T.float32)
                _q8_0_f32_accumulate(acc, rowsum, lambda r, k: qw0[bx * rows + r, k], lambda i: x[i],
                                     rows, blocks, nchunk, ksplit, bper)
                for r in T.Parallel(rows):
                    qr[bx * rows + r] = rowsum[r]
            elif bx < q_tiles + kv_tiles:
                acc2 = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
                rowsum2 = T.alloc_fragment((rows,), T.float32)
                _q8_0_f32_accumulate(acc2, rowsum2, lambda r, k: qw1[(bx - q_tiles) * rows + r, k], lambda i: x[i],
                                     rows, blocks, nchunk, ksplit, bper)
                for r in T.Parallel(rows):
                    kv_raw[(bx - q_tiles) * rows + r] = rowsum2[r]
            elif bx < q_tiles + kv_tiles + c0_tiles:
                f16_pair_tile(0, (bx - q_tiles - kv_tiles) * rows, cw0a, cw0b, cdst_a0, cdst_b0, ape0, state0_kv, state0_sc, dst_row, pos_mod, x)
            else:
                f16_pair_tile(1, (bx - q_tiles - kv_tiles - c0_tiles) * rows, cw1a, cw1b, cdst_a1, cdst_b1, ape1, state1_kv, state1_sc, dst_row, pos_mod, x)

    return kernel


KERNELS += [
    KernelSpec(name="tl_qkv_pair_quad_compressor_store_f32_4096_ape16", build=lambda: qkv_pair_quad_compressor_store(4096, ape_f16=True),
               variants={"r2": lambda: qkv_pair_quad_compressor_store(4096, rows=2, ape_f16=True),
                         "r8": lambda: qkv_pair_quad_compressor_store(4096, rows=8, ape_f16=True)},
               tune_group="quad",
               group="compressor", targets=("metal",),
               notes="decode q_a + kv (Q8_0) and attention / indexer compressor kv + gate (f16) GEMVs with the state append, f16 APE; grid=((q+kv+w0+w1)/4)"),
    KernelSpec(name="tl_qkv_pair_quad_compressor_store_f32_4096_ape32", build=lambda: qkv_pair_quad_compressor_store(4096, ape_f16=False),
               variants={"r2": lambda: qkv_pair_quad_compressor_store(4096, rows=2, ape_f16=False),
                         "r8": lambda: qkv_pair_quad_compressor_store(4096, rows=8, ape_f16=False)},
               tune_group="quad",
               group="compressor", targets=("metal",),
               notes="same with f32 APE"),
]


def compressor_store_one(ape_f16: bool = True, threads: int = 256):
    """kernel_dsv4_compressor_store_one: append one token's compressor kv and
    gate rows to the state ring: dst_row = ratio == 4 ? 4 + pos % 4 : pos % ratio;
    state_kv[dst_row, g] = kv[g]; state_score[dst_row, g] = score[g] + ape[pos % ratio, g].
    width % 4 == 0 (host checks); 4 per thread. Grid: (ceildiv(width4, threads),)."""
    width4 = T.dynamic("width4")   # width / 4
    ape_len = T.dynamic("ape_len")
    state_len = T.dynamic("state_len")
    ape_dtype = T.float16 if ape_f16 else T.float32

    @T.prim_func
    def kernel(
        ape: T.Tensor((ape_len,), ape_dtype),
        kv: T.Tensor((width4 * 4,), T.float32),
        score: T.Tensor((width4 * 4,), T.float32),
        state_kv: T.Tensor((state_len,), T.float32),
        state_score: T.Tensor((state_len,), T.float32),
        pos: T.int32,
        ratio: T.int32,
    ):
        with T.Kernel(T.ceildiv(width4, threads), threads=threads) as bx:
            pos_mod = pos % ratio
            dst_row = T.if_then_else(ratio == 4, 4 + pos_mod, pos_mod)
            for i in T.Parallel(threads):
                c4 = bx * threads + i
                if c4 < width4:
                    for j in T.vectorized(4):
                        state_kv[(dst_row * width4 + c4) * 4 + j] = kv[c4 * 4 + j]
                    for j in T.vectorized(4):
                        state_score[(dst_row * width4 + c4) * 4 + j] = (
                            score[c4 * 4 + j] + T.Cast(T.float32, ape[(pos_mod * width4 + c4) * 4 + j])
                        )

    return kernel


def compressor_shift_ratio4(threads: int = 256):
    """kernel_dsv4_ratio4_shift_f32: state rows 4..7 -> rows 0..3 (both kv and
    score), n = 4 * width floats each, width % 4 == 0; 4 per thread.
    Grid: (ceildiv(n4, threads),)."""
    state_len = T.dynamic("state_len")

    @T.prim_func
    def kernel(
        state_kv: T.Tensor((state_len,), T.float32),
        state_score: T.Tensor((state_len,), T.float32),
        n4: T.int32,
    ):
        with T.Kernel(T.ceildiv(n4, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                c4 = bx * threads + i
                if c4 < n4:
                    for j in T.vectorized(4):
                        state_kv[c4 * 4 + j] = state_kv[(n4 + c4) * 4 + j]
                    for j in T.vectorized(4):
                        state_score[c4 * 4 + j] = state_score[(n4 + c4) * 4 + j]

    return kernel


KERNELS += [
    KernelSpec(name="tl_compressor_store_one_f32_ape16", build=lambda: compressor_store_one(True), group="compressor", targets=("metal",),
               notes="state ring append of one token's compressor kv + (score + f16 APE) row; grid=(ceil(width/4/256))"),
    KernelSpec(name="tl_compressor_store_one_f32_ape32", build=lambda: compressor_store_one(False), group="compressor", targets=("metal",),
               notes="same with f32 APE"),
    KernelSpec(name="tl_compressor_shift_ratio4_f32", build=compressor_shift_ratio4, group="compressor", targets=("metal",),
               notes="ratio-4 state rows 4..7 -> 0..3 (kv and score); grid=(ceil(width/256))"),
]
