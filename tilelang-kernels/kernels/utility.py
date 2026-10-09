"""Utility kernels of the DeepSeek V4 graph: copies, gathers/scatters,
broadcasts, row reductions and the small MoE glue kernels.

References in ds4: metal/cpy.metal, metal/set_rows.metal, metal/get_rows.metal,
metal/repeat.metal, metal/sum_rows.metal, metal/bin.metal, metal/moe.metal
(kernel_dsv4_moe_sum6_f32, kernel_dsv4_moe_swiglu_weight[_f16]),
metal/dsv4_misc.metal (kernel_dsv4_softmax_pool).

ds4's versions are ggml-style kernels with full byte-stride tables (nb0..nb3)
and function-constant specialisations. Every Flash dispatch of them is
row-major with at most a row/column broadcast or a strided gather, so the
TileLang versions take element strides as scalars and the host declines
layouts they do not cover (ds4_tl_metal.h). All lengths are dynamic.

Memory-bound; each kernel is a T.Parallel sweep with one threadgroup per
row (or per chunk of a flat array). Copies move 4 elements per thread like
ds4's *_4 kernels.

No `from __future__ import annotations` here: dtype parameters of the
builder functions appear only in the prim_func signature, and the eager
builder resolves string annotations from the closure, which does not
capture names used solely in annotations.
"""

import tilelang.language as T

from tlk import KernelSpec

# Binary op codes shared with ds4_tl_metal.h (DS4_TL_BIN_*): matches FC_bin_op.
BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV = range(4)


# ---------------------------------------------------------------------------
# Flat contiguous copies with conversion (kernel_cpy_contig_*_4).
# ---------------------------------------------------------------------------

def _cpy_flat(src_dtype, dst_dtype, threads: int = 256):
    """dst[i] = convert(src[i]), 4 elements per thread, scalar tail.
    Grid: ceildiv(ceildiv(n, 4), threads)."""
    n = T.dynamic("n")

    @T.prim_func
    def kernel(dst: T.Tensor((n,), dst_dtype), src: T.Tensor((n,), src_dtype)):
        with T.Kernel(T.ceildiv(T.ceildiv(n, 4), threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                base = (bx * threads + i) * 4
                if base + 4 <= n:
                    for j in T.vectorized(4):
                        dst[base + j] = T.Cast(dst_dtype, src[base + j])
                elif base < n:
                    for j in T.serial(4):
                        if base + j < n:
                            dst[base + j] = T.Cast(dst_dtype, src[base + j])

    return kernel


def cpy_f32_f16(threads: int = 256):
    return _cpy_flat(T.float32, T.float16, threads)


def cpy_f16_f32(threads: int = 256):
    return _cpy_flat(T.float16, T.float32, threads)


def cpy_u16(threads: int = 256):
    """Bitwise f16 -> f16 copy (NaN payloads preserved): kernel_cpy_contig_f16_f16_bits_4."""
    return _cpy_flat(T.uint16, T.uint16, threads)


# ---------------------------------------------------------------------------
# Strided 3-D f32 copy (kernel_cpy_f32_f32): covers the transposes and the
# compressor's 8-row packs. Strides in elements; dst inner dim contiguous.
# ---------------------------------------------------------------------------

def cpy_f32_3d(threads: int = 256):
    """dst[p*d2 + r*d1 + c] = src[p*s2 + r*s1 + c*s0] for p < planes, r < rows,
    c < cols; 4 per thread, vectorised when s0 == 1 (host guarantees the
    strides keep float4 alignment). Grid: (ceildiv(ceildiv(cols, 4), threads), rows, planes)."""
    src_len = T.dynamic("src_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        src: T.Tensor((src_len,), T.float32),
        cols: T.int32,
        rows: T.int32,
        planes: T.int32,
        s0: T.int32,
        s1: T.int32,
        s2: T.int32,
        d1: T.int32,
        d2: T.int32,
        vec: T.int32,   # 1 when s0 == 1 and s1, s2, d1, d2 are multiples of 4
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(cols, 4), threads), rows, planes, threads=threads) as (bx, r, p):
            for i in T.Parallel(threads):
                c = (bx * threads + i) * 4
                if c + 4 <= cols and vec == 1:
                    for j in T.vectorized(4):
                        dst[p * ((d2 // 4) * 4) + r * ((d1 // 4) * 4) + c + j] = src[p * ((s2 // 4) * 4) + r * ((s1 // 4) * 4) + c + j]
                elif c < cols:
                    for j in T.serial(4):
                        if c + j < cols:
                            dst[p * d2 + r * d1 + c + j] = src[p * s2 + r * s1 + (c + j) * s0]

    return kernel


# ---------------------------------------------------------------------------
# Compressor APE tile rows (kernel_cpy_tile_rows_{f16,f32}_f32):
#   dst[t*width + e] = src[((pos0 + t) % ratio) * width + e]
# ---------------------------------------------------------------------------

def _cpy_tile_rows(src_dtype, threads: int = 256):
    n_total = T.dynamic("n_total")
    src_len = T.dynamic("src_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((n_total,), T.float32),
        src: T.Tensor((src_len,), src_dtype),
        width: T.int32,
        ratio: T.int32,
        pos0: T.int32,
    ):
        with T.Kernel(T.ceildiv(n_total, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n_total:
                    t = idx // width
                    e = idx % width
                    r = (pos0 + t) % ratio
                    dst[idx] = T.Cast(T.float32, src[r * width + e])

    return kernel


def cpy_tile_rows_f16_f32(threads: int = 256):
    return _cpy_tile_rows(T.float16, threads)


def cpy_tile_rows_f32_f32(threads: int = 256):
    return _cpy_tile_rows(T.float32, threads)


# ---------------------------------------------------------------------------
# Row scatter / gather by int32 index (kernel_set_rows_f32_i32,
# kernel_get_rows_{f16,f32,i32}).
# ---------------------------------------------------------------------------

def set_rows_f32(threads: int = 256):
    """dst[idx[r], :width] = src[r, :width], 4 per thread (strides must be
    multiples of 4 for the vector path to stay aligned; the host checks).
    Grid: (ceildiv(ceildiv(width, 4), threads), rows)."""
    rows = T.dynamic("rows")
    dst_len = T.dynamic("dst_len")
    src_len = T.dynamic("src_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        src: T.Tensor((src_len,), T.float32),
        idx: T.Tensor((rows,), T.int32),
        width: T.int32,
        src_stride4: T.int32,   # row strides in units of 4 elements, so the
        dst_stride4: T.int32,   # vector path is provably float4-aligned
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(width, 4), threads), rows, threads=threads) as (bx, r):
            row_dst = idx[r]
            for i in T.Parallel(threads):
                c = (bx * threads + i) * 4
                if c + 4 <= width:
                    for j in T.vectorized(4):
                        dst[row_dst * (dst_stride4 * 4) + c + j] = src[r * (src_stride4 * 4) + c + j]
                elif c < width:
                    for j in T.serial(4):
                        if c + j < width:
                            dst[row_dst * (dst_stride4 * 4) + c + j] = src[r * (src_stride4 * 4) + c + j]

    return kernel


def _get_rows(src_dtype, dst_dtype, threads: int):
    """dst[r, :width] = convert(table[idx[r], :width]), 4 per thread.
    Grid: (ceildiv(ceildiv(width, 4), threads), rows)."""
    rows = T.dynamic("rows")
    table_len = T.dynamic("table_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), dst_dtype),
        table: T.Tensor((table_len,), src_dtype),
        idx: T.Tensor((rows,), T.int32),
        width: T.int32,
        table_stride4: T.int32,   # strides in units of 4 elements (alignment proof)
        dst_stride4: T.int32,
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(width, 4), threads), rows, threads=threads) as (bx, r):
            row_src = idx[r]
            for i in T.Parallel(threads):
                c = (bx * threads + i) * 4
                if c + 4 <= width:
                    for j in T.vectorized(4):
                        dst[r * (dst_stride4 * 4) + c + j] = T.Cast(dst_dtype, table[row_src * (table_stride4 * 4) + c + j])
                elif c < width:
                    for j in T.serial(4):
                        if c + j < width:
                            dst[r * (dst_stride4 * 4) + c + j] = T.Cast(dst_dtype, table[row_src * (table_stride4 * 4) + c + j])

    return kernel


def get_rows_f16_f32(threads: int = 256):
    return _get_rows(T.float16, T.float32, threads)


def get_rows_f32(threads: int = 256):
    return _get_rows(T.float32, T.float32, threads)


def get_rows_i32(threads: int = 32):
    """Scalar variant (strides in elements): the hash-router rows are 6 ints
    wide, which the float4 path cannot take. Grid: (ceildiv(width, threads), rows)."""
    rows = T.dynamic("rows")
    table_len = T.dynamic("table_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.int32),
        table: T.Tensor((table_len,), T.int32),
        idx: T.Tensor((rows,), T.int32),
        width: T.int32,
        table_stride: T.int32,
        dst_stride: T.int32,
    ):
        with T.Kernel(T.ceildiv(width, threads), rows, threads=threads) as (bx, r):
            row_src = idx[r]
            for i in T.Parallel(threads):
                c = bx * threads + i
                if c < width:
                    dst[r * dst_stride + c] = table[row_src * table_stride + c]

    return kernel


def gather_scalars_f32(threads: int = 32):
    """Router weight gather (kernel_get_rows_f32 with 1-element rows):
    dst[t, k] = table[t * table_stride + idx[t, k]], k < width (n_expert_used).
    Grid: (rows,)."""
    table_len = T.dynamic("table_len")
    dst_len = T.dynamic("dst_len")
    idx_len = T.dynamic("idx_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        table: T.Tensor((table_len,), T.float32),
        idx: T.Tensor((idx_len,), T.int32),
        rows: T.int32,
        width: T.int32,
        table_stride: T.int32,
        idx_stride: T.int32,
        dst_stride: T.int32,
    ):
        with T.Kernel(rows, threads=threads) as t:
            for k in T.Parallel(threads):
                if k < width:
                    dst[t * dst_stride + k] = table[t * table_stride + idx[t * idx_stride + k]]

    return kernel


# ---------------------------------------------------------------------------
# Broadcast a row over the hyper-connection axis (kernel_repeat_f32):
#   out[t, h, :] = x[t, :]
# ---------------------------------------------------------------------------

def repeat_rows_f32(threads: int = 256):
    """4 per thread. Grid: (ceildiv(ceildiv(width, 4), threads), reps, rows)."""
    src_len = T.dynamic("src_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        src: T.Tensor((src_len,), T.float32),
        rows: T.int32,
        width: T.int32,
        reps: T.int32,
        src_stride4: T.int32,       # strides in units of 4 elements
        dst_rep_stride4: T.int32,
        dst_row_stride4: T.int32,
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(width, 4), threads), reps, rows, threads=threads) as (bx, h, t):
            for i in T.Parallel(threads):
                c = (bx * threads + i) * 4
                if c + 4 <= width:
                    for j in T.vectorized(4):
                        dst[t * (dst_row_stride4 * 4) + h * (dst_rep_stride4 * 4) + c + j] = src[t * (src_stride4 * 4) + c + j]
                elif c < width:
                    for j in T.serial(4):
                        if c + j < width:
                            dst[t * (dst_row_stride4 * 4) + h * (dst_rep_stride4 * 4) + c + j] = src[t * (src_stride4 * 4) + c + j]

    return kernel


# ---------------------------------------------------------------------------
# Row sums (kernel_sum_rows_f32_f32): dst[r] = sum(src[r, :width]).
# Widths in a Flash run are 6, 8 and 128: one thread per row, serial sum.
# ---------------------------------------------------------------------------

def sum_rows_f32(threads: int = 32):
    """Grid: (ceildiv(rows, threads),)."""
    rows = T.dynamic("rows")
    src_len = T.dynamic("src_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((rows,), T.float32),
        src: T.Tensor((src_len,), T.float32),
        width: T.int32,
        src_stride: T.int32,
    ):
        with T.Kernel(T.ceildiv(rows, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                r = bx * threads + i
                if r < rows:
                    acc = T.alloc_var(T.float32)
                    acc = 0.0
                    for c in T.serial(width):
                        acc += src[r * src_stride + c]
                    dst[r] = acc

    return kernel


# ---------------------------------------------------------------------------
# Binary op with broadcast rhs (kernel_bin_fuse_f32_f32_f32):
#   dst[r, i] = a[r, i] OP b[(r % b_rows) * b_stride + (i % b_cols)]
# b_rows = 1: one row broadcast over rows; b_cols = 1: one scalar per row.
# ---------------------------------------------------------------------------

def bin_rows_f32(threads: int = 256):
    """Grid: (ceildiv(width, threads), rows)."""
    a_len = T.dynamic("a_len")
    b_len = T.dynamic("b_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        a: T.Tensor((a_len,), T.float32),
        b: T.Tensor((b_len,), T.float32),
        rows: T.int32,
        width: T.int32,
        a_stride: T.int32,
        dst_stride: T.int32,
        b_rows: T.int32,
        b_cols: T.int32,
        b_stride: T.int32,
        op: T.int32,
    ):
        with T.Kernel(T.ceildiv(width, threads), rows, threads=threads) as (bx, r):
            rb = (r % b_rows) * b_stride
            for i in T.Parallel(threads):
                c = bx * threads + i
                if c < width:
                    x = a[r * a_stride + c]
                    y = b[rb + c % b_cols]
                    dst[r * dst_stride + c] = T.if_then_else(op == BIN_ADD, x + y,
                        T.if_then_else(op == BIN_SUB, x - y,
                        T.if_then_else(op == BIN_MUL, x * y, x / y)))

    return kernel


# ---------------------------------------------------------------------------
# MoE glue: sum of the 6 routed experts' outputs per token
# (kernel_dsv4_moe_sum6_f32), and the routed SwiGLU with the route weight
# applied (kernel_dsv4_moe_swiglu_weight[_f16]).
# ---------------------------------------------------------------------------

def moe_sum6_f32(width: int, threads: int = 256):
    """dst[t, :] = ((((s[t,0,:] + s[t,1,:]) + s[t,2,:]) + s[t,3,:]) + s[t,4,:]) + s[t,5,:]
    (ds4's summation order). Grid: (width/threads, tokens)."""
    assert width % threads == 0
    tokens = T.dynamic("tokens")

    @T.prim_func
    def kernel(
        dst: T.Tensor((tokens, width), T.float32),
        src: T.Tensor((tokens, 6, width), T.float32),
    ):
        with T.Kernel(width // threads, tokens, threads=threads) as (bx, t):
            for i in T.Parallel(threads):
                c = bx * threads + i
                dst[t, c] = ((((src[t, 0, c] + src[t, 1, c]) + src[t, 2, c]) + src[t, 3, c]) + src[t, 4, c]) + src[t, 5, c]

    return kernel


def _moe_swiglu_weight(width: int, mid_dtype, threads: int):
    """mid[r, :] = silu(min(g, c)) * clamp(u, -c, c) * w[r] (c > 1e-6, else
    unclamped), for pair rows r = (token, expert). Grid: (width/threads, rows)."""
    assert width % threads == 0
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        mid: T.Tensor((rows, width), mid_dtype),
        gate: T.Tensor((rows, width), T.float32),
        up: T.Tensor((rows, width), T.float32),
        w: T.Tensor((rows,), T.float32),
        limit: T.float32,
    ):
        with T.Kernel(width // threads, rows, threads=threads) as (bx, r):
            wr = w[r]
            for i in T.Parallel(threads):
                c = bx * threads + i
                g = gate[r, c]
                u = up[r, c]
                gc = T.if_then_else(limit > 1.0e-6, T.min(g, limit), g)
                uc = T.if_then_else(limit > 1.0e-6, T.max(T.min(u, limit), -limit), u)
                mid[r, c] = T.Cast(mid_dtype, gc / (1.0 + T.exp(-gc)) * uc * wr)

    return kernel


def moe_swiglu_weight_f32(width: int, threads: int = 256):
    return _moe_swiglu_weight(width, T.float32, threads)


def moe_swiglu_weight_f16(width: int, threads: int = 256):
    return _moe_swiglu_weight(width, T.float16, threads)


# ---------------------------------------------------------------------------
# Compressor softmax pooling (kernel_dsv4_softmax_pool), n_comp > 1 prefill:
# per (comp, dim) a softmax over `pooled` rows weights the kv rows.
# Strides in elements: (row, dim, comp) for kv and score; dst[comp*d1 + dim].
# ---------------------------------------------------------------------------

def softmax_pool_f32(threads: int = 256):
    """Grid: (ceildiv(head_dim, threads), n_comp)."""
    kv_len = T.dynamic("kv_len")
    score_len = T.dynamic("score_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        kv: T.Tensor((kv_len,), T.float32),
        score: T.Tensor((score_len,), T.float32),
        n_comp: T.int32,
        head_dim: T.int32,
        pooled: T.int32,
        kv_s0: T.int32,
        kv_s1: T.int32,
        kv_s2: T.int32,
        sc_s0: T.int32,
        sc_s1: T.int32,
        sc_s2: T.int32,
        d1: T.int32,
    ):
        with T.Kernel(T.ceildiv(head_dim, threads), n_comp, threads=threads) as (bx, ic):
            for i in T.Parallel(threads):
                d = bx * threads + i
                if d < head_dim:
                    m = T.alloc_var(T.float32)
                    s = T.alloc_var(T.float32)
                    acc = T.alloc_var(T.float32)
                    m = -3.0e38
                    for r in T.serial(pooled):
                        m = T.max(m, score[r * sc_s0 + d * sc_s1 + ic * sc_s2])
                    s = 0.0
                    acc = 0.0
                    for r in T.serial(pooled):
                        e = T.exp(score[r * sc_s0 + d * sc_s1 + ic * sc_s2] - m)
                        s += e
                        acc += kv[r * kv_s0 + d * kv_s1 + ic * kv_s2] * e
                    dst[ic * d1 + d] = acc / s

    return kernel


KERNELS = [
    KernelSpec(name="tl_cpy_f32_f16", build=cpy_f32_f16, group="utility", targets=("metal",),
               notes="dst[i] = half(src[i]); 4 per thread; grid=(ceil(ceil(n/4)/256))"),
    KernelSpec(name="tl_cpy_f16_f32", build=cpy_f16_f32, group="utility", targets=("metal",),
               notes="dst[i] = float(src[i]); 4 per thread; grid=(ceil(ceil(n/4)/256))"),
    KernelSpec(name="tl_cpy_u16", build=cpy_u16, group="utility", targets=("metal",),
               notes="dst[i] = src[i] bitwise 16-bit; grid=(ceil(ceil(n/4)/256))"),
    KernelSpec(name="tl_cpy_f32_3d", build=cpy_f32_3d, group="utility", targets=("metal",),
               notes="dst[p*d2+r*d1+c] = src[p*s2+r*s1+c*s0]; 4 per thread; grid=(ceil(ceil(cols/4)/256), rows, planes)"),
    KernelSpec(name="tl_cpy_tile_rows_f16_f32", build=cpy_tile_rows_f16_f32, group="utility", targets=("metal",),
               notes="dst[t*w+e] = float(src[((pos0+t)%ratio)*w+e]); grid=(ceil(n_total/256))"),
    KernelSpec(name="tl_cpy_tile_rows_f32_f32", build=cpy_tile_rows_f32_f32, group="utility", targets=("metal",),
               notes="dst[t*w+e] = src[((pos0+t)%ratio)*w+e]; grid=(ceil(n_total/256))"),
    KernelSpec(name="tl_set_rows_f32", build=set_rows_f32, group="utility", targets=("metal",),
               notes="dst[idx[r],:] = src[r,:]; 4 per thread; grid=(ceil(ceil(width/4)/256), rows)"),
    KernelSpec(name="tl_get_rows_f16_f32", build=get_rows_f16_f32, group="utility", targets=("metal",),
               notes="dst[r,:] = float(table[idx[r],:]); 4 per thread; grid=(ceil(ceil(width/4)/256), rows)"),
    KernelSpec(name="tl_get_rows_f32", build=get_rows_f32, group="utility", targets=("metal",),
               notes="dst[r,:] = table[idx[r],:]; 4 per thread; grid=(ceil(ceil(width/4)/256), rows)"),
    KernelSpec(name="tl_get_rows_i32", build=get_rows_i32, group="utility", targets=("metal",),
               notes="dst[r,:] = table[idx[r],:] (int32, scalar, element strides); grid=(ceil(width/32), rows)"),
    KernelSpec(name="tl_gather_scalars_f32", build=gather_scalars_f32, group="utility", targets=("metal",),
               notes="dst[t,k] = table[t*ts + idx[t,k]]; grid=(rows)"),
    KernelSpec(name="tl_repeat_rows_f32", build=repeat_rows_f32, group="utility", targets=("metal",),
               notes="dst[t,h,:] = src[t,:]; 4 per thread; grid=(ceil(ceil(width/4)/256), reps, rows)"),
    KernelSpec(name="tl_sum_rows_f32", build=sum_rows_f32, group="utility", targets=("metal",),
               notes="dst[r] = sum(src[r,:width]); one thread per row; grid=(ceil(rows/32))"),
    KernelSpec(name="tl_bin_rows_f32", build=bin_rows_f32, group="utility", targets=("metal",),
               notes="dst[r,i] = a[r,i] op b[(r%b_rows)*b_stride + i%b_cols]; op add/sub/mul/div; grid=(ceil(width/256), rows)"),
    KernelSpec(name="tl_moe_sum6_f32_4096", build=lambda: moe_sum6_f32(4096), group="utility", targets=("metal",),
               notes="dst[t,:] = sum of 6 expert rows in ds4 order; grid=(16, tokens)"),
    KernelSpec(name="tl_moe_swiglu_weight_f32_2048", build=lambda: moe_swiglu_weight_f32(2048), group="utility", targets=("metal",),
               notes="mid[r,:] = silu(min(g,c))*clamp(u,-c,c)*w[r]; grid=(8, rows)"),
    KernelSpec(name="tl_moe_swiglu_weight_f16_2048", build=lambda: moe_swiglu_weight_f16(2048), group="utility", targets=("metal",),
               notes="half mid variant; grid=(8, rows)"),
    KernelSpec(name="tl_softmax_pool_f32", build=softmax_pool_f32, group="utility", targets=("metal",),
               notes="dst[ic*d1+d] = sum_r kv[r,d,ic]*softmax_r(score[r,d,ic]); grid=(ceil(head_dim/256), n_comp)"),
]
