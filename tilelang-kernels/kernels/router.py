"""DeepSeek V4 MoE router kernels.

References in ds4:
  Metal kernel_dsv4_router_project_select_fused            metal/dsv4_misc.metal:5277
        kernel_dsv4_router_transform_finalize_weights_one_simd      :5171
        kernel_dsv4_router_finalize_one / _weights_one               :4887 / :501
        kernel_dsv4_router_shared_gate_up_q8_0 (router part)  metal/dense.metal:937
        kernel_argsort_f32_i32_desc                           metal/argsort.metal:52
  CPU   layer_router_probs_one, topk_desc,
        layer_topk_selected_experts_from_probs                ds4.c:12862-12962

Flash router (n_expert = 256, n_expert_used = 6, scale 1.5, no groups):
    logits = W_f16[256][4096] . x_f32
    probs  = sqrt(softplus(logits))
    score  = probs + bias                        (bias only for the selection)
    sel    = top-6 of score, descending (bitonic sort, strict compares)
    S      = max(((((p[s0]+p[s1])+p[s2])+p[s3])+p[s4])+p[s5], 2^-14)
    w[i]   = p[s_i] * (1.5 / S)

ds4's decode kernel fuses the matvec with the finalize through a
last-arriving-threadgroup atomic (no spinning); TileLang has no portable
device-scope handoff, so this is two dispatches: an f16 GEMV (also the
general f16-weight matvec) and a one-threadgroup finalize. The bitonic
sort runs in threadgroup memory with explicit thread indices: TileLang has
no sort primitive.

No `from __future__ import annotations`: closure names used only in the
prim_func signature (in_dim, n_used) must stay evaluable (see utility.py).
"""

import tilelang.language as T

from tlk import KernelSpec
from kernels.elementwise import _softplus

F16_ROWS = 4
F16_ACC_PER_THREAD = 32


def f16_threads(in_dim: int, rows: int = F16_ROWS) -> int:
    return rows * max(32, min(256, in_dim // 128))


def f16_gemv_f32(in_dim: int, rows: int = F16_ROWS, threads=None):
    """out[r] = sum_k f16 w[r, k] * f32 x[k]; f32 accumulation. The row is
    walked in 4-element chunks (one half4 and one float4 per step); the
    (rows, ksplit) fragment keeps ~32 accumulators per thread and is reduced
    with T.reduce_sum. Grid: (out_dim / rows,)."""
    assert in_dim % 4 == 0
    threads = threads or f16_threads(in_dim, rows)
    chunks = in_dim // 4
    ksplit = min(chunks, max(1, threads * F16_ACC_PER_THREAD // rows))
    while chunks % ksplit:
        ksplit -= 1
    cper = chunks // ksplit
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, in_dim), T.float16),
        x: T.Tensor((in_dim,), T.float32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            acc = T.alloc_fragment((rows, ksplit), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)
            T.clear(acc)
            for r, ks in T.Parallel(rows, ksplit):
                for i in T.serial(cper):
                    c = (i * ksplit + ks) * 4
                    acc[r, ks] += (
                        T.Cast(T.float32, w[bx * rows + r, c]) * x[c]
                        + T.Cast(T.float32, w[bx * rows + r, c + 1]) * x[c + 1]
                        + T.Cast(T.float32, w[bx * rows + r, c + 2]) * x[c + 2]
                        + T.Cast(T.float32, w[bx * rows + r, c + 3]) * x[c + 3]
                    )
            T.reduce_sum(acc, rowsum, dim=1)
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel


def f16_gemv_batch_f32(in_dim: int, rows: int = 1, toks: int = 8, threads=None):
    """out[t, r] = f16 w[r, :] . f32 x[t, :] for a tile of `toks` tokens
    (kernel_mul_mv_ext_f16_f32_r1_* and, for 9..31 tokens, the legacy f16
    GEMM). Tokens past n_tok read token n_tok - 1 and are not written.
    Grid: (out_dim / rows, ceildiv(n_tok, toks))."""
    assert in_dim % 4 == 0
    threads = threads or f16_threads(in_dim, rows)   # 32 threads per row: ~128 inputs per thread per row
    chunks = in_dim // 4
    ksplit = min(chunks, max(1, threads * F16_ACC_PER_THREAD // (rows * toks)))
    while chunks % ksplit:
        ksplit -= 1
    cper = chunks // ksplit
    out_dim = T.dynamic("out_dim")
    n_tok = T.dynamic("n_tok")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_tok, out_dim), T.float32),
        w: T.Tensor((out_dim, in_dim), T.float16),
        x: T.Tensor((n_tok, in_dim), T.float32),
    ):
        with T.Kernel(out_dim // rows, T.ceildiv(n_tok, toks), threads=threads) as (bx, bt):
            acc = T.alloc_fragment((rows, toks, ksplit), T.float32)
            tsum = T.alloc_fragment((rows, toks), T.float32)
            T.clear(acc)
            for r, ks in T.Parallel(rows, ksplit):
                wl = T.alloc_local((4,), T.float16)
                xl = T.alloc_local((4,), T.float32)
                for i in T.serial(cper):
                    c = (i * ksplit + ks) * 4
                    for j in T.vectorized(4):
                        wl[j] = w[bx * rows + r, c + j]
                    for t in T.unroll(toks):
                        tt = T.min(bt * toks + t, n_tok - 1)
                        for j in T.vectorized(4):
                            xl[j] = x[tt, c + j]
                        acc[r, t, ks] += (
                            T.Cast(T.float32, wl[0]) * xl[0] + T.Cast(T.float32, wl[1]) * xl[1]
                            + T.Cast(T.float32, wl[2]) * xl[2] + T.Cast(T.float32, wl[3]) * xl[3]
                        )
            T.reduce_sum(acc, tsum, dim=2)
            for r, t in T.Parallel(rows, toks):
                if bt * toks + t < n_tok:
                    out[bt * toks + t, bx * rows + r] = tsum[r, t]

    return kernel



def hc_rms_norm_mix_f16(in_dim: int = 16384, rows: int = 1, threads: int = 256):
    """HC mixer producer (kernel_dsv4_hc_rms_norm_mix_f16_*, first part):
    mix[r] = sum_k f16 w[r, k] * (x[k] * rsqrt(mean(x^2) + eps)) for the 24
    mixer rows; every threadgroup recomputes the 16384-wide norm scale (as
    ds4's fused kernel does) and produces `rows` rows (rows=1: 24 small
    threadgroups beat 6 wide ones; measured 7.8 vs 8.3 us wall for the
    two-dispatch pair, hand fused kernel 7.1-8.1 us). The split / collapse /
    norm epilogue is tl_hc_split_weighted_sum_norm4.
    Grid: (out_dim / rows,)."""
    assert in_dim % (threads * 4) == 0
    chunks = in_dim // 4
    ksplit = min(chunks, max(1, threads * F16_ACC_PER_THREAD // rows))
    while chunks % ksplit:
        ksplit -= 1
    cper = chunks // ksplit
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, in_dim), T.float16),
        x: T.Tensor((in_dim,), T.float32),
        eps: T.float32,
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            sq = T.alloc_fragment((1, in_dim), T.float32)
            ss = T.alloc_fragment((1,), T.float32)
            T.copy(x[0:in_dim], sq[0, :])
            for i, j in T.Parallel(1, in_dim):
                sq[i, j] = sq[i, j] * sq[i, j]
            T.reduce_sum(sq, ss, dim=1)
            scale = T.rsqrt(ss[0] / in_dim + eps)
            acc = T.alloc_fragment((rows, ksplit), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)
            T.clear(acc)
            for r, ks in T.Parallel(rows, ksplit):
                wl = T.alloc_local((4,), T.float16)
                xl = T.alloc_local((4,), T.float32)
                for i in T.serial(cper):
                    c = (i * ksplit + ks) * 4
                    for j in T.vectorized(4):
                        wl[j] = w[bx * rows + r, c + j]
                    for j in T.vectorized(4):
                        xl[j] = x[c + j]
                    acc[r, ks] += (
                        T.Cast(T.float32, wl[0]) * (xl[0] * scale) + T.Cast(T.float32, wl[1]) * (xl[1] * scale)
                        + T.Cast(T.float32, wl[2]) * (xl[2] * scale) + T.Cast(T.float32, wl[3]) * (xl[3] * scale)
                    )
            T.reduce_sum(acc, rowsum, dim=1)
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel



def hc_mix_gemm_f16(in_dim: int = 16384, mix_dim: int = 24, bm: int = 64, bn: int = 32, bk: int = 64, threads: int = 128):
    """Prefill HC mixer as one GEMM with a fused row sum of squares (after
    TileKernels' mhc "sqrsum" formulation): mix[t, r] = rsqrt(|x_t|^2 / in_dim
    + eps) * sum_k f16 w[r, k] * f16(x[t, k]) for the 24 mixer rows (N padded
    to 32 for the cooperative-tensor tile), x read once in f32 (squares from
    f32, the operand rounded to f16 like the hand GEMM). Replaces ds4's
    rms_norm pass over [n_rows][16384] (read + write) followed by the
    16384->24 f16 GEMV batch (96 GB/s in situ). Grid: (ceildiv(n_rows, bm),)."""
    assert in_dim % bk == 0 and mix_dim <= bn and bn % 32 == 0
    n_rows = T.dynamic("n_rows")

    @T.prim_func
    def kernel(
        mix: T.Tensor((n_rows, mix_dim), T.float32),
        w: T.Tensor((mix_dim, in_dim), T.float16),
        x: T.Tensor((n_rows, in_dim), T.float32),
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(n_rows, bm), threads=threads) as by:
            a_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            b_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            c_s = T.alloc_shared((bm, bn), T.float32, scope="shared")
            rs_s = T.alloc_shared((bm,), T.float32, scope="shared")
            xf = T.alloc_fragment((bm, bk), T.float32)
            sqf = T.alloc_fragment((bm, bk), T.float32)
            rs = T.alloc_fragment((bm,), T.float32)
            rowsq = T.alloc_fragment((bm,), T.float32)
            acc = T.alloc_fragment((bm, bn), T.float32)
            T.clear(acc)
            T.clear(rowsq)
            for ko in T.Pipelined(in_dim // bk, num_stages=0):
                for i, j in T.Parallel(bm, bk):
                    xf[i, j] = x[T.min(by * bm + i, n_rows - 1), ko * bk + j]
                for i, j in T.Parallel(bm, bk):
                    a_s[i, j] = T.Cast(T.float16, xf[i, j])
                    sqf[i, j] = xf[i, j] * xf[i, j]
                T.reduce_sum(sqf, rs, dim=1, clear=True)
                for i in T.Parallel(bm):
                    rowsq[i] += rs[i]
                for kk, n in T.Parallel(bk, bn):
                    b_s[kk, n] = T.if_then_else(n < mix_dim, w[T.min(n, mix_dim - 1), ko * bk + kk], T.float16(0))
                T.gemm(a_s, b_s, acc)
            T.copy(acc, c_s)
            for i in T.Parallel(bm):
                rs_s[i] = rowsq[i]
            T.sync_threads()
            for i, j in T.Parallel(bm, mix_dim):
                if by * bm + i < n_rows:
                    mix[by * bm + i, j] = c_s[i, j] * T.rsqrt(rs_s[i] / in_dim + eps)

    return kernel



@T.macro
def _bitonic_argsort_desc(keys, idx, tid, n_log2):
    """Sort (keys, idx) of length 2**n_log2 in threadgroup memory into
    descending key order; one thread per element, the lower thread of each
    pair does the compare-exchange (kernel_dsv4_router_finalize_one). Strict
    compares: equal keys keep their network order."""
    for ks in T.serial(n_log2):
        k = 2 << ks
        for js in T.serial(ks + 1):
            j = k >> (js + 1)
            other = tid ^ j
            if other > tid:
                a = keys[tid]
                b = keys[other]
                ia = idx[tid]
                ib = idx[other]
                swap = T.if_then_else((tid & k) == 0, a < b, a > b)
                if swap:
                    keys[tid] = b
                    keys[other] = a
                    idx[tid] = ib
                    idx[other] = ia
            # No explicit barrier: TileLang's thread-sync pass places one
            # threadgroup barrier per stage (before the compare-exchange),
            # which is all the network needs.


def router_finalize(n_expert: int = 256, n_used: int = 6, scale: float = 1.5, threads: int = 256):
    """logits -> probs (sqrt softplus, written out), top-n_used by probs +
    bias (if has_bias), normalised weights * scale. One threadgroup of
    n_expert threads. Grid: (1,)."""
    assert n_expert == threads and (n_expert & (n_expert - 1)) == 0
    n_log2 = n_expert.bit_length() - 1

    @T.prim_func
    def kernel(
        selected: T.Tensor((n_used,), T.int32),
        weights: T.Tensor((n_used,), T.float32),
        probs: T.Tensor((n_expert,), T.float32),
        logits: T.Tensor((n_expert,), T.float32),
        bias: T.Tensor((n_expert,), T.float32),
        has_bias: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _bx:
            keys = T.alloc_shared((n_expert,), T.float32)
            idx = T.alloc_shared((n_expert,), T.int32)
            # probs are re-read by other threads after the sort: keep a
            # threadgroup copy (TileLang's sync is a threadgroup-scope barrier;
            # device-memory visibility inside one dispatch is not promised).
            p_s = T.alloc_shared((n_expert,), T.float32)
            tid = T.get_thread_binding()
            p = T.sqrt(_softplus(logits[tid]))
            probs[tid] = p
            p_s[tid] = p
            keys[tid] = T.if_then_else(has_bias != 0, p + bias[tid], p)
            idx[tid] = tid
            T.sync_threads()
            _bitonic_argsort_desc(keys, idx, tid, n_log2)
            if tid < n_used:
                selected[tid] = idx[tid]
                s = T.alloc_var(T.float32)
                s = 0.0
                for i in T.serial(n_used):
                    s += p_s[idx[i]]
                inv = scale / T.max(s, 6.103515625e-5)
                weights[tid] = p_s[idx[tid]] * inv

    return kernel


def router_weights(n_used: int = 6, threads: int = 32):
    """weights[t, i] = probs[t, sel[t, i]] * (scale / max(sum_i probs[t, sel[t, i]], 2^-14)),
    left-fold sum (kernel_dsv4_router_weights_one). Grid: (rows,)."""
    rows = T.dynamic("rows")
    n_expert = T.dynamic("n_expert")

    @T.prim_func
    def kernel(
        weights: T.Tensor((rows, n_used), T.float32),
        probs: T.Tensor((rows, n_expert), T.float32),
        selected: T.Tensor((rows, n_used), T.int32),
        scale: T.float32,
    ):
        with T.Kernel(rows, threads=threads) as t:
            for i in T.Parallel(threads):
                if i < n_used:
                    s = T.alloc_var(T.float32)
                    s = 0.0
                    for k in T.serial(n_used):
                        s += probs[t, selected[t, k]]
                    weights[t, i] = probs[t, selected[t, i]] * (scale / T.max(s, 6.103515625e-5))

    return kernel


def argsort_desc_topk(width: int = 256, threads: int = 256):
    """out[t, 0:k] = indices of the k largest keys[t, :] in descending order
    (kernel_argsort_f32_i32_desc with ne00 = width). One threadgroup per
    row. Grid: (rows,)."""
    assert width == threads and (width & (width - 1)) == 0
    n_log2 = width.bit_length() - 1
    rows = T.dynamic("rows")
    out_len = T.dynamic("out_len")

    @T.prim_func
    def kernel(
        out: T.Tensor((out_len,), T.int32),
        keys_in: T.Tensor((rows, width), T.float32),
        k: T.int32,
    ):
        with T.Kernel(rows, threads=threads) as t:
            keys = T.alloc_shared((width,), T.float32)
            idx = T.alloc_shared((width,), T.int32)
            tid = T.get_thread_binding()
            keys[tid] = keys_in[t, tid]
            idx[tid] = tid
            T.sync_threads()
            _bitonic_argsort_desc(keys, idx, tid, n_log2)
            if tid < k:
                out[t * k + tid] = idx[tid]

    return kernel


F16_IN_DIMS = (4096, 16384, 1024)   # router / compressor / indexer projections, HC mixers, indexer q_b

# Single-token f16 GEMV configs: (in_dim, rows per threadgroup, threads).
# 16384 -> 24 (HC mixers): one row per threadgroup so 24 threadgroups run;
# 4 rows x 512 threads left 6 threadgroups and ran at 1.14x of hand-written.
F16_GEMV_CONFIGS = ((4096, F16_ROWS, None), (16384, 1, 256), (1024, F16_ROWS, None))

KERNELS = [
    KernelSpec(name=f"tl_f16_gemv_f32_{_n}", build=(lambda n=_n, r=_r, t=_t: f16_gemv_f32(n, r, t)), group="router", targets=("metal",),
               notes=f"out[out_dim] = f16 w[out_dim][{_n}] . f32 x; out_dim % {_r} == 0; grid=(out_dim/{_r})",
               variants={"r1": (lambda n=_n: f16_gemv_f32(n, 1)), "r2": (lambda n=_n: f16_gemv_f32(n, 2)),
                         "t2": (lambda n=_n, r=_r, t=_t: f16_gemv_f32(n, r, 2 * (t or f16_threads(n, r))))},
               tune_group="f16_gemv")
    for (_n, _r, _t) in F16_GEMV_CONFIGS
] + [
    KernelSpec(name=f"tl_f16_gemv_batch_f32_{_n}", build=(lambda n=_n: f16_gemv_batch_f32(n)), group="router", targets=("metal",),
               notes=f"out[n_tok][out_dim] = f16 w[out_dim][{_n}] . x[n_tok]; 8 tokens per tile, 1 row per threadgroup; grid=(out_dim, ceil(n_tok/8))",
               variants={"t2": (lambda n=_n: f16_gemv_batch_f32(n, threads=2 * f16_threads(n, 1))),
                         "t4": (lambda n=_n: f16_gemv_batch_f32(n, threads=4 * f16_threads(n, 1)))},
               tune_group="f16_gemv_batch")
    for _n in F16_IN_DIMS
] + [
    KernelSpec(name=f"tl_f16_gemv_batch4_f32_{_n}", build=(lambda n=_n: f16_gemv_batch_f32(n, toks=4)), group="router", targets=("metal",),
               notes=f"same with 4 tokens per tile (n_tok <= 4); grid=(out_dim, ceil(n_tok/4))")
    for _n in F16_IN_DIMS
] + [
    KernelSpec(name="tl_hc_mix_gemm_f16_16384", build=hc_mix_gemm_f16, group="router", targets=("metal",),
               notes="prefill HC mixer GEMM with fused row |x|^2: mix[t][24] = rsqrt(|x_t|^2/16384+eps) * x16_t . fn^T; 64 tokens per threadgroup; grid=(ceil(n_rows/64))",
               variants={"k32": lambda: hc_mix_gemm_f16(bk=32), "k128": lambda: hc_mix_gemm_f16(bk=128), "m32": lambda: hc_mix_gemm_f16(bm=32)},
               tune_group="hc_mix_gemm"),
    KernelSpec(name="tl_hc_rms_norm_mix_f16_16384", build=hc_rms_norm_mix_f16, group="router", targets=("metal",),
               notes="HC mixer producer: mix[r] = f16 w[r] . (x * rsqrt(mean(x^2)+eps)), x [16384], norm recomputed per threadgroup; grid=(out_dim)",
               variants={"r2": lambda: hc_rms_norm_mix_f16(rows=2), "t512": lambda: hc_rms_norm_mix_f16(threads=512),
                         "r2t512": lambda: hc_rms_norm_mix_f16(rows=2, threads=512)},
               tune_group="hc_mix"),
    KernelSpec(name="tl_router_finalize_f32_256_6", build=router_finalize, group="router", targets=("metal",),
               notes="probs = sqrt(softplus(logits)); top-6 of probs(+bias) desc; weights = p*(1.5/max(S,2^-14)); grid=(1)"),
    KernelSpec(name="tl_router_weights_f32_6", build=router_weights, group="router", targets=("metal",),
               notes="weights[t,i] = probs[t,sel[t,i]] * scale/max(sum,2^-14); grid=(rows)"),
    KernelSpec(name="tl_argsort_desc_topk_f32_256", build=argsort_desc_topk, group="router", targets=("metal",),
               notes="out[t,0:k] = argsort desc of keys[t,0:256], first k; grid=(rows)"),
]
