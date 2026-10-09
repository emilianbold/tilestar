"""Q8_0 weight x Q8_0-prequantised activation GEMV (single-token decode).

References in ds4:
  CPU   dot_q8_0_row / matvec_q8_0_rows_prequant   ds4.c       (~9088, ~9505)
  CUDA  matmul_q8_0_preq_warp8_kernel              ds4_cuda.cu (~5256)
  Metal kernel_matmul_q8_0_* (dense.metal)

Layouts (ds4.c:1037, GGUF Q8_0): a row of `in_dim` weights is `blocks`
consecutive 34-byte blocks, each `f16 d` followed by `int8 qs[32]`, so rows
are only 2-byte aligned. The kernel views the weight matrix as bytes
`[out_dim][blocks * 34]`: bytes b*34, b*34+1 are the little-endian f16 scale
and bytes b*34+2 .. b*34+33 the int8 quants, so the inner dot runs over 32
contiguous bytes (which the host C++ compiler can vectorise).

Activations are prequantised per 32-block by quantize_q8_0_activation:
`xq int8[blocks*32]` plus `xscale f32[blocks]`.

out[r] = sum_b d[r,b] * xscale[b] * sum_k qs[r,b,k] * xq[b,k]

Each threadgroup handles ROWS rows: a (ROWS, blocks) fragment of per-block
dots is filled in parallel, then reduced over blocks. out_dim must be a
multiple of ROWS.
"""

from __future__ import annotations

import tilelang.language as T

from tlk import KernelSpec
import shapes
from quant import q8_0 as q8_0_prims

THREADS = 256
ROWS = 8
TOKS = 4  # tokens per tile in the batched kernel (sweep: 16x4 ~5% better, 8-token tiles spill)


def q8_0_gemv(in_dim: int, rows: int = ROWS, threads: int = THREADS):
    assert in_dim % 32 == 0
    blocks = in_dim // 32
    if blocks * rows > 2048:
        threads = 128   # Metal layout inference finds no fragment layout for (8, 384) over 256 threads (in_dim 12288)
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, blocks * 34), T.uint8),
        xq: T.Tensor((blocks * 32,), T.int8),
        xscale: T.Tensor((blocks,), T.float32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            isum = T.alloc_fragment((rows, blocks), T.int32)
            acc = T.alloc_fragment((rows, blocks), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)

            T.clear(isum)
            for r, b in T.Parallel(rows, blocks):
                for k in T.serial(32):
                    isum[r, b] += T.Cast(T.int32, T.reinterpret(T.int8, w[bx * rows + r, b * 34 + 2 + k])) * T.Cast(
                        T.int32, xq[b * 32 + k]
                    )
            for r, b in T.Parallel(rows, blocks):
                bits = T.Cast(T.uint16, w[bx * rows + r, b * 34]) | (T.Cast(T.uint16, w[bx * rows + r, b * 34 + 1]) << 8)
                d = T.Cast(T.float32, T.reinterpret(T.float16, bits))
                acc[r, b] = T.Cast(T.float32, isum[r, b]) * d * xscale[b]
            T.reduce_sum(acc, rowsum, dim=1)
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel


def q8_0_gemv_lanes(in_dim: int, rows: int = ROWS, threads: int = THREADS):
    """CPU-oriented formulation of q8_0_gemv (same contract).

    One float partial sum per (row, lane); the block loop is serial with the
    block scale folded into the same pass, so the only cross-lane reduction
    is one per row at the end. Measured on the llvm target this is at parity
    with q8_0_gemv (each 8 lanes cost a convert + FMA, where ggml first
    reduces 32 int8 products to 8 int32 with vpmaddubsw/vpmaddwd and converts
    once). Writing that 4:1 integer pre-reduction explicitly makes LLVM drop
    to scalar code, and a (rows, 32) T.Parallel body is left unvectorised by
    TileLang's planner; the inner serial lane loop is the shape LLVM
    vectorises. Only 8 threads would be active on a GPU, so this variant is
    generated for the c target only.
    """
    assert in_dim % 32 == 0
    blocks = in_dim // 32
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, blocks * 34), T.uint8),
        xq: T.Tensor((blocks * 32,), T.int8),
        xscale: T.Tensor((blocks,), T.float32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            facc = T.alloc_fragment((rows, 32), T.float32)
            dsc = T.alloc_fragment((rows,), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)

            T.clear(facc)
            for b in T.serial(blocks):
                for r in T.Parallel(rows):
                    bits = T.Cast(T.uint16, w[bx * rows + r, b * 34]) | (T.Cast(T.uint16, w[bx * rows + r, b * 34 + 1]) << 8)
                    dsc[r] = T.Cast(T.float32, T.reinterpret(T.float16, bits)) * xscale[b]
                for r in T.Parallel(rows):
                    for k in T.serial(32):
                        q = T.Cast(T.int32, T.reinterpret(T.int8, w[bx * rows + r, b * 34 + 2 + k])) * T.Cast(T.int32, xq[b * 32 + k])
                        facc[r, k] += dsc[r] * T.Cast(T.float32, q)
            T.reduce_sum(facc, rowsum, dim=1)
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel


def q8_0_gemv_simd(in_dim: int, rows: int = ROWS, threads: int = THREADS):
    """CPU formulation using the hand-written per-block primitive from
    quant/q8_0.py (AVX2 / NEON / scalar), the CPU analogue of T.dp4a.

    TileLang owns the structure (row tile, serial block loop, final per-row
    reduce); the primitive accumulates 8 int32-derived lane partials per row
    in a (rows, 8) fragment that the host compiler keeps in a register.
    """
    assert in_dim % 32 == 0
    blocks = in_dim // 32
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, blocks * 34), T.uint8),
        xq: T.Tensor((blocks * 32,), T.int8),
        xscale: T.Tensor((blocks,), T.float32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            T.import_source(q8_0_prims.BLOCK_ACC_SRC)
            facc = T.alloc_fragment((rows, 8), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)

            T.clear(facc)
            for b in T.serial(blocks):
                for r in T.Parallel(rows):
                    bits = T.Cast(T.uint16, w[bx * rows + r, b * 34]) | (T.Cast(T.uint16, w[bx * rows + r, b * 34 + 1]) << 8)
                    d = T.Cast(T.float32, T.reinterpret(T.float16, bits)) * xscale[b]
                    T.call_extern(
                        "handle",
                        q8_0_prims.BLOCK_ACC_FUNC,
                        T.address_of(facc[r, 0]),
                        T.address_of(w[bx * rows + r, b * 34 + 2]),
                        T.address_of(xq[b * 32]),
                        d,
                    )
            T.reduce_sum(facc, rowsum, dim=1)
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel


def q8_0_gemm_simd(in_dim: int, rows: int = ROWS, toks: int = TOKS, threads: int = THREADS):
    """Batched (n_tok > 1) Q8_0 matmul with the SIMD block primitive.

    Reference: matmul_q8_0_batch_worker / dot_q8_0_row_2 (ds4.c ~9048, ~9151).
    Layouts: xq int8[n_tok][blocks*32], xscale f32[n_tok][blocks],
    out f32[n_tok][out_dim]. Tile = (rows, toks): each 34-byte weight block is
    loaded once and applied to `toks` token blocks, which is the reuse the
    single-token kernel cannot get. n_tok need not be a multiple of toks.
    """
    assert in_dim % 32 == 0
    blocks = in_dim // 32
    out_dim = T.dynamic("out_dim")
    n_tok = T.dynamic("n_tok")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_tok, out_dim), T.float32),
        w: T.Tensor((out_dim, blocks * 34), T.uint8),
        xq: T.Tensor((n_tok, blocks * 32), T.int8),
        xscale: T.Tensor((n_tok, blocks), T.float32),
    ):
        with T.Kernel(out_dim // rows, T.ceildiv(n_tok, toks), threads=threads) as (bx, by):
            T.import_source(q8_0_prims.BLOCK_ACC_SRC)
            facc = T.alloc_fragment((rows, toks, 8), T.float32)
            dsc = T.alloc_fragment((rows,), T.float32)
            tsum = T.alloc_fragment((rows, toks), T.float32)

            T.clear(facc)
            for b in T.serial(blocks):
                for r in T.Parallel(rows):
                    bits = T.Cast(T.uint16, w[bx * rows + r, b * 34]) | (T.Cast(T.uint16, w[bx * rows + r, b * 34 + 1]) << 8)
                    dsc[r] = T.Cast(T.float32, T.reinterpret(T.float16, bits))
                for r in T.Parallel(rows):
                    for t in T.serial(toks):
                        if by * toks + t < n_tok:
                            T.call_extern(
                                "handle",
                                q8_0_prims.BLOCK_ACC_FUNC,
                                T.address_of(facc[r, t, 0]),
                                T.address_of(w[bx * rows + r, b * 34 + 2]),
                                T.address_of(xq[by * toks + t, b * 32]),
                                dsc[r] * xscale[by * toks + t, b],
                            )
            T.reduce_sum(facc, tsum, dim=2)
            for r, t in T.Parallel(rows, toks):
                if by * toks + t < n_tok:
                    out[by * toks + t, bx * rows + r] = tsum[r, t]

    return kernel


# ---------------------------------------------------------------------------
# Metal decode GEMV: Q8_0 weights x f32 activations
# ---------------------------------------------------------------------------

F32_ROWS = 4
F32_ACC_PER_THREAD = 32   # accumulator floats per thread; bounds the fragment as in_dim grows


def f32_threads(in_dim: int, rows: int = F32_ROWS) -> int:
    """Threads per threadgroup: about 128 inputs (4 Q8_0 blocks) per thread
    per row. Measured on the M5 Max for the fused attn_output_b kernel
    (in_dim 8192): 4 rows x 128 threads 1.25-1.39x slower than hand-written,
    4 rows x 256 or 2 rows x 128 at parity; in_dim 1024/4096 are at parity
    with 32 threads per row."""
    per_row = max(32, min(256, in_dim // 128))
    per_row = 1 << (per_row.bit_length() - 1)   # power of two: the reductions are xor butterflies (12288 -> 96 -> 64)
    return rows * per_row


def _q8_0_f32_tile(in_dim: int, rows: int, threads: int, chunk: int, toks: int = 1):
    """Shape of the (rows, ksplit * nchunk) accumulator fragment: `ksplit`
    groups of blocks are reduced in parallel, each thread walking
    blocks / ksplit blocks serially. Keeps the per-thread accumulator at
    F32_ACC_PER_THREAD floats whatever in_dim is (64 per thread at in_dim
    8192 made the fused HC-expand kernel 1.25x slower than hand-written)."""
    assert in_dim % 32 == 0 and chunk in (2, 4, 8)
    blocks = in_dim // 32
    nchunk = 32 // chunk
    ksplit = min(blocks, max(1, threads * F32_ACC_PER_THREAD // (rows * toks * nchunk)))
    while blocks % ksplit:
        ksplit -= 1
    return blocks, nchunk, ksplit, blocks // ksplit


@T.macro
def _q8_0_f32_accumulate(acc, rowsum, w_at, x_at, rows, blocks, nchunk, ksplit, bper):
    """Emit: acc[r, ks*nchunk + c] = sum over this thread's blocks of
    d_b * (4-byte chunk c of block b . x), then rowsum = sum over acc.
    w_at(r, k) is uint16 k of weight row r of this tile, x_at(i) the
    activation; both are plain Python callables returning buffer loads so
    the same body serves flat, grouped and fused callers. Chunk is 4 bytes
    (two uint16) so the block dot is one inlined expression."""
    T.clear(acc)
    for r, kc in T.Parallel(rows, ksplit * nchunk):
        ks = kc // nchunk
        c = kc % nchunk
        for i in T.serial(bper):
            b = i * ksplit + ks
            q0 = w_at(r, b * 17 + 1 + c * 2)
            q1 = w_at(r, b * 17 + 1 + c * 2 + 1)
            d = T.Cast(T.float32, T.reinterpret(T.float16, w_at(r, b * 17)))
            acc[r, kc] += d * (
                T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q0 & 0xFF))) * x_at(b * 32 + c * 4)
                + T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q0 >> 8))) * x_at(b * 32 + c * 4 + 1)
                + T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q1 & 0xFF))) * x_at(b * 32 + c * 4 + 2)
                + T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q1 >> 8))) * x_at(b * 32 + c * 4 + 3)
            )
    T.reduce_sum(acc, rowsum, dim=1)


def q8_0_gemv_f32(in_dim: int, rows: int = F32_ROWS, threads: int | None = None, chunk: int = 4, row_batch: bool = False):
    """Q8_0 weight x *f32* activation GEMV, the contract of ds4's Metal
    decode matvec (kernel_mul_mv_q8_0_f32 takes float activations; only the
    CUDA path prequantises).

    Formulation chosen on the M5 Max (A/B-measured against the alternatives):
    the weight row is viewed as uint16 [blocks*17] (the GGUF
    block is 2-byte aligned: u16 0 is the f16 scale, u16 1..16 hold the 32
    int8 quants as lo/hi bytes) and the fragment's innermost index is a
    4-byte chunk of a block, so consecutive lanes read consecutive 4-byte
    pieces of the same block and the matching x[4] is one float4. 4096x4096:
    ~700 GB/s cache-resident vs ~290 GB/s for a byte view with a serial
    32-wide inner loop (lanes 34 bytes apart); cold, both this and ds4's
    kernel_mul_mv_q8_0_f32 run at DRAM bandwidth (550-600 GB/s). rows=4 x
    128 threads; 8 rows x 256 threads is ~2x slower on in_dim=1024.
    """
    assert chunk == 4
    threads = threads or f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, chunk)
    out_dim = T.dynamic("out_dim")

    if row_batch:
        # ds4's "decode rows exact" contract (the multi-session batch schedule):
        # each activation row gets exactly the single-token kernel's threadgroup
        # work, so a session's result does not depend on its batch companions.
        n_rows = T.dynamic("n_rows")

        @T.prim_func
        def kernel_rows(
            out: T.Tensor((n_rows, out_dim), T.float32),
            w: T.Tensor((out_dim, blocks * 17), T.uint16),
            x: T.Tensor((n_rows, blocks * 32), T.float32),
        ):
            with T.Kernel(out_dim // rows, n_rows, threads=threads) as (bx, by):
                acc = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
                rowsum = T.alloc_fragment((rows,), T.float32)
                _q8_0_f32_accumulate(
                    acc, rowsum,
                    lambda r, k: w[bx * rows + r, k],
                    lambda i: x[by, i],
                    rows, blocks, nchunk, ksplit, bper,
                )
                for r in T.Parallel(rows):
                    out[by, bx * rows + r] = rowsum[r]

        return kernel_rows

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, blocks * 17), T.uint16),
        x: T.Tensor((blocks * 32,), T.float32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            acc = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)
            _q8_0_f32_accumulate(
                acc, rowsum,
                lambda r, k: w[bx * rows + r, k],
                lambda i: x[i],
                rows, blocks, nchunk, ksplit, bper,
            )
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel


def q8_0_gemv_hc_expand4_f32(in_dim: int, rows: int = F32_ROWS, threads: int | None = None, chunk: int = 4, add_routed: bool = False):
    """q8_0_gemv_f32 with ds4's hyper-connection expand epilogue fused in
    (kernel_dsv4_q8_hc_expand4_q8_0_vec_hc, the attn_output_b projection of
    DeepSeek V4 Flash: in=8192, out=n_embd=4096, n_hc=4).

    Per output row d, with v = w[d] . x:
        block_out[d]       = v
        out_hc[h*n + d]    = v * post[h] + sum_j comb[h + 4j] * residual[j*n + d]
    where split = [pre(4) | post(4) | comb(16)] is ds4's hc_split tensor
    (post = split[4..8], comb = split[8..24], comb indexed [dst_hc + j*n_hc]).

    add_routed (kernel_dsv4_shared_down_hc_expand4_q8_0, the shared-expert
    down projection in the non-concurrent decode FFN schedule): block_out[d]
    = v (the shared output alone) and the expand uses v + routed[d].
    """
    assert chunk == 4
    threads = threads or f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, chunk)
    n_hc = 4
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        out_hc: T.Tensor((n_hc * out_dim,), T.float32),
        block_out: T.Tensor((out_dim,), T.float32),
        w: T.Tensor((out_dim, blocks * 17), T.uint16),
        x: T.Tensor((blocks * 32,), T.float32),
        residual: T.Tensor((n_hc * out_dim,), T.float32),
        split: T.Tensor((2 * n_hc + n_hc * n_hc,), T.float32),
        routed: T.Tensor((out_dim,), T.float32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            acc = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)
            _q8_0_f32_accumulate(
                acc, rowsum,
                lambda r, k: w[bx * rows + r, k],
                lambda i: x[i],
                rows, blocks, nchunk, ksplit, bper,
            )
            for r in T.Parallel(rows):
                d = bx * rows + r
                block_out[d] = rowsum[r]
                v = rowsum[r] + routed[d] if add_routed else rowsum[r]
                for h in T.serial(n_hc):
                    out_hc[h * out_dim + d] = (
                        v * split[n_hc + h]
                        + split[2 * n_hc + h + 0 * n_hc] * residual[0 * out_dim + d]
                        + split[2 * n_hc + h + 1 * n_hc] * residual[1 * out_dim + d]
                        + split[2 * n_hc + h + 2 * n_hc] * residual[2 * out_dim + d]
                        + split[2 * n_hc + h + 3 * n_hc] * residual[3 * out_dim + d]
                    )

    return kernel


def q8_0_gemv_grouped_f32(in_dim: int, rows: int = F32_ROWS, threads: int | None = None, chunk: int = 4):
    """n_groups independent Q8_0 GEMVs in one dispatch (grid z = group):
    ds4's kernel_dsv4_attn_out_low_q8_0_f32, the attn_output_a projection of
    DeepSeek V4 Flash (8 head groups x (4096 -> 1024)). Weights are the
    n_groups matrices back to back, x is [n_groups][in_dim], out is
    [n_groups][out_dim]."""
    assert chunk == 4
    threads = threads or f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, chunk)
    out_dim = T.dynamic("out_dim")
    n_groups = T.dynamic("n_groups")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_groups, out_dim), T.float32),
        w: T.Tensor((n_groups, out_dim, blocks * 17), T.uint16),
        x: T.Tensor((n_groups, blocks * 32), T.float32),
    ):
        with T.Kernel(out_dim // rows, n_groups, threads=threads) as (bx, g):
            acc = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)
            _q8_0_f32_accumulate(
                acc, rowsum,
                lambda r, k: w[g, bx * rows + r, k],
                lambda i: x[g, i],
                rows, blocks, nchunk, ksplit, bper,
            )
            for r in T.Parallel(rows):
                out[g, bx * rows + r] = rowsum[r]

    return kernel


def q8_0_gate_up_swiglu_f32(in_dim: int, rows: int = F32_ROWS, threads: int | None = None, chunk: int = 4, row_batch: bool = False):
    """Shared-expert gate and up projections (Q8_0 x f32) with the SwiGLU
    epilogue fused (kernel_dsv4_shared_gate_up_swiglu_q8_0):
        gate[r] = Wg[r].x, up[r] = Wu[r].x
        mid[r]  = silu(min(gate, c)) * clamp(up, -c, c)     (c > 1e-6, else no clamp)
    Both matrices share one activation read per thread."""
    assert chunk == 4
    threads = threads or f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, chunk)
    out_dim = T.dynamic("out_dim")

    if row_batch:
        # row-exact batch form (kernel_dsv4_shared_gate_up_swiglu_q8_0 with ne11 > 1): see q8_0_gemv_f32
        n_rows = T.dynamic("n_rows")

        @T.prim_func
        def kernel_rows(
            gate_out: T.Tensor((n_rows, out_dim), T.float32),
            up_out: T.Tensor((n_rows, out_dim), T.float32),
            mid_out: T.Tensor((n_rows, out_dim), T.float32),
            wg: T.Tensor((out_dim, blocks * 17), T.uint16),
            wu: T.Tensor((out_dim, blocks * 17), T.uint16),
            x: T.Tensor((n_rows, blocks * 32), T.float32),
            clamp: T.float32,
        ):
            with T.Kernel(out_dim // rows, n_rows, threads=threads) as (bx, by):
                accg = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
                accu = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
                sumg = T.alloc_fragment((rows,), T.float32)
                sumu = T.alloc_fragment((rows,), T.float32)
                _q8_0_f32_accumulate(accg, sumg, lambda r, k: wg[bx * rows + r, k], lambda i: x[by, i], rows, blocks, nchunk, ksplit, bper)
                _q8_0_f32_accumulate(accu, sumu, lambda r, k: wu[bx * rows + r, k], lambda i: x[by, i], rows, blocks, nchunk, ksplit, bper)
                for r in T.Parallel(rows):
                    g = sumg[r]
                    u = sumu[r]
                    gate_out[by, bx * rows + r] = g
                    up_out[by, bx * rows + r] = u
                    gc = T.if_then_else(clamp > 1.0e-6, T.min(g, clamp), g)
                    uc = T.if_then_else(clamp > 1.0e-6, T.max(T.min(u, clamp), -clamp), u)
                    mid_out[by, bx * rows + r] = gc / (1.0 + T.exp(-gc)) * uc

        return kernel_rows

    @T.prim_func
    def kernel(
        gate_out: T.Tensor((out_dim,), T.float32),
        up_out: T.Tensor((out_dim,), T.float32),
        mid_out: T.Tensor((out_dim,), T.float32),
        wg: T.Tensor((out_dim, blocks * 17), T.uint16),
        wu: T.Tensor((out_dim, blocks * 17), T.uint16),
        x: T.Tensor((blocks * 32,), T.float32),
        clamp: T.float32,
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            accg = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
            accu = T.alloc_fragment((rows, ksplit * nchunk), T.float32)
            sumg = T.alloc_fragment((rows,), T.float32)
            sumu = T.alloc_fragment((rows,), T.float32)
            _q8_0_f32_accumulate(accg, sumg, lambda r, k: wg[bx * rows + r, k], lambda i: x[i], rows, blocks, nchunk, ksplit, bper)
            _q8_0_f32_accumulate(accu, sumu, lambda r, k: wu[bx * rows + r, k], lambda i: x[i], rows, blocks, nchunk, ksplit, bper)
            for r in T.Parallel(rows):
                g = sumg[r]
                u = sumu[r]
                gate_out[bx * rows + r] = g
                up_out[bx * rows + r] = u
                gc = T.if_then_else(clamp > 1.0e-6, T.min(g, clamp), g)
                uc = T.if_then_else(clamp > 1.0e-6, T.max(T.min(u, clamp), -clamp), u)
                mid_out[bx * rows + r] = gc / (1.0 + T.exp(-gc)) * uc

    return kernel


# ---------------------------------------------------------------------------
# Small-batch Q8_0 x f32 GEMV (2..31 tokens): kernel_mul_mv_ext_q8_0_f32_r1_*.
# ---------------------------------------------------------------------------

BATCH_TOKS = 8      # tokens per tile; a 4-token variant serves n_tok <= 4
BATCH_ROWS = 1      # one output row per threadgroup (sweep at 600 iterations: rows 2/4 were 1.2-1.4x slower)


def q8_0_gemv_batch_f32(in_dim: int, rows: int = BATCH_ROWS, toks: int = BATCH_TOKS, threads: int | None = None, chunk: int = 4):
    """out[t, r] = Q8_0 w[r, :] . x[t, :] for a tile of `toks` tokens: every
    dequantised weight chunk is reused across the tokens of the tile (the
    point of ds4's ext kernels). Tokens past n_tok read token n_tok - 1 and
    are not written. Grid: (out_dim / rows, ceildiv(n_tok, toks))."""
    assert chunk == 4
    threads = threads or f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, chunk, toks)
    out_dim = T.dynamic("out_dim")
    n_tok = T.dynamic("n_tok")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_tok, out_dim), T.float32),
        w: T.Tensor((out_dim, blocks * 17), T.uint16),
        x: T.Tensor((n_tok, blocks * 32), T.float32),
    ):
        with T.Kernel(out_dim // rows, T.ceildiv(n_tok, toks), threads=threads) as (bx, bt):
            blocks_ = blocks  # noqa: F841  (annotation-only name; see utility.py)
            acc = T.alloc_fragment((rows, toks, ksplit * nchunk), T.float32)
            tsum = T.alloc_fragment((rows, toks), T.float32)
            T.clear(acc)
            for r, kc in T.Parallel(rows, ksplit * nchunk):
                ks = kc // nchunk
                c = kc % nchunk
                xl = T.alloc_local((4,), T.float32)
                for i in T.serial(bper):
                    b = i * ksplit + ks
                    q0 = w[bx * rows + r, b * 17 + 1 + c * 2]
                    q1 = w[bx * rows + r, b * 17 + 1 + c * 2 + 1]
                    d = T.Cast(T.float32, T.reinterpret(T.float16, w[bx * rows + r, b * 17]))
                    w0 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q0 & 0xFF)))
                    w1 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q0 >> 8)))
                    w2 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q1 & 0xFF)))
                    w3 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q1 >> 8)))
                    for t in T.unroll(toks):
                        tt = T.min(bt * toks + t, n_tok - 1)
                        for j in T.vectorized(4):
                            xl[j] = x[tt, b * 32 + c * 4 + j]
                        acc[r, t, kc] += w0 * xl[0] + w1 * xl[1] + w2 * xl[2] + w3 * xl[3]
            T.reduce_sum(acc, tsum, dim=2)
            for r, t in T.Parallel(rows, toks):
                if bt * toks + t < n_tok:
                    out[bt * toks + t, bx * rows + r] = tsum[r, t]

    return kernel


def q8_0_gemv_grouped_batch_f32(in_dim: int, rows: int = BATCH_ROWS, toks: int = BATCH_TOKS, threads: int | None = None, chunk: int = 4):
    """attn_output_a for 2..63 tokens: out[t, g, r] = Q8_0 w[g, r, :] . x[t, g, :]
    (kernel_dsv4_attn_out_low_q8_0_f32 with a token batch). Token tiles as
    in q8_0_gemv_batch_f32. Grid: (out_dim / rows, n_groups, ceildiv(n_tok, toks))."""
    assert chunk == 4
    threads = threads or f32_threads(in_dim, rows)
    blocks, nchunk, ksplit, bper = _q8_0_f32_tile(in_dim, rows, threads, chunk, toks)
    out_dim = T.dynamic("out_dim")
    n_groups = T.dynamic("n_groups")
    n_tok = T.dynamic("n_tok")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_tok, n_groups, out_dim), T.float32),
        w: T.Tensor((n_groups, out_dim, blocks * 17), T.uint16),
        x: T.Tensor((n_tok, n_groups, blocks * 32), T.float32),
    ):
        with T.Kernel(out_dim // rows, n_groups, T.ceildiv(n_tok, toks), threads=threads) as (bx, g, bt):
            blocks_ = blocks  # noqa: F841  (annotation-only name)
            acc = T.alloc_fragment((rows, toks, ksplit * nchunk), T.float32)
            tsum = T.alloc_fragment((rows, toks), T.float32)
            T.clear(acc)
            for r, kc in T.Parallel(rows, ksplit * nchunk):
                ks = kc // nchunk
                c = kc % nchunk
                xl = T.alloc_local((4,), T.float32)
                for i in T.serial(bper):
                    b = i * ksplit + ks
                    q0 = w[g, bx * rows + r, b * 17 + 1 + c * 2]
                    q1 = w[g, bx * rows + r, b * 17 + 1 + c * 2 + 1]
                    d = T.Cast(T.float32, T.reinterpret(T.float16, w[g, bx * rows + r, b * 17]))
                    w0 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q0 & 0xFF)))
                    w1 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q0 >> 8)))
                    w2 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q1 & 0xFF)))
                    w3 = d * T.Cast(T.float32, T.reinterpret(T.int8, T.Cast(T.uint8, q1 >> 8)))
                    for t in T.unroll(toks):
                        tt = T.min(bt * toks + t, n_tok - 1)
                        for j in T.vectorized(4):
                            xl[j] = x[tt, g, b * 32 + c * 4 + j]
                        acc[r, t, kc] += w0 * xl[0] + w1 * xl[1] + w2 * xl[2] + w3 * xl[3]
            T.reduce_sum(acc, tsum, dim=2)
            for r, t in T.Parallel(rows, toks):
                if bt * toks + t < n_tok:
                    out[bt * toks + t, g, bx * rows + r] = tsum[r, t]

    return kernel


# ---------------------------------------------------------------------------
# Prefill tier: Q8_0 x f16 activations GEMM on Metal 4 cooperative tensors
# ---------------------------------------------------------------------------

GEMM_BM, GEMM_BN, GEMM_BK, GEMM_THREADS = 64, 64, 32, 64


def q8_0_gemm_f16x(in_dim: int, bm: int = GEMM_BM, bn: int = GEMM_BN, bk: int = GEMM_BK, threads: int = GEMM_THREADS):
    """C[m_pad, out_dim] = X16[m_pad, in_dim] . W[out_dim, in_dim]^T with Q8_0
    weights (uint16 view) dequantised into an f16 [bk, bn] shared tile each
    K step and T.gemm into a persistent cooperative-tensor accumulator
    (TileLang 0334561a). m_pad is a multiple of bm (the host pads the f16
    activation copy and the output scratch), out_dim a multiple of bn.

    M5 Max, 2560x2048x4096: 47.6 TFLOPS with this tile vs 46.5 for ds4's
    kernel_mul_mm_q8_0_f32_nax_direct_rhs; f32 activations converted in the
    copy instead cost 20% (38 TFLOPS), hence the separate f16 conversion.
    The shared B tile is written [K, N]: transpose_B is not implemented on
    TileLang's cooperative-tensor path."""
    assert in_dim % bk == 0 and bk % 32 == 0
    blocks = in_dim // 32
    m_pad = T.dynamic("m_pad")
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        c: T.Tensor((m_pad, out_dim), T.float32),
        x16: T.Tensor((m_pad, in_dim), T.float16),
        w: T.Tensor((out_dim, blocks * 17), T.uint16),
    ):
        with T.Kernel(out_dim // bn, m_pad // bm, threads=threads) as (bx, by):
            blocks_ = blocks  # noqa: F841  (annotation-only name)
            in_dim_ = in_dim  # noqa: F841
            a_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            b_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            acc = T.alloc_fragment((bm, bn), T.float32)
            T.clear(acc)
            for ko in T.Pipelined(in_dim // bk, num_stages=0):
                T.copy(x16[by * bm, ko * bk], a_s)
                for n, kk2 in T.Parallel(bn, bk // 2):
                    blk = (ko * bk) // 32 + kk2 // 16
                    q2 = w[bx * bn + n, blk * 17 + 1 + kk2 % 16]
                    d = T.reinterpret(T.float16, w[bx * bn + n, blk * 17])
                    b_s[2 * kk2, n] = T.Cast(T.float16, T.reinterpret(T.int8, T.Cast(T.uint8, q2 & 0xFF))) * d
                    b_s[2 * kk2 + 1, n] = T.Cast(T.float16, T.reinterpret(T.int8, T.Cast(T.uint8, q2 >> 8))) * d
                T.gemm(a_s, b_s, acc)
            T.copy(acc, c[by * bm, bx * bn])

    return kernel


def f16_gemm_f16x(in_dim: int, bm: int = GEMM_BM, bn: int = GEMM_BN, bk: int = GEMM_BK, threads: int = GEMM_THREADS):
    """C[m_pad, out_dim] = X16[m_pad, in_dim] . W[out_dim, in_dim]^T with f16
    weights (kernel_mul_mm_f16_f32 / _mpp_direct_rhs_n128 in prefill: the
    attention / indexer compressor projections, indexer proj and q_b, the
    router). Same tier as q8_0_gemm_f16x: the host converts the f32
    activations to a padded f16 copy; the B tile is written [K, N] from the
    row-major weights (no transpose_B on the cooperative-tensor path)."""
    assert in_dim % bk == 0
    m_pad = T.dynamic("m_pad")
    out_dim = T.dynamic("out_dim")

    @T.prim_func
    def kernel(
        c: T.Tensor((m_pad, out_dim), T.float32),
        x16: T.Tensor((m_pad, in_dim), T.float16),
        w: T.Tensor((out_dim, in_dim), T.float16),
    ):
        with T.Kernel(out_dim // bn, m_pad // bm, threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            b_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            acc = T.alloc_fragment((bm, bn), T.float32)
            T.clear(acc)
            for ko in T.Pipelined(in_dim // bk, num_stages=0):
                T.copy(x16[by * bm, ko * bk], a_s)
                for n, kk in T.Parallel(bn, bk):
                    b_s[kk, n] = w[bx * bn + n, ko * bk + kk]
                T.gemm(a_s, b_s, acc)
            T.copy(acc, c[by * bm, bx * bn])

    return kernel


def f32_to_f16_rows(width: int, threads: int = 256):
    """x16[r, :] = f16(x[r, :]) for r < n_rows, zero for n_rows <= r < m_pad
    (the GEMM reads whole bm-row tiles). Grid: (m_pad)."""
    assert width % (threads * 4) == 0 or width % 4 == 0
    n_rows = T.dynamic("n_rows")
    m_pad = T.dynamic("m_pad")

    @T.prim_func
    def kernel(
        x16: T.Tensor((m_pad, width), T.float16),
        x: T.Tensor((n_rows, width), T.float32),
    ):
        with T.Kernel(m_pad, threads=threads) as r:
            width_ = width  # noqa: F841
            for j in T.Parallel(width):
                x16[r, j] = T.if_then_else(r < n_rows, T.Cast(T.float16, x[T.min(r, n_rows - 1), j]), T.float16(0))

    return kernel


def copy_rows_f32(width_unused: int = 0, threads: int = 256):
    """out[r, :] = src[r, :] for r < n_rows (width dynamic). Grid: (n_rows)."""
    n_rows = T.dynamic("n_rows")
    m_pad = T.dynamic("m_pad")
    width = T.dynamic("width")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_rows, width), T.float32),
        src: T.Tensor((m_pad, width), T.float32),
    ):
        with T.Kernel(n_rows, threads=threads) as r:
            m_pad_ = m_pad  # noqa: F841  (annotation-only name)
            for j in T.Parallel(width):
                out[r, j] = src[r, j]

    return kernel



def q8_0_gemm_grouped_f16x(in_dim: int, bm: int = GEMM_BM, bn: int = GEMM_BN, bk: int = GEMM_BK, threads: int = GEMM_THREADS):
    """n_groups independent GEMMs sharing the token dimension (attn_output_a
    in prefill: heads[t][g][group_dim] . Wg^T -> low[t][g][rank]). The
    token-major output is viewed 2-D, [m_pad][c_width = n_groups*out_dim],
    because the cooperative-tensor store wants a 2-D destination; the
    activations stay 3-D [m_pad][n_groups][in_dim] f16 so the tile copy is
    vectorised. w is [n_groups][out_dim][blocks*17].
    Grid: (out_dim/bn, m_pad/bm, n_groups)."""
    assert in_dim % bk == 0 and bk % 32 == 0
    blocks = in_dim // 32
    m_pad = T.dynamic("m_pad")
    out_dim = T.dynamic("out_dim")
    n_groups = T.dynamic("n_groups")
    c_width = T.dynamic("c_width")

    @T.prim_func
    def kernel(
        c: T.Tensor((m_pad, c_width), T.float32),
        x16: T.Tensor((m_pad, n_groups, in_dim), T.float16),
        w: T.Tensor((n_groups, out_dim, blocks * 17), T.uint16),
    ):
        with T.Kernel(out_dim // bn, m_pad // bm, n_groups, threads=threads) as (bx, by, g):
            blocks_ = blocks  # noqa: F841  (annotation-only names)
            c_width_ = c_width  # noqa: F841
            a_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            b_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            acc = T.alloc_fragment((bm, bn), T.float32)
            T.clear(acc)
            for ko in T.Pipelined(in_dim // bk, num_stages=0):
                # 3-D view for the load: a static innermost extent keeps the
                # copy vectorised and unguarded (a dynamic row stride made it
                # 32 guarded scalar loads per thread per K step, 2.7x slower).
                T.copy(x16[by * bm:(by + 1) * bm, g, ko * bk:(ko + 1) * bk], a_s)
                for n, kk2 in T.Parallel(bn, bk // 2):
                    blk = (ko * bk) // 32 + kk2 // 16
                    q2 = w[g, bx * bn + n, blk * 17 + 1 + kk2 % 16]
                    d = T.reinterpret(T.float16, w[g, bx * bn + n, blk * 17])
                    b_s[2 * kk2, n] = T.Cast(T.float16, T.reinterpret(T.int8, T.Cast(T.uint8, q2 & 0xFF))) * d
                    b_s[2 * kk2 + 1, n] = T.Cast(T.float16, T.reinterpret(T.int8, T.Cast(T.uint8, q2 >> 8))) * d
                T.gemm(a_s, b_s, acc)
            T.copy(acc, c[by * bm, g * out_dim + bx * bn])

    return kernel


KERNELS = [
    KernelSpec(
        name=f"tl_q8_0_gemm_grouped_f16x_{_n}",
        build=(lambda n=_n: q8_0_gemm_grouped_f16x(n)),
        variants={"k64": (lambda n=_n: q8_0_gemm_grouped_f16x(n, bk=64)),
                  "t128": (lambda n=_n: q8_0_gemm_grouped_f16x(n, threads=128)),
                  "k64t128": (lambda n=_n: q8_0_gemm_grouped_f16x(n, bk=64, threads=128))},
        tune_group="gemm",
        group="q8gemm",
        targets=("metal",),
        notes=f"prefill attn_output_a: c[m_pad][g][out_dim] = x16[m_pad][g][{_n}] . Wg^T; grid=(out_dim/{GEMM_BN}, m_pad/{GEMM_BM}, n_groups)",
    )
    for _n in shapes.Q8_0_GROUPED_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemm_f16x_{_n}",
        build=(lambda n=_n: q8_0_gemm_f16x(n)),
        variants={"k64": (lambda n=_n: q8_0_gemm_f16x(n, bk=64)),
                  "t128": (lambda n=_n: q8_0_gemm_f16x(n, threads=128)),
                  "k64t128": (lambda n=_n: q8_0_gemm_f16x(n, bk=64, threads=128))},
        tune_group="gemm",
        group="q8gemm",
        targets=("metal",),
        notes=f"prefill: C[m_pad][out_dim] = X16[m_pad][{_n}] . Q8_0 W^T; m_pad % {GEMM_BM} == 0, out_dim % {GEMM_BN} == 0; grid=(out_dim/{GEMM_BN}, m_pad/{GEMM_BM})",
    )
    for _n in shapes.Q8_0_GEMM_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_f16_gemm_f16x_{_n}",
        build=(lambda n=_n: f16_gemm_f16x(n)),
        variants={"k64": (lambda n=_n: f16_gemm_f16x(n, bk=64)),
                  "t128": (lambda n=_n: f16_gemm_f16x(n, threads=128)),
                  "k64t128": (lambda n=_n: f16_gemm_f16x(n, bk=64, threads=128))},
        tune_group="gemm",
        group="q8gemm",
        targets=("metal",),
        notes=f"prefill: C[m_pad][out_dim] = X16[m_pad][{_n}] . f16 W^T; m_pad % {GEMM_BM} == 0, out_dim % {GEMM_BN} == 0; grid=(out_dim/{GEMM_BN}, m_pad/{GEMM_BM})",
    )
    for _n in shapes.F16_GEMM_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_f32_to_f16_rows_{_n}",
        build=(lambda n=_n: f32_to_f16_rows(n)),
        group="q8gemm",
        targets=("metal",),
        notes=f"x16[m_pad][{_n}] = f16(x[n_rows][{_n}]), zero rows past n_rows; grid=(m_pad)",
    )
    for _n in shapes.Q8_0_GEMM_IN_DIMS
] + [
    KernelSpec(
        name="tl_copy_rows_f32",
        build=copy_rows_f32,
        group="q8gemm",
        targets=("metal",),
        notes="out[n_rows][width] = src[m_pad][width] rows < n_rows; grid=(n_rows)",
    ),
] + [
    KernelSpec(
        name=f"tl_q8_0_gate_up_swiglu_f32_{_n}",
        build=(lambda n=_n: q8_0_gate_up_swiglu_f32(n)),
        variants={"r2": (lambda n=_n: q8_0_gate_up_swiglu_f32(n, rows=2)),
                  "r8": (lambda n=_n: q8_0_gate_up_swiglu_f32(n, rows=8)),
                  "t2": (lambda n=_n: q8_0_gate_up_swiglu_f32(n, threads=2 * f32_threads(n)))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"shared expert gate/up Q8_0 x f32 + SwiGLU; in_dim={_n}; out_dim % {F32_ROWS} == 0; grid=(out_dim/{F32_ROWS})",
    )
    for _n in shapes.Q8_0_SHARED_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gate_up_swiglu_rows_f32_{_n}",
        build=(lambda n=_n: q8_0_gate_up_swiglu_f32(n, row_batch=True)),
        variants={"r2": (lambda n=_n: q8_0_gate_up_swiglu_f32(n, rows=2, row_batch=True)),
                  "r8": (lambda n=_n: q8_0_gate_up_swiglu_f32(n, rows=8, row_batch=True)),
                  "t2": (lambda n=_n: q8_0_gate_up_swiglu_f32(n, threads=2 * f32_threads(n), row_batch=True))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"row-exact batch form of tl_q8_0_gate_up_swiglu_f32_{_n} (one session row per grid y; the multi-session batch schedule); grid=(out_dim/{F32_ROWS}, n_rows)",
    )
    for _n in shapes.Q8_0_SHARED_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_rows_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_f32(n, row_batch=True)),
        variants={"r2": (lambda n=_n: q8_0_gemv_f32(n, rows=2, row_batch=True)),
                  "r8": (lambda n=_n: q8_0_gemv_f32(n, rows=8, row_batch=True)),
                  "t2": (lambda n=_n: q8_0_gemv_f32(n, threads=2 * f32_threads(n), row_batch=True))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"row-exact batch form of tl_q8_0_gemv_f32_{_n} (ds4_gpu_matmul_q8_0_decode_rows_exact_tensor: one session row per grid y); grid=(out_dim/{F32_ROWS}, n_rows)",
    )
    for _n in shapes.Q8_0_GEMV_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_grouped_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_grouped_f32(n)),
        variants={"r2": (lambda n=_n: q8_0_gemv_grouped_f32(n, rows=2)),
                  "r8": (lambda n=_n: q8_0_gemv_grouped_f32(n, rows=8)),
                  "t2": (lambda n=_n: q8_0_gemv_grouped_f32(n, threads=2 * f32_threads(n)))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"attn_output_a: n_groups x (Q8_0 w[out_dim][{_n}] . x[g]); in_dim={_n}; out_dim % {F32_ROWS} == 0; grid=(out_dim/{F32_ROWS}, n_groups)",
    )
    for _n in shapes.Q8_0_GROUPED_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_hc_expand4_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_hc_expand4_f32(n)),
        variants={"r2": (lambda n=_n: q8_0_gemv_hc_expand4_f32(n, rows=2)),
                  "r8": (lambda n=_n: q8_0_gemv_hc_expand4_f32(n, rows=8)),
                  "t2": (lambda n=_n: q8_0_gemv_hc_expand4_f32(n, threads=2 * f32_threads(n)))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"attn_output_b + HC expand: block_out = Q8_0 w[out_dim][{_n}] . x; out_hc[h] = block*post[h] + comb[h]·residual; in_dim={_n}; out_dim % {F32_ROWS} == 0; grid=(out_dim/{F32_ROWS})",
    )
    for _n in shapes.Q8_0_HC_EXPAND_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_shared_down_hc_expand4_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_hc_expand4_f32(n, add_routed=True)),
        variants={"r2": (lambda n=_n: q8_0_gemv_hc_expand4_f32(n, rows=2, add_routed=True)),
                  "r8": (lambda n=_n: q8_0_gemv_hc_expand4_f32(n, rows=8, add_routed=True)),
                  "t2": (lambda n=_n: q8_0_gemv_hc_expand4_f32(n, threads=2 * f32_threads(n), add_routed=True))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"decode shared-expert down Q8_0 GEMV (in={_n}) + routed add + HC expand (n_hc=4); grid=(out_dim/4)",
    )
    for _n in shapes.MOE_MID_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_batch_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_batch_f32(n)),
        group="q8",
        targets=("metal",),
        notes=f"out[n_tok][out_dim] = Q8_0 w[out_dim][{_n}] . x[n_tok]; {BATCH_TOKS} tokens per tile; grid=(out_dim/{BATCH_ROWS}, ceil(n_tok/{BATCH_TOKS}))",
    )
    for _n in shapes.Q8_0_GEMM_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_grouped_batch_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_grouped_batch_f32(n)),
        group="q8",
        targets=("metal",),
        notes=f"attn_output_a batch: out[t,g,:] = Q8_0 w[g][out_dim][{_n}] . x[t,g]; {BATCH_TOKS} tokens per tile; grid=(out_dim/{BATCH_ROWS}, n_groups, ceil(n_tok/{BATCH_TOKS}))",
    )
    for _n in shapes.Q8_0_GROUPED_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_batch4_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_batch_f32(n, toks=4)),
        group="q8",
        targets=("metal",),
        notes=f"same with 4 tokens per tile (n_tok <= 4); grid=(out_dim/{BATCH_ROWS}, ceil(n_tok/4))",
    )
    for _n in shapes.Q8_0_GEMM_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_f32_{_n}",
        build=(lambda n=_n: q8_0_gemv_f32(n)),
        variants={"r2": (lambda n=_n: q8_0_gemv_f32(n, rows=2)),
                  "r8": (lambda n=_n: q8_0_gemv_f32(n, rows=8)),
                  "t2": (lambda n=_n: q8_0_gemv_f32(n, threads=2 * f32_threads(n)))},
        tune_group="q8_gemv_f32",
        group="q8",
        targets=("metal",),
        notes=f"out[out_dim] = Q8_0 w[out_dim][{_n}] . f32 x; w viewed as u16; in_dim={_n}; out_dim % {F32_ROWS} == 0; grid=(out_dim/{F32_ROWS})",
    )
    for _n in shapes.Q8_0_GEMV_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_preq_{_n}",
        build=(lambda n=_n: q8_0_gemv(n)),
        group="q8",
        notes=f"out[out_dim] = Q8_0 w[out_dim][{_n}] . q8-prequant x; in_dim={_n}; out_dim % {ROWS} == 0; grid=(out_dim/{ROWS})",
    )
    for _n in shapes.Q8_0_GEMV_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_preq_lanes_{_n}",
        build=(lambda n=_n: q8_0_gemv_lanes(n)),
        group="q8",
        targets=("c",),
        notes=f"as tl_q8_0_gemv_preq_{_n}, CPU-oriented lane-accumulator formulation",
    )
    for _n in shapes.Q8_0_GEMV_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemv_preq_simd_{_n}",
        build=(lambda n=_n: q8_0_gemv_simd(n)),
        group="q8",
        targets=("c",),
        notes=f"as tl_q8_0_gemv_preq_{_n}, hand-written AVX2/NEON block primitive via T.import_source",
    )
    for _n in shapes.Q8_0_GEMV_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_q8_0_gemm_preq_simd_{_n}",
        build=(lambda n=_n: q8_0_gemm_simd(n)),
        group="q8",
        targets=("c",),
        notes=f"out[n_tok][out_dim] = Q8_0 w[out_dim][{_n}] . q8-prequant x[n_tok]; tile {ROWS}x{TOKS}; out_dim % {ROWS} == 0",
    )
    for _n in shapes.Q8_0_GEMV_IN_DIMS
]
