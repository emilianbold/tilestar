"""Routed MoE decode kernels (DeepSeek V4 Flash: 6 of 256 experts, 1 token).

References in ds4:
  Metal kernel_mul_mv_id_iq2_xxs_pair_swiglu_pack2_overlap_f32 (metal/moe.metal)
  CUDA  moe_gate_up_mid kernels (ds4_cuda.cu ~20576)

Gate/up for the selected experts, IQ2_XXS weights, f32 activation, with the
SwiGLU + route-weight epilogue fused:
    g = Wg[e] . x,  u = Wu[e] . x
    gate[s] = g, up[s] = u
    mid[s]  = silu(min(g, c)) * clamp(u, -c, c) * route_w[s]   (c > 1e-6, else no clamp)
for slot s with expert id e = ids[s]. Grid: (mid_dim / rows, n_used).
"""

from __future__ import annotations

import tilelang.language as T

from tlk import KernelSpec
import shapes
from quant import iq2_xxs as iq2
from quant import q2_k


def moe_gate_up_iq2_xxs_swiglu(in_dim: int, rows: int = 4, threads: int = 128, rows_per_group: int = 4, batch: bool = False):
    """One fragment element per 32-weight sub-block (like the hand-written
    kernel: one lane per sub-block, the four 8-weight groups unrolled); the
    IQ2 grid (as f32) and sign tables are staged in threadgroup memory.
    Threads are (row_groups x nsb): each thread stages one 32-float
    activation slice and walks rows_per_group rows of both matrices with
    it; the fragment layout is annotated accordingly.

    M5 Max, cold weights, GPU time vs kernel_mul_mv_id_iq2_xxs_pair_swiglu_
    pack2_overlap_f32 (66.5 us): this configuration 73.5 us (1.11x).
    Timing experiments that removed one load stream at a time put the
    cost in the three non-weight streams (activation slice, grid rows,
    sign bytes: ~15 us each), not in the weights or the ALU work; rows x
    threads sweeps from (2,64) to (16,512) all land within 73-88 us, and
    none of the structural variants (tables in device memory, u32 tables,
    hand-kernel simdgroup mapping, forced unrolls) beat this one."""
    assert in_dim % iq2.QK_K == 0 and rows % rows_per_group == 0
    blocks = in_dim // iq2.QK_K
    nsb = in_dim // 32            # 32-weight sub-blocks per row (fragment width)
    row_groups = rows // rows_per_group
    u16_per_row = blocks * iq2.U16_PER_BLOCK
    out_dim = T.dynamic("out_dim")      # expert_mid_dim
    n_expert = T.dynamic("n_expert")    # experts stored (256)
    n_used = T.dynamic("n_used")        # selected experts (6)

    @T.macro
    def sub_block_dot(part, w_at, r, base, blk, xl, grid_s, ksigns_s):
        """part[0..3] += the four 8-weight group dots of sub-block (r, base);
        returns nothing, scale is applied by the caller. Four independent
        accumulation chains per matrix keep the FMA pipeline busy."""
        q0 = w_at(r, base)
        q1 = w_at(r, base + 1)
        q2 = w_at(r, base + 2)
        q3 = w_at(r, base + 3)
        aux32 = T.Cast(T.uint32, q2) | (T.Cast(T.uint32, q3) << 16)
        for l in T.unroll(4):
            gidx = T.Cast(T.int32, (T.if_then_else(l < 2, q0, q1) >> (8 * (l % 2))) & 0xFF)
            signs = iq2.byte_of(ksigns_s, T.Cast(T.int32, (aux32 >> (7 * l)) & 127))
            gv = T.alloc_local((8,), T.float32)
            for j in T.vectorized(8):
                gv[j] = grid_s[gidx * 8 + j]
            part[l] = 0.0
            for j in T.unroll(8):
                part[l] += xl[l * 8 + j] * iq2.signed_f32(gv[j], signs, j)
        part[4] = (part[0] + part[1] + part[2] + part[3]) * (
            T.Cast(T.float32, T.reinterpret(T.float16, w_at(r, blk * iq2.U16_PER_BLOCK)))
            * (0.5 + T.Cast(T.float32, aux32 >> 28))
        )

    @T.macro
    def accumulate(accg, accu, sumg, sumu, wg_at, wu_at, x_at, grid_s, ksigns_s):
        # One thread per 32-weight sub-block column: stage its 32 activations
        # once, then walk the tile's rows for gate and up together (reuse
        # across rows and two matrices, as the hand-written kernel does).
        for rg, sb in T.Parallel(row_groups, nsb):
            xl = T.alloc_local((32,), T.float32)
            pg = T.alloc_local((5,), T.float32)
            pu = T.alloc_local((5,), T.float32)
            for j in T.vectorized(32):
                xl[j] = x_at(sb * 32 + j)
            blk = sb // 8
            ib = sb % 8
            base = blk * iq2.U16_PER_BLOCK + 1 + ib * 4
            for r in T.serial(rows_per_group):
                sub_block_dot(pg, wg_at, rg * rows_per_group + r, base, blk, xl, grid_s, ksigns_s)
                sub_block_dot(pu, wu_at, rg * rows_per_group + r, base, blk, xl, grid_s, ksigns_s)
                accg[rg * rows_per_group + r, sb] = pg[4]
                accu[rg * rows_per_group + r, sb] = pu[4]
        T.reduce_sum(accg, sumg, dim=1)
        T.reduce_sum(accu, sumu, dim=1)

    if batch:
        n_tok = T.dynamic("n_tok")

        @T.prim_func
        def kernel_batch(
            gate_out: T.Tensor((n_tok, n_used, out_dim), T.float32),
            up_out: T.Tensor((n_tok, n_used, out_dim), T.float32),
            mid_out: T.Tensor((n_tok, n_used, out_dim), T.float32),
            wg: T.Tensor((n_expert, out_dim, u16_per_row), T.uint16),
            wu: T.Tensor((n_expert, out_dim, u16_per_row), T.uint16),
            x: T.Tensor((n_tok, in_dim), T.float32),
            ids: T.Tensor((n_tok, n_used), T.int32),
            route_w: T.Tensor((n_tok, n_used), T.float32),
            grid: T.Tensor((iq2.GRID_FLOATS,), T.float32),
            ksigns: T.Tensor((iq2.KSIGNS_WORDS,), T.uint32),
            clamp: T.float32,
        ):
            with T.Kernel(out_dim // rows, n_used, n_tok, threads=threads) as (bx, s, t):
                n_expert_ = n_expert  # noqa: F841
                u16_per_row_ = u16_per_row  # noqa: F841
                in_dim_ = in_dim  # noqa: F841
                accg = T.alloc_fragment((rows, nsb), T.float32)
                accu = T.alloc_fragment((rows, nsb), T.float32)
                sumg = T.alloc_fragment((rows,), T.float32)
                sumu = T.alloc_fragment((rows,), T.float32)
                grid_s = T.alloc_shared((iq2.GRID_FLOATS,), T.float32)
                ksigns_s = T.alloc_shared((iq2.KSIGNS_WORDS,), T.uint32)
                T.annotate_layout({
                    accg: T.Fragment(accg.shape, forward_fn=lambda r, sb: ((r // rows_per_group) * nsb + sb, r % rows_per_group)),
                    accu: T.Fragment(accu.shape, forward_fn=lambda r, sb: ((r // rows_per_group) * nsb + sb, r % rows_per_group)),
                })
                T.copy(grid, grid_s)
                T.copy(ksigns, ksigns_s)
                e = ids[t, s]
                accumulate(accg, accu, sumg, sumu,
                           lambda r, k: wg[e, bx * rows + r, k],
                           lambda r, k: wu[e, bx * rows + r, k],
                           lambda i: x[t, i], grid_s, ksigns_s)
                for r in T.Parallel(rows):
                    g = sumg[r] * 0.25
                    u = sumu[r] * 0.25
                    gate_out[t, s, bx * rows + r] = g
                    up_out[t, s, bx * rows + r] = u
                    gc = T.if_then_else(clamp > 1.0e-6, T.min(g, clamp), g)
                    uc = T.if_then_else(clamp > 1.0e-6, T.max(T.min(u, clamp), -clamp), u)
                    mid_out[t, s, bx * rows + r] = gc / (1.0 + T.exp(-gc)) * uc * route_w[t, s]

        return kernel_batch

    @T.prim_func
    def kernel(
        gate_out: T.Tensor((n_used, out_dim), T.float32),
        up_out: T.Tensor((n_used, out_dim), T.float32),
        mid_out: T.Tensor((n_used, out_dim), T.float32),
        wg: T.Tensor((n_expert, out_dim, u16_per_row), T.uint16),
        wu: T.Tensor((n_expert, out_dim, u16_per_row), T.uint16),
        x: T.Tensor((in_dim,), T.float32),
        ids: T.Tensor((n_used,), T.int32),
        route_w: T.Tensor((n_used,), T.float32),
        grid: T.Tensor((iq2.GRID_FLOATS,), T.float32),
        ksigns: T.Tensor((iq2.KSIGNS_WORDS,), T.uint32),
        clamp: T.float32,
    ):
        with T.Kernel(out_dim // rows, n_used, threads=threads) as (bx, s):
            # The parser only binds closure names the body mentions; these
            # are otherwise annotation-only.
            n_expert_ = n_expert  # noqa: F841
            u16_per_row_ = u16_per_row  # noqa: F841
            in_dim_ = in_dim  # noqa: F841
            accg = T.alloc_fragment((rows, nsb), T.float32)
            accu = T.alloc_fragment((rows, nsb), T.float32)
            sumg = T.alloc_fragment((rows,), T.float32)
            sumu = T.alloc_fragment((rows,), T.float32)
            grid_s = T.alloc_shared((iq2.GRID_FLOATS,), T.float32)
            ksigns_s = T.alloc_shared((iq2.KSIGNS_WORDS,), T.uint32)
            # thread = (row group, sub-block); local index = row within group
            T.annotate_layout({
                accg: T.Fragment(accg.shape, forward_fn=lambda r, sb: ((r // rows_per_group) * nsb + sb, r % rows_per_group)),
                accu: T.Fragment(accu.shape, forward_fn=lambda r, sb: ((r // rows_per_group) * nsb + sb, r % rows_per_group)),
            })
            T.copy(grid, grid_s)
            T.copy(ksigns, ksigns_s)
            e = ids[s]
            accumulate(accg, accu, sumg, sumu,
                       lambda r, k: wg[e, bx * rows + r, k],
                       lambda r, k: wu[e, bx * rows + r, k],
                       lambda i: x[i], grid_s, ksigns_s)
            for r in T.Parallel(rows):
                g = sumg[r] * 0.25
                u = sumu[r] * 0.25
                gate_out[s, bx * rows + r] = g
                up_out[s, bx * rows + r] = u
                gc = T.if_then_else(clamp > 1.0e-6, T.min(g, clamp), g)
                uc = T.if_then_else(clamp > 1.0e-6, T.max(T.min(u, clamp), -clamp), u)
                mid_out[s, bx * rows + r] = gc / (1.0 + T.exp(-gc)) * uc * route_w[s]

    return kernel


def moe_down_q2_k_sum(in_dim: int, rows: int = 4, threads: int = 128, rows_per_group: int = 4, batch: bool = False):
    """Routed down projection, Q2_K weights, summed over the selected experts
    (kernel_mul_mv_id_q2_K_sum6_f32):
        out[d] = sum_s Wd[ids[s]][d] . mid[s]
    where mid[s] already carries the route weight. One thread per 16-weight
    sub-block column of a row group, serial over slots and rows; scale and
    min are applied per sub-block as d*sc*(q.x) - dmin*m*sum(x)."""
    assert in_dim % q2_k.QK_K == 0 and rows % rows_per_group == 0
    blocks = in_dim // q2_k.QK_K
    nsb = in_dim // 16
    row_groups = rows // rows_per_group
    assert threads == row_groups * nsb
    u32_per_row = blocks * q2_k.U32_PER_BLOCK
    out_dim = T.dynamic("out_dim")
    n_expert = T.dynamic("n_expert")
    n_used = T.dynamic("n_used")

    @T.macro
    def sub_block_dot(part, w_at, r, blk, is_, xl, xsum):
        """part[0] = contribution of sub-block (r, blk, is_) with xl staged."""
        shift = q2_k.sub_block_shift(is_)
        sc = q2_k.scale_byte(w_at, blk, is_)
        d, dmin = q2_k.d_dmin(w_at, blk)
        part[1] = 0.0
        for k in T.unroll(4):
            word = w_at(q2_k.sub_block_qs_word(blk, is_, k))
            for b in T.unroll(4):
                q = T.Cast(T.float32, (word >> (8 * b + shift)) & 3)
                part[1] += xl[4 * k + b] * q
        part[0] = d * T.Cast(T.float32, sc & 0xF) * part[1] - dmin * T.Cast(T.float32, sc >> 4) * xsum

    if batch:
        n_tok = T.dynamic("n_tok")

        @T.prim_func
        def kernel_batch(
            out: T.Tensor((n_tok, out_dim), T.float32),
            wd: T.Tensor((n_expert, out_dim, u32_per_row), T.uint32),
            mid: T.Tensor((n_tok, n_used, in_dim), T.float32),
            ids: T.Tensor((n_tok, n_used), T.int32),
        ):
            with T.Kernel(out_dim // rows, n_tok, threads=threads) as (bx, t):
                n_expert_ = n_expert  # noqa: F841  (annotation-only names)
                u32_per_row_ = u32_per_row  # noqa: F841
                in_dim_ = in_dim  # noqa: F841
                acc = T.alloc_fragment((rows, nsb), T.float32)
                rowsum = T.alloc_fragment((rows,), T.float32)
                T.annotate_layout({
                    acc: T.Fragment(acc.shape, forward_fn=lambda r, sb: ((r // rows_per_group) * nsb + sb, r % rows_per_group)),
                })
                T.clear(acc)
                for rg, sb in T.Parallel(row_groups, nsb):
                    xl = T.alloc_local((16,), T.float32)
                    part = T.alloc_local((2,), T.float32)
                    blk = sb // q2_k.SUB_BLOCKS
                    is_ = sb % q2_k.SUB_BLOCKS
                    for s in T.serial(n_used):
                        e = ids[t, s]
                        for j in T.vectorized(16):
                            xl[j] = mid[t, s, sb * 16 + j]
                        xsum = T.alloc_var(T.float32)
                        xsum = 0.0
                        for j in T.unroll(16):
                            xsum += xl[j]
                        for r in T.serial(rows_per_group):
                            sub_block_dot(part, lambda k: wd[e, bx * rows + rg * rows_per_group + r, k], r, blk, is_, xl, xsum)
                            acc[rg * rows_per_group + r, sb] += part[0]
                T.reduce_sum(acc, rowsum, dim=1)
                for r in T.Parallel(rows):
                    out[t, bx * rows + r] = rowsum[r]

        return kernel_batch

    @T.prim_func
    def kernel(
        out: T.Tensor((out_dim,), T.float32),
        wd: T.Tensor((n_expert, out_dim, u32_per_row), T.uint32),
        mid: T.Tensor((n_used, in_dim), T.float32),
        ids: T.Tensor((n_used,), T.int32),
    ):
        with T.Kernel(out_dim // rows, threads=threads) as bx:
            n_expert_ = n_expert  # noqa: F841  (annotation-only names)
            u32_per_row_ = u32_per_row  # noqa: F841
            in_dim_ = in_dim  # noqa: F841
            acc = T.alloc_fragment((rows, nsb), T.float32)
            rowsum = T.alloc_fragment((rows,), T.float32)
            T.annotate_layout({
                acc: T.Fragment(acc.shape, forward_fn=lambda r, sb: ((r // rows_per_group) * nsb + sb, r % rows_per_group)),
            })
            T.clear(acc)
            for rg, sb in T.Parallel(row_groups, nsb):
                xl = T.alloc_local((16,), T.float32)
                part = T.alloc_local((2,), T.float32)
                blk = sb // q2_k.SUB_BLOCKS
                is_ = sb % q2_k.SUB_BLOCKS
                for s in T.serial(n_used):
                    e = ids[s]
                    for j in T.vectorized(16):
                        xl[j] = mid[s, sb * 16 + j]
                    xsum = T.alloc_var(T.float32)
                    xsum = 0.0
                    for j in T.unroll(16):
                        xsum += xl[j]
                    for r in T.serial(rows_per_group):
                        sub_block_dot(part, lambda k: wd[e, bx * rows + rg * rows_per_group + r, k], r, blk, is_, xl, xsum)
                        acc[rg * rows_per_group + r, sb] += part[0]
            T.reduce_sum(acc, rowsum, dim=1)
            for r in T.Parallel(rows):
                out[bx * rows + r] = rowsum[r]

    return kernel


KERNELS = [
    KernelSpec(
        name=f"tl_moe_down_q2_k_sum_batch_{_n}",
        build=(lambda n=_n: moe_down_q2_k_sum(n, batch=True)),
        group="moe",
        targets=("metal",),
        notes=f"routed down Q2_K summed over selected experts, n_tok tokens; in_dim={_n}; grid=(out_dim/4, n_tok)",
    )
    for _n in shapes.MOE_MID_DIMS
] + [
    KernelSpec(
        name=f"tl_moe_gate_up_iq2_xxs_swiglu_batch_{_n}",
        build=(lambda n=_n: moe_gate_up_iq2_xxs_swiglu(n, batch=True)),
        group="moe",
        targets=("metal",),
        notes=f"routed gate/up IQ2_XXS + SwiGLU*route_w, n_tok tokens; in_dim={_n}; grid=(mid_dim/4, n_used, n_tok)",
    )
    for _n in shapes.MOE_IN_DIMS
] + [
    KernelSpec(
        name=f"tl_moe_down_q2_k_sum_{_n}",
        build=(lambda n=_n: moe_down_q2_k_sum(n)),
        variants={"r8": (lambda n=_n: moe_down_q2_k_sum(n, rows=8, threads=128, rows_per_group=8)),
                  "r8g4": (lambda n=_n: moe_down_q2_k_sum(n, rows=8, threads=256, rows_per_group=4)),
                  "r2": (lambda n=_n: moe_down_q2_k_sum(n, rows=2, threads=128, rows_per_group=2))},
        tune_group="moe_down",
        group="moe",
        targets=("metal",),
        notes=f"routed down Q2_K x f32 summed over selected experts; in_dim={_n}; grid=(out_dim/4)",
    )
    for _n in shapes.MOE_MID_DIMS
] + [
    KernelSpec(
        name=f"tl_moe_gate_up_iq2_xxs_swiglu_{_n}",
        build=(lambda n=_n: moe_gate_up_iq2_xxs_swiglu(n)),
        variants={"r8": (lambda n=_n: moe_gate_up_iq2_xxs_swiglu(n, rows=8, threads=128, rows_per_group=8)),
                  "r8g4": (lambda n=_n: moe_gate_up_iq2_xxs_swiglu(n, rows=8, threads=256, rows_per_group=4)),
                  "r2": (lambda n=_n: moe_gate_up_iq2_xxs_swiglu(n, rows=2, threads=128, rows_per_group=2))},
        tune_group="moe_gate_up",
        group="moe",
        targets=("metal",),
        notes=f"routed gate/up IQ2_XXS x f32 + SwiGLU*route_w; in_dim={_n}; grid=(mid_dim/4, n_used)",
    )
    for _n in shapes.MOE_IN_DIMS
]


# ---------------------------------------------------------------------------
# Prefill tier (>= 512 tokens): per-expert packed GEMMs on cooperative tensors
# ---------------------------------------------------------------------------

PACKED_BM, PACKED_BN, PACKED_BK, PACKED_THREADS = 32, 64, 32, 128


def moe_packed_gemm_iq2_xxs(in_dim: int, bm: int = PACKED_BM, bn: int = PACKED_BN, bk: int = PACKED_BK, threads: int = PACKED_THREADS):
    """kernel_mul_mm_id_iq2_xxs_mpp_packed: one threadgroup per (32-row work
    item of one expert, 64 output rows). ds4's map0 kernel lists the pair
    rows (token*6 + slot) routed to each expert (ids[e][0..counts[e])) and
    emits work items (expert, r1) in steps of 32; pack_rhs copies the
    activations of those rows into rhs[offsets[e] + i][in_dim] (f16). This
    kernel: A = the 32 packed rows, B = 64 expert rows dequantised to f16 per
    32-wide K step (one IQ2_XXS sub-block; grid / ksigns tables in shared
    memory), T.gemm into a cooperative-tensor accumulator, then the tile is
    scattered to dst[pair_row][r0 + col] for the nr1 valid rows. The packed
    buffer carries 32 rows of slack so the A copy of the last tile may read
    past the valid rows (the host adds it). Grid: (work_cap, ne0 / 64)."""
    assert in_dim % iq2.QK_K == 0 and bk == 32 and bn * 4 == threads * 2
    blocks = in_dim // iq2.QK_K
    u16_per_row = blocks * iq2.U16_PER_BLOCK
    ne0 = T.dynamic("ne0")
    n_expert = T.dynamic("n_expert")
    n_tok = T.dynamic("n_tok")
    pair_rows = T.dynamic("pair_rows")
    rhs_rows = T.dynamic("rhs_rows")
    work_len = T.dynamic("work_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((pair_rows, ne0), T.float32),
        w: T.Tensor((n_expert, ne0, u16_per_row), T.uint16),
        rhs: T.Tensor((rhs_rows, in_dim), T.float16),
        offsets: T.Tensor((n_expert,), T.int32),
        counts: T.Tensor((n_expert,), T.int32),
        ids: T.Tensor((n_expert, n_tok), T.int32),
        work: T.Tensor((work_len,), T.int32),
        grid: T.Tensor((iq2.GRID_WORDS,), T.uint32),   # the codebook as packed bytes (2 KB in shared)
        ksigns: T.Tensor((iq2.KSIGNS_WORDS,), T.uint32),
        work_cap: T.int32,
    ):
        with T.Kernel(work_cap, ne0 // bn, threads=threads) as (bx, by):
            u16_per_row_ = u16_per_row  # noqa: F841  (annotation-only names)
            blocks_ = blocks  # noqa: F841
            n_expert_ = n_expert  # noqa: F841
            n_tok_ = n_tok  # noqa: F841
            pair_rows_ = pair_rows  # noqa: F841
            rhs_rows_ = rhs_rows  # noqa: F841
            work_len_ = work_len  # noqa: F841
            a_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            b_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            c_s = T.alloc_shared((bm, bn), T.float32, scope="shared")
            grid_s = T.alloc_shared((iq2.GRID_WORDS,), T.uint32, scope="shared")
            ksigns_s = T.alloc_shared((iq2.KSIGNS_WORDS,), T.uint32, scope="shared")
            acc = T.alloc_fragment((bm, bn), T.float32)
            if bx < work[0]:
                e = work[2 + 2 * bx]
                r1 = work[3 + 2 * bx]
                r0 = by * bn
                nr1 = T.min(bm, counts[e] - r1)
                row_base = offsets[e] + r1
                T.copy(grid, grid_s)
                T.copy(ksigns, ksigns_s)
                T.clear(acc)
                for ko in T.Pipelined(in_dim // bk, num_stages=0):
                    T.copy(rhs[row_base, ko * bk], a_s)
                    # two 8-weight groups (16 weights) of one expert row per thread,
                    # like the hand kernel's half4x4 dequant
                    for n, h in T.Parallel(bn, 2):
                        blk = (ko * bk) // iq2.QK_K
                        ib = ((ko * bk) % iq2.QK_K) // 32
                        base = blk * iq2.U16_PER_BLOCK + 1 + ib * 4
                        qg = w[e, r0 + n, base + h]          # grid indices of groups 2h, 2h+1
                        q2 = w[e, r0 + n, base + 2]
                        q3 = w[e, r0 + n, base + 3]
                        aux32 = T.Cast(T.uint32, q2) | (T.Cast(T.uint32, q3) << 16)
                        scale = T.Cast(T.float32, T.reinterpret(T.float16, w[e, r0 + n, blk * iq2.U16_PER_BLOCK])) * (0.5 + T.Cast(T.float32, aux32 >> 28)) * 0.25
                        for g in T.unroll(2):
                            l = 2 * h + g
                            gidx = T.Cast(T.int32, (qg >> (8 * g)) & 0xFF)
                            signs = iq2.byte_of(ksigns_s, T.Cast(T.int32, (aux32 >> (7 * l)) & 127))
                            gw0 = grid_s[gidx * 2]
                            gw1 = grid_s[gidx * 2 + 1]
                            for j in T.unroll(8):
                                mag = T.Cast(T.float32, ((T.if_then_else(j < 4, gw0, gw1)) >> (8 * (j % 4))) & 0xFF)
                                b_s[l * 8 + j, n] = T.Cast(T.float16, scale * iq2.signed_f32(mag, signs, j))
                    T.gemm(a_s, b_s, acc)
                T.copy(acc, c_s)
                T.sync_threads()
                for i in T.Parallel(bm * (bn // 4)):
                    row = i // (bn // 4)
                    c4 = (i % (bn // 4)) * 4
                    if row < nr1:
                        pr = ids[e, r1 + row]
                        for j in T.vectorized(4):
                            dst[pr, r0 + c4 + j] = c_s[row, c4 + j]

    return kernel


def moe_packed_gemm_q2_k(in_dim: int, bm: int = PACKED_BM, bn: int = PACKED_BN, bk: int = PACKED_BK, threads: int = PACKED_THREADS):
    """kernel_mul_mm_id_q2_K_mpp_packed (the routed down projection): same
    structure with Q2_K expert rows, two 16-weight sub-blocks per K step
    (value = d*(sc&15)*q - dmin*(sc>>4)). Grid: (work_cap, ne0 / 64)."""
    assert in_dim % q2_k.QK_K == 0 and bk == 32 and bn * 2 == threads
    blocks = in_dim // q2_k.QK_K
    u32_per_row = blocks * q2_k.U32_PER_BLOCK
    ne0 = T.dynamic("ne0")
    n_expert = T.dynamic("n_expert")
    n_tok = T.dynamic("n_tok")
    pair_rows = T.dynamic("pair_rows")
    rhs_rows = T.dynamic("rhs_rows")
    work_len = T.dynamic("work_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((pair_rows, ne0), T.float32),
        w: T.Tensor((n_expert, ne0, u32_per_row), T.uint32),
        rhs: T.Tensor((rhs_rows, in_dim), T.float16),
        offsets: T.Tensor((n_expert,), T.int32),
        counts: T.Tensor((n_expert,), T.int32),
        ids: T.Tensor((n_expert, n_tok), T.int32),
        work: T.Tensor((work_len,), T.int32),
        work_cap: T.int32,
    ):
        with T.Kernel(work_cap, ne0 // bn, threads=threads) as (bx, by):
            u32_per_row_ = u32_per_row  # noqa: F841  (annotation-only names)
            blocks_ = blocks  # noqa: F841
            n_expert_ = n_expert  # noqa: F841
            n_tok_ = n_tok  # noqa: F841
            pair_rows_ = pair_rows  # noqa: F841
            rhs_rows_ = rhs_rows  # noqa: F841
            work_len_ = work_len  # noqa: F841
            a_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            b_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            c_s = T.alloc_shared((bm, bn), T.float32, scope="shared")
            acc = T.alloc_fragment((bm, bn), T.float32)
            if bx < work[0]:
                e = work[2 + 2 * bx]
                r1 = work[3 + 2 * bx]
                r0 = by * bn
                nr1 = T.min(bm, counts[e] - r1)
                row_base = offsets[e] + r1
                T.clear(acc)
                for ko in T.Pipelined(in_dim // bk, num_stages=0):
                    T.copy(rhs[row_base, ko * bk], a_s)
                    # one 16-weight sub-block of one expert row per thread
                    for n, h in T.Parallel(bn, 2):
                        blk = (ko * bk) // q2_k.QK_K
                        is_ = ((ko * bk) % q2_k.QK_K) // 16 + h
                        shift = q2_k.sub_block_shift(is_)
                        sc_word = w[e, r0 + n, blk * q2_k.U32_PER_BLOCK + is_ // 4]
                        sc = T.Cast(T.int32, (sc_word >> (8 * (is_ % 4))) & 0xFF)
                        dm = w[e, r0 + n, blk * q2_k.U32_PER_BLOCK + 20]
                        d = T.Cast(T.float32, T.reinterpret(T.float16, T.Cast(T.uint16, dm & 0xFFFF))) * T.Cast(T.float32, sc & 0xF)
                        mn = T.Cast(T.float32, T.reinterpret(T.float16, T.Cast(T.uint16, dm >> 16))) * T.Cast(T.float32, sc >> 4)
                        for k in T.unroll(4):
                            word = w[e, r0 + n, q2_k.sub_block_qs_word(blk, is_, k)]
                            for b in T.unroll(4):
                                q = T.Cast(T.float32, (word >> (8 * b + shift)) & 3)
                                b_s[h * 16 + 4 * k + b, n] = T.Cast(T.float16, d * q - mn)
                    T.gemm(a_s, b_s, acc)
                T.copy(acc, c_s)
                T.sync_threads()
                for i in T.Parallel(bm * (bn // 4)):
                    row = i // (bn // 4)
                    c4 = (i % (bn // 4)) * 4
                    if row < nr1:
                        pr = ids[e, r1 + row]
                        for j in T.vectorized(4):
                            dst[pr, r0 + c4 + j] = c_s[row, c4 + j]

    return kernel


KERNELS += [
    KernelSpec(name=f"tl_moe_packed_gemm_iq2_xxs_{_n}", build=(lambda n=_n: moe_packed_gemm_iq2_xxs(n)), group="moe", targets=("metal",),
               notes=f"prefill routed gate/up: per (32-row expert work item, 64 rows) GEMM of packed f16 rows x IQ2_XXS expert rows (K={_n}), scattered to dst[pair_row]; grid=(work_cap, ne0/64)")
    for _n in shapes.MOE_IN_DIMS
] + [
    KernelSpec(name=f"tl_moe_packed_gemm_q2_k_{_n}", build=(lambda n=_n: moe_packed_gemm_q2_k(n)), group="moe", targets=("metal",),
               notes=f"prefill routed down: per (32-row expert work item, 64 rows) GEMM of packed f16 rows x Q2_K expert rows (K={_n}); grid=(work_cap, ne0/64)")
    for _n in shapes.MOE_MID_DIMS
]


def moe_map0(n_used: int = 6, threads: int = 256):
    """kernel_mul_mm_id_map0_ne20_6: one thread per expert (n_expert <= threads).
    hids[e][0..counts[e]) = the pair rows token*n_used + slot routed to e (token
    order), counts[e], and the 32-row work items (expert, tile*32) packed
    expert-major into work[2 + 2*i], work[3 + 2*i] with work[0] = their count.
    The routing ids are staged `threads` tokens at a time in shared memory.
    Grid: (1,)."""
    n_tok = T.dynamic("n_tok")
    n_expert = T.dynamic("n_expert")
    work_len = T.dynamic("work_len")

    @T.prim_func
    def kernel(
        ids: T.Tensor((n_tok, n_used), T.int32),
        counts: T.Tensor((n_expert,), T.int32),
        hids: T.Tensor((n_expert, n_tok), T.int32),
        work: T.Tensor((work_len,), T.int32),
    ):
        with T.Kernel(1, threads=threads):
            work_len_ = work_len  # noqa: F841
            sids = T.alloc_shared((threads, n_used), T.int32, scope="shared")
            tiles_s = T.alloc_shared((threads,), T.int32, scope="shared")
            tid = T.get_thread_binding()
            n_all = T.alloc_var(T.int32)
            n_all = 0
            for i0 in T.serial(T.ceildiv(n_tok, threads)):
                if i0 * threads + tid < n_tok:
                    for s in T.unroll(n_used):
                        sids[tid, s] = ids[i0 * threads + tid, s]
                T.sync_threads()
                for t in T.serial(T.min(threads, n_tok - i0 * threads)):
                    sel = T.alloc_var(T.int32)
                    sel = -1
                    for s in T.unroll(n_used):
                        if sids[t, s] == tid:
                            sel = s
                    if sel >= 0 and tid < n_expert:
                        hids[tid, n_all] = (i0 * threads + t) * n_used + sel
                        n_all += 1
                T.sync_threads()
            if tid < n_expert:
                counts[tid] = n_all
            n_tiles = T.if_then_else(tid < n_expert, (n_all + 31) // 32, 0)
            tiles_s[tid] = n_tiles
            # inclusive prefix sum of the tile counts (Hillis-Steele)
            for d in T.serial(8):
                step = 1 << d
                v = T.alloc_var(T.int32)
                v = T.if_then_else(tid >= step, tiles_s[T.max(tid - step, 0)], 0)
                T.sync_threads()
                tiles_s[tid] = tiles_s[tid] + v
                T.sync_threads()
            tile_base = tiles_s[tid] - n_tiles
            for k in T.serial(n_tiles):
                work[2 + 2 * (tile_base + k)] = tid
                work[3 + 2 * (tile_base + k)] = k * 32
            if tid == threads - 1:
                work[0] = tiles_s[threads - 1]
                work[1] = 0

    return kernel


def moe_packed_offsets(threads: int = 256):
    """kernel_moe_packed_offsets: offsets = exclusive prefix sum of counts
    (n_expert <= threads). Grid: (1,)."""
    n_expert = T.dynamic("n_expert")
    off_len = T.dynamic("off_len")

    @T.prim_func
    def kernel(
        offsets: T.Tensor((off_len,), T.int32),
        counts: T.Tensor((n_expert,), T.int32),
    ):
        with T.Kernel(1, threads=threads):
            off_len_ = off_len  # noqa: F841
            sc = T.alloc_shared((threads,), T.int32, scope="shared")
            tid = T.get_thread_binding()
            own = T.if_then_else(tid < n_expert, counts[T.min(tid, n_expert - 1)], 0)
            sc[tid] = own
            for d in T.serial(8):
                step = 1 << d
                v = T.alloc_var(T.int32)
                v = T.if_then_else(tid >= step, sc[T.max(tid - step, 0)], 0)
                T.sync_threads()
                sc[tid] = sc[tid] + v
                T.sync_threads()
            if tid < n_expert:
                offsets[tid] = sc[tid] - own

    return kernel


def moe_pack_rhs(src_f16: bool, threads: int = 128):
    """kernel_moe_pack_rhs_{f32,f16}: one threadgroup per packed row:
    row -> work item row/32, local row ri = item.y + row%32 (skipped past the
    expert's count); dst[offsets[e] + ri][:] = f16(src[token*src_tok_stride +
    slot*src_slot_stride][:]) with (token, slot) from the pair row
    hids[e][ri]. K % 4 == 0, strides in elements, multiples of 4.
    Grid: (32 * work_cap,)."""
    src_dtype = T.float16 if src_f16 else T.float32
    n_expert = T.dynamic("n_expert")
    n_tok = T.dynamic("n_tok")
    work_len = T.dynamic("work_len")
    src_len = T.dynamic("src_len")
    dst_rows = T.dynamic("dst_rows")
    k_dim = T.dynamic("k_dim")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_rows, k_dim), T.float16),
        src: T.Tensor((src_len,), src_dtype),
        offsets: T.Tensor((n_expert,), T.int32),
        counts: T.Tensor((n_expert,), T.int32),
        hids: T.Tensor((n_expert, n_tok), T.int32),
        work: T.Tensor((work_len,), T.int32),
        n_used: T.int32,
        src_tok_stride4: T.int32,    # strides in float4 / half4 units
        src_slot_stride4: T.int32,
        work_cap: T.int32,
    ):
        with T.Kernel(32 * work_cap, threads=threads) as row:
            n_expert_ = n_expert  # noqa: F841
            n_tok_ = n_tok  # noqa: F841
            work_len_ = work_len  # noqa: F841
            src_len_ = src_len  # noqa: F841
            dst_rows_ = dst_rows  # noqa: F841
            src_dtype_ = src_dtype  # noqa: F841
            wi = row // 32
            if wi < work[0]:
                e = work[2 + 2 * wi]
                ri = work[3 + 2 * wi] + row % 32
                if ri < counts[e]:
                    pr = offsets[e] + ri
                    pid = hids[e, ri]
                    base4 = (pid // n_used) * src_tok_stride4 + (pid % n_used) * src_slot_stride4
                    for i in T.Parallel(threads):
                        for c4 in T.serial(T.ceildiv(k_dim // 4, threads)):
                            c = (c4 * threads + i) * 4
                            if c < k_dim:
                                for j in T.vectorized(4):
                                    dst[pr, c + j] = T.Cast(T.float16, src[(base4 + c // 4) * 4 + j])

    return kernel


KERNELS += [
    KernelSpec(name="tl_moe_map0_6", build=moe_map0, group="moe", targets=("metal",),
               notes="per-expert pair-row lists, counts and 32-row work items from the routing ids [n_tok][6]; one threadgroup of 256 (one thread per expert); grid=(1)"),
    KernelSpec(name="tl_moe_packed_offsets", build=moe_packed_offsets, group="moe", targets=("metal",),
               notes="offsets = exclusive prefix sum of the expert counts; grid=(1)"),
    KernelSpec(name="tl_moe_pack_rhs_f32", build=lambda: moe_pack_rhs(False), group="moe", targets=("metal",),
               notes="pack the f32 activation rows of each expert's pair rows into contiguous f16 rows; grid=(32*work_cap)"),
    KernelSpec(name="tl_moe_pack_rhs_f16", build=lambda: moe_pack_rhs(True), group="moe", targets=("metal",),
               notes="same from an f16 source (the mid activations)"),
]
