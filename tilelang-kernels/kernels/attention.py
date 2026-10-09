"""DeepSeek V4 decode attention (one query token, 64 heads, one shared
K = V latent of 512 dims, f16 caches).

References in ds4:
  Metal kernel_dsv4_flash_kv_stage_f16                        metal/cpy.metal:145
        kernel_flash_attn_ext_vec_f16_dk512_dv512 (+ _reduce)  metal/flash_attn.metal:951, 1404
        kernel_dsv4_flash_attn_vec_packed32_reduce_rope_...   metal/dsv4_rope.metal:832
  CPU   layer_attention_mixed_one                             ds4.c:14841

Key set per layer = [n_raw rows of the raw SWA ring (f32, converted to f16)]
++ [n_comp compressed rows (f16)], attended jointly with a per-head sink
logit; the softmax is f32 throughout; Q is rounded to f16 like the
hand-written kernel; V is the same buffer as K (MLA latent).

Three dispatches replace ds4's two (stage + packed kernel):
  tl_flash_kv_stage_f16          gather raw ring (ring wrap, f32 -> f16) + copy comp rows
  tl_flash_decode_partial_...    one threadgroup per (key block, head group): S = Q K^T on
                                 simdgroup matrices, block max / exp / sum, P V in f32,
                                 unnormalised partial O + (m, l) per split
  tl_flash_decode_reduce_...     per head: merge splits + sink, 1/L, optional inverse RoPE
                                 tail (ds4's rope fusion) -> heads[h][512]
Keys past n_keys are masked inside the partial kernel, so no pad buffer.
"""

import tilelang.language as T

from tlk import KernelSpec
from kernels.dsv4 import _yarn_corr_dims, _yarn_cos_sin

HEAD_DIM = 512
NEG_HALF_MAX = -1.7014118e38   # -FLT_MAX / 2, ds4's running-max init


def flash_kv_stage_f16(head_dim: int = HEAD_DIM, threads: int = 256):
    """dst[r, :] = f16(raw[(raw_start + r) % raw_cap, :]) for r < n_raw, then
    comp[r - n_raw, :] (bitwise) for the n_comp compressed rows. 4 elements
    per thread. Grid: (ceildiv((n_raw + n_comp) * head_dim / 4, threads),)."""
    assert head_dim % 4 == 0
    raw_len = T.dynamic("raw_len")
    comp_len = T.dynamic("comp_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float16),
        raw: T.Tensor((raw_len,), T.float32),
        comp: T.Tensor((comp_len,), T.float16),
        n_raw: T.int32,
        n_comp: T.int32,
        raw_cap: T.int32,
        raw_start: T.int32,
    ):
        with T.Kernel(T.ceildiv((n_raw + n_comp) * (head_dim // 4), threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                v = bx * threads + i
                if v < (n_raw + n_comp) * (head_dim // 4):
                    r = v // (head_dim // 4)
                    c = (v % (head_dim // 4)) * 4
                    if r < n_raw:
                        pr = raw_start + r
                        pr = T.if_then_else(pr >= raw_cap, pr - raw_cap, pr)
                        for j in T.vectorized(4):
                            dst[r * head_dim + c + j] = T.Cast(T.float16, raw[pr * head_dim + c + j])
                    else:
                        for j in T.vectorized(4):
                            dst[r * head_dim + c + j] = comp[(r - n_raw) * head_dim + c + j]

    return kernel


def flash_decode_partial(head_dim: int = HEAD_DIM, bm: int = 32, bn: int = 32, bk: int = 64, threads: int = 256):
    """Grid: (ceildiv(n_keys, bn), n_head / bm). For the key block [bx*bn, +bn)
    and heads [by*bm, +bm): s = scale * Q K^T (Q rounded to f16, f32
    accumulation), keys >= n_keys masked, m = max_k s, p = exp(s - m),
    l = sum_k p, O = sum_k p V (f32 FMAs, V = K rows). Writes
    out_part[bx, h, :] (unnormalised) and stats[bx, h, 0:2] = (m, l).
    Reading the raw ring / comp cache in place instead of the staged f16
    copy was tried and is 2x slower (branchy scalar loads)."""
    assert head_dim % bk == 0 and (head_dim * bm) % threads == 0
    dpt = (head_dim * bm) // threads          # output dims per thread in the PV phase
    hpt = threads // bm                       # threads per head in the PV phase
    assert dpt * hpt == head_dim and dpt % 4 == 0
    n_head = T.dynamic("n_head")
    n_keys_len = T.dynamic("n_keys_len")
    nsplit = T.dynamic("nsplit")

    @T.prim_func
    def kernel(
        out_part: T.Tensor((nsplit, n_head, head_dim), T.float32),
        stats: T.Tensor((nsplit, n_head, 2), T.float32),
        q: T.Tensor((n_head, head_dim), T.float32),
        kv: T.Tensor((n_keys_len, head_dim), T.float16),
        n_keys: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(n_keys, bn), n_head // bm, threads=threads) as (bx, by):
            q_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            k_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            s_acc = T.alloc_fragment((bm, bn), T.float32)
            m_f = T.alloc_fragment((bm,), T.float32)
            l_f = T.alloc_fragment((bm,), T.float32)
            p_s = T.alloc_shared((bm, bn), T.float32, scope="shared")
            T.clear(s_acc)
            for ko in T.Pipelined(head_dim // bk, num_stages=0):
                for h, kk in T.Parallel(bm, bk):
                    q_s[h, kk] = T.Cast(T.float16, q[by * bm + h, ko * bk + kk])
                for kk, n in T.Parallel(bk, bn):
                    key = bx * bn + n
                    k_s[kk, n] = T.if_then_else(key < n_keys, kv[T.min(key, n_keys - 1), ko * bk + kk], T.float16(0))
                T.gemm(q_s, k_s, s_acc)
            # The cooperative-tensor accumulator cannot be reduced directly
            # (TileLang Metal: no reduce in metal.simdgroup scope): round-trip
            # through shared memory into a plain fragment. The simdgroup_store
            # is opaque to the sync pass: order it explicitly.
            T.copy(s_acc, p_s)
            T.sync_threads()
            s_f = T.alloc_fragment((bm, bn), T.float32)
            T.copy(p_s, s_f)
            for h, n in T.Parallel(bm, bn):
                s_f[h, n] = T.if_then_else(bx * bn + n < n_keys, s_f[h, n] * scale, NEG_HALF_MAX)
            T.reduce_max(s_f, m_f, dim=1, clear=True)
            for h, n in T.Parallel(bm, bn):
                s_f[h, n] = T.if_then_else(bx * bn + n < n_keys, T.exp(s_f[h, n] - m_f[h]), 0.0)
            T.reduce_sum(s_f, l_f, dim=1)
            T.copy(s_f, p_s)
            for h in T.Parallel(bm):
                stats[bx, by * bm + h, 0] = m_f[h]
                stats[bx, by * bm + h, 1] = l_f[h]
            # P V: thread (h, chunk) owns dpt consecutive output dims of head h.
            tid = T.get_thread_binding()
            h = tid // hpt
            d0 = (tid % hpt) * dpt
            o = T.alloc_local((dpt,), T.float32)
            vl = T.alloc_local((4,), T.float16)
            for j in T.unroll(dpt):
                o[j] = 0.0
            n_valid = T.min(bn, n_keys - bx * bn)
            for n in T.serial(n_valid):
                pw = p_s[h, n]
                for j4 in T.serial(dpt // 4):
                    for j in T.vectorized(4):
                        vl[j] = kv[bx * bn + n, d0 + j4 * 4 + j]
                    for j in T.unroll(4):
                        o[j4 * 4 + j] += pw * T.Cast(T.float32, vl[j])
            for j4 in T.serial(dpt // 4):
                for j in T.vectorized(4):
                    out_part[bx, by * bm + h, d0 + j4 * 4 + j] = o[j4 * 4 + j]

    return kernel


def flash_decode_reduce(head_dim: int = HEAD_DIM, n_rot: int = 64, threads: int = 128):
    """heads[h, :] = (sum_i out_part[i, h, :] * exp(m_i - M)) / (sum_i l_i * exp(m_i - M) + exp(sink_h - M)),
    M = max(max_i m_i, sink_h); then, if do_rope, the inverse YaRN RoPE
    tail on the last n_rot dims at position pos0. Grid: (n_head,)."""
    assert (head_dim % threads) == 0 and n_rot % 2 == 0 and n_rot // 2 <= threads
    dpt = head_dim // threads
    n_head = T.dynamic("n_head")
    nsplit = T.dynamic("nsplit")
    n_nope = head_dim - n_rot

    @T.prim_func
    def kernel(
        heads: T.Tensor((n_head, head_dim), T.float32),
        out_part: T.Tensor((nsplit, n_head, head_dim), T.float32),
        stats: T.Tensor((nsplit, n_head, 2), T.float32),
        sinks: T.Tensor((n_head,), T.float32),
        do_rope: T.int32,
        pos0: T.int32,
        n_ctx_orig: T.int32,
        freq_base: T.float32,
        freq_scale: T.float32,
        ext_factor: T.float32,
        attn_factor: T.float32,
        beta_fast: T.float32,
        beta_slow: T.float32,
    ):
        with T.Kernel(n_head, threads=threads) as h:
            row = T.alloc_shared((head_dim,), T.float32)
            ml = T.alloc_shared((2,), T.float32)
            tid = T.get_thread_binding()
            if tid == 0:
                mx = T.alloc_var(T.float32)
                mx = sinks[h]
                for i in T.serial(nsplit):
                    mx = T.max(mx, stats[i, h, 0])
                lsum = T.alloc_var(T.float32)
                lsum = T.exp(sinks[h] - mx)
                for i in T.serial(nsplit):
                    lsum += stats[i, h, 1] * T.exp(stats[i, h, 0] - mx)
                ml[0] = mx
                ml[1] = T.if_then_else(lsum == 0.0, 0.0, 1.0 / lsum)
            T.sync_threads()
            acc = T.alloc_local((dpt,), T.float32)
            for j in T.unroll(dpt):
                acc[j] = 0.0
            for i in T.serial(nsplit):
                sc = T.exp(stats[i, h, 0] - ml[0])
                for j in T.vectorized(dpt):
                    acc[j] += out_part[i, h, tid * dpt + j] * sc
            for j in T.unroll(dpt):
                row[tid * dpt + j] = acc[j] * ml[1]
            T.sync_threads()
            if do_rope != 0 and tid < n_rot // 2:
                low, high = _yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
                r = 2 * tid
                theta = T.Cast(T.float32, pos0) * T.pow(freq_base, T.Cast(T.float32, -r) / n_rot)
                c, s = _yarn_cos_sin(theta, r, freq_scale, ext_factor, attn_factor, low, high, 1)
                x0 = row[n_nope + r]
                x1 = row[n_nope + r + 1]
                row[n_nope + r] = x0 * c - x1 * s
                row[n_nope + r + 1] = x0 * s + x1 * c
            T.sync_threads()
            for j in T.vectorized(dpt):
                heads[h, tid * dpt + j] = row[tid * dpt + j]

    return kernel


def flash_prefill(head_dim: int = HEAD_DIM, bm: int = 32, bn: int = 32, bk: int = 64, threads: int = 256):
    """Prefill attention with a mask: rows are (token, head) pairs of
    q[n_q][n_head][512] (the 64 heads share K = V, so they are just more
    query rows), a threadgroup owns bm consecutive rows (one token, half the
    heads) and walks the key blocks with an online softmax:
      s = scale * Q K^T + mask[token][key]   (f16 mask, -inf = masked)
      m' = max(m, max_k s); p = exp(s - m'); l = l*exp(m-m') + sum p;
      O = O*exp(m-m') + P V
    then the per-head sink logit, 1/l, and heads[token][head][:] = O.
    Key blocks whose mask row is entirely -inf are skipped (ds4's blk
    summary, computed in place). Grid: (ceildiv(n_q * n_head, bm),)."""
    assert head_dim % bk == 0 and (head_dim * bm) % threads == 0 and 64 % bm == 0
    dpt = (head_dim * bm) // threads
    hpt = threads // bm
    assert dpt * hpt == head_dim and dpt % 4 == 0
    n_rows = T.dynamic("n_rows")
    n_keys_len = T.dynamic("n_keys_len")
    mask_len = T.dynamic("mask_len")
    n_head_len = T.dynamic("n_head_len")

    @T.prim_func
    def kernel(
        heads: T.Tensor((n_rows, head_dim), T.float32),
        q: T.Tensor((n_rows, head_dim), T.float32),
        kv: T.Tensor((n_keys_len, head_dim), T.float16),
        mask: T.Tensor((mask_len,), T.float16),
        sinks: T.Tensor((n_head_len,), T.float32),
        n_keys: T.int32,
        n_head: T.int32,   # multiple of bm, so a threadgroup's rows share one token
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(n_rows, bm), threads=threads) as bx:
            q_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            k_s = T.alloc_shared((bk, bn), T.float16, scope="shared")
            s_acc = T.alloc_fragment((bm, bn), T.float32)
            s_f = T.alloc_fragment((bm, bn), T.float32)
            m_f = T.alloc_fragment((bm,), T.float32)
            mnew_f = T.alloc_fragment((bm,), T.float32)
            l_f = T.alloc_fragment((bm,), T.float32)
            rs_f = T.alloc_fragment((bm,), T.float32)
            p_s = T.alloc_shared((bm, bn), T.float32, scope="shared")
            alpha_s = T.alloc_shared((bm,), T.float32, scope="shared")
            mvalid = T.alloc_shared((bn,), T.float32, scope="shared")
            tid = T.get_thread_binding()
            row0 = bx * bm
            token = row0 // n_head
            h = tid // hpt
            d0 = (tid % hpt) * dpt
            o = T.alloc_local((dpt,), T.float32)
            vl = T.alloc_local((4,), T.float16)
            for j in T.unroll(dpt):
                o[j] = 0.0
            T.fill(m_f, NEG_HALF_MAX)
            T.clear(l_f)
            for kb in T.serial(T.ceildiv(n_keys, bn)):
                # block visibility: any unmasked key in this block for this token
                if tid < bn:
                    key = kb * bn + tid
                    mvalid[tid] = T.if_then_else(key < n_keys, T.Cast(T.float32, mask[token * n_keys + T.min(key, n_keys - 1)]), -65504.0)
                T.sync_threads()
                anyv = T.alloc_var(T.float32)
                anyv = -65504.0
                for n in T.serial(bn):
                    anyv = T.max(anyv, mvalid[n])
                if anyv > -60000.0:
                    T.clear(s_acc)
                    for ko in T.Pipelined(head_dim // bk, num_stages=0):
                        for r, kk in T.Parallel(bm, bk):
                            q_s[r, kk] = T.Cast(T.float16, q[T.min(row0 + r, n_rows - 1), ko * bk + kk])
                        for kk, n in T.Parallel(bk, bn):
                            key = kb * bn + n
                            k_s[kk, n] = T.if_then_else(key < n_keys, kv[T.min(key, n_keys - 1), ko * bk + kk], T.float16(0))
                        T.gemm(q_s, k_s, s_acc)
                    T.copy(s_acc, p_s)
                    T.sync_threads()
                    T.copy(p_s, s_f)
                    for r, n in T.Parallel(bm, bn):
                        key = kb * bn + n
                        s_f[r, n] = T.if_then_else(key < n_keys, s_f[r, n] * scale + mvalid[n], NEG_HALF_MAX)
                    T.reduce_max(s_f, rs_f, dim=1, clear=True)
                    for r in T.Parallel(bm):
                        mnew_f[r] = T.max(m_f[r], rs_f[r])
                        alpha_s[r] = T.exp(m_f[r] - mnew_f[r])
                    for r, n in T.Parallel(bm, bn):
                        s_f[r, n] = T.if_then_else(kb * bn + n < n_keys, T.exp(s_f[r, n] - mnew_f[r]), 0.0)
                    T.reduce_sum(s_f, rs_f, dim=1)
                    for r in T.Parallel(bm):
                        l_f[r] = l_f[r] * T.exp(m_f[r] - mnew_f[r]) + rs_f[r]
                        m_f[r] = mnew_f[r]
                    T.copy(s_f, p_s)
                    T.sync_threads()
                    a = alpha_s[h]
                    for j in T.unroll(dpt):
                        o[j] = o[j] * a
                    n_valid = T.min(bn, n_keys - kb * bn)
                    for n in T.serial(n_valid):
                        pw = p_s[h, n]
                        for j4 in T.serial(dpt // 4):
                            for j in T.vectorized(4):
                                vl[j] = kv[kb * bn + n, d0 + j4 * 4 + j]
                            for j in T.unroll(4):
                                o[j4 * 4 + j] += pw * T.Cast(T.float32, vl[j])
                T.sync_threads()
            # sink logit and normalisation: l_f / m_f live in the fragment layout,
            # pass the per-row scale through shared memory
            for r in T.Parallel(bm):
                sk = sinks[(row0 + r) % n_head]
                mfin = T.max(m_f[r], sk)
                lfin = l_f[r] * T.exp(m_f[r] - mfin) + T.exp(sk - mfin)
                alpha_s[r] = T.if_then_else(lfin == 0.0, 0.0, T.exp(m_f[r] - mfin) / lfin)
            T.sync_threads()
            a = alpha_s[h]
            if row0 + h < n_rows:
                for j4 in T.serial(dpt // 4):
                    for j in T.vectorized(4):
                        heads[row0 + h, d0 + j4 * 4 + j] = o[j4 * 4 + j] * a

    return kernel



def indexed_attention(head_dim: int = HEAD_DIM, bm: int = 32, bn: int = 16, bk: int = 64, threads: int = 256, comp_f16: bool = True):
    """Indexed mixed attention (kernel_dsv4_indexed_mixed_attention_heads16_dual /
    heads8_split + _reduce, metal/dsv4_misc.metal:5841 / 6083): per query token
    the key set is [the raw ring rows of positions max(first_raw_pos,
    qpos+1-window) .. min(qpos, last_pos), chronological] ++ [the indexer's
    top-k compressed rows (int32 per token; idx < 0 skipped, idx >= visible =
    min((qpos+1)/ratio, n_comp) invisible; sorted_topk != 0 means the row is
    ascending so the valid entries are one contiguous range)], then the
    per-head sink. A threadgroup owns one token and bm heads and walks the
    gathered keys in blocks of bn: each block's rows are staged once as f16
    (raw f32 -> f16, comp f16 or f32) and used for both Q K^T (T.gemm) and
    P V; online softmax as in flash_prefill. q is the f16 copy of the f32
    queries (tl_cpy_f32_f16 by the host), re-staged per key block in bk
    chunks. Measured on the M5 Max (256 tokens, 1100 comp rows): the Q K^T
    simdgroup gemm and the scalar P V each take ~45% of the time; the row
    gathers, Q re-staging and the softmax are hidden. Tried and not better:
    software-prefetching the next block's rows (+3%), 64 heads x 512
    threads per threadgroup (+18%), a [dim][key] copy for the gemm B
    operand instead of transpose_B (+10%), bk 128 (+4%); the +8 row padding
    is worth 1%. Decode uses ~40 splits (one block each). Grid z = n_splits: with one
    split the kernel writes heads[token][head][:]; with several (decode) each
    split handles a range of key blocks and writes unnormalised out_part[z,
    head, :] + stats[z, head, 0:2] = (m, l) in tl_flash_decode_reduce's
    layout (sink applied there). Grid: (n_tokens, n_head / bm, n_splits)."""
    assert head_dim % bk == 0 and (head_dim * bm) % threads == 0 and head_dim % 4 == 0
    # P V thread mapping: a simdgroup (32 threads) covers one full row of 512
    # dims (16 per thread) for hpg = 4 heads, so each staged row is read from
    # threadgroup memory once per 4 heads (the hand kernel: once per 2 heads).
    hpg = 4
    dpg = head_dim // 32
    assert dpg % 4 == 0 and threads == 32 * (bm // hpg)
    dpt = hpg * dpg
    assert threads % bn == 0 and bn <= threads
    comp_dtype = T.float16 if comp_f16 else T.float32
    n_rows = T.dynamic("n_rows")
    raw_len = T.dynamic("raw_len")
    comp_len = T.dynamic("comp_len")
    topk_len = T.dynamic("topk_len")
    n_head_len = T.dynamic("n_head_len")
    part_len = T.dynamic("part_len")
    stats_len = T.dynamic("stats_len")

    @T.prim_func
    def kernel(
        heads: T.Tensor((n_rows, head_dim), T.float32),
        out_part: T.Tensor((part_len, head_dim), T.float32),
        stats: T.Tensor((stats_len, 2), T.float32),
        q: T.Tensor((n_rows, head_dim), T.float16),   # f16 copy of the f32 q (tl_cpy_f32_f16)
        raw_kv: T.Tensor((raw_len, head_dim), T.float32),
        comp_kv: T.Tensor((comp_len, head_dim), comp_dtype),
        topk: T.Tensor((topk_len,), T.int32),
        sinks: T.Tensor((n_head_len,), T.float32),
        n_tokens: T.int32,
        n_head: T.int32,
        n_raw: T.int32,
        raw_cap: T.int32,
        raw_start: T.int32,
        n_comp: T.int32,
        top_k: T.int32,
        pos0: T.int32,
        window: T.int32,
        ratio: T.int32,
        sorted_topk: T.int32,
        n_splits: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(n_tokens, n_head // bm, n_splits, threads=threads) as (token, hg, split):
            # rows padded by 8 halves (16 B): the 8x8 simdgroup tile loads of the
            # Q K^T gemm then hit different banks per row (TileLang pads only on
            # its cooperative-tensor path)
            kv_s = T.alloc_shared((bn, head_dim + 8), T.float16, scope="shared")
            q_s = T.alloc_shared((bm, bk), T.float16, scope="shared")
            s_acc = T.alloc_fragment((bm, bn), T.float32)
            s_f = T.alloc_fragment((bm, bn), T.float32)
            m_f = T.alloc_fragment((bm,), T.float32)
            mnew_f = T.alloc_fragment((bm,), T.float32)
            l_f = T.alloc_fragment((bm,), T.float32)
            rs_f = T.alloc_fragment((bm,), T.float32)
            p_s = T.alloc_shared((bm, bn), T.float32, scope="shared")
            alpha_s = T.alloc_shared((bm,), T.float32, scope="shared")
            kind_s = T.alloc_shared((bn,), T.int32, scope="shared")
            src_s = T.alloc_shared((bn,), T.int32, scope="shared")
            cnt_f = T.alloc_fragment((1, threads), T.int32)
            neg_f = T.alloc_fragment((1,), T.int32)
            end_f = T.alloc_fragment((1,), T.int32)
            tid = T.get_thread_binding()
            hq = tid // 32          # heads hq*hpg .. +hpg
            d0 = (tid % 32) * dpg   # dims d0 .. d0+dpg
            row0 = token * n_head + hg * bm
            qpos = pos0 + token
            last_pos = pos0 + n_tokens - 1
            first_raw_pos = last_pos + 1 - n_raw
            first = T.max(first_raw_pos, qpos + 1 - window)
            last = T.min(qpos, last_pos)
            nraw = T.max(0, last - first + 1)
            visible = T.min((qpos + 1) // ratio, n_comp)
            # valid top-k range: sorted rows -> [n_neg, n_end); unsorted -> all, checked per key
            for i in T.Parallel(threads):
                cnt_f[0, i] = 0
            for e0 in T.serial(T.ceildiv(top_k, threads)):
                for i in T.Parallel(threads):
                    e = e0 * threads + i
                    if e < top_k and sorted_topk != 0:
                        cnt_f[0, i] += T.if_then_else(topk[token * top_k + e] < 0, 1, 0)
            T.reduce_sum(cnt_f, neg_f, dim=1, clear=True)
            for i in T.Parallel(threads):
                cnt_f[0, i] = 0
            for e0 in T.serial(T.ceildiv(top_k, threads)):
                for i in T.Parallel(threads):
                    e = e0 * threads + i
                    if e < top_k and sorted_topk != 0:
                        cnt_f[0, i] += T.if_then_else(topk[token * top_k + e] < visible, 1, 0)
            T.reduce_sum(cnt_f, end_f, dim=1, clear=True)
            n_neg = T.if_then_else(sorted_topk != 0, neg_f[0], 0)
            n_end = T.if_then_else(sorted_topk != 0, end_f[0], top_k)
            total = nraw + (n_end - n_neg)
            nblocks = T.ceildiv(total, bn)
            # dynamic divisor: floordiv lowers to a Let that the Metal codegen
            # cannot print inside a var assignment; all operands are >= 0
            bps = T.truncdiv(nblocks + n_splits - 1, n_splits)
            kb0 = split * bps
            kb1 = T.min(nblocks, kb0 + bps)
            o = T.alloc_local((dpt,), T.float32)
            vl = T.alloc_local((4,), T.float16)
            pw = T.alloc_local((hpg,), T.float32)
            for j in T.unroll(dpt):
                o[j] = 0.0
            T.fill(m_f, NEG_HALF_MAX)
            T.clear(l_f)
            for kb in T.serial(kb1 - kb0):
                # key descriptors for this block: kind 1 = raw ring row, 2 = comp row, 0 = none
                if tid < bn:
                    jk = (kb0 + kb) * bn + tid
                    kind = T.alloc_var(T.int32)
                    src = T.alloc_var(T.int32)
                    kind = 0
                    src = 0
                    if jk < nraw:
                        rr = raw_start + (first + jk) - first_raw_pos   # < 2 * raw_cap
                        src = T.if_then_else(rr >= raw_cap, rr - raw_cap, rr)
                        kind = 1
                    elif jk < total:
                        idx = topk[token * top_k + n_neg + (jk - nraw)]
                        if idx >= 0 and idx < visible:
                            src = idx
                            kind = 2
                    kind_s[tid] = kind
                    src_s[tid] = src
                T.sync_threads()
                for n, c4 in T.Parallel(bn, head_dim // 4):
                    if kind_s[n] == 1:
                        for j in T.vectorized(4):
                            kv_s[n, c4 * 4 + j] = T.Cast(T.float16, raw_kv[src_s[n], c4 * 4 + j])
                    elif kind_s[n] == 2:
                        for j in T.vectorized(4):
                            kv_s[n, c4 * 4 + j] = T.Cast(T.float16, comp_kv[src_s[n], c4 * 4 + j])
                    else:
                        for j in T.vectorized(4):
                            kv_s[n, c4 * 4 + j] = T.float16(0)
                T.sync_threads()
                T.clear(s_acc)
                for ko in T.Pipelined(head_dim // bk, num_stages=0):
                    for r, kk4 in T.Parallel(bm, bk // 4):
                        for j in T.vectorized(4):
                            q_s[r, kk4 * 4 + j] = q[row0 + r, ko * bk + kk4 * 4 + j]
                    # B = the staged rows themselves ([key][dim] -> transposed load,
                    # simdgroup path; no [dim][key] copy)
                    T.gemm(q_s[0:bm, 0:bk], kv_s[0:bn, ko * bk:(ko + 1) * bk], s_acc, transpose_B=True)
                T.copy(s_acc, p_s)
                T.sync_threads()
                T.copy(p_s, s_f)
                for r, n in T.Parallel(bm, bn):
                    s_f[r, n] = T.if_then_else(kind_s[n] != 0, s_f[r, n] * scale, NEG_HALF_MAX)
                T.reduce_max(s_f, rs_f, dim=1, clear=True)
                for r in T.Parallel(bm):
                    mnew_f[r] = T.max(m_f[r], rs_f[r])
                    alpha_s[r] = T.exp(m_f[r] - mnew_f[r])
                for r, n in T.Parallel(bm, bn):
                    s_f[r, n] = T.if_then_else(kind_s[n] != 0, T.exp(s_f[r, n] - mnew_f[r]), 0.0)
                T.reduce_sum(s_f, rs_f, dim=1)
                for r in T.Parallel(bm):
                    l_f[r] = l_f[r] * T.exp(m_f[r] - mnew_f[r]) + rs_f[r]
                    m_f[r] = mnew_f[r]
                T.copy(s_f, p_s)
                T.sync_threads()
                for i in T.unroll(hpg):
                    a = alpha_s[hq * hpg + i]
                    for j in T.unroll(dpg):
                        o[i * dpg + j] = o[i * dpg + j] * a
                for n in T.serial(bn):
                    if kind_s[n] != 0:
                        for i in T.unroll(hpg):
                            pw[i] = p_s[hq * hpg + i, n]
                        for j4 in T.unroll(dpg // 4):
                            for j in T.vectorized(4):
                                vl[j] = kv_s[n, d0 + j4 * 4 + j]
                            for i in T.unroll(hpg):
                                for j in T.unroll(4):
                                    o[i * dpg + j4 * 4 + j] += pw[i] * T.Cast(T.float32, vl[j])
                T.sync_threads()
            if n_splits == 1:
                for r in T.Parallel(bm):
                    sk = sinks[hg * bm + r]
                    mfin = T.max(m_f[r], sk)
                    lfin = l_f[r] * T.exp(m_f[r] - mfin) + T.exp(sk - mfin)
                    alpha_s[r] = T.if_then_else(lfin == 0.0, 0.0, T.exp(m_f[r] - mfin) / lfin)
                T.sync_threads()
                for i in T.unroll(hpg):
                    a = alpha_s[hq * hpg + i]
                    for j4 in T.unroll(dpg // 4):
                        for j in T.vectorized(4):
                            heads[row0 + hq * hpg + i, d0 + j4 * 4 + j] = o[i * dpg + j4 * 4 + j] * a
            else:
                for r in T.Parallel(bm):
                    stats[(split * n_tokens + token) * n_head + hg * bm + r, 0] = m_f[r]
                    stats[(split * n_tokens + token) * n_head + hg * bm + r, 1] = l_f[r]
                for i in T.unroll(hpg):
                    for j4 in T.unroll(dpg // 4):
                        for j in T.vectorized(4):
                            out_part[(split * n_tokens + token) * n_head + hg * bm + hq * hpg + i, d0 + j4 * 4 + j] = o[i * dpg + j4 * 4 + j]

    return kernel



KERNELS = [
    KernelSpec(name="tl_flash_kv_stage_f16_512", build=flash_kv_stage_f16, group="attention", targets=("metal",),
               notes="dst f16 [(n_raw+n_comp)][512] = raw ring rows (f32->f16, wrap) ++ comp rows; grid=(ceil(rows*128/256))"),
    KernelSpec(name="tl_flash_decode_partial_f16_512", build=flash_decode_partial, group="attention", targets=("metal",),
               variants={"k128": lambda: flash_decode_partial(bk=128)}, tune_group="flash_decode",
               notes="per (32-key block, 32 heads): s = scale*Q K^T (f16 operands, f32 acc), m, p, l, O = P V (f32); grid=(ceil(n_keys/32), n_head/32)"),
    KernelSpec(name="tl_flash_decode_reduce_f16_512", build=flash_decode_reduce, group="attention", targets=("metal",),
               notes="merge splits + sink, normalise, optional inverse RoPE tail; grid=(n_head)"),
    KernelSpec(name="tl_indexed_attention_f16_512_comp16", build=lambda: indexed_attention(comp_f16=True), group="attention", targets=("metal",),
               variants={"k128": lambda: indexed_attention(comp_f16=True, bk=128)}, tune_group="indexed_attn",
               notes="indexed mixed attention (raw window ++ top-k comp rows, f16 comp cache): one token x 32 heads per threadgroup, 16-key blocks staged once; grid=(n_tokens, n_head/32, n_splits)"),
    KernelSpec(name="tl_indexed_attention_f16_512_comp32", build=lambda: indexed_attention(comp_f16=False), group="attention", targets=("metal",),
               variants={"k128": lambda: indexed_attention(comp_f16=False, bk=128)}, tune_group="indexed_attn",
               notes="same with an f32 comp cache"),
    KernelSpec(name="tl_flash_prefill_f16_512", build=flash_prefill, group="attention", targets=("metal",),
               variants={"n64": lambda: flash_prefill(bn=64), "k128": lambda: flash_prefill(bk=128)},   # n64 + k128 together: 33 KB of threadgroup memory
               tune_group="flash_prefill",
               notes="masked prefill attention over (token, head) rows with online softmax, sink; grid=(ceil(n_q*n_head/32))"),
]
