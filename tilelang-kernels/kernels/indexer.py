"""DeepSeek V4 lightning indexer (long context, > 2048 tokens): the scores
that select the top-k compressed rows per query token.

References in ds4:
  Metal kernel_dsv4_indexer_scores_nax / _tiled / _tiled_f32   metal/dsv4_misc.metal:6572 / 6431 / 6298
        kernel_dsv4_indexer_score_one_direct (decode)            metal/dsv4_misc.metal:437
  CPU   indexer_allowed_decode_one                               ds4.c:15084

  score[t, c] = c < visible(t) ? scale * sum_h relu(q[t, h, :] . k[c, :]) * w[t, h] : -inf
  visible(t) = min((pos0 + t + 1) / ratio, n_comp)

q f32 [n_tokens][n_head][head_dim] (after RoPE tail + Hadamard/FP4 fake
quant), w f32 [n_tokens][n_head], k f32 [n_comp][head_dim] (the indexer's
compressed cache), scores f32 [n_tokens][n_comp].
"""

import tilelang.language as T

from tlk import KernelSpec

NEG_INF = -3.4e38   # the hand kernels store -INFINITY; this sorts the same (the top-k only compares)


def indexer_scores_f32(n_head: int = 64, head_dim: int = 128, tn: int = 32, bk: int = 64, threads: int = 128):
    """One threadgroup per (tn compressed rows, token): S = Q_t K^T for the
    64 heads of the token on simdgroup matrices (f16 operands, f32
    accumulation, like the hand NAX / tiled kernels), then per column the
    weighted relu sum over the heads. Tiles entirely past the token's
    visibility write -inf without the gemm. Grid: (ceildiv(n_comp, tn), n_tokens)."""
    assert head_dim % bk == 0 and n_head % 8 == 0 and tn % 8 == 0
    n_rows = T.dynamic("n_rows")        # n_tokens * n_head
    n_tokens = T.dynamic("n_tokens")
    comp_len = T.dynamic("comp_len")

    @T.prim_func
    def kernel(
        scores: T.Tensor((n_tokens, comp_len), T.float32),
        q: T.Tensor((n_rows, head_dim), T.float32),
        w: T.Tensor((n_rows,), T.float32),
        k: T.Tensor((comp_len, head_dim), T.float32),
        n_comp: T.int32,
        pos0: T.int32,
        ratio: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(n_comp, tn), n_tokens, threads=threads) as (bx, t):
            q_s = T.alloc_shared((n_head, bk), T.float16, scope="shared")
            k_s = T.alloc_shared((bk, tn), T.float16, scope="shared")
            s_acc = T.alloc_fragment((n_head, tn), T.float32)
            p_s = T.alloc_shared((n_head, tn), T.float32, scope="shared")
            tid = T.get_thread_binding()
            c0 = bx * tn
            visible = T.min(T.truncdiv(pos0 + t + 1, ratio), n_comp)
            if c0 >= visible:
                if tid < tn and c0 + tid < n_comp:
                    scores[t, c0 + tid] = NEG_INF
            else:
                T.clear(s_acc)
                for ko in T.Pipelined(head_dim // bk, num_stages=0):
                    for h, kk in T.Parallel(n_head, bk):
                        q_s[h, kk] = T.Cast(T.float16, q[t * n_head + h, ko * bk + kk])
                    for kk, n in T.Parallel(bk, tn):
                        k_s[kk, n] = T.if_then_else(c0 + n < n_comp, T.Cast(T.float16, k[T.min(c0 + n, n_comp - 1), ko * bk + kk]), T.float16(0))
                    T.gemm(q_s, k_s, s_acc)
                T.copy(s_acc, p_s)
                T.sync_threads()
                if tid < tn and c0 + tid < n_comp:
                    acc = T.alloc_var(T.float32)
                    acc = 0.0
                    for h in T.serial(n_head):
                        acc += T.max(p_s[h, tid], 0.0) * w[t * n_head + h]
                    scores[t, c0 + tid] = T.if_then_else(c0 + tid < visible, acc * scale, NEG_INF)

    return kernel


KERNELS = [
    KernelSpec(name="tl_indexer_scores_f32_64_128", build=indexer_scores_f32, group="indexer", targets=("metal",),
               notes="indexer scores: scale * sum_h relu(q[t,h].k[c]) * w[t,h], causal -inf; 64 heads x 128 dims, 32 comp rows per threadgroup; grid=(ceil(n_comp/32), n_tokens)"),
]


def topk_radix_select_f32(threads: int = 256):
    """Top-k indices per row by radix select (kernel_argsort_f32_i32_desc +
    kernel_argsort_merge_f32_i32_desc in ds4: a 1024-wide bitonic sort per
    block plus merge passes). One threadgroup per row: four 8-bit histogram
    passes over order-preserving int32 keys of the scores find the k-th
    largest key exactly; then every key above it is selected, and enough of
    the keys equal to it (lowest column index first) to reach top_k. The bin
    walk and the output offsets are parallel prefix scans (8 steps each).
    Deterministic; the selected indices come out in ascending column order
    within the two groups (all > threshold, then == threshold), so entries
    with finite scores always precede the -inf ones. Thread t owns the
    contiguous columns [t*cpt, (t+1)*cpt). Grid: (n_rows,)."""
    assert threads == 256   # one histogram bin per thread
    n_rows = T.dynamic("n_rows")
    width = T.dynamic("width")
    top_k = T.dynamic("top_k")

    @T.macro
    def ordered_key(x):
        # signed int32 that orders like the float: positives keep their bits,
        # negatives become ~magnitude (so -inf is the smallest key)
        bits = T.reinterpret(T.int32, x)
        return T.if_then_else(bits >= 0, bits, ~(bits & 0x7FFFFFFF))

    @T.macro
    def scan_inclusive(arr, tid):
        # Hillis-Steele inclusive prefix sum of arr[0:threads] in place
        for d in T.serial(8):
            step = 1 << d
            v = T.alloc_var(T.int32)
            v = T.if_then_else(tid >= step, arr[T.max(tid - step, 0)], 0)
            T.sync_threads()
            arr[tid] = arr[tid] + v
            T.sync_threads()

    @T.prim_func
    def kernel(
        out: T.Tensor((n_rows, top_k), T.int32),
        scores: T.Tensor((n_rows, width), T.float32),
    ):
        with T.Kernel(n_rows, threads=threads) as row:
            hist = T.alloc_shared((threads,), T.int32, scope="shared")
            cnt = T.alloc_shared((threads,), T.int32, scope="shared")
            sel = T.alloc_shared((4,), T.int32, scope="shared")   # prefix, remaining, n_gt
            tid = T.get_thread_binding()
            cpt = T.ceildiv(width, threads)
            c_begin = tid * cpt
            c_end = T.min(width, c_begin + cpt)
            if tid == 0:
                sel[0] = 0
                sel[1] = top_k
            T.sync_threads()
            for p in T.serial(4):
                shift = 24 - 8 * p
                hist[tid] = 0
                T.sync_threads()
                prefix = sel[0]
                remaining = sel[1]
                # bytes above this one must match the prefix found so far
                mask = T.if_then_else(p == 0, 0, ~((1 << (shift + 8)) - 1))
                # the top byte's sign bit is flipped so negative keys land in the low bins
                flip = T.if_then_else(p == 0, 128, 0)
                for c in T.serial(c_end - c_begin):
                    key = ordered_key(scores[row, c_begin + c])
                    if (key & mask) == prefix:
                        T.atomic_add(hist[((key >> shift) & 255) ^ flip], 1)
                T.sync_threads()
                # cnt[i] = number of keys in bins >= 255 - i (suffix counts from the top bin)
                cnt[tid] = hist[threads - 1 - tid]
                scan_inclusive(cnt, tid)
                above = T.if_then_else(tid == 0, 0, cnt[T.max(tid - 1, 0)])
                if cnt[tid] >= remaining and above < remaining:
                    b = threads - 1 - tid
                    sel[0] = prefix | ((b ^ flip) << shift)
                    sel[1] = remaining - above
                T.sync_threads()
            thr = sel[0]
            n_eq_take = sel[1]
            n_gt = T.alloc_var(T.int32)
            n_eq = T.alloc_var(T.int32)
            n_gt = 0
            n_eq = 0
            for c in T.serial(c_end - c_begin):
                key = ordered_key(scores[row, c_begin + c])
                n_gt += T.if_then_else(key > thr, 1, 0)
                n_eq += T.if_then_else(key == thr, 1, 0)
            cnt[tid] = n_gt
            scan_inclusive(cnt, tid)
            base_gt = cnt[tid] - n_gt
            total_gt = cnt[threads - 1]
            T.sync_threads()
            cnt[tid] = n_eq
            scan_inclusive(cnt, tid)
            base_eq = cnt[tid] - n_eq
            slot_gt = T.alloc_var(T.int32)
            slot_eq = T.alloc_var(T.int32)
            slot_gt = base_gt
            slot_eq = base_eq
            for c in T.serial(c_end - c_begin):
                key = ordered_key(scores[row, c_begin + c])
                if key > thr:
                    out[row, slot_gt] = c_begin + c
                    slot_gt += 1
                elif key == thr:
                    if slot_eq < n_eq_take:
                        out[row, total_gt + slot_eq] = c_begin + c
                    slot_eq += 1

    return kernel


KERNELS += [
    KernelSpec(name="tl_topk_radix_select_f32", build=topk_radix_select_f32, group="indexer", targets=("metal",),
               notes="top-k indices per row (descending scores) by 4-pass radix select; deterministic, ascending column order per group; grid=(n_rows)"),
]
