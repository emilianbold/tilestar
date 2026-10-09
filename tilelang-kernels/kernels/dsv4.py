"""DeepSeek V4 per-head kernels: RoPE tail (YaRN) and head RMSNorm + RoPE.

References in ds4:
  Metal kernel_dsv4_rope_tail_f32 / _inplace_pair_affine / _shared4,
        kernel_dsv4_head_rms_norm_rope_tail_f32           metal/dsv4_rope.metal
  CPU   rope_tail / head rms norm                         ds4.c

DeepSeek V4 applies RoPE (mode 0, interleaved pairs, no frequency factors)
to the last n_rot of the head_dim dims of every head; positions are affine
within a dispatch (pos0 + t * pos_step). YaRN parameters are per-call
scalars. Everything is in place on x[n_tok][n_head][head_dim] f32.

The YaRN schedule follows ggml/ds4 exactly (log/floor/ceil corr dims,
ramp, pow-based theta) so results match the hand-written kernels to float
rounding; the host's "exp2/log2" drift flag is not replicated (ds4 runs
with it off).
"""

from __future__ import annotations

import math

import tilelang.language as T

from tlk import KernelSpec
import shapes


def _yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow):
    """(low, high) correction dims, as float expressions."""
    two_pi = 2.0 * math.pi

    def corr(beta):
        return n_dims * T.log(T.Cast(T.float32, n_ctx_orig) / (beta * two_pi)) / (2.0 * T.log(freq_base))

    low = T.max(0.0, T.floor(corr(beta_fast)))
    high = T.min(n_dims - 1.0, T.ceil(corr(beta_slow)))
    return low, high


def _yarn_cos_sin(theta_extrap, r, freq_scale, ext_factor, attn_factor, low, high, inverse):
    """rope_yarn(): cos/sin with YaRN interpolation and magnitude scaling."""
    theta_interp = freq_scale * theta_extrap
    y = (T.Cast(T.float32, r // 2) - low) / T.max(0.001, high - low)
    ramp = (1.0 - T.min(1.0, T.max(0.0, y))) * ext_factor
    theta = T.if_then_else(ext_factor != 0.0, theta_interp * (1.0 - ramp) + theta_extrap * ramp, theta_interp)
    mscale = T.if_then_else(ext_factor != 0.0, attn_factor * (1.0 + 0.1 * T.log(1.0 / freq_scale)), attn_factor)
    c = T.cos(theta) * mscale
    s = T.sin(theta) * mscale
    s = T.if_then_else(inverse != 0, -s, s)
    return c, s


def rope_tail(head_dim: int, n_rot: int, n_head: int, threads: int = 256):
    """In-place RoPE on the last n_rot dims of every head. One (head, pair)
    per thread; grid (n_tok, n_head / heads_per_tg) so a single decode token
    still spreads over several threadgroups."""
    assert n_rot % 2 == 0 and n_rot <= head_dim
    n_nope = head_dim - n_rot
    pairs = n_rot // 2
    heads_per_tg = max(1, min(n_head, threads // pairs))
    threads = heads_per_tg * pairs
    assert n_head % heads_per_tg == 0
    head_groups = n_head // heads_per_tg
    n_tok = T.dynamic("n_tok")

    @T.prim_func
    def kernel(
        x: T.Tensor((n_tok, n_head, head_dim), T.float32),
        pos0: T.int32,
        pos_step: T.int32,
        n_ctx_orig: T.int32,
        inverse: T.int32,
        freq_base: T.float32,
        freq_scale: T.float32,
        ext_factor: T.float32,
        attn_factor: T.float32,
        beta_fast: T.float32,
        beta_slow: T.float32,
    ):
        with T.Kernel(n_tok, head_groups, threads=threads) as (t, hg):
            head_dim_ = head_dim  # noqa: F841  (annotation-only names)
            n_head_ = n_head  # noqa: F841
            low, high = _yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
            theta_base = T.Cast(T.float32, pos0 + t * pos_step)
            for hl, p in T.Parallel(heads_per_tg, pairs):
                h = hg * heads_per_tg + hl
                r = 2 * p
                theta = theta_base * T.pow(freq_base, T.Cast(T.float32, -r) / n_rot)
                c, s = _yarn_cos_sin(theta, r, freq_scale, ext_factor, attn_factor, low, high, inverse)
                x0 = x[t, h, n_nope + r]
                x1 = x[t, h, n_nope + r + 1]
                x[t, h, n_nope + r] = x0 * c - x1 * s
                x[t, h, n_nope + r + 1] = x0 * s + x1 * c

    return kernel


def head_rms_norm_rope_tail(head_dim: int, n_rot: int, threads: int = 64):
    """Per (token, head): x *= rsqrt(mean(x^2) + eps), then RoPE on the last
    n_rot dims (kernel_dsv4_head_rms_norm_rope_tail_f32). Grid: (n_head, n_tok)."""
    assert n_rot % 2 == 0 and n_rot <= head_dim and head_dim % 4 == 0
    n_nope = head_dim - n_rot
    assert n_nope % threads == 0 and head_dim % threads == 0
    pairs = n_rot // 2
    n_tok = T.dynamic("n_tok")
    n_head = T.dynamic("n_head")

    @T.prim_func
    def kernel(
        x: T.Tensor((n_tok, n_head, head_dim), T.float32),
        pos0: T.int32,
        n_ctx_orig: T.int32,
        inverse: T.int32,
        eps: T.float32,
        freq_base: T.float32,
        freq_scale: T.float32,
        ext_factor: T.float32,
        attn_factor: T.float32,
        beta_fast: T.float32,
        beta_slow: T.float32,
    ):
        with T.Kernel(n_head, n_tok, threads=threads) as (h, t):
            sq = T.alloc_fragment((1, head_dim), T.float32)
            ss = T.alloc_fragment((1,), T.float32)
            T.copy(x[t, h : h + 1, :], sq)
            for i, j in T.Parallel(1, head_dim):
                sq[i, j] = sq[i, j] * sq[i, j]
            T.reduce_sum(sq, ss, dim=1)
            scale = T.rsqrt(ss[0] / head_dim + eps)
            # Only the non-rotary dims are written here: a store to the tail
            # followed by the rotation's read of it from another thread has
            # no barrier between them (TileLang orders shared-memory
            # dependencies, not global ones). n_nope must divide by threads.
            for j in T.Parallel(n_nope):
                x[t, h, j] = x[t, h, j] * scale
            low, high = _yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
            theta_base = T.Cast(T.float32, pos0 + t)
            for p in T.Parallel(pairs):
                r = 2 * p
                theta = theta_base * T.pow(freq_base, T.Cast(T.float32, -r) / n_rot)
                c, s = _yarn_cos_sin(theta, r, freq_scale, ext_factor, attn_factor, low, high, inverse)
                x0 = x[t, h, n_nope + r] * scale
                x1 = x[t, h, n_nope + r + 1] * scale
                x[t, h, n_nope + r] = x0 * c - x1 * s
                x[t, h, n_nope + r + 1] = x0 * s + x1 * c

    return kernel


def hc_expand4(n_embd: int, threads: int = 256):
    """Hyper-connection expand for n_hc = 4 (kernel_dsv4_hc_expand4):
        v            = block[t, d] (+ add[t, d] if has_add)
        out[t, h, d] = v * post[t*ps + h] + sum_j comb[t*cs2 + j*cs1 + h] * residual[t, j, d]
    post/comb are flat views with caller-provided strides (they live either
    in dedicated tensors or inside the per-token split vector). When has_add
    is 0 the add buffer is ignored (the host binds block again). Grid:
    (n_embd / threads, n_tok)."""
    assert n_embd % threads == 0
    n_tok = T.dynamic("n_tok")
    post_len = T.dynamic("post_len")
    comb_len = T.dynamic("comb_len")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_tok, 4, n_embd), T.float32),
        block: T.Tensor((n_tok, n_embd), T.float32),
        add: T.Tensor((n_tok, n_embd), T.float32),
        residual: T.Tensor((n_tok, 4, n_embd), T.float32),
        post: T.Tensor((post_len,), T.float32),
        comb: T.Tensor((comb_len,), T.float32),
        ps: T.int32,
        cs1: T.int32,
        cs2: T.int32,
        has_add: T.int32,
    ):
        with T.Kernel(n_embd // threads, n_tok, threads=threads) as (bd, t):
            post_len_ = post_len  # noqa: F841  (annotation-only names)
            comb_len_ = comb_len  # noqa: F841
            for i in T.Parallel(threads):
                d = bd * threads + i
                v = T.if_then_else(has_add != 0, block[t, d] + add[t, d], block[t, d])
                r0 = residual[t, 0, d]
                r1 = residual[t, 1, d]
                r2 = residual[t, 2, d]
                r3 = residual[t, 3, d]
                for h in T.unroll(4):
                    acc = v * post[t * ps + h]
                    acc = acc + comb[t * cs2 + 0 * cs1 + h] * r0
                    acc = acc + comb[t * cs2 + 1 * cs1 + h] * r1
                    acc = acc + comb[t * cs2 + 2 * cs1 + h] * r2
                    acc = acc + comb[t * cs2 + 3 * cs1 + h] * r3
                    out[t, h, d] = acc

    return kernel


def hc_weighted_sum4(n_embd: int, threads: int = 256):
    """out[t, d] = sum_h x[t, h, d] * w[t*ws + h] (kernel_dsv4_hc_weighted_sum,
    n_hc = 4; w is either a dedicated [n_tok][4] tensor or the pre part of the
    per-token split vector). Grid: (n_embd / threads, n_tok)."""
    assert n_embd % threads == 0
    n_tok = T.dynamic("n_tok")
    w_len = T.dynamic("w_len")

    @T.prim_func
    def kernel(
        out: T.Tensor((n_tok, n_embd), T.float32),
        x: T.Tensor((n_tok, 4, n_embd), T.float32),
        w: T.Tensor((w_len,), T.float32),
        ws: T.int32,
    ):
        with T.Kernel(n_embd // threads, n_tok, threads=threads) as (bd, t):
            w_len_ = w_len  # noqa: F841
            for i in T.Parallel(threads):
                d = bd * threads + i
                # same accumulation order as the hand-written kernel
                acc = x[t, 0, d] * w[t * ws + 0]
                acc = acc + x[t, 1, d] * w[t * ws + 1]
                acc = acc + x[t, 2, d] * w[t * ws + 2]
                acc = acc + x[t, 3, d] * w[t * ws + 3]
                out[t, d] = acc

    return kernel


# (head_dim, n_rot, n_head) combinations ds4 dispatches for DeepSeek V4 Flash:
# q / heads (64 heads x 512), kv (1 x 512), indexer q (64 x 128), all with
# the 64-dim rotary tail.
ROPE_SHAPES = ((512, 64, 64), (512, 64, 1), (128, 64, 64), (128, 64, 1))   # last: indexer compressor rows


def qkv_rms_norm(q_n: int, kv_n: int, threads: int = 256):
    """kernel_dsv4_qkv_rms_norm_f32_4: the q-lora row (q_n) and the KV row
    (kv_n) of each token normalised in one dispatch; grid (rows, 2), y = 0 is
    the q task, y = 1 the kv task. scale = rsqrt(mean(x^2) + eps), out = x *
    scale * w. Rows are contiguous [rows][n]."""
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        q_out: T.Tensor((rows, q_n), T.float32),
        q_x: T.Tensor((rows, q_n), T.float32),
        q_w: T.Tensor((q_n,), T.float32),
        kv_out: T.Tensor((rows, kv_n), T.float32),
        kv_x: T.Tensor((rows, kv_n), T.float32),
        kv_w: T.Tensor((kv_n,), T.float32),
        eps: T.float32,
    ):
        with T.Kernel(rows, 2, threads=threads) as (row, task):
            if task == 0:
                sq = T.alloc_fragment((1, q_n), T.float32)
                ss = T.alloc_fragment((1,), T.float32)
                T.copy(q_x[row : row + 1, :], sq)
                for i, j in T.Parallel(1, q_n):
                    sq[i, j] = sq[i, j] * sq[i, j]
                T.reduce_sum(sq, ss, dim=1)
                for i, j in T.Parallel(1, q_n):
                    q_out[row + i, j] = q_x[row + i, j] * T.rsqrt(ss[i] / q_n + eps) * q_w[j]
            else:
                sk = T.alloc_fragment((1, kv_n), T.float32)
                sk_s = T.alloc_fragment((1,), T.float32)
                T.copy(kv_x[row : row + 1, :], sk)
                for i, j in T.Parallel(1, kv_n):
                    sk[i, j] = sk[i, j] * sk[i, j]
                T.reduce_sum(sk, sk_s, dim=1)
                for i, j in T.Parallel(1, kv_n):
                    kv_out[row + i, j] = kv_x[row + i, j] * T.rsqrt(sk_s[i] / kv_n + eps) * kv_w[j]

    return kernel


def _e4m3fn_round(y):
    """Round y (|y| <= 448) to the nearest E4M3FN value, ties to even, as
    f32. Normal range (|y| >= 2^-6): round-to-nearest-even on the top 3
    mantissa bits through the float32 bit pattern; subnormal range: the
    grid is uniform 2^-9, so rint(y * 512) / 512."""
    bits = T.reinterpret(T.uint32, y)
    rounded = (bits + T.uint32(0x7FFFF) + ((bits >> 20) & T.uint32(1))) & T.uint32(0xFFF00000)
    normal = T.reinterpret(T.float32, rounded)
    sub = T.round(y * 512.0) * (1.0 / 512.0)
    return T.if_then_else(T.abs(y) < 0.015625, sub, normal)


def fp8_kv_quantize(head_dim: int, n_rot: int, block: int = 64):
    """kernel_dsv4_fp8_kv_quantize_f32: per 64-element block of the non-RoPE
    part of each KV row, scale = exp2(ceil(log2(max(amax, 1e-4) / 448))),
    out = e4m3fn(clamp(x / scale, -448, 448)) * scale; the RoPE tail is
    copied. In place (out may alias x). Grid: (rows,), 64 threads."""
    assert (head_dim - n_rot) % block == 0
    n_blocks = (head_dim - n_rot) // block
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        out: T.Tensor((rows, head_dim), T.float32),
        x: T.Tensor((rows, head_dim), T.float32),
    ):
        with T.Kernel(rows, threads=block) as row:
            v = T.alloc_fragment((1, block), T.float32)
            av = T.alloc_fragment((1, block), T.float32)
            amax = T.alloc_fragment((1,), T.float32)
            for b in T.serial(n_blocks):
                T.copy(x[row : row + 1, b * block : (b + 1) * block], v)
                for i, j in T.Parallel(1, block):
                    av[i, j] = T.abs(v[i, j])
                T.reduce_max(av, amax, dim=1, clear=True)
                for i, j in T.Parallel(1, block):
                    scale = T.exp2(T.ceil(T.log2(T.max(amax[i], 1.0e-4) / 448.0)))
                    y = T.max(T.min(v[i, j] / scale, 448.0), -448.0)
                    out[row + i, b * block + j] = _e4m3fn_round(y) * scale
            for i, j in T.Parallel(1, n_rot):
                out[row + i, head_dim - n_rot + j] = x[row + i, head_dim - n_rot + j]

    return kernel


def kv_fp8_store(head_dim: int, n_rot: int, block: int = 64):
    """kernel_dsv4_kv_fp8_store_f32 (decode, after RoPE): the fp8_kv_quantize
    round trip in place on one KV row plus raw[raw_row][:] = f16-rounded copy of
    the whole (quantized) row for FlashAttention's raw ring. Grid: (1,), 64 threads."""
    assert (head_dim - n_rot) % block == 0
    n_blocks = (head_dim - n_rot) // block
    raw_len = T.dynamic("raw_len")

    @T.prim_func
    def kernel(
        kv: T.Tensor((head_dim,), T.float32),
        raw: T.Tensor((raw_len,), T.float32),
        raw_row: T.int32,
    ):
        with T.Kernel(1, threads=block):
            raw_len_ = raw_len  # noqa: F841  (annotation-only name)
            v = T.alloc_fragment((1, block), T.float32)
            av = T.alloc_fragment((1, block), T.float32)
            amax = T.alloc_fragment((1,), T.float32)
            for b in T.serial(n_blocks):
                for i, j in T.Parallel(1, block):
                    v[i, j] = kv[b * block + j]
                for i, j in T.Parallel(1, block):
                    av[i, j] = T.abs(v[i, j])
                T.reduce_max(av, amax, dim=1, clear=True)
                for i, j in T.Parallel(1, block):
                    scale = T.exp2(T.ceil(T.log2(T.max(amax[i], 1.0e-4) / 448.0)))
                    y = T.max(T.min(v[i, j] / scale, 448.0), -448.0)
                    q = _e4m3fn_round(y) * scale
                    kv[b * block + j] = q
                    raw[raw_row * head_dim + b * block + j] = T.Cast(T.float32, T.Cast(T.float16, q))
            for j in T.Parallel(n_rot):
                raw[raw_row * head_dim + head_dim - n_rot + j] = T.Cast(
                    T.float32, T.Cast(T.float16, kv[head_dim - n_rot + j]))

    return kernel


def hc_split_weighted_sum_norm4(n_embd: int, threads: int = 1024, with_norm: bool = True):
    """kernel_dsv4_hc_split_weighted_sum_norm4 (n_hc = 4): per row,
    thread 0 turns the 24 mixer outputs into the HC split vector
      split[0:4]  = sigmoid(mix[0:4]  * scale[0] + base[0:4]) + eps        (pre)
      split[4:8]  = 2 sigmoid(mix[4:8] * scale[1] + base[4:8])             (post)
      split[8:24] = sinkhorn(softmax_rows(mix[8:24] * scale[2] + base[8:24]) + eps)
    then all threads collapse the residual with the pre gates in ds4's order
      v = ((x[0]*pre0 + x[1]*pre1) + x[2]*pre2) + x[3]*pre3
    and write dst = v and norm_dst = v * norm_scale * w with
      norm_scale = rows > 1 ? 1/sqrt(mean(v^2) + norm_eps) : rsqrt(...).
    Grid: (rows,)."""
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        dst: T.Tensor((rows, n_embd), T.float32),
        norm_dst: T.Tensor((rows, n_embd), T.float32),
        split: T.Tensor((rows, 24), T.float32),
        mixes: T.Tensor((rows, 24), T.float32),
        x: T.Tensor((rows, 4, n_embd), T.float32),
        scale: T.Tensor((3,), T.float32),
        base: T.Tensor((24,), T.float32),
        w: T.Tensor((n_embd,), T.float32),
        eps: T.float32,
        norm_eps: T.float32,
        sinkhorn_iters: T.int32,
    ):
        with T.Kernel(rows, threads=threads) as row:
            pre_s = T.alloc_shared((4,), T.float32)
            tid = T.get_thread_binding()
            if tid == 0:
                r = T.alloc_local((16,), T.float32)
                rs = T.alloc_local((4,), T.float32)
                cs = T.alloc_local((4,), T.float32)
                for h in T.unroll(4):
                    z = mixes[row, h] * scale[0] + base[h]
                    pre = 1.0 / (1.0 + T.exp(-z)) + eps
                    split[row, h] = pre
                    pre_s[h] = pre
                for h in T.unroll(4):
                    z = mixes[row, 4 + h] * scale[1] + base[4 + h]
                    split[row, 4 + h] = 2.0 / (1.0 + T.exp(-z))
                for i in T.unroll(4):
                    for j in T.unroll(4):
                        r[i * 4 + j] = mixes[row, 8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]
                for i in T.unroll(4):
                    m = T.max(T.max(r[i * 4 + 0], r[i * 4 + 1]), T.max(r[i * 4 + 2], r[i * 4 + 3]))
                    for j in T.unroll(4):
                        r[i * 4 + j] = T.exp(r[i * 4 + j] - m)
                    inv = 1.0 / (r[i * 4 + 0] + r[i * 4 + 1] + r[i * 4 + 2] + r[i * 4 + 3])
                    for j in T.unroll(4):
                        r[i * 4 + j] = r[i * 4 + j] * inv + eps
                for j in T.unroll(4):
                    cs[j] = 1.0 / (r[0 + j] + r[4 + j] + r[8 + j] + r[12 + j] + eps)
                for i in T.unroll(4):
                    for j in T.unroll(4):
                        r[i * 4 + j] = r[i * 4 + j] * cs[j]
                for it in T.serial(sinkhorn_iters - 1):
                    for i in T.unroll(4):
                        rs[i] = 1.0 / (r[i * 4 + 0] + r[i * 4 + 1] + r[i * 4 + 2] + r[i * 4 + 3] + eps)
                        for j in T.unroll(4):
                            r[i * 4 + j] = r[i * 4 + j] * rs[i]
                    for j in T.unroll(4):
                        cs[j] = 1.0 / (r[0 + j] + r[4 + j] + r[8 + j] + r[12 + j] + eps)
                    for i in T.unroll(4):
                        for j in T.unroll(4):
                            r[i * 4 + j] = r[i * 4 + j] * cs[j]
                for k in T.unroll(16):
                    split[row, 8 + k] = r[k]
            T.sync_threads()
            v = T.alloc_fragment((1, n_embd), T.float32)
            sq = T.alloc_fragment((1, n_embd), T.float32)
            ss = T.alloc_fragment((1,), T.float32)
            for i, j in T.Parallel(1, n_embd):
                acc = x[row + i, 0, j] * pre_s[0]
                acc = acc + x[row + i, 1, j] * pre_s[1]
                acc = acc + x[row + i, 2, j] * pre_s[2]
                acc = acc + x[row + i, 3, j] * pre_s[3]
                v[i, j] = acc
                sq[i, j] = acc * acc
            if with_norm:
                T.reduce_sum(sq, ss, dim=1)
                for i, j in T.Parallel(1, n_embd):
                    norm_arg = ss[i] / n_embd + norm_eps
                    ns = T.if_then_else(rows > 1, 1.0 / T.sqrt(norm_arg), T.rsqrt(norm_arg))
                    dst[row + i, j] = v[i, j]
                    norm_dst[row + i, j] = (v[i, j] * ns) * w[j]
            else:
                for i, j in T.Parallel(1, n_embd):
                    dst[row + i, j] = v[i, j]

    return kernel



def qkv_rms_norm_kv_rope_fp8_store(q_n: int = 1024, kv_n: int = 512, n_rot: int = 64, threads: int = 256):
    """kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32 (decode): grid (1, 2).
    y = 0: q_out = rms_norm(q_x) * q_w. y = 1: kv = rms_norm(kv_x) * kv_w,
    then YaRN RoPE on the last n_rot dims at position pos0, then the E4M3FN
    round trip of the first kv_n - n_rot dims per 64-block; kv keeps the
    f32 result and raw[raw_row][:] receives the f16-rounded copy (the
    RoPE dims f16-rounded, not quantised). Rows staged in shared memory
    between the steps."""
    assert kv_n == 512 and n_rot == 64 and threads == 256 and (kv_n - n_rot) % 64 == 0
    n_nope = kv_n - n_rot
    pairs = n_rot // 2
    raw_len = T.dynamic("raw_len")

    @T.prim_func
    def kernel(
        q_out: T.Tensor((q_n,), T.float32),
        q_x: T.Tensor((q_n,), T.float32),
        q_w: T.Tensor((q_n,), T.float32),
        kv_out: T.Tensor((kv_n,), T.float32),
        kv_x: T.Tensor((kv_n,), T.float32),
        kv_w: T.Tensor((kv_n,), T.float32),
        raw: T.Tensor((raw_len,), T.float32),
        eps: T.float32,
        raw_row: T.int32,
        pos0: T.int32,
        n_ctx_orig: T.int32,
        freq_base: T.float32,
        freq_scale: T.float32,
        ext_factor: T.float32,
        attn_factor: T.float32,
        beta_fast: T.float32,
        beta_slow: T.float32,
    ):
        with T.Kernel(1, 2, threads=threads) as (_bx, task):
            raw_len_ = raw_len  # noqa: F841  (annotation-only name)
            tid = T.get_thread_binding()
            if task == 0:
                sq = T.alloc_fragment((1, q_n), T.float32)
                ss = T.alloc_fragment((1,), T.float32)
                T.copy(q_x[0:q_n], sq[0, :])
                for i, j in T.Parallel(1, q_n):
                    sq[i, j] = sq[i, j] * sq[i, j]
                T.reduce_sum(sq, ss, dim=1)
                for i, j in T.Parallel(1, q_n):
                    q_out[j] = (q_x[j] * T.rsqrt(ss[i] / q_n + eps)) * q_w[j]
            else:
                row = T.alloc_shared((kv_n,), T.float32)
                absbuf = T.alloc_shared((kv_n,), T.float32)
                amax_s = T.alloc_shared((8,), T.float32)
                sk = T.alloc_fragment((1, kv_n), T.float32)
                sk_s = T.alloc_fragment((1,), T.float32)
                T.copy(kv_x[0:kv_n], sk[0, :])
                for i, j in T.Parallel(1, kv_n):
                    sk[i, j] = sk[i, j] * sk[i, j]
                T.reduce_sum(sk, sk_s, dim=1)
                scale = T.rsqrt(sk_s[0] / kv_n + eps)
                for j in T.serial(2):
                    e = tid + j * threads
                    row[e] = (kv_x[e] * scale) * kv_w[e]
                T.sync_threads()
                if tid < pairs:
                    low, high = _yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow)
                    r = 2 * tid
                    theta = T.Cast(T.float32, pos0) * T.pow(freq_base, T.Cast(T.float32, -r) / n_rot)
                    c, s = _yarn_cos_sin(theta, r, freq_scale, ext_factor, attn_factor, low, high, 0)
                    x0 = row[n_nope + r]
                    x1 = row[n_nope + r + 1]
                    row[n_nope + r] = x0 * c - x1 * s
                    row[n_nope + r + 1] = x0 * s + x1 * c
                T.sync_threads()
                for j in T.serial(2):
                    e = tid + j * threads
                    absbuf[e] = T.abs(row[e])
                T.sync_threads()
                if tid < n_nope // 64:
                    mx = T.alloc_var(T.float32)
                    mx = absbuf[tid * 64]
                    for k in T.serial(63):
                        mx = T.max(mx, absbuf[tid * 64 + 1 + k])
                    amax_s[tid] = mx
                T.sync_threads()
                for j in T.serial(2):
                    e = tid + j * threads
                    if e < n_nope:
                        amax = T.max(amax_s[e >> 6], 1.0e-4)
                        qscale = T.exp2(T.ceil(T.log2(amax / 448.0)))
                        y = T.max(T.min(row[e] / qscale, 448.0), -448.0)
                        row[e] = _e4m3fn_round(y) * qscale
                T.sync_threads()
                for j in T.serial(2):
                    e = tid + j * threads
                    kv_out[e] = row[e]
                    raw[raw_row * kv_n + e] = T.Cast(T.float32, T.Cast(T.float16, row[e]))

    return kernel


KERNELS = [
    KernelSpec(
        name=f"tl_rope_tail_f32_{_hd}_{_nr}_h{_nh}",
        build=(lambda hd=_hd, nr=_nr, nh=_nh: rope_tail(hd, nr, nh)),
        group="dsv4",
        targets=("metal",),
        notes=f"in-place YaRN RoPE on the last {_nr} of {_hd} dims, {_nh} heads; grid=(n_tok, head groups)",
    )
    for (_hd, _nr, _nh) in ROPE_SHAPES
] + [
    KernelSpec(
        name="tl_hc_expand4_f32_4096",
        build=(lambda: hc_expand4(4096)),
        group="dsv4",
        targets=("metal",),
        notes="HC expand n_hc=4: out[t,h,:] = v*post[h] + comb[h]·residual; grid=(n_embd/256, n_tok)",
    ),
    KernelSpec(
        name="tl_hc_weighted_sum4_f32_4096",
        build=(lambda: hc_weighted_sum4(4096)),
        group="dsv4",
        targets=("metal",),
        notes="HC collapse n_hc=4: out[t,:] = sum_h w[t,h] x[t,h,:]; grid=(n_embd/256, n_tok)",
    ),
    KernelSpec(
        name="tl_head_rms_norm_rope_tail_f32_512_64",
        build=(lambda: head_rms_norm_rope_tail(512, 64)),
        group="dsv4",
        targets=("metal",),
        notes="per-head RMSNorm then YaRN RoPE tail, in place; grid=(n_head, n_tok)",
    ),
    KernelSpec(
        name="tl_qkv_rms_norm_f32_1024_512",
        build=(lambda: qkv_rms_norm(1024, 512)),
        group="dsv4",
        targets=("metal",),
        notes="q-lora (1024) and kv (512) rows rms-normed in one dispatch; grid=(rows, 2)",
    ),
    KernelSpec(
        name="tl_qkv_rms_norm_kv_rope_fp8_store_f32_1024_512",
        build=(lambda: qkv_rms_norm_kv_rope_fp8_store()),
        group="dsv4",
        targets=("metal",),
        notes="decode: q rms norm | kv rms norm + RoPE tail + E4M3FN round trip + f16-rounded raw ring store; grid=(1, 2)",
    ),
    KernelSpec(
        name="tl_fp8_kv_quantize_f32_512_64",
        build=(lambda: fp8_kv_quantize(512, 64)),
        group="dsv4",
        targets=("metal",),
        notes="E4M3FN fake-quant of the 448 non-RoPE dims per 64-block (power-of-two scale), RoPE tail copied; in place; grid=(rows)",
    ),
    KernelSpec(
        name="tl_kv_fp8_store_f32_512_64",
        build=(lambda: kv_fp8_store(512, 64)),
        group="dsv4",
        targets=("metal",),
        notes="decode after RoPE (unfused: the multi-session batch schedule): E4M3FN round trip of the 448 non-RoPE dims in place + f16-rounded raw ring row; grid=(1)",
    ),
    KernelSpec(
        name="tl_hc_split_weighted_sum_norm4_f32_4096",
        build=(lambda: hc_split_weighted_sum_norm4(4096)),
        group="dsv4",
        targets=("metal",),
        notes="HC split (sigmoid gates + 4x4 sinkhorn) + pre-gated collapse + rms norm; grid=(rows)",
    ),
    KernelSpec(
        name="tl_hc_split_weighted_sum4_f32_4096",
        build=(lambda: hc_split_weighted_sum_norm4(4096, with_norm=False)),
        group="dsv4",
        targets=("metal",),
        notes="HC split (sigmoid gates + 4x4 sinkhorn) + pre-gated collapse, no norm (the unfused batch path: DSpark's 1-token verifier batches); grid=(rows)",
    ),
]
