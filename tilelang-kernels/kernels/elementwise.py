"""Elementwise kernels of the DeepSeek V4 graph.

References in ds4: metal/unary.metal (kernel_unary_impl with a function-
constant op, kernel_dsv4_softplus_sqrt_f32_4), metal/bin.metal
(kernel_add2_f32), metal/glu.metal (kernel_swiglu_flat_f32).

All dynamic in length; strides are in elements. The unary op is a runtime
scalar (one kernel, a select chain) rather than one kernel per op: the ops
are a few instructions and the kernels are bandwidth-bound.
"""

from __future__ import annotations

import tilelang.language as T

from tlk import KernelSpec

# Op codes shared with ds4_tl_metal.h (DS4_TL_UNARY_*).
UNARY_SCALE, UNARY_CLAMP, UNARY_SQRT, UNARY_SIGMOID, UNARY_SILU, UNARY_SOFTPLUS, UNARY_SOFTPLUS_SQRT, UNARY_FILL = range(8)


def _softplus(x):
    """ds4's branch-free softplus: log1p series below exp(x) < 1/32, identity above 20."""
    ex = T.exp(x)
    em = T.min(ex, 0.03125)
    poly = em * (1.0 - em * (0.5 - em * (1.0 / 3.0 - 0.25 * em)))
    return T.if_then_else(x > 20.0, x, T.if_then_else(ex < 0.03125, poly, T.log(1.0 + ex)))


def unary_rows(threads: int = 256):
    """dst[r, i] = op(src[r, i]) over rows x width with element strides."""
    src_len = T.dynamic("src_len")
    dst_len = T.dynamic("dst_len")

    @T.prim_func
    def kernel(
        dst: T.Tensor((dst_len,), T.float32),
        src: T.Tensor((src_len,), T.float32),
        rows: T.int32,
        width: T.int32,
        src_stride: T.int32,
        dst_stride: T.int32,
        op: T.int32,
        p0: T.float32,   # scale  | clamp min
        p1: T.float32,   # bias   | clamp max | fill value
    ):
        with T.Kernel(T.ceildiv(width, threads), rows, threads=threads) as (bx, r):
            src_len_ = src_len  # noqa: F841  (annotation-only names)
            dst_len_ = dst_len  # noqa: F841
            for i in T.Parallel(threads):
                col = bx * threads + i
                if col < width:
                    x = src[r * src_stride + col]
                    y = T.if_then_else(op == UNARY_SCALE, p0 * x + p1,
                        T.if_then_else(op == UNARY_CLAMP, T.min(T.max(x, p0), p1),
                        T.if_then_else(op == UNARY_SQRT, T.sqrt(x),
                        T.if_then_else(op == UNARY_SIGMOID, 1.0 / (1.0 + T.exp(-x)),
                        T.if_then_else(op == UNARY_SILU, x / (1.0 + T.exp(-x)),
                        T.if_then_else(op == UNARY_SOFTPLUS, _softplus(x),
                        T.if_then_else(op == UNARY_SOFTPLUS_SQRT, T.sqrt(_softplus(x)), p1)))))))
                    dst[r * dst_stride + col] = y

    return kernel


def add2(threads: int = 256):
    """out[i] = a[i] + b[i]. Grid: ceildiv(n, threads)."""
    n = T.dynamic("n")

    @T.prim_func
    def kernel(out: T.Tensor((n,), T.float32), a: T.Tensor((n,), T.float32), b: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    out[idx] = a[idx] + b[idx]

    return kernel


def swiglu_flat(threads: int = 256):
    """out[i] = silu(min(g, c)) * clamp(u, -c, c) * alpha (c > 1e-6, else no
    clamp), kernel_swiglu_flat_f32. Grid: ceildiv(n, threads)."""
    n = T.dynamic("n")

    @T.prim_func
    def kernel(
        out: T.Tensor((n,), T.float32),
        gate: T.Tensor((n,), T.float32),
        up: T.Tensor((n,), T.float32),
        alpha: T.float32,
        limit: T.float32,
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < n:
                    g = gate[idx]
                    u = up[idx]
                    gc = T.if_then_else(limit > 1.0e-6, T.min(g, limit), g)
                    uc = T.if_then_else(limit > 1.0e-6, T.max(T.min(u, limit), -limit), u)
                    out[idx] = gc / (1.0 + T.exp(-gc)) * uc * alpha

    return kernel


KERNELS = [
    KernelSpec(name="tl_unary_rows_f32", build=unary_rows, group="elementwise", targets=("metal",),
               notes="dst[r,i] = op(src[r,i]); op: scale/clamp/sqrt/sigmoid/silu/softplus/softplus_sqrt/fill; grid=(ceil(width/256), rows)"),
    KernelSpec(name="tl_add2_f32", build=add2, group="elementwise", targets=("metal",),
               notes="out = a + b; grid=(ceil(n/256))"),
    KernelSpec(name="tl_swiglu_flat_f32", build=swiglu_flat, group="elementwise", targets=("metal",),
               notes="out = silu(min(g,c)) * clamp(u,-c,c) * alpha; grid=(ceil(n/256))"),
]
