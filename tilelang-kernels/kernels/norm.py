"""RMSNorm kernels.

References in ds4:
  CPU   rms_norm_weight / rms_norm_no_weight      ds4.c       (~8902-8918)
  CUDA  rms_norm_weight_kernel                    ds4_cuda.cu (~6697)
  Metal kernel_rms_norm_mul_f32_4 / _f32_4        metal/norm.metal

Semantics: per row, scale = rsqrt(mean(x^2) + eps); out = x * scale [* w].
The CPU reference accumulates the sum of squares in double; GPU kernels use
float. Tests compare with a tolerance, not bit-exactly.

Written at tile level (fragments + T.Parallel + T.reduce_sum) so the same
source lowers to every target: on Metal each threadgroup handles one row with
`threads` threads and a simdgroup/threadgroup AllReduce; on CPU a block is a
single thread and the fragment is a plain local array.

Metal (M5 Max, GPU time per dispatch, 500 back-to-back in one command
buffer, tests/test_tl_metal + DS4_METAL_CB_TIMES): with 32-bit indices
(gen.py sets tl.disable_index_legalize) and the two-level simdgroup
AllReduce, the generated kernels run at 0.65-0.9x of ds4's hand-written
kernel_rms_norm_{mul_,}f32_4 on every shape in {512,2048,4096} x {1,64}.
With the default int64 index promotion they were 1.5-3x slower: Apple GPUs
have no 64-bit integer ALU.
"""

from __future__ import annotations

import tilelang.language as T

from tlk import KernelSpec
import shapes



def default_threads(n: int) -> int:
    """One float4 per thread, capped at a full 1024-thread threadgroup: on
    the M5 Max 512/1024 threads for n=2048/4096 measured 10-30% faster than
    256 threads with 2-4 float4 each (more threadgroups in flight per core),
    and it matches ds4's own `ds4_gpu_rms_norm_threads`."""
    return max(32, min(1024, n // 4))



def rms_norm_weight(n: int, threads: int | None = None):
    threads = threads or default_threads(n)
    rows = T.dynamic("rows")
    dtype = T.float32

    @T.prim_func
    def kernel(
        out: T.Tensor((rows, n), dtype),
        x: T.Tensor((rows, n), dtype),
        w: T.Tensor((n,), dtype),
        eps: T.float32,
    ):
        with T.Kernel(rows, threads=threads) as row:
            sq = T.alloc_fragment((1, n), dtype)
            ss = T.alloc_fragment((1,), dtype)

            T.copy(x[row : row + 1, :], sq)
            for i, j in T.Parallel(1, n):
                sq[i, j] = sq[i, j] * sq[i, j]
            T.reduce_sum(sq, ss, dim=1)
            for i, j in T.Parallel(1, n):
                out[row + i, j] = x[row + i, j] * T.rsqrt(ss[i] / n + eps) * w[j]

    return kernel


def rms_norm_plain(n: int, threads: int | None = None):
    threads = threads or default_threads(n)
    rows = T.dynamic("rows")
    dtype = T.float32

    @T.prim_func
    def kernel(
        out: T.Tensor((rows, n), dtype),
        x: T.Tensor((rows, n), dtype),
        eps: T.float32,
    ):
        with T.Kernel(rows, threads=threads) as row:
            sq = T.alloc_fragment((1, n), dtype)
            ss = T.alloc_fragment((1,), dtype)

            T.copy(x[row : row + 1, :], sq)
            for i, j in T.Parallel(1, n):
                sq[i, j] = sq[i, j] * sq[i, j]
            T.reduce_sum(sq, ss, dim=1)
            for i, j in T.Parallel(1, n):
                out[row + i, j] = x[row + i, j] * T.rsqrt(ss[i] / n + eps)

    return kernel


KERNELS = []
for _n in shapes.RMS_NORM_WIDTHS:
    KERNELS.append(
        KernelSpec(
            name=f"tl_rms_norm_weight_f32_{_n}",
            build=(lambda n=_n: rms_norm_weight(n)),
            group="norm",
            notes=f"out[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * w; n={_n}; grid=(rows)",
        )
    )
    KERNELS.append(
        KernelSpec(
            name=f"tl_rms_norm_plain_f32_{_n}",
            build=(lambda n=_n: rms_norm_plain(n)),
            group="norm",
            notes=f"out[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps); n={_n}; grid=(rows)",
        )
    )
