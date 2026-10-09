#!/usr/bin/env python3
"""Benchmark ds4's TileLang kernels on the TileLang `llvm` CPU target.

The `c` target emits portable C and leaves SIMD to the host compiler; the
`llvm` target runs TVM's LLVM codegen in-process, which is TileLang's real
CPU performance path. This script lowers the same kernel sources with
`llvm -mcpu=<cpu>` and times them through TVM's time_evaluator (no Python
overhead in the timed region), so the numbers compare directly with the C
harness in speed-bench/tl_cpu_bench.c run on the same machine.

    TILELANG_ROOT=~/tilelang ~/tilelang/.venv/bin/python tilelang-kernels/bench_llvm.py [--mcpu native] [--only q8]

Requires a TileLang build with USE_LLVM.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT", Path.home() / "tilelang"))
sys.path.insert(0, str(HERE))
if (TILELANG_ROOT / "tilelang").is_dir():
    sys.path.insert(0, str(TILELANG_ROOT))

import tilelang  # noqa: E402
from tilelang import tvm  # noqa: E402
from tvm import runtime  # noqa: E402

import kernels  # noqa: E402


def f32_to_f16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).view(np.uint16)


def q8_0_rows(out_dim: int, in_dim: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Q8_0 rows in the on-disk layout. Returns (bytes[out_dim, blocks*34], d[out_dim, blocks], qs[out_dim, blocks, 32])."""
    blocks = in_dim // 32
    d = (0.01 + 0.02 * rng.random((out_dim, blocks))).astype(np.float32)
    qs = rng.integers(-127, 128, size=(out_dim, blocks, 32), dtype=np.int8)
    raw = np.zeros((out_dim, blocks, 34), dtype=np.uint8)
    raw[:, :, 0:2] = f32_to_f16_bits(d).view(np.uint8).reshape(out_dim, blocks, 2)
    raw[:, :, 2:] = qs.view(np.uint8)
    return raw.reshape(out_dim, blocks * 34), d.astype(np.float16).astype(np.float32), qs


def quantize_q8_0_activation(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ds4.c quantize_q8_0_activation."""
    blocks = x.shape[0] // 32
    xb = x.reshape(blocks, 32)
    amax = np.abs(xb).max(axis=1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0)
    q = np.clip(np.rint(xb * inv[:, None]), -128, 127).astype(np.int8)
    return q.reshape(-1), d.astype(np.float32)


def build(spec, target):
    func = spec.build().with_attr("global_symbol", spec.name)
    with tvm.transform.PassContext(), target:
        art = tilelang.lower(func, target=target, target_host=target, enable_host_codegen=True, enable_device_compile=True)
    return art


def timed(rt_mod, name, dev, args, number, repeat=7):
    ev = rt_mod.time_evaluator(name, dev, number=number, repeat=repeat)
    res = ev(*args)
    costs = sorted(res.results)
    return costs[0] * 1e6, costs[len(costs) // 2] * 1e6


def report(label, best_us, med_us, nbytes):
    print(f"  {label:<44s} best {best_us:9.1f} us  med {med_us:9.1f} us  {nbytes / best_us / 1e3:7.2f} GB/s")


def bench_rms_norm(specs, target, dev, n, rows):
    spec = specs[f"tl_rms_norm_weight_f32_{n}"]
    art = build(spec, target)
    fn_name = spec.name
    rng = np.random.default_rng(1)
    x = (rng.random((rows, n), dtype=np.float32) * 2 - 1) * 3
    w = rng.random(n, dtype=np.float32) * 2 - 1
    out = np.zeros((rows, n), dtype=np.float32)
    eps = 1e-6
    args = [runtime.tensor(out, dev), runtime.tensor(x, dev), runtime.tensor(w, dev), eps]
    art.rt_mod[fn_name](*args)
    ref = x * (1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + eps)) * w
    got = args[0].numpy()
    err = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-6))
    print(f"rms_norm_weight n={n} rows={rows}  (llvm, max_rel={err:.2e})")
    best, med = timed(art.rt_mod, fn_name, dev, args, number=500 if rows == 1 else 20)
    report("tilelang llvm target", best, med, rows * n * 4 * 2 + n * 4)


def bench_q8_0_gemv(specs, target, dev, in_dim, out_dim, variant=""):
    spec = specs[f"tl_q8_0_gemv_preq_{variant}{in_dim}"]
    art = build(spec, target)
    fn_name = spec.name
    rng = np.random.default_rng(2)
    blocks = in_dim // 32
    raw, d, qs = q8_0_rows(out_dim, in_dim, rng)
    x = (rng.random(in_dim, dtype=np.float32) * 2 - 1) * 2
    xq, xscale = quantize_q8_0_activation(x)
    out = np.zeros(out_dim, dtype=np.float32)
    # out_dim is a symbolic shape; the host entry derives it from `out`.
    args = [runtime.tensor(out, dev), runtime.tensor(raw, dev), runtime.tensor(xq, dev), runtime.tensor(xscale, dev)]
    art.rt_mod[fn_name](*args)
    isum = (qs.astype(np.int32) * xq.reshape(1, blocks, 32).astype(np.int32)).sum(axis=2)
    ref = (isum.astype(np.float32) * d * xscale[None, :]).sum(axis=1)
    got = args[0].numpy()
    err = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-6))
    nbytes = out_dim * blocks * 34 + blocks * 36 + out_dim * 4
    print(f"q8_0_gemv{variant and ' ' + variant.rstrip('_')} in={in_dim} out={out_dim} ({out_dim * blocks * 34 / 1e6:.1f} MB weights)  (llvm, max_rel={err:.2e})")
    best, med = timed(art.rt_mod, fn_name, dev, args, number=10 if out_dim >= 2048 else 200)
    report("tilelang llvm target", best, med, nbytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcpu", default="skylake", help="LLVM CPU name; `native` is rejected by TVM (e.g. skylake, skylake-avx512, apple-m1)")
    ap.add_argument("--only")
    ap.add_argument("--no-fast-math", action="store_true", help="disable LLVM fast-math flags (ds4 and the c target build with -ffast-math)")
    ap.add_argument("--dump-asm", action="store_true", help="print the generated assembly of the q8 kernel")
    args = ap.parse_args()

    cfg = {"kind": "llvm", "mcpu": args.mcpu, "opt-level": 3}
    if not args.no_fast_math:
        cfg["fast-math"] = True
    target = tvm.target.Target(cfg)
    # On the llvm target the whole kernel is one host function named after
    # the spec; device_mod is empty.
    dev = runtime.cpu(0)
    specs = {s.name: s for s in kernels.all_kernels()}
    print(f"target: {target}")

    if not args.only or re.search(args.only, "rms_norm"):
        bench_rms_norm(specs, target, dev, 4096, 1)
        bench_rms_norm(specs, target, dev, 4096, 64)
        bench_rms_norm(specs, target, dev, 512, 64)
    if not args.only or re.search(args.only, "q8_0"):
        for variant in ("", "lanes_"):
            bench_q8_0_gemv(specs, target, dev, 4096, 256, variant)
            bench_q8_0_gemv(specs, target, dev, 4096, 4096, variant)
            bench_q8_0_gemv(specs, target, dev, 1024, 4096, variant)
        if args.dump_asm:
            art = build(specs["tl_q8_0_gemv_preq_4096"], target)
            print(art.rt_mod.inspect_source("asm") if hasattr(art.rt_mod, "inspect_source") else art.rt_mod.get_source("asm"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
