# tilelang-kernels

TileLang sources for ds4's generated kernels, about 90 specs across ten
modules. This directory is a **build-time tool**: the generated sources in
`../gen/` are checked in, and building or running `ds4-server` needs nothing
here.

## How it fits together

```
kernels/*.py  --gen.py-->  gen/metal/tl_*.metal   (MSL, embedded in the binary)
 (KernelSpec)              gen/cpu/tl_*.cpp       (C reference, tests only)
                           gen/*/tl_launch_meta.h (symbols, grids, arg structs)
                                  |
                  ds4_tl_metal.h: ds4_tl_encode_* binds buffers and dispatches
                                  |
                  ds4_metal.m: each entry point tries its TileLang encoder first
```

1. **Spec.** A kernel is a TileLang `PrimFunc` wrapped in a `tlk.KernelSpec`
   (name, builder, output group, targets). Each module in `kernels/` exposes a
   `KERNELS` list; shapes come from `shapes.py`, which mirrors the DeepSeek V4
   Flash dimensions in `ds4.c`, and kernels are specialised on them (`in_dim`
   4096, 64 heads, ...). A kernel declined at runtime for a shape it does not
   cover has no hand-written fallback in this branch, so every shape the model
   uses must be listed.
2. **Generation.** `gen.py` lowers every spec for the `metal` and `c` targets
   and writes the sources plus `tl_launch_meta.h`, which records each device
   symbol, its launch grid and its scalar-argument struct. `gen/VERSION` pins
   the TileLang commit and a hash of the generator.
3. **Embedding.** The build concatenates `prelude.metal` and `gen/metal/*.metal`
   (plus stand-ins, see the top-level README) and links them compressed into
   the binary; Metal compiles them at startup.
4. **Dispatch.** `ds4_tl_metal.h` holds one `ds4_tl_encode_*` function per
   kernel family. It checks shape and alignment, looks up the pipeline, binds
   buffers in signature order, packs the scalars into the args struct and
   dispatches with the grid from the launch meta.
5. **Tuning.** Kernels declared with `variants` are generated in several tile
   shapes; `tune.py` profiles the real model and writes the winning variant per
   tune group to `tuned/apple-m5-max.json`, which the build embeds (`DS4_TL_TUNED`
   overrides or disables it).

## Kernel families

Each module's docstring names the upstream ds4 kernels it replaces. Paths such as
`metal/dense.metal` or `ds4_cuda.cu` in those headers refer to the upstream
repository, not to files here.

| Module | Contents |
| --- | --- |
| `norm.py` | RMSNorm, plain and weighted, per width |
| `q8.py` | Q8_0 GEMV / batch GEMV / GEMM, attention-output projections, shared gate/up + SwiGLU, row-exact batch forms |
| `moe.py` | routed MoE: IQ2_XXS gate/up + SwiGLU and Q2_K down, in several tile variants |
| `dsv4.py` | RoPE tail (YaRN), head RMSNorm + RoPE, hyper-connection split / weighted sum / expand, fp8 KV quantize and store |
| `attention.py` | decode attention: K/V staging, split partials, merge; masked prefill attention |
| `compressor.py` | compressor pools, state shift and stores, row finalize |
| `indexer.py` | lightning-indexer scores and top-k for long contexts |
| `router.py` | MoE router: f16 GEMV, top-6 select, hash weights, argsort |
| `elementwise.py` | unary ops, add, SwiGLU |
| `utility.py` | copies with conversion, gathers/scatters, broadcasts, row sums, MoE glue |

`quant/` holds the quantisation helpers (IQ2_XXS, Q2_K, Q8_0 dot products and
tables) injected into kernels with `T.import_source`; the CPU target has no
dot-product lowering, so these are written per ISA.

## Conventions

- Kernel names start with `tl_` and carry dtype and specialised sizes
  (`tl_rms_norm_weight_f32_4096`).
- Buffers come first in the PrimFunc signature, then scalars. On Metal the
  buffers bind to `[[buffer(i)]]` in order and the scalars are packed into the
  `<name>_args_t` struct from `tl_launch_meta.h`.
- Write against the common dialect (`import tilelang.language as T`) so one
  source lowers to every target.
- `gen.py` sets `tl.disable_index_legalize` for Metal: Apple GPUs have no 64-bit
  integer ALU and TileLang's default int64 index promotion made kernels up to 3x
  slower. Every buffer bound to a generated kernel must stay below 2^31 elements.

## Testing

```sh
make test-tl-metal   # every Metal kernel vs a double-precision reference, plus a throughput batch per shape
make test-tl-cpu     # the generated C kernels vs ds4's CPU reference
make gen-check       # gen/ matches the sources (needs the TileLang fork)
```

To confirm a whole run uses only TileLang, start the server with
`DS4_METAL_DUMMY_HAND_KERNELS=1`; it counts every hand-kernel dispatch and prints
the report at exit.

## Regenerating

Needs the **custom TileLang fork** (branch `ds4-metal-cpu-reduce`, based on
TileLang 0.1.15) with a Metal codegen target, a 32-bit index pass config, a
simdgroup AllReduce lowering and a CPU reference target; see the top-level
README. The fork is private, so only its author can regenerate; everyone else
uses the checked-in `gen/`. Default checkout `~/tilelang` (override with
`TILELANG_ROOT`):

```sh
cd ~/tilelang && git submodule update --init --recursive
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt -r requirements.txt
cmake -S . -B build -G Ninja -DUSE_CUDA=OFF -DUSE_LLVM=OFF -DUSE_METAL=ON \
      -DCMAKE_BUILD_TYPE=Release            # ~2.5 minutes on an M5 Max
cd - && make gen-kernels                    # regenerate gen/
```

On macOS the Command Line Tools are enough (`brew install python@3.12 ccache`;
cmake and ninja come from pip); Metal source is compiled at runtime by ds4.
Codegen is host-side, no GPU needed. Upgrading TileLang is a deliberate commit:
rebuild, regenerate, review the diff in `gen/`. `bench_llvm.py` benchmarks the
same sources on TileLang's `llvm` target (needs `-DUSE_LLVM`) for comparison with
the C harness.
