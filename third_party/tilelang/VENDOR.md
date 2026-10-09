# Vendored TileLang runtime headers

Header-only support files that TileLang-generated kernel sources in `gen/`
include. Copied verbatim from the TileLang checkout used by
`tilelang-kernels/gen.py`; nothing here is compiled on its own.

## Upstream pin

| Field   | Value |
|---------|-------|
| Source  | https://github.com/tile-ai/tilelang |
| Commit  | `32e2531a` (branch `ds4-metal-cpu-reduce`: v0.1.15 + Metal reduce lowering, CPU fragment reduce/fill, CPU stack allocations, no fwd decls for imported externs) |
| License | MIT (`LICENSE` in this directory). `half.hpp` is MIT, copyright Christian Rau. |

## File inventory

| File | Origin | Used by |
|------|--------|---------|
| `tl_templates/cpp/common.h` | `src/tl_templates/cpp/common.h` | `gen/cpu/*.cpp` (target `c`) |
| `tl_templates/cpp/half.hpp` | `src/tl_templates/cpp/half.hpp` | included by `common.h` |

Metal output needs no support headers (`#include <metal_stdlib>` only).
CUDA/HIP output will need `tl_templates/cuda` / `tl_templates/hip` plus a
CUTLASS/CuTe header subset; add them here when those targets are enabled.

Re-sync whenever `gen/VERSION` moves to a new TileLang commit.
