#!/usr/bin/env python3
"""Generate ds4's TileLang kernels into ../gen/.

    python3 gen.py                 # regenerate gen/ for all targets
    python3 gen.py --targets metal # subset
    python3 gen.py --check         # regenerate to a temp dir and diff against gen/

Requires the TileLang checkout referenced by TILELANG_ROOT (default
~/tilelang) to be built; see README.md.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEN = ROOT / "gen"
TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT", Path.home() / "tilelang"))

sys.path.insert(0, str(HERE))
if (TILELANG_ROOT / "tilelang").is_dir():
    sys.path.insert(0, str(TILELANG_ROOT))

import tilelang  # noqa: E402
from tilelang import tvm  # noqa: E402
from tvm import tirx  # noqa: E402

from tlk import KernelSpec  # noqa: E402
import kernels  # noqa: E402

TARGET_EXT = {"metal": "metal", "c": "cpp"}
TARGET_DIR = {"metal": "metal", "c": "cpu"}


# ----------------------------------------------------------------------------
# PrimExpr -> C expression (for launch grid rules)
# ----------------------------------------------------------------------------

def expr_to_c(e) -> str:
    if isinstance(e, tirx.IntImm):
        return str(int(e.value))
    if isinstance(e, tirx.Var):
        return e.name
    if isinstance(e, tirx.Cast):
        return expr_to_c(e.value)
    bin_ops = {
        tirx.Add: "({a} + {b})",
        tirx.Sub: "({a} - {b})",
        tirx.Mul: "({a} * {b})",
        tirx.FloorDiv: "({a} / {b})",
        tirx.FloorMod: "({a} % {b})",
        tirx.Min: "(({a}) < ({b}) ? ({a}) : ({b}))",
        tirx.Max: "(({a}) > ({b}) ? ({a}) : ({b}))",
    }
    for cls, fmt in bin_ops.items():
        if isinstance(e, cls):
            return fmt.format(a=expr_to_c(e.a), b=expr_to_c(e.b))
    raise NotImplementedError(f"cannot print grid expression {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# Lowering
# ----------------------------------------------------------------------------

@dataclass
class Param:
    name: str
    dtype: str      # TVM dtype string; for buffers, the element dtype
    is_buffer: bool


@dataclass
class Lowered:
    spec: KernelSpec
    target: str
    symbol: str                 # device kernel symbol in the generated source
    source: str                 # full generated source
    params: list[Param]         # device kernel params, in order
    block: tuple[int, int, int]
    grid: tuple[str, str, str]  # C expressions over scalar params
    smem_bytes: int
    variants: list["Lowered"] = field(default_factory=list)   # autotune variants (suffix in .variant)
    variant: str = ""


def lower_spec(spec: KernelSpec, target: str, variant: str = "", build=None) -> Lowered:
    func = (build or spec.build)()
    name = spec.name + (f"__{variant}" if variant else "")
    func = func.with_attr("global_symbol", name)
    if target == "metal":
        # Generate for Metal 4 (M5: cooperative tensors for T.gemm). ds4 only
        # adds the GEMM-tier sources to its shader library when the device
        # reports the Metal 4 tensor API (ds4_gpu_mpp_available).
        from tilelang.metal.target import _metal_target_config
        tvm_target = tvm.target.Target(_metal_target_config(enable_metal4=True))
    else:
        tvm_target = tvm.target.Target(target)
    # 32-bit buffer indices on the GPU targets. With a dynamic grid extent
    # TileLang cannot bound blockIdx-based indices and promotes them to
    # int64; Apple GPUs emulate 64-bit integer math and that made the
    # generated rms_norm 3x slower than ds4's hand-written kernel. ds4 binds
    # every kernel argument as its own (sub-)buffer well below 2^31
    # elements, so the promotion is never needed. (CPU keeps the default:
    # 64-bit is native there.)
    pass_configs = {}
    if target == "metal":
        pass_configs["tl.disable_index_legalize"] = True
        # No per-access bounds guards either: the generated kernels are
        # dispatched with exact grids by ds4 and the hand-written kernels
        # they replace do not guard. With a data-dependent expert index the
        # legalizer wrapped every weight load in a select.
        pass_configs["tl.disable_safe_memory_legalize"] = True
    # Pin the host target: with USE_LLVM builds the default host becomes
    # `llvm` with the generating machine's triple, which leaks into the
    # emitted source and makes gen/ differ between machines.
    with tvm.transform.PassContext(config=pass_configs), tvm_target:
        art = tilelang.lower(func, target=tvm_target, target_host="c")

    dev_funcs = list(art.device_mod.functions.items())
    if len(dev_funcs) != 1:
        raise RuntimeError(f"{spec.name}/{target}: expected 1 device function, got {len(dev_funcs)}")
    gvar, dfunc = dev_funcs[0]
    symbol = str(dfunc.attrs["global_symbol"]) if "global_symbol" in dfunc.attrs else gvar.name_hint

    params = []
    for v in dfunc.params:
        is_buf = v.dtype == "handle"
        dtype = str(v.dtype)
        if is_buf:
            ann = v.type_annotation
            elem = getattr(getattr(ann, "element_type", None), "dtype", None)
            dtype = str(elem) if elem is not None else "uint8"
        params.append(Param(name=v.name, dtype=dtype, is_buffer=is_buf))

    block = [1, 1, 1]
    grid = ["1", "1", "1"]
    thread_extent = dfunc.attrs.get("thread_extent", {})
    for tag, extent in thread_extent.items():
        tag = str(tag)
        axis = "xyz".index(tag[-1])
        if tag.startswith("threadIdx"):
            if not isinstance(extent, tirx.IntImm):
                raise RuntimeError(f"{spec.name}/{target}: dynamic threadgroup size unsupported ({tag}={extent})")
            block[axis] = int(extent.value)
        elif tag.startswith("blockIdx"):
            grid[axis] = expr_to_c(extent)
    if spec.grid_override:
        grid = list(spec.grid_override)

    smem = 0
    if "dyn_shared_memory_buf" in dfunc.attrs:
        smem_buf = dfunc.attrs["dyn_shared_memory_buf"]
        smem = int(smem_buf) if isinstance(smem_buf, (int, tirx.IntImm)) else 0

    return Lowered(
        spec=spec,
        target=target,
        symbol=symbol,
        source=art.kernel_source,
        params=params,
        block=tuple(block),
        grid=tuple(grid),
        smem_bytes=smem,
    )


# ----------------------------------------------------------------------------
# Metal: split TileLang's module-level boilerplate from per-kernel code
# ----------------------------------------------------------------------------

_METAL_KERNEL_START = re.compile(r"^(struct \w+_args_t \{|\[\[kernel|kernel void )", re.M)


def metal_top_level_blocks(prelude: str) -> list[str]:
    """Split a Metal prelude into top-level declarations: a block ends where
    the brace depth returns to zero (so `namespace tl { ... }` and
    `union X { ... };` stay whole) and single lines (#include, #define,
    using) are blocks of their own. Blank lines separate nothing."""
    blocks: list[str] = []
    cur: list[str] = []
    depth = 0
    for line in prelude.splitlines():
        if not line.strip() and depth == 0:
            continue
        cur.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def split_metal(source: str, symbol: str) -> tuple[str, str]:
    """Return (prelude, kernel_body)."""
    m = _METAL_KERNEL_START.search(source)
    if not m:
        raise RuntimeError(f"{symbol}: could not locate kernel start in Metal source")
    prelude_lines = [l for l in source[: m.start()].splitlines() if not l.startswith("// Function:")]
    prelude = "\n".join(prelude_lines).strip() + "\n"
    return prelude, source[m.start():].rstrip() + "\n"


# ----------------------------------------------------------------------------
# Metadata header
# ----------------------------------------------------------------------------

C_SCALAR_TYPES = {
    "float32": "float",
    "float16": "uint16_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "bool": "uint8_t",
}


def c_type(dtype: str) -> str:
    if dtype not in C_SCALAR_TYPES:
        raise RuntimeError(f"no C type mapping for {dtype}")
    return C_SCALAR_TYPES[dtype]


def metal_args_struct(lw: Lowered) -> str:
    """Mirror codegen_metal.cc:233-260: 32-bit scalars as T x[2], 64-bit as T x,
    anything else through an 8-byte union."""
    scalars = [p for p in lw.params if not p.is_buffer]
    if not scalars:
        return ""
    lines = [f"typedef struct {{"]
    for p in scalars:
        bits = int(re.search(r"\d+", p.dtype).group()) if re.search(r"\d+", p.dtype) else 8
        if bits == 32:
            lines.append(f"    {c_type(p.dtype)} {p.name}[2];")
        elif bits == 64:
            lines.append(f"    {c_type(p.dtype)} {p.name};")
        else:
            lines.append(f"    union {{ {c_type(p.dtype)} v; uint64_t pad; }} {p.name};")
    lines.append(f"}} {lw.spec.name}_args_t;")
    return "\n".join(lines)


def _grid_fn(out: list[str], fn_name: str, lw: Lowered, sig: str, scalars) -> None:
    out.append(f"static inline void {fn_name}(uint32_t grid[3]{sig}) {{")
    grid_text = " ".join(lw.grid)
    for p in scalars:
        if not re.search(rf"\b{re.escape(p.name)}\b", grid_text):
            out.append(f"    (void){p.name};")
    for i in range(3):
        out.append(f"    grid[{i}] = (uint32_t)({lw.grid[i]});")
    out.append("}")


def meta_header(target: str, lowered: list[Lowered]) -> str:
    guard = f"DS4_TL_{TARGET_DIR[target].upper()}_LAUNCH_META_H"
    out = [
        "/* Generated by tilelang-kernels/gen.py. Do not edit. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "#include <stdint.h>",
        "",
    ]
    tune_groups: list[str] = []
    group_suffixes: dict[str, list[str]] = {}
    if target == "metal":
        for lw in lowered:
            if lw.variants:
                g = lw.spec.tune_group or lw.spec.name
                sufs = [v.variant for v in lw.variants]
                if g in group_suffixes and group_suffixes[g] != sufs:
                    raise RuntimeError(f"tune group {g}: {lw.spec.name} declares variants {sufs}, group has {group_suffixes[g]}")
                group_suffixes.setdefault(g, sufs)
                if g not in tune_groups:
                    tune_groups.append(g)
        out.append("/* Autotune: kernels with variants select one per tune group at run time;")
        out.append(" * ds4_tl_variant_index(group) is defined by the host (ds4_tl_metal.h). */")
        out.append("typedef struct { const char *suffix; const char *symbol; unsigned tx, ty, tz; } ds4_tl_variant_meta;")
        out.append("static int ds4_tl_variant_index(int tune_group);")
        out.append(f"#define DS4_TL_NUM_TUNE_GROUPS {len(tune_groups)}")
        out.append("static const char *const ds4_tl_tune_group_names[] = { " + ", ".join(f'"{g}"' for g in tune_groups) + (", " if tune_groups else "") + "NULL };")
        out.append("/* variant suffixes per tune group, index 0 = default, NULL-terminated */")
        for g in tune_groups:
            sufs = ['"default"'] + [f'"{x}"' for x in group_suffixes[g]]
            out.append(f"static const char *const ds4_tl_tune_suffixes_{tune_groups.index(g)}[] = {{ {', '.join(sufs)}, NULL }};")
        out.append("static const char *const *const ds4_tl_tune_group_suffixes[] = { " + ", ".join(f"ds4_tl_tune_suffixes_{i}" for i in range(len(tune_groups))) + (", " if tune_groups else "") + "NULL };")
        out.append("")
    for lw in lowered:
        bufs = [p for p in lw.params if p.is_buffer]
        scalars = [p for p in lw.params if not p.is_buffer]
        up = lw.spec.name.upper()
        out.append(f"/* {lw.spec.name}: {lw.spec.notes} */")
        out.append(f'#define {up}_SYMBOL "{lw.symbol}"')
        out.append(f"#define {up}_NUM_BUFFERS {len(bufs)}")
        out.append(f"#define {up}_NUM_SCALARS {len(scalars)}")
        out.append(f"#define {up}_THREADS_X {lw.block[0]}")
        out.append(f"#define {up}_THREADS_Y {lw.block[1]}")
        out.append(f"#define {up}_THREADS_Z {lw.block[2]}")
        out.append(f"#define {up}_SMEM_BYTES {lw.smem_bytes} /* declared statically inside the kernel; informational */")
        out.append("/* buffer order: " + ", ".join(p.name for p in bufs) + " */")
        if target == "metal":
            s = metal_args_struct(lw)
            if s:
                out.append(s)
            out.append(f"#define {up}_ARGS_BUFFER_INDEX {len(bufs)}")
        if target == "c":
            # CPU kernels loop over the grid internally; expose the C prototype.
            args = [f"{c_type(p.dtype)} *{p.name}" for p in bufs] + [f"{c_type(p.dtype)} {p.name}" for p in scalars]
            out.append("#ifdef __cplusplus")
            out.append('extern "C"')
            out.append("#endif")
            out.append(f"int32_t {lw.symbol}({', '.join(args)});")
        else:
            sig = "".join(f", {c_type(p.dtype)} {p.name}" for p in scalars)
            if not lw.variants:
                out.append(f"#define {up}_NUM_VARIANTS 1")
                out.append(f"#define {up}_VARIANTS NULL")
                out.append(f"#define {up}_TUNE_ID -1")
                _grid_fn(out, f"{lw.spec.name}_grid", lw, sig, scalars)
            else:
                g = lw.spec.tune_group or lw.spec.name
                allv = [lw] + lw.variants
                out.append(f"#define {up}_NUM_VARIANTS {len(allv)}")
                out.append(f"#define {up}_TUNE_ID {tune_groups.index(g)}")
                out.append(f"static const ds4_tl_variant_meta {lw.spec.name}_variants_[] = {{")
                for v in allv:
                    out.append(f'    {{ "{v.variant or "default"}", "{v.symbol}", {v.block[0]}, {v.block[1]}, {v.block[2]} }},')
                out.append("};")
                out.append(f"#define {up}_VARIANTS {lw.spec.name}_variants_")
                for i, v in enumerate(allv):
                    _grid_fn(out, f"{lw.spec.name}_grid_v{i}", v, sig, scalars)
                call = "".join(f", {p.name}" for p in scalars)
                out.append(f"static inline void {lw.spec.name}_grid(uint32_t grid[3]{sig}) {{")
                out.append(f"    switch (ds4_tl_variant_index({up}_TUNE_ID)) {{")
                for i in range(1, len(allv)):
                    out.append(f"    case {i}: {lw.spec.name}_grid_v{i}(grid{call}); break;")
                out.append(f"    default: {lw.spec.name}_grid_v0(grid{call}); break;")
                out.append("    }")
                out.append("}")
        out.append("")
    out.append(f"#endif /* {guard} */")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def tilelang_commit() -> str:
    """Commit of the TileLang checkout, with a -dirty suffix for uncommitted
    changes under src/ or tilelang/ so gen/VERSION never claims a clean pin
    it did not use."""
    try:
        head = subprocess.check_output(["git", "-C", str(TILELANG_ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(TILELANG_ROOT), "status", "--porcelain", "--", "src", "tilelang", "3rdparty/tvm"],
            text=True,
        ).strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def tilelang_version() -> str:
    """Release number only. The local build suffix (``+cuda.git...``,
    ``+git...``) encodes which backends the generating machine compiled, not
    what was generated; the commit from tilelang_commit() is the real pin."""
    return tilelang.__version__.split("+", 1)[0]


def file_header(target: str) -> str:
    return (
        f"// Generated by tilelang-kernels/gen.py for target '{target}'.\n"
        f"// TileLang {tilelang_version()} ({tilelang_commit()[:12]}). Do not edit; edit tilelang-kernels/ and rerun.\n"
    )


def write_target(outdir: Path, target: str, lowered: list[Lowered]) -> None:
    tdir = outdir / TARGET_DIR[target]
    tdir.mkdir(parents=True, exist_ok=True)
    ext = TARGET_EXT[target]

    groups: dict[str, list[Lowered]] = {}
    for lw in lowered:
        groups.setdefault(lw.spec.group, []).append(lw)

    if target == "metal":
        # Kernels pull in different helper sets (the AllReduce template, the
        # MetalPerformancePrimitives include for cooperative tensors, ...).
        # The prelude is the ordered union of every kernel's top-level
        # declarations; TileLang emits each as a self-contained block, so a
        # block seen before is dropped and anything new is appended.
        prelude_blocks: list[str] = []
        seen: set[str] = set()
        for lw in [x for l in lowered for x in [l] + l.variants]:
            p, _ = split_metal(lw.source, lw.symbol)
            for block in metal_top_level_blocks(p):
                key = block.strip()
                if key and key not in seen:
                    seen.add(key)
                    prelude_blocks.append(block.rstrip())
        prelude = "\n".join(prelude_blocks) + "\n" if prelude_blocks else ""
        (tdir / "tl_prelude.metal").write_text(file_header(target) + prelude)

    for group, items in groups.items():
        parts = [file_header(target)]
        for lw in items:
            for v in [lw] + lw.variants:
                parts.append(f"\n// ---- {lw.spec.name}{'__' + v.variant if v.variant else ''}: {lw.spec.notes}\n")
                if target == "metal":
                    _, body = split_metal(v.source, v.symbol)
                    parts.append(body)
                else:
                    parts.append(v.source.rstrip() + "\n")
        (tdir / f"tl_{group}.{ext}").write_text("".join(parts))

    (tdir / "tl_launch_meta.h").write_text(meta_header(target, lowered))


def write_version(outdir: Path, targets: list[str]) -> None:
    h = hashlib.sha256()
    for p in sorted(HERE.rglob("*.py")):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(HERE).as_posix().encode())
        h.update(p.read_bytes())
    (outdir / "VERSION").write_text(
        f"tilelang_version={tilelang_version()}\n"
        f"tilelang_commit={tilelang_commit()}\n"
        f"targets={','.join(targets)}\n"
        f"generator_sha256={h.hexdigest()}\n"
    )


def generate(outdir: Path, targets: list[str], only: str | None) -> None:
    specs = [s for s in kernels.all_kernels() if not only or re.search(only, s.name)]
    if not specs:
        raise SystemExit("no kernels matched")
    for target in targets:
        lowered = []
        for spec in specs:
            if target not in spec.targets:
                continue
            print(f"[{target}] {spec.name}", file=sys.stderr)
            lw = lower_spec(spec, target)
            if spec.variants and target == "metal":
                for suffix, build in spec.variants.items():
                    print(f"[{target}] {spec.name}__{suffix}", file=sys.stderr)
                    v = lower_spec(spec, target, suffix, build)
                    v.variant = suffix
                    if [p.name for p in v.params] != [p.name for p in lw.params]:
                        raise RuntimeError(f"{spec.name}__{suffix}: parameter list differs from the default variant")
                    lw.variants.append(v)
            lowered.append(lw)
        write_target(outdir, target, lowered)
    write_version(outdir, targets)


_WORKSPACE_ALIAS = re.compile(r"\bworkspace_\d+\b")


def normalise_source(text: str) -> list[str]:
    """TileLang's Metal codegen names the per-reduce threadgroup workspaces
    `workspace`, `workspace_1`, ... in an order that varies between runs
    (unordered-map iteration in shared-memory merging); they alias the same
    bytes. Fold them so --check compares the kernels, not the run."""
    return _WORKSPACE_ALIAS.sub("workspace", text).splitlines()


def diff_dirs(a: Path, b: Path) -> int:
    rc = 0
    def listing(d: Path) -> set[Path]:
        return {p.relative_to(d) for p in d.rglob("*") if p.is_file() and p.suffix not in (".o", ".d")}

    files = sorted(listing(a) | listing(b))
    for rel in files:
        fa, fb = a / rel, b / rel
        if not fa.exists() or not fb.exists():
            print(f"only in {'gen' if fb.exists() else 'regenerated'}: {rel}")
            rc = 1
            continue
        ta, tb = normalise_source(fa.read_text()), normalise_source(fb.read_text())
        if ta != tb:
            rc = 1
            sys.stdout.writelines(l + "\n" for l in difflib.unified_diff(tb, ta, f"gen/{rel}", f"regenerated/{rel}", lineterm=""))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(TARGET_EXT), help="comma-separated: metal,c")
    ap.add_argument("--only", help="regex on kernel name")
    ap.add_argument("--check", action="store_true", help="regenerate to a temp dir and diff against gen/")
    ap.add_argument("--out", type=Path, default=GEN)
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    if args.check:
        with tempfile.TemporaryDirectory(prefix="ds4-tl-gen-") as tmp:
            generate(Path(tmp), targets, args.only)
            rc = diff_dirs(Path(tmp), args.out)
            print("gen/ is up to date" if rc == 0 else "gen/ is stale: run `make gen-kernels`")
            return rc

    generate(args.out, targets, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
