#!/usr/bin/env python3
"""Whole-model autotune for the TileLang kernels (profile-guided).

Kernels declared with `variants` in kernels/*.py are generated in every
variant. Under DS4_TL_TUNE=1 ds4 rotates through the variants of each tune
group, one step per dispatch, so a single run of the real model measures
every variant in situ (real shapes, caches, concurrency). The encoder
timeline (DS4_METAL_ENCODER_TIMELINE) attributes GPU time per variant
symbol; this script runs a few representative workloads, estimates for
each group the total time of the workload set under each variant and
writes tuned/<gpu-key>.json with the selections and the evidence. ds4
loads that file at start-up (DS4_TL_TUNED=0 disables, DS4_TL_TUNED=<file>
overrides).

    python3 tilelang-kernels/tune.py --model ~/gguf/....gguf [--validate]

Run from the ds4 checkout (needs ./ds4 and gen/). Combinatorics: groups are
tuned independently (their GPU times add), so the cost is one pass of the
workloads, not the product of the variant counts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

WORKLOADS = [
    # name, extra ds4 args; prompts are the jotter's e2e prompts when present
    ("decode_short", ["-p", "Write a haiku about autumn.", "-n", "65"]),
    ("prefill_2604", ["--prompt-file", "/tmp/prompt2k.txt", "-n", "8"]),
    ("long_5107", ["--prompt-file", "/tmp/prompt5k.txt", "-n", "8"]),
]


def tune_groups() -> dict[str, dict]:
    """Parse gen/metal/tl_launch_meta.h: group -> {kernels: [...], suffixes: [...]}"""
    text = (ROOT / "gen/metal/tl_launch_meta.h").read_text()
    names = re.search(r"ds4_tl_tune_group_names\[\] = \{ (.*?) \};", text).group(1)
    groups = [g.strip().strip('"') for g in names.split(",") if g.strip() and g.strip() != "NULL"]
    info = {g: {"kernels": [], "suffixes": []} for g in groups}
    for gi, g in enumerate(groups):
        sufs = re.search(rf"ds4_tl_tune_suffixes_{gi}\[\] = \{{ (.*?) \}};", text).group(1)
        info[g]["suffixes"] = [x.strip().strip('"') for x in sufs.split(",") if x.strip() and x.strip() != "NULL"]
    for m in re.finditer(r"#define (TL_\w+)_TUNE_ID (\d+)", text):
        info[groups[int(m.group(2))]]["kernels"].append(m.group(1).lower())
    return info


INVALID: set[str] = set()   # variant symbols ds4 could not build on this device


def run(model: str, args: list[str], timeline: Path, env_extra: dict) -> str:
    env = dict(os.environ, DS4_TL_ALL="1", **env_extra)
    env["DS4_METAL_ENCODER_TIMELINE"] = str(timeline)
    p = subprocess.run(["./ds4", "-m", model, "--temp", "0", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise SystemExit(f"ds4 failed on {args}")
    for m in re.finditer(r"TileLang variant invalid: (\S+) \((\w+)\)", p.stderr):
        if m.group(1) not in INVALID:
            print(f"  invalid variant (not buildable here): {m.group(1)}")
        INVALID.add(m.group(1))
    m = re.search(r"prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s", p.stderr)
    return m.group(0) if m else "?"


SYM = re.compile(r"^(tl_[a-z0-9_]+?)(?:__([a-z0-9]+))?_kernel$")


def attribute(timeline: Path) -> dict[tuple[str, str, str], tuple[float, int]]:
    """(base kernel, caller, suffix) -> (total us, dispatches). The caller
    (host call site, timeline field 8) separates the shapes a kernel is used
    for (the 4096-wide Q8_0 GEMV serves both lm_head at ~950 us and 512-row
    projections at ~18 us; a per-kernel mean would just reflect which shape
    the rotation handed to which variant). Encoders with several kernels
    charge their whole time to each TileLang kernel inside (the other parts
    are the same for every variant of a kernel)."""
    acc: dict[tuple[str, str, str], list] = defaultdict(lambda: [0.0, 0])
    for line in timeline.read_text().splitlines():
        if not line.startswith("E "):
            continue
        f = line.split()
        dur = float(f[5])
        caller = f[7]
        for sym in f[-1].split("+"):
            m = SYM.match(sym)
            if not m:
                continue
            key = (m.group(1), caller, m.group(2) or "default")
            acc[key][0] += dur
            acc[key][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def gpu_key(model: str) -> str:
    p = subprocess.run(["./ds4", "-m", model, "-p", "hi", "-n", "1", "--temp", "0"], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, DS4_TL_TUNED="0"))
    m = re.search(r"Metal device ([^,\n]+)", p.stderr)
    name = m.group(1) if m else "unknown"
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", name.lower())).strip("-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", help="tuned config path (default tilelang-kernels/tuned/<gpu-key>.json)")
    ap.add_argument("--min-gain", type=float, default=0.01, help="keep the default unless a variant is this much faster (fraction)")
    ap.add_argument("--validate", action="store_true", help="after writing, run the workloads with defaults and with the tuned file")
    a = ap.parse_args()

    groups = tune_groups()
    key = gpu_key(a.model)
    out = Path(a.out) if a.out else HERE / "tuned" / f"{key}.json"
    print(f"gpu key {key}; {len(groups)} tune groups: {', '.join(groups)}")

    # measurements[workload][(kernel, suffix)] = (us, count)
    measurements: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        for name, args in WORKLOADS:
            if "--prompt-file" in args and not Path(args[args.index("--prompt-file") + 1]).exists():
                print(f"skip {name}: prompt file missing")
                continue
            tl = Path(td) / f"{name}.txt"
            rate = run(a.model, args, tl, {"DS4_TL_TUNE": "1"})
            measurements[name] = attribute(tl)
            print(f"  {name}: {rate}")

    # Groups whose kernels share an encoder with another tunable group (the
    # MoE section, for one) cannot be separated in a joint round-robin: the
    # encoder time mixes both rotations. Re-measure those groups alone
    # (DS4_TL_TUNE=<group>: only that group rotates).
    kernel_group = {k: g for g, info in groups.items() for k in info["kernels"]}
    entangled: set[str] = set()
    with tempfile.TemporaryDirectory() as td:
        for name, args in WORKLOADS:
            if name not in measurements:
                continue
            tl = Path(td) / f"{name}.txt"
            run(a.model, args, tl, {"DS4_TL_TUNE": "1"})
            for line in tl.read_text().splitlines():
                if not line.startswith("E "):
                    continue
                gs = set()
                for sym in line.split()[-1].split("+"):
                    m = SYM.match(sym)
                    if m and m.group(1) in kernel_group:
                        gs.add(kernel_group[m.group(1)])
                if len(gs) > 1:
                    entangled |= gs
        for g in sorted(entangled):
            print(f"  re-measuring entangled group {g} alone")
            for name, args in WORKLOADS:
                if name not in measurements:
                    continue
                tl = Path(td) / f"{g}_{name}.txt"
                run(a.model, args, tl, {"DS4_TL_TUNE": g})
                meas = attribute(tl)
                for k in groups[g]["kernels"]:
                    for key in [key for key in measurements[name] if key[0] == k]:
                        del measurements[name][key]
                    for key, v in meas.items():
                        if key[0] == k:
                            measurements[name][key] = v

    selection, evidence = {}, {}
    print()
    print(f"{'group':16s} {'variant':10s} {'est. total ms':>14s} {'vs default':>10s}")
    for g, info in groups.items():
        est = {}
        invalid = [s for s in info["suffixes"] if any(f"{k}__{s}_kernel" in INVALID for k in info["kernels"])]
        for suf in info["suffixes"]:
            if suf in invalid:
                continue
            total = 0.0
            ok = True
            per_kernel = {}
            for w, meas in measurements.items():
                cells = sorted({(k, c) for (k, c, _s) in meas if k in info["kernels"]})
                for k, c in cells:
                    counts = [meas[(k, c, s)][1] for s in info["suffixes"] if (k, c, s) in meas]
                    n_all = sum(counts)
                    if (k, c, suf) not in meas:
                        # too few dispatches at this call site to have reached this variant:
                        # neutral (the cell's mean over the variants that were seen)
                        us_all = sum(meas[(k, c, s)][0] for s in info["suffixes"] if (k, c, s) in meas)
                        total += us_all / n_all * n_all / 1e3
                        continue
                    us, n = meas[(k, c, suf)]
                    mean = us / n
                    total += mean * n_all / 1e3
                    per_kernel[f"{w}:{k}@{c}"] = {"mean_us": round(mean, 2), "dispatches": n}
            if ok and per_kernel:
                est[suf] = (total, per_kernel)
        if "default" not in est:
            print(f"{g:16s} (not dispatched by the workloads)")
            continue
        base = est["default"][0]
        best = min(est, key=lambda s: est[s][0])
        if best != "default" and est[best][0] > base * (1 - a.min_gain):
            best = "default"
        selection[g] = best
        evidence[g] = {s: {"est_total_ms": round(v[0], 3), "per_kernel": v[1]} for s, v in est.items()}
        for s_ in invalid:
            evidence[g][s_] = {"invalid": "pipeline failed on this device"}
        for s in info["suffixes"]:
            if s in est:
                mark = " <-" if s == best else ""
                print(f"{g:16s} {s:10s} {est[s][0]:14.3f} {est[s][0] / base:10.3f}{mark}")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "gpu": key,
        "date": dt.date.today().isoformat(),
        "model": os.path.basename(a.model),
        "workloads": [w for w, _ in WORKLOADS if w in measurements],
        "note": "selection: tune group -> variant suffix (ds4 reads only these pairs); evidence: estimated total GPU ms of the workload set per variant",
        "selection": selection,
        "evidence": evidence,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\nwrote {out}: {sum(1 for v in selection.values() if v != 'default')} of {len(selection)} groups changed")

    if a.validate:
        print("\nvalidation (DS4_TL_ALL=1): defaults vs tuned")
        with tempfile.TemporaryDirectory() as td:
            for name, args in WORKLOADS:
                if name not in measurements:
                    continue
                for label, env in (("default", {"DS4_TL_TUNED": "0"}), ("tuned", {"DS4_TL_TUNED": str(out)})):
                    rates = [run(a.model, args, Path(td) / f"v_{name}_{label}_{i}.txt", env) for i in range(2)]
                    print(f"  {name:14s} {label:8s} {' | '.join(rates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
