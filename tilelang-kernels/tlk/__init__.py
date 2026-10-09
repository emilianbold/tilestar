"""Shared infrastructure for ds4's TileLang kernel sources.

Every module under ``kernels/`` exposes ``KERNELS``: a list of ``KernelSpec``.
``gen.py`` lowers each spec for each requested target and writes the generated
sources plus launch metadata into ``../gen/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

TARGETS = ("metal", "c")


@dataclass(frozen=True)
class KernelSpec:
    # Unique kernel name, becomes the device symbol (plus TileLang's suffix).
    name: str
    # Returns the T.prim_func for this kernel. Called once per target.
    build: Callable[[], object]
    # Output file stem under gen/<target>/ (groups related kernels).
    group: str
    # Targets this kernel is generated for. Defaults to all.
    targets: tuple[str, ...] = TARGETS
    # Free-form notes written as a comment above the kernel in gen/.
    notes: str = ""
    # Extra C expressions for the launch grid are derived from the PrimFunc;
    # kernels needing a hand-written grid rule can override here.
    grid_override: tuple[str, str, str] | None = None
    extra: dict = field(default_factory=dict)
    # Autotune variants: suffix -> build function (the default build is the
    # variant selected when no tuned configuration says otherwise). Kernels in
    # the same tune_group share one selection (they also share host code and
    # grid helpers), so they must declare the same suffixes.
    variants: dict[str, Callable[[], object]] | None = None
    tune_group: str | None = None
