"""Quantisation primitives injected into TileLang kernels with T.import_source.

TileLang has no CPU dot-product lowering (no analogue of T.dp4a), so the
per-block int8 dots are hand-written here per ISA and called from the
TileLang kernel with T.call_extern, the way TileLang's own LOP3 fast-dequant
helpers work on CUDA. Each helper has AVX2, NEON and scalar bodies selected
by the host compiler's predefined macros, so one generated source compiles
on x86 and Apple silicon.
"""
