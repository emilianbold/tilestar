#!/usr/bin/env python3
"""Assemble the embedded Metal source for the machine-specific build.

The kernel source is fully regenerable from checked-in files:

  1. `prelude.metal` - the ds4 base prelude (includes, shared macros,
     quant structs) plus the two no-op housekeeping kernels.
  2. `gen/metal/tl_*.metal` - the TileLang-generated kernels, in the
     runtime's dispatch order (tl_prelude carries TileLang's MSL helpers).
  3. One-line no-op stand-ins for every hand-written kernel name the host
     looks up (extracted from the string literals in ds4_metal.m) that is
     not defined above, so availability checks, preconditions, and the two
     argument-encoder creations behave exactly as with the full upstream
     source. The stand-ins are never dispatched in the verified flows; if
     a workload dispatches one for real work, DS4_METAL_DUMMY_HAND_KERNELS
     names it in the exit report.
  4. `kernel_ds4_hand_noop` - the host-side fallback pipeline for lookups
     of names with no stand-in (dynamically built names).

Usage: assemble-metal-source.py <prelude.metal> <gen/metal> <output.metal>
"""
import re
import sys

# Runtime dispatch order (see the required_sources list in ds4_metal.m
# history): tl_prelude first, it carries TileLang's MSL helpers.
TL_ORDER = [
    'tl_prelude.metal', 'tl_norm.metal', 'tl_q8.metal', 'tl_moe.metal',
    'tl_dsv4.metal', 'tl_elementwise.metal', 'tl_utility.metal',
    'tl_router.metal', 'tl_compressor.metal', 'tl_attention.metal',
    'tl_indexer.metal', 'tl_q8gemm.metal',
]

# Stand-ins for these names must declare the buffer slots the init section
# creates argument encoders for (see the newArgumentEncoderWithBufferIndex
# call sites in ds4_metal.m).
ENCODER_BUFFERS = {
    'kernel_mul_mv_table_q4_K_pair_swiglu_f32': [2, 3],
    'kernel_mul_mv_table_q4_K_sum6_f32': [1],
}

NOOP = 'kernel_ds4_hand_noop'


def standin(name):
    bufs = ENCODER_BUFFERS.get(name)
    if bufs:
        params = ', '.join(f'device int *unused{b} [[buffer({b})]]' for b in bufs)
        return f'kernel void {name}({params}) {{ }}'
    return f'kernel void {name}(device int *unused [[buffer(0)]]) {{ }}'


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    prelude_path, gen_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    parts = [open(prelude_path).read().rstrip('\n') + '\n']
    for name in TL_ORDER:
        parts.append(f'\n// appended gen/metal/{name}\n'
                     + open(f'{gen_dir}/{name}').read().rstrip('\n') + '\n')
    src = ''.join(parts)

    # every hand-kernel name the host looks up as a string literal
    host = open('ds4_metal.m').read()
    host_names = sorted(set(re.findall(r'"(kernel_[A-Za-z0-9_]+)"', host)))

    # names already defined in the assembled source (directly or via a
    # host_name instantiation) get no stand-in
    defined = set(re.findall(r'kernel void (kernel_[A-Za-z0-9_]+)\(', src))
    defined |= set(re.findall(r'host_name\("(kernel_[A-Za-z0-9_]+)"', src))

    standins = [standin(n) for n in host_names if n not in defined and n != NOOP]
    src += ('\n/* No-op stand-ins for the hand-written kernels this\n'
            ' * machine-specific build does not carry (their sources live\n'
            ' * upstream; see README "Building the single-file server").\n'
            ' * Availability checks, preconditions, and the two\n'
            ' * argument-encoder creations behave as before; the ops\n'
            ' * themselves are covered by the TileLang kernels. */\n'
            + '\n'.join(standins) + '\n')
    src += ('\n/* Host-side fallback pipeline for lookups of names with no stand-in. */\n'
            f'kernel void {NOOP}(device int *unused [[buffer(0)]]) {{ }}\n')

    # sanity checks
    opens = len(re.findall(r'^\s*#(if|ifdef|ifndef)\b', src, re.M))
    closes = len(re.findall(r'^\s*#endif\b', src, re.M))
    assert opens == closes, f'preprocessor imbalance: {opens} vs {closes}'
    for k in ('kernel_touch_u8_stride', 'kernel_gpu_keepalive'):
        assert f'kernel void {k}(' in src, k

    open(out_path, 'w').write(src)
    print(f'{out_path}: {len(src)} bytes ({len(standins)} stand-ins, '
          f'{len(defined)} kernels defined)')


if __name__ == '__main__':
    main()
