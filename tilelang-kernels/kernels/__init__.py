from . import norm, q8, moe, dsv4, elementwise, utility, router, compressor, attention, indexer

MODULES = (norm, q8, moe, dsv4, elementwise, utility, router, compressor, attention, indexer)


def all_kernels():
    for mod in MODULES:
        yield from mod.KERNELS
