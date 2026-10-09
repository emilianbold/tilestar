/* ds4_tl_metal.h - host glue for the TileLang-generated Metal kernels.
 *
 * Included by ds4_metal.m only. The kernels live in gen/metal/tl_*.metal
 * (concatenated into the Metal library with the hand-written shaders) and
 * their launch contract in gen/metal/tl_launch_meta.h; both are produced by
 * tilelang-kernels/gen.py. See tilelang-kernels/README.md.
 *
 * Conventions enforced here:
 *   - buffers bind to [[buffer(i)]] in the order recorded in tl_launch_meta.h
 *     (which is NOT the Python parameter order);
 *   - scalars travel in the generated <kernel>_args_t struct via setBytes at
 *     the index after the last buffer;
 *   - TileLang types buffers by the vector width it chose (float4 ...), so
 *     every bound offset must be a multiple of DS4_TL_BUFFER_ALIGN;
 *   - threadgroup memory is declared statically in the kernel, so there is
 *     no setThreadgroupMemoryLength.
 *
 * Each kernel family has an env toggle (DS4_TL_<OP>=1) or the blanket
 * DS4_TL=1; the ds4_gpu_* entry points fall back to the hand-written kernels
 * whenever the toggle is off, the shape is not specialised, or a buffer
 * offset is misaligned. ds4_gpu_tl_* entry points call the TileLang kernel
 * unconditionally so tests can compare both paths without touching the
 * environment.
 */
#ifndef DS4_TL_METAL_H
#define DS4_TL_METAL_H

#include "gen/metal/tl_launch_meta.h"
#include "ds4_iq2_tables.h"

#define DS4_TL_BUFFER_ALIGN 16u

typedef struct {
    id<MTLBuffer> buf;
    NSUInteger    off;
    /* Optional inline data (setBytes, <= 4 KB) instead of a buffer. */
    const void   *bytes;
    NSUInteger    bytes_len;
} ds4_tl_buf;

/* (buffer, offset) binding; the inline-data fields stay empty. Spelled out
 * so -Wmissing-field-initializers stays quiet at the ~400 call sites. */
static inline ds4_tl_buf ds4_tl_mk(id<MTLBuffer> buf, NSUInteger off) {
    ds4_tl_buf b;
    b.buf = buf;
    b.off = off;
    b.bytes = NULL;
    b.bytes_len = 0;
    return b;
}

/* Toggle state, read once. DS4_TL=1 enables the families at parity or
 * better, DS4_TL_ALL=1 every family (for an all-TileLang comparison),
 * DS4_TL_<FAMILY>=1 one family. */
static int ds4_tl_env_on(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && v[0] != '0';
}
static int ds4_tl_env_flag(const char *name) {
    return ds4_tl_env_on(name) || ds4_tl_env_on("DS4_TL") || ds4_tl_env_on("DS4_TL_ALL");
}

/* Kernel families, for the per-family env toggles and the test hook. */
enum ds4_tl_family {
    DS4_TL_FAMILY_RMS_NORM = 0,
    DS4_TL_FAMILY_Q8_GEMV,
    DS4_TL_FAMILY_MOE,
    DS4_TL_FAMILY_GEMM,
    DS4_TL_FAMILY_ROPE,
    DS4_TL_FAMILY_HC,
    DS4_TL_FAMILY_ELEMENTWISE,
    DS4_TL_FAMILY_KV,
    DS4_TL_FAMILY_ROUTER,
    DS4_TL_FAMILY_ATTN,
    DS4_TL_FAMILY_UTILITY,
    DS4_TL_FAMILY_F16_GEMV,
    DS4_TL_FAMILY_GEMM_SMALL,   /* opt-in: 32..63-token GEMMs with narrow outputs (TileLang pads to 64 rows: ~1.5x of the hand NAX GEMM) */
    DS4_TL_FAMILY_COUNT
};

static const char *const ds4_tl_family_env[DS4_TL_FAMILY_COUNT] = {
    "DS4_TL_RMS_NORM", "DS4_TL_Q8_GEMV", "DS4_TL_MOE", "DS4_TL_GEMM", "DS4_TL_ROPE",
    "DS4_TL_HC", "DS4_TL_ELEMENTWISE", "DS4_TL_KV", "DS4_TL_ROUTER", "DS4_TL_ATTN",
    "DS4_TL_UTILITY", "DS4_TL_F16_GEMV", "DS4_TL_GEMM_SMALL",
};
/* Families still slower than the hand-written kernels are not part of
 * DS4_TL=1 and need their own variable (or DS4_TL_ALL=1): MOE (1.09x) and
 * ROUTER (decode project+select is two dispatches, 1.4x of the fused
 * last-threadgroup kernel; ~0.5% of a decode token). */
static const int ds4_tl_family_optin_only[DS4_TL_FAMILY_COUNT] = {
    0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
};

static int g_tl_force[DS4_TL_FAMILY_COUNT];   /* test hook, see ds4_gpu_tl_force */
/* Some ds4 dispatch chains exist only to be bit-identical with a fused
 * hand-written kernel (ds4_test --metal-kernels pins them). TileLang
 * reductions have their own summation order, so those chains stay
 * hand-written: wrap them in ds4_tl_suppress_begin/end. */
static int g_tl_suppress;
static void ds4_tl_suppress_begin(void) { g_tl_suppress++; }
static void ds4_tl_suppress_end(void) { g_tl_suppress--; }

static int ds4_tl_family_enabled(enum ds4_tl_family f) {
    (void)f;
    return 1;
    static int initialized[DS4_TL_FAMILY_COUNT], enabled[DS4_TL_FAMILY_COUNT];
    if (g_tl_suppress) return 0;
    if (g_tl_force[f]) return 1;
    if (!initialized[f]) {
        if (ds4_tl_family_optin_only[f]) {
            enabled[f] = ds4_tl_env_on(ds4_tl_family_env[f]) || ds4_tl_env_on("DS4_TL_ALL");
        } else {
            enabled[f] = ds4_tl_env_flag(ds4_tl_family_env[f]);
        }
        initialized[f] = 1;
    }
    return enabled[f];
}

static int ds4_tl_rms_norm_enabled(void) {
    return ds4_tl_family_enabled(DS4_TL_FAMILY_RMS_NORM);
}

static int ds4_tl_q8_gemv_enabled(void) {
    return ds4_tl_family_enabled(DS4_TL_FAMILY_Q8_GEMV);
}

/* Not part of DS4_TL=1 yet: the routed gate/up kernel is at 1.16x of the
 * hand-written one in isolation and costs ~10% on the MoE section of a
 * decode step. Opt in explicitly while it is being tuned. */
static int ds4_tl_moe_enabled(void) {
    return ds4_tl_family_enabled(DS4_TL_FAMILY_MOE);
}

/* Prefill GEMM tier: DS4_TL_GEMM=1 / DS4_TL=1, and only with the Metal 4
 * tensor API (the kernels are not in the library otherwise). */
static int ds4_tl_gemm_enabled(void) {
    return 1;
    return ds4_tl_family_enabled(DS4_TL_FAMILY_GEMM) && g_metal4_tensor_api_enabled;
}

static int ds4_tl_aligned(NSUInteger off) {
    return (off % DS4_TL_BUFFER_ALIGN) == 0;
}

static void ds4_tl_dispatch(
        id<MTLComputeCommandEncoder> enc,
        id<MTLComputePipelineState> pso,
        const ds4_tl_buf *bufs,
        unsigned n_bufs,
        const void *args,
        size_t args_len,
        const uint32_t grid[3],
        unsigned tx, unsigned ty, unsigned tz) {
    [enc setComputePipelineState:pso];
    for (unsigned i = 0; i < n_bufs; i++) {
        if (bufs[i].bytes) [enc setBytes:bufs[i].bytes length:bufs[i].bytes_len atIndex:i];
        else [enc setBuffer:bufs[i].buf offset:bufs[i].off atIndex:i];
    }
    [enc setBytes:args length:args_len atIndex:n_bufs];
    [enc dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
         threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}

/* ---- rms_norm ------------------------------------------------------------
 * Specialised on n (the row width); rows is a runtime scalar.
 */

#define DS4_TL_MAX_VARIANTS 8

typedef struct {
    const char *symbol;                /* default variant */
    unsigned    tx, ty, tz;            /* threads of the variant selected by the last ds4_tl_pipeline() */
    id<MTLComputePipelineState> pso;   /* memoised on first use (default variant) */
    const ds4_tl_variant_meta *variants;   /* autotune variants, index 0 = default; NULL if none */
    int         n_variants;
    int         tune_id;               /* tune group, -1 if none */
    id<MTLComputePipelineState> pso_v[DS4_TL_MAX_VARIANTS];
    unsigned char bad_v[DS4_TL_MAX_VARIANTS];   /* variant pipelines that failed to build */
} ds4_tl_kernel;

#define DS4_TL_KERNEL(PREFIX) \
    { PREFIX##_SYMBOL, PREFIX##_THREADS_X, PREFIX##_THREADS_Y, PREFIX##_THREADS_Z, nil, \
      PREFIX##_VARIANTS, PREFIX##_NUM_VARIANTS, PREFIX##_TUNE_ID, { nil }, { 0 } }

/* ---- autotune variant selection ---------------------------------------------
 * Every tune group has one selected variant index. Sources, in order:
 *   DS4_TL_TUNE=1          round-robin over the variants, one step per
 *                          pipeline fetch (the encoder timeline then shows
 *                          every variant's in-situ time: tilelang-kernels/tune.py)
 *   DS4_TL_TUNED=0         defaults only
 *   DS4_TL_TUNED=<file>    a tuned configuration file
 *   tilelang-kernels/tuned/<gpu-key>.json   (gpu-key from the device name,
 *                          e.g. apple-m5-max), if present
 * A configuration file is JSON; the host only reads its "selection" pairs,
 * "group": "suffix". The grid helpers in tl_launch_meta.h ask
 * ds4_tl_variant_index(): encoders must fetch the pipeline before computing
 * the grid so both see the same selection (DS4_TL_TUNE_CHECK=1 verifies). */
static int g_tl_variant_sel[DS4_TL_NUM_TUNE_GROUPS > 0 ? DS4_TL_NUM_TUNE_GROUPS : 1];
static unsigned g_tl_variant_rr[DS4_TL_NUM_TUNE_GROUPS > 0 ? DS4_TL_NUM_TUNE_GROUPS : 1];
static int g_tl_tune_mode;        /* 1: round-robin over every group; 2: over the group g_tl_tune_only */
static int g_tl_tune_only = -1;
static int g_tl_tuned_loaded;

static const char *ds4_tl_gpu_key(char *buf, size_t n) {
    const char *name = g_device && [g_device name] ? [[g_device name] UTF8String] : "unknown";
    size_t j = 0;
    for (size_t i = 0; name[i] && j + 1 < n; i++) {
        const char c = name[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) buf[j++] = c;
        else if (c >= 'A' && c <= 'Z') buf[j++] = (char)(c - 'A' + 'a');
        else if (j && buf[j - 1] != '-') buf[j++] = '-';
    }
    while (j && buf[j - 1] == '-') j--;
    buf[j] = 0;
    return buf;
}

static void ds4_tl_tuned_apply(const char *text, const char *path) {
    /* scan "group": "suffix" string pairs */
    int applied = 0;
    const char *p = text;
    while ((p = strchr(p, '"'))) {
        const char *k0 = p + 1, *k1 = strchr(k0, '"');
        if (!k1) break;
        const char *q = k1 + 1;
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
        if (*q != ':') { p = k1 + 1; continue; }
        q++;
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
        if (*q != '"') { p = k1 + 1; continue; }
        const char *v0 = q + 1, *v1 = strchr(v0, '"');
        if (!v1) break;
        for (int g = 0; g < DS4_TL_NUM_TUNE_GROUPS; g++) {
            const char *gname = ds4_tl_tune_group_names[g];
            if (strlen(gname) != (size_t)(k1 - k0) || strncmp(gname, k0, (size_t)(k1 - k0)) != 0) continue;
            int idx = -1;
            for (int i = 0; ds4_tl_tune_group_suffixes[g][i]; i++) {
                const char *sfx = ds4_tl_tune_group_suffixes[g][i];
                if (strlen(sfx) == (size_t)(v1 - v0) && strncmp(sfx, v0, (size_t)(v1 - v0)) == 0) { idx = i; break; }
            }
            if (idx >= 0) { g_tl_variant_sel[g] = idx; applied++; }
            else fprintf(stderr, "ds4: TileLang tuned config %s: unknown variant '%.*s' for group %s\n", path, (int)(v1 - v0), v0, gname);
        }
        p = v1 + 1;
    }
    fprintf(stderr, "ds4: TileLang tuned config %s: %d of %d groups set\n", path, applied, DS4_TL_NUM_TUNE_GROUPS);
}

static void ds4_tl_tuned_init(void) {
    if (g_tl_tuned_loaded) return;
    g_tl_tuned_loaded = 1;
    const char *tune = getenv("DS4_TL_TUNE");
    if (tune && tune[0] && tune[0] != '0') {
        g_tl_tune_mode = 1;
        for (int g = 0; g < DS4_TL_NUM_TUNE_GROUPS; g++)
            if (strcmp(ds4_tl_tune_group_names[g], tune) == 0) { g_tl_tune_mode = 2; g_tl_tune_only = g; }
        fprintf(stderr, "ds4: TileLang autotune mode: round-robin over kernel variants%s%s\n",
                g_tl_tune_mode == 2 ? " of group " : "", g_tl_tune_mode == 2 ? tune : "");
        return;
    }
    const char *force = getenv("DS4_TL_VARIANT");   /* test hook: every group that has this suffix uses it */
    if (force && force[0]) {
        int n = 0;
        for (int g = 0; g < DS4_TL_NUM_TUNE_GROUPS; g++)
            for (int i = 0; ds4_tl_tune_group_suffixes[g][i]; i++)
                if (strcmp(ds4_tl_tune_group_suffixes[g][i], force) == 0) { g_tl_variant_sel[g] = i; n++; }
        fprintf(stderr, "ds4: TileLang variant '%s' forced in %d tune groups\n", force, n);
        return;
    }
    const char *env = getenv("DS4_TL_TUNED");
    if (env && env[0] == '0' && !env[1]) return;
    char path[512];
    if (env && env[0]) snprintf(path, sizeof(path), "%s", env);
    else { char key[64]; snprintf(path, sizeof(path), "tilelang-kernels/tuned/%s.json", ds4_tl_gpu_key(key, sizeof(key))); }
    FILE *f = fopen(path, "rb");
    if (!f) {
        /* Embedded tuned config (ds4_kernels_gz.c): used when the file is absent. */
        if (!env || !env[0]) {
            extern const unsigned long ds4_tuned_gz_len;
            extern const unsigned long ds4_tuned_src_len;
            extern const unsigned char ds4_tuned_gz[];
            extern int puff(unsigned char *dest, unsigned long *destlen,
                            const unsigned char *source, unsigned long *sourcelen);
            char *text = (char *)malloc(ds4_tuned_src_len + 1);
            if (text) {
                unsigned long destlen = ds4_tuned_src_len, sourcelen = ds4_tuned_gz_len;
                if (puff((unsigned char *)text, &destlen, ds4_tuned_gz, &sourcelen) == 0 &&
                    destlen == ds4_tuned_src_len) {
                    text[destlen] = 0;
                    ds4_tl_tuned_apply(text, "embedded tuned config");
                }
                free(text);
            }
        }
        if (env && env[0]) fprintf(stderr, "ds4: TileLang tuned config %s not found\n", path);
        return;
    }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = n > 0 && n < (1 << 22) ? malloc((size_t)n + 1) : NULL;
    if (text && fread(text, 1, (size_t)n, f) == (size_t)n) { text[n] = 0; ds4_tl_tuned_apply(text, path); }
    free(text);
    fclose(f);
}

static int __attribute__((unused)) ds4_tl_variant_index(int tune_id) {
    if (tune_id < 0 || tune_id >= DS4_TL_NUM_TUNE_GROUPS) return 0;
    return g_tl_variant_sel[tune_id];
}

/* Pipeline lookup through the shared NSCache keyed by symbol name, memoised
 * in the kernel record like the other hot ds4 pipelines. Selects the tune
 * variant and publishes its thread counts in k->tx/ty/tz. */
static id<MTLComputePipelineState> ds4_tl_pipeline(ds4_tl_kernel *k) {
    if (k->n_variants > 1 && k->tune_id >= 0) {
        ds4_tl_tuned_init();
        const int rotating = g_tl_tune_mode == 1 || (g_tl_tune_mode == 2 && k->tune_id == g_tl_tune_only);
        for (int attempt = 0; attempt < k->n_variants; attempt++) {
            if (rotating) {
                g_tl_variant_sel[k->tune_id] = (int)(g_tl_variant_rr[k->tune_id]++ % (unsigned)k->n_variants);
                if (k->bad_v[g_tl_variant_sel[k->tune_id]]) continue;   /* invalid variants are skipped, never substituted */
            }
            const int idx = g_tl_variant_sel[k->tune_id] < k->n_variants ? g_tl_variant_sel[k->tune_id] : 0;
            const ds4_tl_variant_meta *v = &k->variants[idx];
            if (!k->pso_v[idx] || ds4_gpu_disable_hot_pipeline_statics()) k->pso_v[idx] = ds4_gpu_get_pipeline(v->symbol);
            if (k->pso_v[idx]) {
                k->tx = v->tx; k->ty = v->ty; k->tz = v->tz;
                return k->pso_v[idx];
            }
            /* The variant cannot be built on this device (e.g. threadgroup memory). */
            if (!k->bad_v[idx]) {
                k->bad_v[idx] = 1;
                fprintf(stderr, "ds4: TileLang variant invalid: %s (%s) - pipeline failed%s\n", v->symbol, v->suffix,
                        rotating ? ", skipped in the rotation" : "; the tuned selection for this group is dropped");
            }
            if (!rotating) {
                if (idx == 0) return nil;        /* the default itself failed: decline, hand-written path */
                g_tl_variant_sel[k->tune_id] = 0;   /* explicit: selection dropped, back to default (the grid helper must agree) */
            }
        }
        return nil;
    }
    if (!k->pso || ds4_gpu_disable_hot_pipeline_statics()) {
        k->pso = ds4_gpu_get_pipeline(k->symbol);
    }
    return k->pso;
}

static ds4_tl_kernel *ds4_tl_rms_norm_kernel_for(uint32_t n, int weighted) {
    static ds4_tl_kernel k[] = {
        DS4_TL_KERNEL(TL_RMS_NORM_WEIGHT_F32_4096),  DS4_TL_KERNEL(TL_RMS_NORM_PLAIN_F32_4096),
        DS4_TL_KERNEL(TL_RMS_NORM_WEIGHT_F32_2048),  DS4_TL_KERNEL(TL_RMS_NORM_PLAIN_F32_2048),
        DS4_TL_KERNEL(TL_RMS_NORM_WEIGHT_F32_512),   DS4_TL_KERNEL(TL_RMS_NORM_PLAIN_F32_512),
        DS4_TL_KERNEL(TL_RMS_NORM_WEIGHT_F32_16384), DS4_TL_KERNEL(TL_RMS_NORM_PLAIN_F32_16384),
        DS4_TL_KERNEL(TL_RMS_NORM_WEIGHT_F32_128),   DS4_TL_KERNEL(TL_RMS_NORM_PLAIN_F32_128),
        DS4_TL_KERNEL(TL_RMS_NORM_WEIGHT_F32_1024),  DS4_TL_KERNEL(TL_RMS_NORM_PLAIN_F32_1024),
    };
    int base;
    switch (n) {
    case 4096:  base = 0; break;
    case 2048:  base = 2; break;
    case 512:   base = 4; break;
    case 16384: base = 6; break;
    case 128:   base = 8; break;
    case 1024:  base = 10; break;
    default:    return NULL;
    }
    return &k[base + (weighted ? 0 : 1)];
}

/* The args structs of all rms_norm variants have the same layout. */
typedef tl_rms_norm_weight_f32_4096_args_t ds4_tl_rms_norm_args_t;

/* Encode one rms_norm dispatch. w is ignored for the plain variant.
 * Returns 0 (nothing encoded) when the shape is not specialised, the
 * pipeline is missing or an offset is misaligned; the caller then takes the
 * hand-written path. */
static int ds4_tl_encode_rms_norm(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out, ds4_tl_buf x, ds4_tl_buf w, int weighted,
        uint32_t n, uint32_t rows, float eps) {
    ds4_tl_kernel *k = ds4_tl_rms_norm_kernel_for(n, weighted);
    if (!k) return 0;
    if (!ds4_tl_aligned(out.off) || !ds4_tl_aligned(x.off) ||
        (weighted && !ds4_tl_aligned(w.off))) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    ds4_tl_rms_norm_args_t args;
    memset(&args, 0, sizeof(args));
    args.eps[0] = eps;
    args.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_rms_norm_weight_f32_4096_grid(grid, eps, (int32_t)rows);

    ds4_tl_buf bufs[3];
    unsigned n_bufs = 0;
    bufs[n_bufs++] = out;              /* buffer order: out, [w,] x */
    if (weighted) bufs[n_bufs++] = w;
    bufs[n_bufs++] = x;
    ds4_tl_dispatch(enc, pso, bufs, n_bufs, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* ---- Q8_0 x f32 GEMV (single-token decode) ------------------------------
 * Specialised on in_dim; out_dim is a runtime scalar and must be a multiple
 * of the kernel's row tile. The weight buffer is viewed as uint16 (GGUF
 * Q8_0 blocks are 2-byte aligned), x/out as float.
 */

static ds4_tl_kernel *ds4_tl_q8_0_gemv_f32_kernel_for(uint64_t in_dim) {
    static ds4_tl_kernel k[] = {
        DS4_TL_KERNEL(TL_Q8_0_GEMV_F32_4096),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_F32_1024),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_F32_2048),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_F32_12288),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_F32_256),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_F32_8192),
    };
    switch (in_dim) {
    case 4096:  return &k[0];
    case 1024:  return &k[1];
    case 2048:  return &k[2];
    case 12288: return &k[3];   /* DSpark support model main_proj */
    case 256:   return &k[4];   /* DSpark Markov heads */
    case 8192:  return &k[5];   /* attn_output_b as a plain GEMV (1-token batch paths) */
    default:    return NULL;
    }
}

#define DS4_TL_Q8_0_GEMV_F32_ROWS 8   /* alignment required of out dims: the largest row tile among the tune variants (default 4) */

static int ds4_tl_encode_q8_0_gemv_f32(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
        uint64_t in_dim, uint64_t out_dim) {
    ds4_tl_kernel *k = ds4_tl_q8_0_gemv_f32_kernel_for(in_dim);
    if (!k || out_dim == 0 || out_dim > INT32_MAX ||
        (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0) {
        return 0;
    }
    if ((out.off % 4u) != 0 || (x.off % 4u) != 0 || (w.off % 2u) != 0) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    tl_q8_0_gemv_f32_4096_args_t args;   /* same layout for every in_dim */
    memset(&args, 0, sizeof(args));
    args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gemv_f32_4096_grid(grid, (int32_t)out_dim);

    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* Row-exact batch form (ds4_gpu_matmul_q8_0_decode_rows_exact_tensor): out[r] = w . x[r] for
 * n_rows activation rows with exactly the single-token kernel's per-row arithmetic (same tune
 * group, so the tuned variant stays the same for both forms). */
static ds4_tl_kernel *ds4_tl_q8_0_gemv_rows_f32_kernel_for(uint64_t in_dim) {
    static ds4_tl_kernel k[] = {
        DS4_TL_KERNEL(TL_Q8_0_GEMV_ROWS_F32_4096),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_ROWS_F32_1024),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_ROWS_F32_2048),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_ROWS_F32_12288),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_ROWS_F32_256),
        DS4_TL_KERNEL(TL_Q8_0_GEMV_ROWS_F32_8192),
    };
    switch (in_dim) {
    case 4096:  return &k[0];
    case 1024:  return &k[1];
    case 2048:  return &k[2];
    case 12288: return &k[3];
    case 256:   return &k[4];
    case 8192:  return &k[5];
    default:    return NULL;
    }
}

static int ds4_tl_encode_q8_0_gemv_rows_f32(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_rows) {
    ds4_tl_kernel *k = ds4_tl_q8_0_gemv_rows_f32_kernel_for(in_dim);
    if (!k || out_dim == 0 || out_dim > INT32_MAX || n_rows == 0 || n_rows > (uint32_t)INT32_MAX ||
        (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0) {
        return 0;
    }
    if ((out.off % 4u) != 0 || (x.off % 4u) != 0 || (w.off % 2u) != 0) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    tl_q8_0_gemv_rows_f32_4096_args_t args;   /* same layout for every in_dim */
    memset(&args, 0, sizeof(args));
    args.n_rows[0] = (int32_t)n_rows; args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gemv_rows_f32_4096_grid(grid, args.n_rows[0], args.out_dim[0]);

    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* ---- Q8_0 x f32 GEMV + hyper-connection expand (attn_output_b) ---------- */

static ds4_tl_kernel *ds4_tl_q8_0_gemv_hc_expand4_kernel_for(uint64_t in_dim, uint32_t n_hc) {
    static ds4_tl_kernel k8192 = DS4_TL_KERNEL(TL_Q8_0_GEMV_HC_EXPAND4_F32_8192);
    if (n_hc != 4) return NULL;
    return in_dim == 8192 ? &k8192 : NULL;
}

static int ds4_tl_encode_q8_0_gemv_hc_expand4(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out_hc, ds4_tl_buf block_out, ds4_tl_buf w, ds4_tl_buf x,
        ds4_tl_buf residual, ds4_tl_buf split,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_hc) {
    ds4_tl_kernel *k = ds4_tl_q8_0_gemv_hc_expand4_kernel_for(in_dim, n_hc);
    if (!k || out_dim == 0 || out_dim > INT32_MAX ||
        (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0) {
        return 0;
    }
    if ((out_hc.off % 4u) != 0 || (block_out.off % 4u) != 0 || (x.off % 4u) != 0 ||
        (residual.off % 4u) != 0 || (split.off % 4u) != 0 || (w.off % 2u) != 0) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    tl_q8_0_gemv_hc_expand4_f32_8192_args_t args;
    memset(&args, 0, sizeof(args));
    args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gemv_hc_expand4_f32_8192_grid(grid, (int32_t)out_dim);

    /* buffer order: block_out, out_hc, residual, split, w, x */
    const ds4_tl_buf bufs[6] = { block_out, out_hc, residual, split, w, x };
    ds4_tl_dispatch(enc, pso, bufs, 6, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* ---- grouped Q8_0 x f32 GEMV (attn_output_a: n_groups x (group_dim -> rank)) */

static ds4_tl_kernel *ds4_tl_q8_0_gemv_grouped_kernel_for(uint64_t in_dim) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_Q8_0_GEMV_GROUPED_F32_4096);
    return in_dim == 4096 ? &k4096 : NULL;
}

static int ds4_tl_encode_q8_0_gemv_grouped(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_groups) {
    ds4_tl_kernel *k = ds4_tl_q8_0_gemv_grouped_kernel_for(in_dim);
    if (!k || out_dim == 0 || out_dim > INT32_MAX || n_groups == 0 ||
        (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0) {
        return 0;
    }
    if ((out.off % 4u) != 0 || (x.off % 4u) != 0 || (w.off % 2u) != 0) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    tl_q8_0_gemv_grouped_f32_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.n_groups[0] = (int32_t)n_groups;
    args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gemv_grouped_f32_4096_grid(grid, (int32_t)n_groups, (int32_t)out_dim);

    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* ---- routed MoE gate/up, IQ2_XXS x f32, SwiGLU * route weight ---------- */

/* The IQ2_XXS grid as 2048 floats (the 256 x 8 codebook bytes, converted)
 * followed by ksigns (128 bytes), uploaded once. */
#define DS4_TL_IQ2_GRID_FLOATS 2048
#define DS4_TL_IQ2_KSIGNS_OFFSET (DS4_TL_IQ2_GRID_FLOATS * sizeof(float))
/* the same grid as 256 x 8 packed bytes (2 KB; the packed GEMM stages this one) */
#define DS4_TL_IQ2_GRID_WORDS_OFFSET (DS4_TL_IQ2_KSIGNS_OFFSET + sizeof(ds4_ksigns_iq2xs))
static id<MTLBuffer> ds4_tl_iq2_tables_buffer(void) {
    static id<MTLBuffer> buf;
    if (!buf) {
        uint8_t bytes[DS4_TL_IQ2_GRID_WORDS_OFFSET + sizeof(ds4_iq2xxs_grid)];
        float *grid = (float *)bytes;
        for (int g = 0; g < 256; g++) {
            const uint8_t *row = (const uint8_t *)&ds4_iq2xxs_grid[g];
            for (int j = 0; j < 8; j++) grid[g * 8 + j] = (float)row[j];
        }
        memcpy(bytes + DS4_TL_IQ2_KSIGNS_OFFSET, ds4_ksigns_iq2xs, sizeof(ds4_ksigns_iq2xs));
        memcpy(bytes + DS4_TL_IQ2_GRID_WORDS_OFFSET, ds4_iq2xxs_grid, sizeof(ds4_iq2xxs_grid));
        buf = [g_device newBufferWithBytes:bytes length:sizeof(bytes)
                                   options:MTLResourceStorageModeShared];
    }
    return buf;
}

static ds4_tl_kernel *ds4_tl_moe_gate_up_iq2_xxs_kernel_for(uint64_t in_dim) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_MOE_GATE_UP_IQ2_XXS_SWIGLU_4096);
    return in_dim == 4096 ? &k4096 : NULL;
}

#define DS4_TL_MOE_GATE_UP_ROWS 4

/* gate_out/up_out/mid_out: [n_used][mid_dim] f32. wg/wu: expert matrices
 * back to back from expert 0 (the TP-sharded case is not handled). ids:
 * int32[n_used] expert indices, route_w: f32[n_used]. */
static int ds4_tl_encode_moe_gate_up_iq2_xxs(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf gate_out, ds4_tl_buf up_out, ds4_tl_buf mid_out,
        ds4_tl_buf wg, ds4_tl_buf wu, ds4_tl_buf x,
        ds4_tl_buf ids, ds4_tl_buf route_w,
        uint64_t in_dim, uint64_t mid_dim, uint32_t n_used, float clamp) {
    ds4_tl_kernel *k = ds4_tl_moe_gate_up_iq2_xxs_kernel_for(in_dim);
    if (!k || mid_dim == 0 || mid_dim > INT32_MAX || n_used == 0 ||
        (mid_dim % DS4_TL_MOE_GATE_UP_ROWS) != 0) {
        return 0;
    }
    if ((gate_out.off % 4u) != 0 || (up_out.off % 4u) != 0 || (mid_out.off % 4u) != 0 ||
        (x.off % 4u) != 0 || (ids.off % 4u) != 0 || (route_w.off % 4u) != 0 ||
        (wg.off % 2u) != 0 || (wu.off % 2u) != 0) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    id<MTLBuffer> tables = ds4_tl_iq2_tables_buffer();
    if (!pso || !tables) return 0;

    tl_moe_gate_up_iq2_xxs_swiglu_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.clamp[0] = clamp;
    args.n_used[0] = (int32_t)n_used;
    args.out_dim[0] = (int32_t)mid_dim;
    uint32_t grid[3];
    tl_moe_gate_up_iq2_xxs_swiglu_4096_grid(grid, clamp, (int32_t)n_used, (int32_t)mid_dim);

    const ds4_tl_buf grid_t = ds4_tl_mk(tables, 0);
    const ds4_tl_buf ksigns_t = ds4_tl_mk(tables, DS4_TL_IQ2_KSIGNS_OFFSET);
    /* buffer order: gate_out, grid, ids, ksigns, mid_out, route_w, up_out, wg, wu, x */
    const ds4_tl_buf bufs[10] = { gate_out, grid_t, ids, ksigns_t, mid_out, route_w, up_out, wg, wu, x };
    ds4_tl_dispatch(enc, pso, bufs, 10, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* Batch form (2..31 tokens of a prefill chunk): x [n_tok][in_dim], ids /
 * route_w [n_tok][n_used], gate/up/mid [n_tok][n_used][mid_dim] f32. */
static int ds4_tl_encode_moe_gate_up_iq2_xxs_batch(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf gate_out, ds4_tl_buf up_out, ds4_tl_buf mid_out,
        ds4_tl_buf wg, ds4_tl_buf wu, ds4_tl_buf x,
        ds4_tl_buf ids, ds4_tl_buf route_w,
        uint64_t in_dim, uint64_t mid_dim, uint32_t n_used, uint32_t n_tok, float clamp) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_MOE_GATE_UP_IQ2_XXS_SWIGLU_BATCH_4096);
    ds4_tl_kernel *k = in_dim == 4096 ? &k4096 : NULL;
    if (!k || mid_dim == 0 || mid_dim > INT32_MAX || n_used == 0 || n_tok == 0 || (uint64_t)n_tok * n_used * mid_dim > INT32_MAX ||
        (mid_dim % DS4_TL_MOE_GATE_UP_ROWS) != 0) return 0;
    if ((gate_out.off % 4u) || (up_out.off % 4u) || (mid_out.off % 4u) || (x.off % 4u) || (ids.off % 4u) || (route_w.off % 4u) ||
        (wg.off % 2u) || (wu.off % 2u)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    id<MTLBuffer> tables = ds4_tl_iq2_tables_buffer();
    if (!pso || !tables) return 0;
    tl_moe_gate_up_iq2_xxs_swiglu_batch_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.clamp[0] = clamp; args.n_tok[0] = (int32_t)n_tok; args.n_used[0] = (int32_t)n_used; args.out_dim[0] = (int32_t)mid_dim;
    uint32_t grid[3];
    tl_moe_gate_up_iq2_xxs_swiglu_batch_4096_grid(grid, clamp, args.n_tok[0], args.n_used[0], args.out_dim[0]);
    const ds4_tl_buf grid_t = ds4_tl_mk(tables, 0);
    const ds4_tl_buf ksigns_t = ds4_tl_mk(tables, DS4_TL_IQ2_KSIGNS_OFFSET);
    /* buffer order: gate_out, grid, ids, ksigns, mid_out, route_w, up_out, wg, wu, x */
    const ds4_tl_buf bufs[10] = { gate_out, grid_t, ids, ksigns_t, mid_out, route_w, up_out, wg, wu, x };
    ds4_tl_dispatch(enc, pso, bufs, 10, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* ---- routed MoE down, Q2_K x f32, summed over the selected experts ------ */

static ds4_tl_kernel *ds4_tl_moe_down_q2_k_kernel_for(uint64_t mid_dim) {
    static ds4_tl_kernel k2048 = DS4_TL_KERNEL(TL_MOE_DOWN_Q2_K_SUM_2048);
    return mid_dim == 2048 ? &k2048 : NULL;
}

#define DS4_TL_MOE_DOWN_ROWS 4

/* out: f32[out_dim]; wd: expert matrices from expert 0; mid: [n_used][mid_dim]
 * (route weight already applied); ids: int32[n_used]. */
static int ds4_tl_encode_moe_down_q2_k(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out, ds4_tl_buf wd, ds4_tl_buf mid, ds4_tl_buf ids,
        uint64_t mid_dim, uint64_t out_dim, uint32_t n_used) {
    ds4_tl_kernel *k = ds4_tl_moe_down_q2_k_kernel_for(mid_dim);
    if (!k || out_dim == 0 || out_dim > INT32_MAX || n_used == 0 ||
        (out_dim % DS4_TL_MOE_DOWN_ROWS) != 0) {
        return 0;
    }
    if ((out.off % 4u) != 0 || (mid.off % 4u) != 0 || (ids.off % 4u) != 0 || (wd.off % 4u) != 0) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    tl_moe_down_q2_k_sum_2048_args_t args;
    memset(&args, 0, sizeof(args));
    args.n_used[0] = (int32_t)n_used;
    args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_moe_down_q2_k_sum_2048_grid(grid, (int32_t)n_used, (int32_t)out_dim);

    /* buffer order: ids, mid, out, wd */
    const ds4_tl_buf bufs[4] = { ids, mid, out, wd };
    ds4_tl_dispatch(enc, pso, bufs, 4, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* Batch form: out [n_tok][out_dim], mid [n_tok][n_used][mid_dim], ids [n_tok][n_used]. */
static int ds4_tl_encode_moe_down_q2_k_batch(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out, ds4_tl_buf wd, ds4_tl_buf mid, ds4_tl_buf ids,
        uint64_t mid_dim, uint64_t out_dim, uint32_t n_used, uint32_t n_tok) {
    static ds4_tl_kernel k2048 = DS4_TL_KERNEL(TL_MOE_DOWN_Q2_K_SUM_BATCH_2048);
    ds4_tl_kernel *k = mid_dim == 2048 ? &k2048 : NULL;
    if (!k || out_dim == 0 || out_dim > INT32_MAX || n_used == 0 || n_tok == 0 || (uint64_t)n_tok * out_dim > INT32_MAX ||
        (out_dim % DS4_TL_MOE_DOWN_ROWS) != 0) return 0;
    if ((out.off % 4u) || (mid.off % 4u) || (ids.off % 4u) || (wd.off % 4u)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_moe_down_q2_k_sum_batch_2048_args_t args;
    memset(&args, 0, sizeof(args));
    args.n_tok[0] = (int32_t)n_tok; args.n_used[0] = (int32_t)n_used; args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_moe_down_q2_k_sum_batch_2048_grid(grid, args.n_tok[0], args.n_used[0], args.out_dim[0]);
    const ds4_tl_buf bufs[4] = { ids, mid, out, wd };   /* buffer order: ids, mid, out, wd */
    ds4_tl_dispatch(enc, pso, bufs, 4, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* ---- shared expert gate/up (Q8_0 x f32) + SwiGLU -------------------------- */

static ds4_tl_kernel *ds4_tl_q8_0_gate_up_swiglu_kernel_for(uint64_t in_dim) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_Q8_0_GATE_UP_SWIGLU_F32_4096);
    return in_dim == 4096 ? &k4096 : NULL;
}

static int ds4_tl_encode_q8_0_gate_up_swiglu(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf gate_out, ds4_tl_buf up_out, ds4_tl_buf mid_out,
        ds4_tl_buf wg, ds4_tl_buf wu, ds4_tl_buf x,
        uint64_t in_dim, uint64_t out_dim, float clamp) {
    ds4_tl_kernel *k = ds4_tl_q8_0_gate_up_swiglu_kernel_for(in_dim);
    if (!k || out_dim == 0 || out_dim > INT32_MAX ||
        (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0) {
        return 0;
    }
    if ((gate_out.off % 4u) != 0 || (up_out.off % 4u) != 0 || (mid_out.off % 4u) != 0 ||
        (x.off % 4u) != 0 || (wg.off % 2u) != 0 || (wu.off % 2u) != 0) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;

    tl_q8_0_gate_up_swiglu_f32_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.clamp[0] = clamp;
    args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gate_up_swiglu_f32_4096_grid(grid, clamp, (int32_t)out_dim);

    /* buffer order: gate_out, mid_out, up_out, wg, wu, x */
    const ds4_tl_buf bufs[6] = { gate_out, mid_out, up_out, wg, wu, x };
    ds4_tl_dispatch(enc, pso, bufs, 6, &args, sizeof(args), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* Row-exact batch form (kernel_dsv4_shared_gate_up_swiglu_q8_0 with several rows): see ds4_tl_encode_q8_0_gemv_rows_f32. */
static int ds4_tl_encode_q8_0_gate_up_swiglu_rows(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf gate_out, ds4_tl_buf up_out, ds4_tl_buf mid_out,
        ds4_tl_buf wg, ds4_tl_buf wu, ds4_tl_buf x,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_rows, float clamp) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_Q8_0_GATE_UP_SWIGLU_ROWS_F32_4096);
    if (in_dim != 4096 || out_dim == 0 || out_dim > INT32_MAX || n_rows == 0 || n_rows > (uint32_t)INT32_MAX ||
        (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0) {
        return 0;
    }
    if ((gate_out.off % 4u) != 0 || (up_out.off % 4u) != 0 || (mid_out.off % 4u) != 0 ||
        (x.off % 4u) != 0 || (wg.off % 2u) != 0 || (wu.off % 2u) != 0) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k4096);
    if (!pso) return 0;
    tl_q8_0_gate_up_swiglu_rows_f32_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.clamp[0] = clamp; args.n_rows[0] = (int32_t)n_rows; args.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gate_up_swiglu_rows_f32_4096_grid(grid, clamp, args.n_rows[0], args.out_dim[0]);
    const ds4_tl_buf bufs[6] = { gate_out, mid_out, up_out, wg, wu, x };   /* buffer order: gate_out, mid_out, up_out, wg, wu, x */
    ds4_tl_dispatch(enc, pso, bufs, 6, &args, sizeof(args), grid, k4096.tx, k4096.ty, k4096.tz);
    return 1;
}

/* ---- prefill: Q8_0 GEMM on cooperative tensors ---------------------------
 * out[n_tok][out_dim] = x[n_tok][in_dim] . W^T in three dispatches: f32->f16
 * activation copy padded to a multiple of the row tile, the GEMM, and (when
 * n_tok is not a tile multiple) a row copy out of a padded scratch.
 */

#define DS4_TL_GEMM_BM 64
#define DS4_TL_GEMM_BN 64
#define DS4_TL_GEMM_MIN_TOKENS 64
/* 17..31 tokens: the cooperative-tensor GEMM (f16 activations, m_pad 64)
 * beats the batched GEMV once the weight matrix is large (8192 -> 4096 at
 * 24 tokens: 310 vs 804 us; 1024 -> 32768 at 20: 139 vs 342), and loses on
 * 4096 -> 1024 (128 vs 84 us). */
#define DS4_TL_GEMM_MID_TOKENS 17
#define DS4_TL_GEMM_NARROW_MIN_TOKENS 32   /* narrow outputs (< 4096) join the GEMM tier from here (GEMM_SMALL family) */

/* Between 17 and 63 tokens the TileLang GEMM pads to 64 rows. ds4's Metal 4
 * NAX GEMM takes exactly the multiples of 32 there and is faster on them
 * (32 tokens: 0.12 vs 0.35 ms per attn_output_b); at every other count the
 * hand path is the legacy GEMM or a batched GEMV, which the padded GEMM
 * beats. The opt-in GEMM_SMALL family forces TileLang on the NAX counts too. */
static int ds4_tl_gemm_mid_tier_ok(uint64_t n_tok) {
    return n_tok >= DS4_TL_GEMM_MIN_TOKENS || (n_tok % 32u) != 0 || ds4_tl_family_enabled(DS4_TL_FAMILY_GEMM_SMALL);
}
#define DS4_TL_GEMM_MID_MIN_OUT 4096

typedef struct {
    ds4_tl_kernel gemm;
    ds4_tl_kernel convert;
} ds4_tl_gemm_kernels;

static ds4_tl_gemm_kernels *ds4_tl_q8_0_gemm_kernels_for(uint64_t in_dim) {
    static ds4_tl_gemm_kernels k4096 = { DS4_TL_KERNEL(TL_Q8_0_GEMM_F16X_4096), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_4096) };
    static ds4_tl_gemm_kernels k1024 = { DS4_TL_KERNEL(TL_Q8_0_GEMM_F16X_1024), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_1024) };
    static ds4_tl_gemm_kernels k2048 = { DS4_TL_KERNEL(TL_Q8_0_GEMM_F16X_2048), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_2048) };
    static ds4_tl_gemm_kernels k8192 = { DS4_TL_KERNEL(TL_Q8_0_GEMM_F16X_8192), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_8192) };
    static ds4_tl_gemm_kernels k12288 = { DS4_TL_KERNEL(TL_Q8_0_GEMM_F16X_12288), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_12288) };
    static ds4_tl_gemm_kernels k256 = { DS4_TL_KERNEL(TL_Q8_0_GEMM_F16X_256), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_256) };
    switch (in_dim) {
    case 4096:  return &k4096;
    case 1024:  return &k1024;
    case 2048:  return &k2048;
    case 8192:  return &k8192;
    case 12288: return &k12288;   /* DSpark support model main_proj */
    case 256:   return &k256;     /* DSpark Markov heads */
    default:    return NULL;
    }
}

static id<MTLBuffer> g_tl_x16_scratch;
static NSUInteger    g_tl_x16_scratch_bytes;
static id<MTLBuffer> g_tl_c_scratch;
static NSUInteger    g_tl_c_scratch_bytes;

/* Encodes the whole tier into `enc` (one serial encoder: dispatch order is
 * the data dependency). Returns 0 without encoding when unsupported. */
static ds4_tl_gemm_kernels *ds4_tl_f16_gemm_kernels_for(uint64_t in_dim) {
    static ds4_tl_gemm_kernels k4096 = { DS4_TL_KERNEL(TL_F16_GEMM_F16X_4096), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_4096) };
    static ds4_tl_gemm_kernels k1024 = { DS4_TL_KERNEL(TL_F16_GEMM_F16X_1024), DS4_TL_KERNEL(TL_F32_TO_F16_ROWS_1024) };
    switch (in_dim) {
    case 4096: return &k4096;
    case 1024: return &k1024;
    default:   return NULL;
    }
}

/* Dense prefill GEMM tier, Q8_0 (default) or f16 weights: f32 -> f16 row
 * conversion, the GEMM into a bm-padded scratch when needed, valid rows out. */
static int ds4_tl_encode_dense_gemm(
        id<MTLComputeCommandEncoder> enc, int f16_weights,
        ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    ds4_tl_gemm_kernels *k = f16_weights ? ds4_tl_f16_gemm_kernels_for(in_dim) : ds4_tl_q8_0_gemm_kernels_for(in_dim);
    if (!k || n_tok == 0 || n_tok > INT32_MAX || out_dim == 0 || out_dim > INT32_MAX ||
        (out_dim % DS4_TL_GEMM_BN) != 0 || (w.off % (f16_weights ? 8u : 2u)) != 0 ||
        (x.off % 4u) != 0 || (out.off % 4u) != 0) {
        return 0;
    }
    const uint64_t m_pad = (n_tok + DS4_TL_GEMM_BM - 1) / DS4_TL_GEMM_BM * DS4_TL_GEMM_BM;
    const NSUInteger x16_bytes = (NSUInteger)(m_pad * in_dim * sizeof(uint16_t));
    if (!ds4_gpu_ensure_scratch_buffer(&g_tl_x16_scratch, &g_tl_x16_scratch_bytes, x16_bytes, "ds4_tl_x16")) return 0;
    const int padded = m_pad != n_tok;
    if (padded) {
        const NSUInteger c_bytes = (NSUInteger)(m_pad * out_dim * sizeof(float));
        if (!ds4_gpu_ensure_scratch_buffer(&g_tl_c_scratch, &g_tl_c_scratch_bytes, c_bytes, "ds4_tl_gemm_c")) return 0;
    }
    id<MTLComputePipelineState> pso_gemm = ds4_tl_pipeline(&k->gemm);
    id<MTLComputePipelineState> pso_cvt = ds4_tl_pipeline(&k->convert);
    static ds4_tl_kernel copy_k = DS4_TL_KERNEL(TL_COPY_ROWS_F32);
    id<MTLComputePipelineState> pso_copy = padded ? ds4_tl_pipeline(&copy_k) : nil;
    if (!pso_gemm || !pso_cvt || (padded && !pso_copy)) return 0;

    const ds4_tl_buf x16 = ds4_tl_mk(g_tl_x16_scratch, 0);
    const ds4_tl_buf c = padded ? ds4_tl_mk(g_tl_c_scratch, 0) : out;
    uint32_t grid[3];

    /* 1. x -> f16, rows past n_tok zeroed. buffer order: x, x16 */
    tl_f32_to_f16_rows_4096_args_t cvt_args;      /* same layout for every width */
    memset(&cvt_args, 0, sizeof(cvt_args));
    cvt_args.m_pad[0] = (int32_t)m_pad;
    cvt_args.n_rows[0] = (int32_t)n_tok;
    tl_f32_to_f16_rows_4096_grid(grid, (int32_t)m_pad, (int32_t)n_tok);
    { const ds4_tl_buf bufs[2] = { x, x16 };
      ds4_tl_dispatch(enc, pso_cvt, bufs, 2, &cvt_args, sizeof(cvt_args), grid, k->convert.tx, k->convert.ty, k->convert.tz); }

    /* 2. GEMM. buffer order: c, w, x16 (both weight types; same args layout and grid) */
    tl_q8_0_gemm_f16x_4096_args_t gemm_args;
    memset(&gemm_args, 0, sizeof(gemm_args));
    gemm_args.m_pad[0] = (int32_t)m_pad;
    gemm_args.out_dim[0] = (int32_t)out_dim;
    tl_q8_0_gemm_f16x_4096_grid(grid, (int32_t)m_pad, (int32_t)out_dim);
    { const ds4_tl_buf bufs[3] = { c, w, x16 };
      ds4_tl_dispatch(enc, pso_gemm, bufs, 3, &gemm_args, sizeof(gemm_args), grid, k->gemm.tx, k->gemm.ty, k->gemm.tz); }

    /* 3. valid rows out of the padded scratch. buffer order: out, src */
    if (padded) {
        tl_copy_rows_f32_args_t copy_args;
        memset(&copy_args, 0, sizeof(copy_args));
        copy_args.n_rows[0] = (int32_t)n_tok;
        copy_args.width[0] = (int32_t)out_dim;
        tl_copy_rows_f32_grid(grid, (int32_t)n_tok, (int32_t)out_dim);
        const ds4_tl_buf bufs[2] = { out, c };
        ds4_tl_dispatch(enc, pso_copy, bufs, 2, &copy_args, sizeof(copy_args), grid, copy_k.tx, copy_k.ty, copy_k.tz);
    }
    return 1;
}

static int ds4_tl_encode_q8_0_gemm(id<MTLComputeCommandEncoder> enc, ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
                                   uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    return ds4_tl_encode_dense_gemm(enc, 0, out, w, x, in_dim, out_dim, n_tok);
}

static int ds4_tl_encode_f16_gemm(id<MTLComputeCommandEncoder> enc, ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
                                  uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    return ds4_tl_encode_dense_gemm(enc, 1, out, w, x, in_dim, out_dim, n_tok);
}

/* Grouped variant (attn_output_a in prefill): heads [n_tok][n_groups][group_dim]
 * f32 -> low [n_tok][n_groups][rank] f32, weights n_groups matrices back to
 * back. Same three dispatches. */
static int ds4_tl_encode_q8_0_gemm_grouped(
        id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf low, ds4_tl_buf w, ds4_tl_buf heads,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t n_tok) {
    static ds4_tl_kernel gemm_k = DS4_TL_KERNEL(TL_Q8_0_GEMM_GROUPED_F16X_4096);
    static ds4_tl_kernel copy_k = DS4_TL_KERNEL(TL_COPY_ROWS_F32);
    ds4_tl_gemm_kernels *k = ds4_tl_q8_0_gemm_kernels_for(group_dim);
    if (group_dim != 4096 || !k || n_tok == 0 || n_tok > INT32_MAX || rank == 0 ||
        (rank % DS4_TL_GEMM_BN) != 0 || n_groups == 0 ||
        (w.off % 2u) != 0 || (heads.off % 4u) != 0 || (low.off % 4u) != 0) {
        return 0;
    }
    const uint64_t m_pad = (n_tok + DS4_TL_GEMM_BM - 1) / DS4_TL_GEMM_BM * DS4_TL_GEMM_BM;
    const uint64_t x_width = (uint64_t)n_groups * group_dim;
    const uint64_t c_width = (uint64_t)n_groups * rank;
    if (x_width > INT32_MAX || c_width > INT32_MAX || m_pad * (uint64_t)n_groups > INT32_MAX) return 0;
    const NSUInteger x16_bytes = (NSUInteger)(m_pad * x_width * sizeof(uint16_t));
    if (!ds4_gpu_ensure_scratch_buffer(&g_tl_x16_scratch, &g_tl_x16_scratch_bytes, x16_bytes, "ds4_tl_x16")) return 0;
    const int padded = m_pad != n_tok;
    if (padded) {
        const NSUInteger c_bytes = (NSUInteger)(m_pad * c_width * sizeof(float));
        if (!ds4_gpu_ensure_scratch_buffer(&g_tl_c_scratch, &g_tl_c_scratch_bytes, c_bytes, "ds4_tl_gemm_c")) return 0;
    }
    id<MTLComputePipelineState> pso_gemm = ds4_tl_pipeline(&gemm_k);
    id<MTLComputePipelineState> pso_cvt = ds4_tl_pipeline(&k->convert);
    id<MTLComputePipelineState> pso_copy = padded ? ds4_tl_pipeline(&copy_k) : nil;
    if (!pso_gemm || !pso_cvt || (padded && !pso_copy)) return 0;

    const ds4_tl_buf x16 = ds4_tl_mk(g_tl_x16_scratch, 0);
    const ds4_tl_buf c = padded ? ds4_tl_mk(g_tl_c_scratch, 0) : low;
    uint32_t grid[3];

    /* 1. heads viewed as (n_tok * n_groups) rows of group_dim -> f16, padded
     * to m_pad * n_groups rows (token padding == trailing rows). */
    tl_f32_to_f16_rows_4096_args_t cvt_args;
    memset(&cvt_args, 0, sizeof(cvt_args));
    cvt_args.m_pad[0] = (int32_t)(m_pad * n_groups);
    cvt_args.n_rows[0] = (int32_t)(n_tok * n_groups);
    tl_f32_to_f16_rows_4096_grid(grid, cvt_args.m_pad[0], cvt_args.n_rows[0]);
    { const ds4_tl_buf bufs[2] = { heads, x16 };
      ds4_tl_dispatch(enc, pso_cvt, bufs, 2, &cvt_args, sizeof(cvt_args), grid, k->convert.tx, k->convert.ty, k->convert.tz); }

    /* 2. grouped GEMM. buffer order: c, w, x16 */
    tl_q8_0_gemm_grouped_f16x_4096_args_t gemm_args;
    memset(&gemm_args, 0, sizeof(gemm_args));
    gemm_args.c_width_[0] = (int32_t)c_width;
    gemm_args.m_pad[0] = (int32_t)m_pad;
    gemm_args.n_groups[0] = (int32_t)n_groups;
    gemm_args.out_dim[0] = (int32_t)rank;
    tl_q8_0_gemm_grouped_f16x_4096_grid(grid, (int32_t)c_width, (int32_t)m_pad, (int32_t)n_groups, (int32_t)rank);
    { const ds4_tl_buf bufs[3] = { c, w, x16 };
      ds4_tl_dispatch(enc, pso_gemm, bufs, 3, &gemm_args, sizeof(gemm_args), grid, gemm_k.tx, gemm_k.ty, gemm_k.tz); }

    if (padded) {
        tl_copy_rows_f32_args_t copy_args;
        memset(&copy_args, 0, sizeof(copy_args));
        copy_args.n_rows[0] = (int32_t)n_tok;
        copy_args.width[0] = (int32_t)c_width;
        tl_copy_rows_f32_grid(grid, (int32_t)n_tok, (int32_t)c_width);
        const ds4_tl_buf bufs[2] = { low, c };
        ds4_tl_dispatch(enc, pso_copy, bufs, 2, &copy_args, sizeof(copy_args), grid, copy_k.tx, copy_k.ty, copy_k.tz);
    }
    return 1;
}

/* ---- RoPE tail (YaRN) and head RMSNorm + RoPE, in place ---------------- */

typedef struct {
    uint32_t head_dim, n_rot, n_head, head_groups;
    ds4_tl_kernel k;
} ds4_tl_rope_kernel;

static ds4_tl_rope_kernel *ds4_tl_rope_kernel_for(uint32_t head_dim, uint32_t n_rot, uint32_t n_head) {
    /* head_groups mirrors kernels/dsv4.py: n_head / (threads / pairs). */
    static ds4_tl_rope_kernel k[] = {
        { 512, 64, 64, 8, DS4_TL_KERNEL(TL_ROPE_TAIL_F32_512_64_H64) },
        { 512, 64, 1,  1, DS4_TL_KERNEL(TL_ROPE_TAIL_F32_512_64_H1) },
        { 128, 64, 64, 8, DS4_TL_KERNEL(TL_ROPE_TAIL_F32_128_64_H64) },
        { 128, 64, 1,  1, DS4_TL_KERNEL(TL_ROPE_TAIL_F32_128_64_H1) },
    };
    for (size_t i = 0; i < sizeof(k) / sizeof(*k); i++) {
        if (k[i].head_dim == head_dim && k[i].n_rot == n_rot && k[i].n_head == n_head) return &k[i];
    }
    return NULL;
}

static int ds4_tl_encode_rope_tail(
        id<MTLComputeCommandEncoder> enc, ds4_tl_buf x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t pos_step, uint32_t n_ctx_orig, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    ds4_tl_rope_kernel *rk = ds4_tl_rope_kernel_for(head_dim, n_rot, n_head);
    if (!rk || n_tok == 0 || n_tok > INT32_MAX || (x.off % 4u) != 0) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&rk->k);
    if (!pso) return 0;
    tl_rope_tail_f32_512_64_h64_args_t a;   /* same layout for every variant */
    memset(&a, 0, sizeof(a));
    a.attn_factor[0] = attn_factor; a.beta_fast[0] = beta_fast; a.beta_slow[0] = beta_slow;
    a.ext_factor[0] = ext_factor; a.freq_base[0] = freq_base; a.freq_scale[0] = freq_scale;
    a.inverse[0] = inverse ? 1 : 0; a.n_ctx_orig[0] = (int32_t)n_ctx_orig; a.n_tok[0] = (int32_t)n_tok;
    a.pos0[0] = (int32_t)pos0; a.pos_step[0] = (int32_t)pos_step;
    const uint32_t grid[3] = { n_tok, rk->head_groups, 1 };
    ds4_tl_dispatch(enc, pso, &x, 1, &a, sizeof(a), grid, rk->k.tx, rk->k.ty, rk->k.tz);
    return 1;
}

static int ds4_tl_encode_head_rms_norm_rope_tail(
        id<MTLComputeCommandEncoder> enc, ds4_tl_buf x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, int inverse, float eps,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HEAD_RMS_NORM_ROPE_TAIL_F32_512_64);
    if (head_dim != 512 || n_rot != 64 || n_tok == 0 || n_head == 0 || (x.off % 16u) != 0) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_head_rms_norm_rope_tail_f32_512_64_args_t a;
    memset(&a, 0, sizeof(a));
    a.attn_factor[0] = attn_factor; a.beta_fast[0] = beta_fast; a.beta_slow[0] = beta_slow; a.eps[0] = eps;
    a.ext_factor[0] = ext_factor; a.freq_base[0] = freq_base; a.freq_scale[0] = freq_scale;
    a.inverse[0] = inverse ? 1 : 0; a.n_ctx_orig[0] = (int32_t)n_ctx_orig; a.n_head[0] = (int32_t)n_head;
    a.n_tok[0] = (int32_t)n_tok; a.pos0[0] = (int32_t)pos0;
    const uint32_t grid[3] = { n_head, n_tok, 1 };
    ds4_tl_dispatch(enc, pso, &x, 1, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- hyper-connection expand / collapse ---------------------------------
 * Both take the hand-written kernels' argument structs and decline when the
 * strides are not the canonical [token][hc][n_embd] / [token][n_embd]
 * layouts (post/comb/weights may live inside the per-token split vector:
 * their token strides are passed through).
 */

static int ds4_tl_encode_hc_expand4(
        id<MTLComputeCommandEncoder> enc, const ds4_gpu_hc_expand_args *a,
        ds4_tl_buf block, ds4_tl_buf residual, ds4_tl_buf post, ds4_tl_buf comb,
        ds4_tl_buf add, ds4_tl_buf out) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HC_EXPAND4_F32_4096);
    const uint64_t n = (uint64_t)a->n_embd;
    if (a->n_hc != 4 || n != 4096 || a->n_tokens <= 0 || a->n_tokens > INT32_MAX ||
        a->nb_block0 != 4 || a->nb_block1 != n * 4 || a->nb_add0 != 4 || a->nb_add1 != n * 4 ||
        a->nb_res0 != 4 || a->nb_res1 != n * 4 || a->nb_res2 != 4 * n * 4 ||
        a->nb_post0 != 4 || (a->nb_post1 % 4) != 0 || a->nb_comb0 != 4 ||
        (a->nb_comb1 % 4) != 0 || (a->nb_comb2 % 4) != 0 ||
        a->nb0 != 4 || a->nb1 != n * 4 || a->nb2 != 4 * n * 4 ||
        (block.off % 4) || (residual.off % 4) || (post.off % 4) || (comb.off % 4) ||
        (add.off % 4) || (out.off % 4)) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_hc_expand4_f32_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.cs1[0] = (int32_t)(a->nb_comb1 / 4);
    args.cs2[0] = (int32_t)(a->nb_comb2 / 4);
    args.has_add[0] = a->has_add ? 1 : 0;
    args.n_tok[0] = (int32_t)a->n_tokens;
    args.ps[0] = (int32_t)(a->nb_post1 / 4);
    uint32_t grid[3];
    tl_hc_expand4_f32_4096_grid(grid, args.cs1[0], args.cs2[0], args.has_add[0], args.n_tok[0], args.ps[0]);
    /* buffer order: add, block, comb, out, post, residual */
    const ds4_tl_buf bufs[6] = { add, block, comb, out, post, residual };
    ds4_tl_dispatch(enc, pso, bufs, 6, &args, sizeof(args), grid, k.tx, k.ty, k.tz);
    return 1;
}

static int ds4_tl_encode_hc_weighted_sum4(
        id<MTLComputeCommandEncoder> enc, const ds4_gpu_hc_weighted_sum_args *a,
        ds4_tl_buf x, ds4_tl_buf w, ds4_tl_buf out) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HC_WEIGHTED_SUM4_F32_4096);
    const uint64_t n = (uint64_t)a->n_embd;
    if (a->n_hc != 4 || n != 4096 || a->n_tokens <= 0 || a->n_tokens > INT32_MAX ||
        a->nb_x0 != 4 || a->nb_x1 != n * 4 || a->nb_x2 != 4 * n * 4 ||
        a->nb_w0 != 4 || (a->nb_w1 % 4) != 0 || a->nb0 != 4 || a->nb1 != n * 4 ||
        (x.off % 4) || (w.off % 4) || (out.off % 4)) {
        return 0;
    }
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_hc_weighted_sum4_f32_4096_args_t args;
    memset(&args, 0, sizeof(args));
    args.n_tok[0] = (int32_t)a->n_tokens;
    args.ws[0] = (int32_t)(a->nb_w1 / 4);
    uint32_t grid[3];
    tl_hc_weighted_sum4_f32_4096_grid(grid, args.n_tok[0], args.ws[0]);
    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &args, sizeof(args), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- elementwise -------------------------------------------------------- */

enum { DS4_TL_UNARY_SCALE = 0, DS4_TL_UNARY_CLAMP, DS4_TL_UNARY_SQRT, DS4_TL_UNARY_SIGMOID,
       DS4_TL_UNARY_SILU, DS4_TL_UNARY_SOFTPLUS, DS4_TL_UNARY_SOFTPLUS_SQRT, DS4_TL_UNARY_FILL };

/* rows x width f32, element strides; dst may alias src. */
static int ds4_tl_encode_unary_rows(
        id<MTLComputeCommandEncoder> enc, int op, ds4_tl_buf src, ds4_tl_buf dst,
        uint32_t width, uint32_t rows, uint32_t src_stride, uint32_t dst_stride, float p0, float p1) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_UNARY_ROWS_F32);
    if (width == 0 || rows == 0 || width > INT32_MAX || rows > INT32_MAX || (src.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_unary_rows_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.dst_stride[0] = (int32_t)dst_stride; a.op[0] = op; a.p0[0] = p0; a.p1[0] = p1;
    a.rows[0] = (int32_t)rows; a.src_stride[0] = (int32_t)src_stride; a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_unary_rows_f32_grid(grid, a.dst_stride[0], op, p0, p1, a.rows[0], a.src_stride[0], a.width[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

static int ds4_tl_encode_add2(id<MTLComputeCommandEncoder> enc, ds4_tl_buf a, ds4_tl_buf b, ds4_tl_buf out, uint32_t n) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_ADD2_F32);
    if (n == 0 || n > INT32_MAX || (a.off % 4) || (b.off % 4) || (out.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_add2_f32_args_t args;
    memset(&args, 0, sizeof(args));
    args.n[0] = (int32_t)n;
    uint32_t grid[3];
    tl_add2_f32_grid(grid, (int32_t)n);
    const ds4_tl_buf bufs[3] = { a, b, out };   /* buffer order: a, b, out */
    ds4_tl_dispatch(enc, pso, bufs, 3, &args, sizeof(args), grid, k.tx, k.ty, k.tz);
    return 1;
}

static int ds4_tl_encode_swiglu_flat(id<MTLComputeCommandEncoder> enc, ds4_tl_buf gate, ds4_tl_buf up, ds4_tl_buf out,
                                     uint32_t n, float alpha, float limit) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_SWIGLU_FLAT_F32);
    if (n == 0 || n > INT32_MAX || (gate.off % 4) || (up.off % 4) || (out.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_swiglu_flat_f32_args_t args;
    memset(&args, 0, sizeof(args));
    args.alpha[0] = alpha; args.limit[0] = limit; args.n[0] = (int32_t)n;
    uint32_t grid[3];
    tl_swiglu_flat_f32_grid(grid, alpha, limit, (int32_t)n);
    const ds4_tl_buf bufs[3] = { gate, out, up };   /* buffer order: gate, out, up */
    ds4_tl_dispatch(enc, pso, bufs, 3, &args, sizeof(args), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- utility: copies, gathers, broadcasts, row sums, MoE glue -----------
 * ds4's versions are ggml-style with byte-stride tables; these take element
 * strides and the callers decline layouts they do not cover. All lengths
 * must fit int32 (32-bit indexing in the generated kernels).
 */

static int ds4_tl_i32_ok(uint64_t v) { return v <= (uint64_t)INT32_MAX; }

/* Flat 1-D converting copies. The vector path casts to float4/half4, so
 * buffer offsets must be 16-byte aligned. */
static int ds4_tl_encode_cpy_flat(id<MTLComputeCommandEncoder> enc, ds4_tl_kernel *k,
                                  ds4_tl_buf src, ds4_tl_buf dst, uint64_t n) {
    if (n == 0 || !ds4_tl_i32_ok(n) || !ds4_tl_aligned(src.off) || !ds4_tl_aligned(dst.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_cpy_f32_f16_args_t a;   /* all three copies share the {int n[2]} layout */
    memset(&a, 0, sizeof(a));
    a.n[0] = (int32_t)n;
    uint32_t grid[3];
    tl_cpy_f32_f16_grid(grid, a.n[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

static int ds4_tl_encode_cpy_f32_f16(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst, uint64_t n) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_CPY_F32_F16);
    return ds4_tl_encode_cpy_flat(enc, &k, src, dst, n);
}

static int ds4_tl_encode_cpy_f16_f32(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst, uint64_t n) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_CPY_F16_F32);
    return ds4_tl_encode_cpy_flat(enc, &k, src, dst, n);
}

static int ds4_tl_encode_cpy_u16(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst, uint64_t n) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_CPY_U16);
    return ds4_tl_encode_cpy_flat(enc, &k, src, dst, n);
}

/* dst[p*d2 + r*d1 + c] = src[p*s2 + r*s1 + c*s0]; strides in floats. */
static int ds4_tl_encode_cpy_f32_3d(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst,
                                    uint64_t cols, uint64_t rows, uint64_t planes,
                                    uint64_t s0, uint64_t s1, uint64_t s2, uint64_t d1, uint64_t d2) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_CPY_F32_3D);
    if (cols == 0 || rows == 0 || planes == 0 || (src.off % 4) || (dst.off % 4)) return 0;
    if (!ds4_tl_i32_ok(cols) || !ds4_tl_i32_ok(rows) || !ds4_tl_i32_ok(planes) ||
        !ds4_tl_i32_ok(s0) || !ds4_tl_i32_ok(s1) || !ds4_tl_i32_ok(s2) || !ds4_tl_i32_ok(d1) || !ds4_tl_i32_ok(d2)) return 0;
    /* The flattened extents must stay below 2^31 elements. */
    if ((planes - 1) * s2 + (rows - 1) * s1 + (cols - 1) * s0 >= (uint64_t)INT32_MAX ||
        (planes - 1) * d2 + (rows - 1) * d1 + cols >= (uint64_t)INT32_MAX) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_cpy_f32_3d_args_t a;
    memset(&a, 0, sizeof(a));
    a.cols[0] = (int32_t)cols; a.rows[0] = (int32_t)rows; a.planes[0] = (int32_t)planes;
    a.s0[0] = (int32_t)s0; a.s1[0] = (int32_t)s1; a.s2[0] = (int32_t)s2; a.d1[0] = (int32_t)d1; a.d2[0] = (int32_t)d2;
    /* float4 path: contiguous source columns, all strides and offsets 16-byte aligned. */
    a.vec[0] = s0 == 1 && (s1 % 4) == 0 && (s2 % 4) == 0 && (d1 % 4) == 0 && (d2 % 4) == 0 &&
               ds4_tl_aligned(src.off) && ds4_tl_aligned(dst.off);
    uint32_t grid[3];
    tl_cpy_f32_3d_grid(grid, a.cols[0], a.d1[0], a.d2[0], a.planes[0], a.rows[0], a.s0[0], a.s1[0], a.s2[0], a.vec[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* dst[t*width + e] = src[((pos0 + t) % ratio) * width + e], n_total = n_tokens*width. */
static int ds4_tl_encode_cpy_tile_rows(id<MTLComputeCommandEncoder> enc, int src_is_f16, ds4_tl_buf src, ds4_tl_buf dst,
                                       uint64_t n_total, uint32_t width, uint32_t ratio, uint32_t pos0) {
    static ds4_tl_kernel k16 = DS4_TL_KERNEL(TL_CPY_TILE_ROWS_F16_F32);
    static ds4_tl_kernel k32 = DS4_TL_KERNEL(TL_CPY_TILE_ROWS_F32_F32);
    ds4_tl_kernel *k = src_is_f16 ? &k16 : &k32;
    if (n_total == 0 || !ds4_tl_i32_ok(n_total) || width == 0 || ratio == 0 || (src.off % 4) || (dst.off % 4)) return 0;
    if (width > (uint32_t)INT32_MAX || ratio > (uint32_t)INT32_MAX || pos0 > (uint32_t)INT32_MAX) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_cpy_tile_rows_f16_f32_args_t a;   /* same layout for both variants */
    memset(&a, 0, sizeof(a));
    a.n_total[0] = (int32_t)n_total; a.pos0[0] = (int32_t)pos0; a.ratio[0] = (int32_t)ratio; a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_cpy_tile_rows_f16_f32_grid(grid, a.n_total[0], a.pos0[0], a.ratio[0], a.width[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* dst[idx[r], :width] = src[r, :width]; idx is int32 in a buffer (setBytes
 * also works through ds4_tl_buf with a transient buffer on the caller). */
static int ds4_tl_encode_set_rows_f32(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf idx, ds4_tl_buf dst,
                                      uint32_t width, uint32_t rows, uint64_t src_stride, uint64_t dst_stride) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_SET_ROWS_F32);
    /* Strides in float4 units and 16-byte offsets: the kernel's vector path relies on it. */
    if (width == 0 || rows == 0 || width > (uint32_t)INT32_MAX || rows > (uint32_t)INT32_MAX ||
        !ds4_tl_i32_ok(src_stride) || !ds4_tl_i32_ok(dst_stride) || (src_stride % 4) || (dst_stride % 4) ||
        !ds4_tl_aligned(src.off) || (idx.off % 4) || !ds4_tl_aligned(dst.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_set_rows_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.dst_stride4[0] = (int32_t)(dst_stride / 4); a.rows[0] = (int32_t)rows; a.src_stride4[0] = (int32_t)(src_stride / 4); a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_set_rows_f32_grid(grid, a.dst_stride4[0], a.rows[0], a.src_stride4[0], a.width[0]);
    const ds4_tl_buf bufs[3] = { dst, idx, src };   /* buffer order: dst, idx, src */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

enum { DS4_TL_GET_ROWS_F16_F32 = 0, DS4_TL_GET_ROWS_F32, DS4_TL_GET_ROWS_I32 };

/* dst[r, :width] = table[idx[r], :width] (f16 table -> f32 dst, f32, or i32). */
static int ds4_tl_encode_get_rows(id<MTLComputeCommandEncoder> enc, int kind, ds4_tl_buf table, ds4_tl_buf idx, ds4_tl_buf dst,
                                  uint32_t width, uint32_t rows, uint64_t table_stride, uint64_t dst_stride) {
    static ds4_tl_kernel k[3] = { DS4_TL_KERNEL(TL_GET_ROWS_F16_F32), DS4_TL_KERNEL(TL_GET_ROWS_F32), DS4_TL_KERNEL(TL_GET_ROWS_I32) };
    if (width == 0 || rows == 0 || width > (uint32_t)INT32_MAX || rows > (uint32_t)INT32_MAX ||
        !ds4_tl_i32_ok(table_stride) || !ds4_tl_i32_ok(dst_stride) || (idx.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k[kind]);
    if (!pso) return 0;
    uint32_t grid[3];
    const ds4_tl_buf bufs[3] = { dst, idx, table };   /* buffer order: dst, idx, table */
    if (kind == DS4_TL_GET_ROWS_I32) {
        /* scalar variant, element strides */
        if ((table.off % 4) || (dst.off % 4)) return 0;
        tl_get_rows_i32_args_t a;
        memset(&a, 0, sizeof(a));
        a.dst_stride[0] = (int32_t)dst_stride; a.rows[0] = (int32_t)rows; a.table_stride[0] = (int32_t)table_stride; a.width[0] = (int32_t)width;
        tl_get_rows_i32_grid(grid, a.dst_stride[0], a.rows[0], a.table_stride[0], a.width[0]);
        ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k[kind].tx, k[kind].ty, k[kind].tz);
        return 1;
    }
    /* float4 path: strides in 4-element units, offsets aligned to 4 elements. */
    const unsigned elem = kind == DS4_TL_GET_ROWS_F16_F32 ? 2u : 4u;
    if ((table_stride % 4) || (dst_stride % 4) || (table.off % (4 * elem)) || !ds4_tl_aligned(dst.off)) return 0;
    tl_get_rows_f32_args_t a;   /* same layout for the f16 and f32 variants */
    memset(&a, 0, sizeof(a));
    a.dst_stride4[0] = (int32_t)(dst_stride / 4); a.rows[0] = (int32_t)rows; a.table_stride4[0] = (int32_t)(table_stride / 4); a.width[0] = (int32_t)width;
    tl_get_rows_f32_grid(grid, a.dst_stride4[0], a.rows[0], a.table_stride4[0], a.width[0]);
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k[kind].tx, k[kind].ty, k[kind].tz);
    return 1;
}

/* dst[t*dst_stride + k] = table[t*table_stride + idx[t*idx_stride + k]], k < width (<= 32). */
static int ds4_tl_encode_gather_scalars_f32(id<MTLComputeCommandEncoder> enc, ds4_tl_buf table, ds4_tl_buf idx, ds4_tl_buf dst,
                                            uint32_t width, uint32_t rows, uint64_t table_stride, uint64_t idx_stride, uint64_t dst_stride) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_GATHER_SCALARS_F32);
    if (width == 0 || width > k.tx || rows == 0 || rows > (uint32_t)INT32_MAX ||
        !ds4_tl_i32_ok(table_stride) || !ds4_tl_i32_ok(idx_stride) || !ds4_tl_i32_ok(dst_stride) ||
        (table.off % 4) || (idx.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_gather_scalars_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.dst_stride[0] = (int32_t)dst_stride; a.idx_stride[0] = (int32_t)idx_stride; a.rows[0] = (int32_t)rows;
    a.table_stride[0] = (int32_t)table_stride; a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_gather_scalars_f32_grid(grid, a.dst_stride[0], a.idx_stride[0], a.rows[0], a.table_stride[0], a.width[0]);
    const ds4_tl_buf bufs[3] = { dst, idx, table };   /* buffer order: dst, idx, table */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* dst[t*dst_row_stride + h*dst_rep_stride + c] = src[t*src_stride + c]. */
static int ds4_tl_encode_repeat_rows_f32(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst,
                                         uint32_t width, uint32_t reps, uint32_t rows,
                                         uint64_t src_stride, uint64_t dst_rep_stride, uint64_t dst_row_stride) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_REPEAT_ROWS_F32);
    if (width == 0 || reps == 0 || rows == 0 || width > (uint32_t)INT32_MAX || reps > (uint32_t)INT32_MAX || rows > (uint32_t)INT32_MAX ||
        !ds4_tl_i32_ok(src_stride) || !ds4_tl_i32_ok(dst_rep_stride) || !ds4_tl_i32_ok(dst_row_stride) ||
        (src_stride % 4) || (dst_rep_stride % 4) || (dst_row_stride % 4) ||
        (rows - 1) * dst_row_stride + (reps - 1) * dst_rep_stride + width >= (uint64_t)INT32_MAX ||
        !ds4_tl_aligned(src.off) || !ds4_tl_aligned(dst.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_repeat_rows_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.dst_rep_stride4[0] = (int32_t)(dst_rep_stride / 4); a.dst_row_stride4[0] = (int32_t)(dst_row_stride / 4); a.reps[0] = (int32_t)reps;
    a.rows[0] = (int32_t)rows; a.src_stride4[0] = (int32_t)(src_stride / 4); a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_repeat_rows_f32_grid(grid, a.dst_rep_stride4[0], a.dst_row_stride4[0], a.reps[0], a.rows[0], a.src_stride4[0], a.width[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* dst[r] = sum(src[r*src_stride + 0..width)); one thread per row (widths in
 * a Flash run are 6, 8, 128). */
static int ds4_tl_encode_sum_rows_f32(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst,
                                      uint32_t width, uint32_t rows, uint64_t src_stride) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_SUM_ROWS_F32);
    if (width == 0 || rows == 0 || width > 1024 || rows > (uint32_t)INT32_MAX || !ds4_tl_i32_ok(src_stride) ||
        (src.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_sum_rows_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.rows[0] = (int32_t)rows; a.src_stride[0] = (int32_t)src_stride; a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_sum_rows_f32_grid(grid, a.rows[0], a.src_stride[0], a.width[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

enum { DS4_TL_BIN_ADD = 0, DS4_TL_BIN_SUB, DS4_TL_BIN_MUL, DS4_TL_BIN_DIV };

/* dst[r, i] = a[r, i] op b[(r % b_rows) * b_stride + (i % b_cols)]; b_rows = 1
 * broadcasts one row, b_cols = 1 one scalar per row. dst may alias a. */
static int ds4_tl_encode_bin_rows_f32(id<MTLComputeCommandEncoder> enc, int op, ds4_tl_buf a_buf, ds4_tl_buf b_buf, ds4_tl_buf dst,
                                      uint32_t width, uint32_t rows, uint64_t a_stride, uint64_t dst_stride,
                                      uint32_t b_rows, uint32_t b_cols, uint64_t b_stride) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_BIN_ROWS_F32);
    if (width == 0 || rows == 0 || b_rows == 0 || b_cols == 0 || width > (uint32_t)INT32_MAX || rows > (uint32_t)INT32_MAX ||
        !ds4_tl_i32_ok(a_stride) || !ds4_tl_i32_ok(dst_stride) || !ds4_tl_i32_ok(b_stride) ||
        (a_buf.off % 4) || (b_buf.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_bin_rows_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.a_stride[0] = (int32_t)a_stride; a.b_cols[0] = (int32_t)b_cols; a.b_rows[0] = (int32_t)b_rows; a.b_stride[0] = (int32_t)b_stride;
    a.dst_stride[0] = (int32_t)dst_stride; a.op[0] = op; a.rows[0] = (int32_t)rows; a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_bin_rows_f32_grid(grid, a.a_stride[0], a.b_cols[0], a.b_rows[0], a.b_stride[0], a.dst_stride[0], op, a.rows[0], a.width[0]);
    const ds4_tl_buf bufs[3] = { a_buf, b_buf, dst };   /* buffer order: a, b, dst */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* dst[t, :4096] = sum of the 6 expert rows src[t, e, :4096], contiguous. */
static int ds4_tl_encode_moe_sum6(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst, uint32_t width, uint32_t tokens) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_MOE_SUM6_F32_4096);
    if (width != 4096 || tokens == 0 || tokens > (uint32_t)INT32_MAX / (6u * 4096u) || (src.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_moe_sum6_f32_4096_args_t a;
    memset(&a, 0, sizeof(a));
    a.tokens[0] = (int32_t)tokens;
    uint32_t grid[3];
    tl_moe_sum6_f32_4096_grid(grid, a.tokens[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* mid[r, :2048] = silu(min(g, c)) * clamp(u, -c, c) * w[r] for contiguous
 * [rows][2048] gate/up (f32) and mid (f32 or f16); w is [rows] f32. */
static int ds4_tl_encode_moe_swiglu_weight(id<MTLComputeCommandEncoder> enc, int mid_is_f16, ds4_tl_buf gate, ds4_tl_buf up,
                                           ds4_tl_buf mid, ds4_tl_buf w, uint32_t width, uint32_t rows, float limit) {
    static ds4_tl_kernel k32 = DS4_TL_KERNEL(TL_MOE_SWIGLU_WEIGHT_F32_2048);
    static ds4_tl_kernel k16 = DS4_TL_KERNEL(TL_MOE_SWIGLU_WEIGHT_F16_2048);
    ds4_tl_kernel *k = mid_is_f16 ? &k16 : &k32;
    if (width != 2048 || rows == 0 || rows > (uint32_t)INT32_MAX / 2048u || (gate.off % 4) || (up.off % 4) || (mid.off % 4) || (w.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_moe_swiglu_weight_f32_2048_args_t a;   /* same layout for both variants */
    memset(&a, 0, sizeof(a));
    a.limit[0] = limit; a.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_moe_swiglu_weight_f32_2048_grid(grid, limit, a.rows[0]);
    const ds4_tl_buf bufs[4] = { gate, mid, up, w };   /* buffer order: gate, mid, up, w */
    ds4_tl_dispatch(enc, pso, bufs, 4, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* dst[ic*d1 + d] = sum_r kv[r*kv_s0 + d*kv_s1 + ic*kv_s2] * softmax_r(score[...]). */
static int ds4_tl_encode_softmax_pool(id<MTLComputeCommandEncoder> enc, ds4_tl_buf kv, ds4_tl_buf score, ds4_tl_buf dst,
                                      uint32_t head_dim, uint32_t pooled, uint32_t n_comp,
                                      uint64_t kv_s0, uint64_t kv_s1, uint64_t kv_s2,
                                      uint64_t sc_s0, uint64_t sc_s1, uint64_t sc_s2, uint64_t d1) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_SOFTMAX_POOL_F32);
    if (head_dim == 0 || pooled == 0 || n_comp == 0 || head_dim > (uint32_t)INT32_MAX || pooled > (uint32_t)INT32_MAX || n_comp > (uint32_t)INT32_MAX ||
        !ds4_tl_i32_ok(kv_s0) || !ds4_tl_i32_ok(kv_s1) || !ds4_tl_i32_ok(kv_s2) || !ds4_tl_i32_ok(sc_s0) || !ds4_tl_i32_ok(sc_s1) ||
        !ds4_tl_i32_ok(sc_s2) || !ds4_tl_i32_ok(d1) ||
        (pooled - 1) * kv_s0 + (head_dim - 1) * kv_s1 + (n_comp - 1) * kv_s2 >= (uint64_t)INT32_MAX ||
        (pooled - 1) * sc_s0 + (head_dim - 1) * sc_s1 + (n_comp - 1) * sc_s2 >= (uint64_t)INT32_MAX ||
        (kv.off % 4) || (score.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_softmax_pool_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.d1[0] = (int32_t)d1; a.head_dim[0] = (int32_t)head_dim; a.kv_s0[0] = (int32_t)kv_s0; a.kv_s1[0] = (int32_t)kv_s1; a.kv_s2[0] = (int32_t)kv_s2;
    a.n_comp[0] = (int32_t)n_comp; a.pooled[0] = (int32_t)pooled; a.sc_s0[0] = (int32_t)sc_s0; a.sc_s1[0] = (int32_t)sc_s1; a.sc_s2[0] = (int32_t)sc_s2;
    uint32_t grid[3];
    tl_softmax_pool_f32_grid(grid, a.d1[0], a.head_dim[0], a.kv_s0[0], a.kv_s1[0], a.kv_s2[0], a.n_comp[0], a.pooled[0], a.sc_s0[0], a.sc_s1[0], a.sc_s2[0]);
    const ds4_tl_buf bufs[3] = { dst, kv, score };   /* buffer order: dst, kv, score */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- DSv4 fused small kernels (kernels/dsv4.py) -------------------------- */

/* q (1024) and kv (512) rows rms-normed in one dispatch; contiguous rows. */
static int ds4_tl_encode_qkv_rms_norm(id<MTLComputeCommandEncoder> enc,
                                      ds4_tl_buf q_x, ds4_tl_buf q_w, ds4_tl_buf q_out,
                                      ds4_tl_buf kv_x, ds4_tl_buf kv_w, ds4_tl_buf kv_out,
                                      uint32_t q_n, uint32_t kv_n, uint32_t rows, float eps) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_QKV_RMS_NORM_F32_1024_512);
    if (q_n != 1024 || kv_n != 512 || rows == 0 || rows > (uint32_t)INT32_MAX / 1024u ||
        !ds4_tl_aligned(q_x.off) || !ds4_tl_aligned(q_w.off) || !ds4_tl_aligned(q_out.off) ||
        !ds4_tl_aligned(kv_x.off) || !ds4_tl_aligned(kv_w.off) || !ds4_tl_aligned(kv_out.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_qkv_rms_norm_f32_1024_512_args_t a;
    memset(&a, 0, sizeof(a));
    a.eps[0] = eps; a.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_qkv_rms_norm_f32_1024_512_grid(grid, eps, a.rows[0]);
    const ds4_tl_buf bufs[6] = { kv_out, kv_w, kv_x, q_out, q_w, q_x };   /* buffer order: kv_out, kv_w, kv_x, q_out, q_w, q_x */
    ds4_tl_dispatch(enc, pso, bufs, 6, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* In-place E4M3FN fake quantisation of [rows][512] KV rows (64 RoPE dims copied). */
static int ds4_tl_encode_fp8_kv_quantize(id<MTLComputeCommandEncoder> enc, ds4_tl_buf x, uint32_t rows, uint32_t head_dim, uint32_t n_rot) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_FP8_KV_QUANTIZE_F32_512_64);
    if (head_dim != 512 || n_rot != 64 || rows == 0 || rows > (uint32_t)INT32_MAX / 512u || !ds4_tl_aligned(x.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_fp8_kv_quantize_f32_512_64_args_t a;
    memset(&a, 0, sizeof(a));
    a.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_fp8_kv_quantize_f32_512_64_grid(grid, a.rows[0]);
    const ds4_tl_buf bufs[2] = { x, x };   /* buffer order: out, x (in place) */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* Decode KV finalizer after RoPE (kernel_dsv4_kv_fp8_store_f32): fp8 round trip in place + f16-rounded raw ring row. */
static int ds4_tl_encode_kv_fp8_store(id<MTLComputeCommandEncoder> enc, ds4_tl_buf kv, ds4_tl_buf raw, uint32_t raw_row,
                                      uint32_t head_dim, uint32_t n_rot) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_KV_FP8_STORE_F32_512_64);
    if (head_dim != 512 || n_rot != 64 || raw_row > (uint32_t)INT32_MAX / 512u || !ds4_tl_aligned(kv.off) || !ds4_tl_aligned(raw.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_kv_fp8_store_f32_512_64_args_t a;
    memset(&a, 0, sizeof(a));
    a.raw_row[0] = (int32_t)raw_row;
    uint32_t grid[3];
    tl_kv_fp8_store_f32_512_64_grid(grid, a.raw_row[0]);
    const ds4_tl_buf bufs[2] = { kv, raw };   /* buffer order: kv, raw */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* HC split (gates + sinkhorn) + pre-gated collapse + rms norm, n_hc = 4, n_embd = 4096. */
static int ds4_tl_encode_hc_split_weighted_sum_norm4(id<MTLComputeCommandEncoder> enc,
                                                     ds4_tl_buf mixes, ds4_tl_buf scale, ds4_tl_buf base, ds4_tl_buf x,
                                                     ds4_tl_buf split, ds4_tl_buf dst, ds4_tl_buf w, ds4_tl_buf norm_dst,
                                                     uint32_t n_embd, uint32_t rows, uint32_t sinkhorn_iters, float eps, float norm_eps) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HC_SPLIT_WEIGHTED_SUM_NORM4_F32_4096);
    if (n_embd != 4096 || rows == 0 || rows > (uint32_t)INT32_MAX / (4u * 4096u) || sinkhorn_iters == 0 || sinkhorn_iters > (uint32_t)INT32_MAX ||
        !ds4_tl_aligned(mixes.off) || (scale.off % 4) || !ds4_tl_aligned(base.off) || !ds4_tl_aligned(x.off) ||
        !ds4_tl_aligned(split.off) || !ds4_tl_aligned(dst.off) || !ds4_tl_aligned(w.off) || !ds4_tl_aligned(norm_dst.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_hc_split_weighted_sum_norm4_f32_4096_args_t a;
    memset(&a, 0, sizeof(a));
    a.eps[0] = eps; a.norm_eps[0] = norm_eps; a.rows[0] = (int32_t)rows; a.sinkhorn_iters[0] = (int32_t)sinkhorn_iters;
    uint32_t grid[3];
    tl_hc_split_weighted_sum_norm4_f32_4096_grid(grid, eps, norm_eps, a.rows[0], a.sinkhorn_iters[0]);
    const ds4_tl_buf bufs[8] = { base, dst, mixes, norm_dst, scale, split, w, x };   /* buffer order: base, dst, mixes, norm_dst, scale, split, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 8, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- router (kernels/router.py) ------------------------------------------ */

static int ds4_tl_encode_f16_gemv(id<MTLComputeCommandEncoder> enc, ds4_tl_buf w, ds4_tl_buf x, ds4_tl_buf out,
                                  uint64_t in_dim, uint64_t out_dim, uint64_t n_tok);

/* out[out_dim] = f16 w[out_dim][in_dim] . f32 x (single token). */
static int ds4_tl_encode_f16_gemv_f32(id<MTLComputeCommandEncoder> enc, ds4_tl_buf w, ds4_tl_buf x, ds4_tl_buf out,
                                      uint32_t in_dim, uint32_t out_dim) {
    return ds4_tl_encode_f16_gemv(enc, w, x, out, in_dim, out_dim, 1);
}

/* One token: probs = sqrt(softplus(logits)), top-6 by probs (+ bias), weights.
 * When has_bias is 0 the bias buffer is never read (bind probs). */
static int ds4_tl_encode_router_finalize(id<MTLComputeCommandEncoder> enc, ds4_tl_buf logits, ds4_tl_buf bias, int has_bias,
                                         ds4_tl_buf probs, ds4_tl_buf selected, ds4_tl_buf weights,
                                         uint32_t n_expert, uint32_t n_used, float scale) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_ROUTER_FINALIZE_F32_256_6);
    if (n_expert != 256 || n_used != 6 || scale != 1.5f ||
        (logits.off % 4) || (bias.off % 4) || (probs.off % 4) || (selected.off % 4) || (weights.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_router_finalize_f32_256_6_args_t a;
    memset(&a, 0, sizeof(a));
    a.has_bias[0] = has_bias ? 1 : 0;
    uint32_t grid[3];
    tl_router_finalize_f32_256_6_grid(grid, a.has_bias[0]);
    const ds4_tl_buf bufs[5] = { has_bias ? bias : probs, logits, probs, selected, weights };   /* buffer order: bias, logits, probs, selected, weights */
    ds4_tl_dispatch(enc, pso, bufs, 5, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* weights[t, i] = probs[t, sel[t, i]] * scale / max(sum, 2^-14), rows tokens. */
static int ds4_tl_encode_router_weights(id<MTLComputeCommandEncoder> enc, ds4_tl_buf probs, ds4_tl_buf selected, ds4_tl_buf weights,
                                        uint32_t n_expert, uint32_t n_used, uint32_t rows, float scale) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_ROUTER_WEIGHTS_F32_6);
    if (n_used != 6 || n_expert == 0 || rows == 0 || rows > (uint32_t)INT32_MAX / n_expert ||
        (probs.off % 4) || (selected.off % 4) || (weights.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_router_weights_f32_6_args_t a;
    memset(&a, 0, sizeof(a));
    a.n_expert[0] = (int32_t)n_expert; a.rows[0] = (int32_t)rows; a.scale[0] = scale;
    uint32_t grid[3];
    tl_router_weights_f32_6_grid(grid, a.n_expert[0], a.rows[0], scale);
    const ds4_tl_buf bufs[3] = { probs, selected, weights };   /* buffer order: probs, selected, weights */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* out[t, 0:k] = indices of the k largest of keys[t, 0:256], descending. */
static int ds4_tl_encode_argsort_desc_topk(id<MTLComputeCommandEncoder> enc, ds4_tl_buf keys, ds4_tl_buf out,
                                           uint32_t width, uint32_t rows, uint32_t k_top) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_ARGSORT_DESC_TOPK_F32_256);
    if (width != 256 || rows == 0 || k_top == 0 || k_top > 256 || rows > (uint32_t)INT32_MAX / 256u ||
        (keys.off % 4) || (out.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_argsort_desc_topk_f32_256_args_t a;
    memset(&a, 0, sizeof(a));
    a.k[0] = (int32_t)k_top; a.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_argsort_desc_topk_f32_256_grid(grid, a.k[0], a.rows[0]);
    const ds4_tl_buf bufs[2] = { keys, out };   /* buffer order: keys_in, out */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- compressor / indexer (kernels/compressor.py) ------------------------- */

/* Ratio-4 decode pool over the 8-row state: dst[head_dim] f32. */
static int ds4_tl_encode_compressor_pool_ratio4(id<MTLComputeCommandEncoder> enc, ds4_tl_buf state_kv, ds4_tl_buf state_score,
                                                ds4_tl_buf dst, uint32_t head_dim) {
    static ds4_tl_kernel k512 = DS4_TL_KERNEL(TL_COMPRESSOR_POOL_RATIO4_F32_512);
    static ds4_tl_kernel k128 = DS4_TL_KERNEL(TL_COMPRESSOR_POOL_RATIO4_F32_128);
    ds4_tl_kernel *k = head_dim == 512 ? &k512 : head_dim == 128 ? &k128 : NULL;
    if (!k || (state_kv.off % 4) || (state_score.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    uint32_t grid[3];
    if (head_dim == 512) tl_compressor_pool_ratio4_f32_512_grid(grid); else tl_compressor_pool_ratio4_f32_128_grid(grid);
    const uint32_t dummy = 0;   /* no scalar arguments */
    const ds4_tl_buf bufs[3] = { dst, state_kv, state_score };   /* buffer order: dst, state_kv, state_score */
    ds4_tl_dispatch(enc, pso, bufs, 3, &dummy, sizeof(dummy), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* In-place Hadamard + E2M1 round trip of [rows][128] f32 (element row stride). */
static int ds4_tl_encode_indexer_hadamard_fp4(id<MTLComputeCommandEncoder> enc, ds4_tl_buf x, uint32_t rows, uint32_t head_dim, uint64_t row_stride) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_INDEXER_HADAMARD_FP4_F32_128);
    if (head_dim != 128 || rows == 0 || rows > (uint32_t)INT32_MAX / 128u || !ds4_tl_i32_ok(row_stride) || row_stride < 128 || (x.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_indexer_hadamard_fp4_f32_128_args_t a;
    memset(&a, 0, sizeof(a));
    a.row_stride[0] = (int32_t)row_stride; a.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_indexer_hadamard_fp4_f32_128_grid(grid, a.row_stride[0], a.rows[0]);
    const ds4_tl_buf bufs[1] = { x };
    ds4_tl_dispatch(enc, pso, bufs, 1, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* dst[t, :512] = ascending sort of src[t, :512] (int32). */
static int ds4_tl_encode_sort_i32_rows_asc(id<MTLComputeCommandEncoder> enc, ds4_tl_buf src, ds4_tl_buf dst, uint32_t width, uint32_t rows) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_SORT_I32_ROWS_ASC_512);
    if (width != 512 || rows == 0 || rows > (uint32_t)INT32_MAX / 512u || (src.off % 4) || (dst.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_sort_i32_rows_asc_512_args_t a;
    memset(&a, 0, sizeof(a));
    a.rows[0] = (int32_t)rows;
    uint32_t grid[3];
    tl_sort_i32_rows_asc_512_grid(grid, a.rows[0]);
    const ds4_tl_buf bufs[2] = { dst, src };   /* buffer order: dst, src */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* Decode compressor row finalize (ratio 4): see kernels/compressor.py. */
static int ds4_tl_encode_comp_row_finalize(id<MTLComputeCommandEncoder> enc,
                                           ds4_tl_buf attn_row, ds4_tl_buf attn_norm_w, ds4_tl_buf attn_cache,
                                           ds4_tl_buf index_row, ds4_tl_buf index_norm_w,
                                           ds4_tl_buf attn_state_kv, ds4_tl_buf attn_state_score,
                                           ds4_tl_buf index_state_kv, ds4_tl_buf index_state_score,
                                           float rms_eps, uint32_t pos0, uint32_t n_ctx_orig, float freq_base, float freq_scale,
                                           float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_COMP_ROW_FINALIZE_F32_512_128);
    if (pos0 > (uint32_t)INT32_MAX || n_ctx_orig > (uint32_t)INT32_MAX ||
        (attn_row.off % 4) || (attn_norm_w.off % 4) || (attn_cache.off % 2) || (index_row.off % 4) || (index_norm_w.off % 4) ||
        (attn_state_kv.off % 4) || (attn_state_score.off % 4) || (index_state_kv.off % 4) || (index_state_score.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_comp_row_finalize_f32_512_128_args_t a;
    memset(&a, 0, sizeof(a));
    a.attn_factor[0] = attn_factor; a.beta_fast[0] = beta_fast; a.beta_slow[0] = beta_slow; a.ext_factor[0] = ext_factor;
    a.freq_base[0] = freq_base; a.freq_scale[0] = freq_scale; a.n_ctx_orig[0] = (int32_t)n_ctx_orig; a.pos0[0] = (int32_t)pos0; a.rms_eps[0] = rms_eps;
    uint32_t grid[3];
    tl_comp_row_finalize_f32_512_128_grid(grid, attn_factor, beta_fast, beta_slow, ext_factor, freq_base, freq_scale, a.n_ctx_orig[0], a.pos0[0], rms_eps);
    /* buffer order: attn_cache, attn_norm_w, attn_row, attn_state_kv, attn_state_score, index_norm_w, index_row, index_state_kv, index_state_score */
    const ds4_tl_buf bufs[9] = { attn_cache, attn_norm_w, attn_row, attn_state_kv, attn_state_score, index_norm_w, index_row, index_state_kv, index_state_score };
    ds4_tl_dispatch(enc, pso, bufs, 9, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- small-batch GEMVs (2..31 tokens): Q8_0 and f16 weights ---------------- */

#define DS4_TL_BATCH_GEMV_MAX_TOKENS 31u

static ds4_tl_kernel *ds4_tl_q8_0_gemv_batch_kernel_for(uint64_t in_dim, uint64_t n_tok) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH_F32_4096), k4096_4 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH4_F32_4096);
    static ds4_tl_kernel k1024 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH_F32_1024), k1024_4 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH4_F32_1024);
    static ds4_tl_kernel k2048 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH_F32_2048), k2048_4 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH4_F32_2048);
    static ds4_tl_kernel k8192 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH_F32_8192), k8192_4 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH4_F32_8192);
    static ds4_tl_kernel k12288 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH_F32_12288), k12288_4 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH4_F32_12288);
    static ds4_tl_kernel k256 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH_F32_256), k256_4 = DS4_TL_KERNEL(TL_Q8_0_GEMV_BATCH4_F32_256);
    const int small = n_tok <= 4;
    switch (in_dim) {
    case 4096: return small ? &k4096_4 : &k4096;
    case 1024: return small ? &k1024_4 : &k1024;
    case 2048: return small ? &k2048_4 : &k2048;
    case 8192: return small ? &k8192_4 : &k8192;
    case 12288: return small ? &k12288_4 : &k12288;
    case 256: return small ? &k256_4 : &k256;
    default: return NULL;
    }
}

/* out[n_tok][out_dim] = Q8_0 w[out_dim][in_dim] . x[n_tok][in_dim], contiguous rows. */
static int ds4_tl_encode_q8_0_gemv_batch(id<MTLComputeCommandEncoder> enc, ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
                                         uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    ds4_tl_kernel *k = ds4_tl_q8_0_gemv_batch_kernel_for(in_dim, n_tok);
    if (!k || out_dim == 0 || n_tok == 0 || out_dim > INT32_MAX || n_tok > INT32_MAX || out_dim * n_tok > INT32_MAX ||
        (out.off % 4u) || (x.off % 4u) || (w.off % 2u)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_q8_0_gemv_batch_f32_4096_args_t a;   /* same layout for every variant */
    memset(&a, 0, sizeof(a));
    a.n_tok[0] = (int32_t)n_tok; a.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    if (n_tok <= 4) tl_q8_0_gemv_batch4_f32_4096_grid(grid, a.n_tok[0], a.out_dim[0]);
    else tl_q8_0_gemv_batch_f32_4096_grid(grid, a.n_tok[0], a.out_dim[0]);
    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* attn_output_a for 2..63 tokens: out [n_tok][n_groups][rank], x [n_tok][n_groups][group_dim]. */
static int ds4_tl_encode_q8_0_gemv_grouped_batch(id<MTLComputeCommandEncoder> enc, ds4_tl_buf out, ds4_tl_buf w, ds4_tl_buf x,
                                                 uint64_t in_dim, uint64_t out_dim, uint64_t n_groups, uint64_t n_tok) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_Q8_0_GEMV_GROUPED_BATCH_F32_4096);
    ds4_tl_kernel *k = in_dim == 4096 ? &k4096 : NULL;
    if (!k || out_dim == 0 || n_groups == 0 || n_tok == 0 || out_dim > INT32_MAX || n_groups > INT32_MAX || n_tok > INT32_MAX ||
        out_dim * n_groups * n_tok > INT32_MAX || in_dim * n_groups * n_tok > INT32_MAX ||
        (out.off % 4u) || (x.off % 4u) || (w.off % 2u)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_q8_0_gemv_grouped_batch_f32_4096_args_t a;
    memset(&a, 0, sizeof(a));
    a.n_groups[0] = (int32_t)n_groups; a.n_tok[0] = (int32_t)n_tok; a.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gemv_grouped_batch_f32_4096_grid(grid, a.n_groups[0], a.n_tok[0], a.out_dim[0]);
    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

static ds4_tl_kernel *ds4_tl_f16_gemv_kernel_for(uint64_t in_dim, uint64_t n_tok) {
    static ds4_tl_kernel k4096 = DS4_TL_KERNEL(TL_F16_GEMV_F32_4096), kb4096 = DS4_TL_KERNEL(TL_F16_GEMV_BATCH_F32_4096), kb4096_4 = DS4_TL_KERNEL(TL_F16_GEMV_BATCH4_F32_4096);
    static ds4_tl_kernel k16384 = DS4_TL_KERNEL(TL_F16_GEMV_F32_16384), kb16384 = DS4_TL_KERNEL(TL_F16_GEMV_BATCH_F32_16384), kb16384_4 = DS4_TL_KERNEL(TL_F16_GEMV_BATCH4_F32_16384);
    static ds4_tl_kernel k1024 = DS4_TL_KERNEL(TL_F16_GEMV_F32_1024), kb1024 = DS4_TL_KERNEL(TL_F16_GEMV_BATCH_F32_1024), kb1024_4 = DS4_TL_KERNEL(TL_F16_GEMV_BATCH4_F32_1024);
    switch (in_dim) {
    case 4096: return n_tok == 1 ? &k4096 : n_tok <= 4 ? &kb4096_4 : &kb4096;
    case 16384: return n_tok == 1 ? &k16384 : n_tok <= 4 ? &kb16384_4 : &kb16384;
    case 1024: return n_tok == 1 ? &k1024 : n_tok <= 4 ? &kb1024_4 : &kb1024;
    default: return NULL;
    }
}

/* out[n_tok][out_dim] = f16 w[out_dim][in_dim] . x[n_tok][in_dim]; n_tok == 1
 * uses the single-token kernel, otherwise the 4-token tiles. out_dim % 4 == 0. */
static int ds4_tl_encode_f16_gemv(id<MTLComputeCommandEncoder> enc, ds4_tl_buf w, ds4_tl_buf x, ds4_tl_buf out,
                                  uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    ds4_tl_kernel *k = ds4_tl_f16_gemv_kernel_for(in_dim, n_tok);
    if (!k || out_dim == 0 || n_tok == 0 || out_dim > INT32_MAX || n_tok > INT32_MAX || out_dim * n_tok > INT32_MAX || in_dim * n_tok > INT32_MAX ||
        (out_dim % 4) || (w.off % 8) || !ds4_tl_aligned(x.off) || (out.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    uint32_t grid[3];
    const ds4_tl_buf bufs[3] = { out, w, x };   /* buffer order: out, w, x */
    if (n_tok == 1) {
        tl_f16_gemv_f32_4096_args_t a;   /* same layout for every in_dim */
        memset(&a, 0, sizeof(a));
        a.out_dim[0] = (int32_t)out_dim;
        /* rows per threadgroup differ per in_dim: use each kernel's grid rule */
        if (in_dim == 16384) tl_f16_gemv_f32_16384_grid(grid, a.out_dim[0]);
        else if (in_dim == 1024) tl_f16_gemv_f32_1024_grid(grid, a.out_dim[0]);
        else tl_f16_gemv_f32_4096_grid(grid, a.out_dim[0]);
        ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    } else {
        tl_f16_gemv_batch_f32_4096_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_tok[0] = (int32_t)n_tok; a.out_dim[0] = (int32_t)out_dim;
        if (n_tok <= 4) tl_f16_gemv_batch4_f32_4096_grid(grid, a.n_tok[0], a.out_dim[0]);
        else tl_f16_gemv_batch_f32_4096_grid(grid, a.n_tok[0], a.out_dim[0]);
        ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    }
    return 1;
}

/* ---- decode attention (kernels/attention.py) ------------------------------ */

#define DS4_TL_ATTN_BM 32    /* heads per threadgroup of the partial kernel */
#define DS4_TL_INDEXED_BM 32 /* heads per threadgroup of the indexed attention kernel (all of them) */
#define DS4_TL_ATTN_BN 32    /* keys per split */

/* Three dispatches in one encoder: stage K=V (f16), per-split partials, merge
 * (+ optional inverse RoPE tail). kv_stage/out_part/stats are scratch
 * buffers the caller sizes: kv_stage >= n_keys*1024 B, out_part >=
 * nsplit*n_head*2048 B, stats >= nsplit*n_head*8 B. */
static int ds4_tl_encode_flash_decode(id<MTLComputeCommandEncoder> enc,
                                      ds4_tl_buf heads, ds4_tl_buf q, ds4_tl_buf raw, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
                                      ds4_tl_buf comp, uint32_t n_comp, ds4_tl_buf sinks,
                                      ds4_tl_buf kv_stage, ds4_tl_buf out_part, ds4_tl_buf stats,
                                      uint32_t n_head, uint32_t head_dim, float scale,
                                      int do_rope, uint32_t pos0, uint32_t n_ctx_orig, float freq_base, float freq_scale,
                                      float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    static ds4_tl_kernel k_stage = DS4_TL_KERNEL(TL_FLASH_KV_STAGE_F16_512);
    static ds4_tl_kernel k_part = DS4_TL_KERNEL(TL_FLASH_DECODE_PARTIAL_F16_512);
    static ds4_tl_kernel k_red = DS4_TL_KERNEL(TL_FLASH_DECODE_REDUCE_F16_512);
    const uint32_t n_keys = n_raw + n_comp;
    if (head_dim != 512 || n_head == 0 || (n_head % DS4_TL_ATTN_BM) || n_keys == 0 || n_keys > 65536 || raw_cap < n_raw || raw_start >= raw_cap ||
        !ds4_tl_aligned(heads.off) || !ds4_tl_aligned(q.off) || !ds4_tl_aligned(raw.off) || (n_comp && !ds4_tl_aligned(comp.off)) ||
        (sinks.off % 4) || !ds4_tl_aligned(kv_stage.off) || !ds4_tl_aligned(out_part.off) || (stats.off % 4)) return 0;
    id<MTLComputePipelineState> p_stage = ds4_tl_pipeline(&k_stage), p_part = ds4_tl_pipeline(&k_part), p_red = ds4_tl_pipeline(&k_red);
    if (!p_stage || !p_part || !p_red) return 0;
    const uint32_t nsplit = (n_keys + DS4_TL_ATTN_BN - 1) / DS4_TL_ATTN_BN;
    uint32_t grid[3];
    {
        tl_flash_kv_stage_f16_512_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_comp[0] = (int32_t)n_comp; a.n_raw[0] = (int32_t)n_raw; a.raw_cap[0] = (int32_t)raw_cap; a.raw_start[0] = (int32_t)raw_start;
        tl_flash_kv_stage_f16_512_grid(grid, a.n_comp[0], a.n_raw[0], a.raw_cap[0], a.raw_start[0]);
        const ds4_tl_buf bufs[3] = { n_comp ? comp : kv_stage, kv_stage, raw };   /* buffer order: comp, dst, raw */
        ds4_tl_dispatch(enc, p_stage, bufs, 3, &a, sizeof(a), grid, k_stage.tx, k_stage.ty, k_stage.tz);
    }
    {
        tl_flash_decode_partial_f16_512_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_head[0] = (int32_t)n_head; a.n_keys[0] = (int32_t)n_keys; a.scale[0] = scale;
        tl_flash_decode_partial_f16_512_grid(grid, a.n_head[0], a.n_keys[0], scale);
        const ds4_tl_buf bufs[4] = { kv_stage, out_part, q, stats };   /* buffer order: kv, out_part, q, stats */
        ds4_tl_dispatch(enc, p_part, bufs, 4, &a, sizeof(a), grid, k_part.tx, k_part.ty, k_part.tz);
    }
    {
        tl_flash_decode_reduce_f16_512_args_t a;
        memset(&a, 0, sizeof(a));
        a.attn_factor[0] = attn_factor; a.beta_fast[0] = beta_fast; a.beta_slow[0] = beta_slow; a.do_rope[0] = do_rope ? 1 : 0;
        a.ext_factor[0] = ext_factor; a.freq_base[0] = freq_base; a.freq_scale[0] = freq_scale; a.n_ctx_orig[0] = (int32_t)n_ctx_orig;
        a.n_head[0] = (int32_t)n_head; a.nsplit[0] = (int32_t)nsplit; a.pos0[0] = (int32_t)pos0;
        tl_flash_decode_reduce_f16_512_grid(grid, attn_factor, beta_fast, beta_slow, a.do_rope[0], ext_factor, freq_base, freq_scale,
                                            a.n_ctx_orig[0], a.n_head[0], a.nsplit[0], a.pos0[0]);
        const ds4_tl_buf bufs[4] = { heads, out_part, sinks, stats };   /* buffer order: heads, out_part, sinks, stats */
        ds4_tl_dispatch(enc, p_red, bufs, 4, &a, sizeof(a), grid, k_red.tx, k_red.ty, k_red.tz);
    }
    return 1;
}

/* Masked prefill attention over (token, head) rows: heads/q are [n_q][n_head][512]
 * f32, kv the staged f16 [n_keys][512] (K = V), mask f16 [n_q][n_keys],
 * sinks f32 [n_head]. One dispatch; no pad / blk buffers. */
static int ds4_tl_encode_flash_prefill(id<MTLComputeCommandEncoder> enc, ds4_tl_buf heads, ds4_tl_buf q, ds4_tl_buf kv,
                                       ds4_tl_buf mask, ds4_tl_buf sinks, uint32_t n_q, uint32_t n_head, uint32_t n_keys, float scale) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_FLASH_PREFILL_F16_512);
    const uint64_t n_rows = (uint64_t)n_q * n_head;
    /* a threadgroup's 32 rows must belong to one token: n_head % 32 == 0 */
    if (n_q == 0 || n_head == 0 || (n_head % 32) || n_keys == 0 || n_rows > INT32_MAX / 512 ||
        (uint64_t)n_q * n_keys > INT32_MAX || (uint64_t)n_keys * 512 > INT32_MAX ||
        !ds4_tl_aligned(heads.off) || !ds4_tl_aligned(q.off) || !ds4_tl_aligned(kv.off) || (mask.off % 2) || (sinks.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_flash_prefill_f16_512_args_t a;
    memset(&a, 0, sizeof(a));
    a.n_head[0] = (int32_t)n_head; a.n_keys[0] = (int32_t)n_keys; a.n_rows[0] = (int32_t)n_rows; a.scale[0] = scale;
    uint32_t grid[3];
    tl_flash_prefill_f16_512_grid(grid, a.n_head[0], a.n_keys[0], a.n_rows[0], scale);
    const ds4_tl_buf bufs[5] = { heads, kv, mask, q, sinks };   /* buffer order: heads, kv, mask, q, sinks */
    ds4_tl_dispatch(enc, pso, bufs, 5, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- fused decode kernels (one dispatch each) ------------------------------ */

/* q rms norm | kv rms norm + RoPE + E4M3FN + f16-rounded raw ring row store. */
static int ds4_tl_encode_qkv_rms_norm_kv_rope_fp8_store(id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf q_x, ds4_tl_buf q_w, ds4_tl_buf q_out, ds4_tl_buf kv_x, ds4_tl_buf kv_w, ds4_tl_buf kv_out, ds4_tl_buf raw,
        uint32_t q_n, uint32_t kv_n, uint32_t n_rot, uint32_t raw_row, uint32_t pos0, uint32_t n_ctx_orig, float eps,
        float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_QKV_RMS_NORM_KV_ROPE_FP8_STORE_F32_1024_512);
    if (q_n != 1024 || kv_n != 512 || n_rot != 64 || raw_row > (uint32_t)INT32_MAX / 512u || pos0 > (uint32_t)INT32_MAX || n_ctx_orig > (uint32_t)INT32_MAX ||
        !ds4_tl_aligned(q_x.off) || !ds4_tl_aligned(q_w.off) || !ds4_tl_aligned(q_out.off) ||
        !ds4_tl_aligned(kv_x.off) || !ds4_tl_aligned(kv_w.off) || (kv_out.off % 4) || (raw.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_qkv_rms_norm_kv_rope_fp8_store_f32_1024_512_args_t a;
    memset(&a, 0, sizeof(a));
    a.attn_factor[0] = attn_factor; a.beta_fast[0] = beta_fast; a.beta_slow[0] = beta_slow; a.eps[0] = eps; a.ext_factor[0] = ext_factor;
    a.freq_base[0] = freq_base; a.freq_scale[0] = freq_scale; a.n_ctx_orig[0] = (int32_t)n_ctx_orig; a.pos0[0] = (int32_t)pos0; a.raw_row[0] = (int32_t)raw_row;
    uint32_t grid[3];
    tl_qkv_rms_norm_kv_rope_fp8_store_f32_1024_512_grid(grid, attn_factor, beta_fast, beta_slow, eps, ext_factor, freq_base, freq_scale,
                                                        a.n_ctx_orig[0], a.pos0[0], a.raw_row[0]);
    const ds4_tl_buf bufs[7] = { kv_out, kv_w, kv_x, q_out, q_w, q_x, raw };   /* buffer order: kv_out, kv_w, kv_x, q_out, q_w, q_x, raw */
    ds4_tl_dispatch(enc, pso, bufs, 7, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* q_a + kv (Q8_0) and the attention / indexer compressor kv + gate (f16) GEMVs
 * with the compressor state append. width1 == 0 -> no indexer compressor
 * (its buffers are then bound to the attention ones and never touched). */
static int ds4_tl_encode_qkv_pair_quad_compressor_store(id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf qr, ds4_tl_buf kv_raw, ds4_tl_buf qw0, ds4_tl_buf qw1,
        ds4_tl_buf cw0a, ds4_tl_buf cw0b, ds4_tl_buf cw1a, ds4_tl_buf cw1b, ds4_tl_buf x,
        ds4_tl_buf cdst_a0, ds4_tl_buf cdst_b0, ds4_tl_buf cdst_a1, ds4_tl_buf cdst_b1,
        ds4_tl_buf ape0, ds4_tl_buf ape1, int ape_f16,
        ds4_tl_buf state0_kv, ds4_tl_buf state0_sc, ds4_tl_buf state1_kv, ds4_tl_buf state1_sc,
        uint32_t in_dim, uint32_t q_dim, uint32_t kv_dim, uint32_t width0, uint32_t width1, uint32_t ratio, uint32_t pos) {
    static ds4_tl_kernel k16 = DS4_TL_KERNEL(TL_QKV_PAIR_QUAD_COMPRESSOR_STORE_F32_4096_APE16);
    static ds4_tl_kernel k32 = DS4_TL_KERNEL(TL_QKV_PAIR_QUAD_COMPRESSOR_STORE_F32_4096_APE32);
    ds4_tl_kernel *k = ape_f16 ? &k16 : &k32;
    /* q_dim, kv_dim and width1 may be 0: the kernel's role ranges are empty then (the
     * f16-only pair / quad compressor stores of the multi-session batch schedule). */
    if (in_dim != 4096 || width0 == 0 || ratio == 0 ||
        (q_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) || (kv_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) || (width0 % DS4_TL_Q8_0_GEMV_F32_ROWS) || (width1 % DS4_TL_Q8_0_GEMV_F32_ROWS) ||
        q_dim > 65536 || kv_dim > 65536 || width0 > 65536 || width1 > 65536 || pos > (uint32_t)INT32_MAX ||
        (qr.off % 4) || (kv_raw.off % 4) || (qw0.off % 2) || (qw1.off % 2) ||
        (cw0a.off % 8) || (cw0b.off % 8) || (cw1a.off % 8) || (cw1b.off % 8) || !ds4_tl_aligned(x.off) ||
        (cdst_a0.off % 4) || (cdst_b0.off % 4) || (cdst_a1.off % 4) || (cdst_b1.off % 4) ||
        (ape0.off % (ape_f16 ? 2 : 4)) || (ape1.off % (ape_f16 ? 2 : 4)) ||
        (state0_kv.off % 4) || (state0_sc.off % 4) || (state1_kv.off % 4) || (state1_sc.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_qkv_pair_quad_compressor_store_f32_4096_ape16_args_t a;   /* same layout for both APE types */
    memset(&a, 0, sizeof(a));
    a.kv_dim[0] = (int32_t)kv_dim; a.pos[0] = (int32_t)pos; a.q_dim[0] = (int32_t)q_dim; a.ratio[0] = (int32_t)ratio;
    a.width0[0] = (int32_t)width0; a.width1[0] = (int32_t)width1;
    uint32_t grid[3];
    tl_qkv_pair_quad_compressor_store_f32_4096_ape16_grid(grid, a.kv_dim[0], a.pos[0], a.q_dim[0], a.ratio[0], a.width0[0], a.width1[0]);
    /* buffer order: ape0, ape1, cdst_a0, cdst_a1, cdst_b0, cdst_b1, cw0a, cw0b, cw1a, cw1b, kv_raw, qr, qw0, qw1, state0_kv, state0_sc, state1_kv, state1_sc, x */
    const ds4_tl_buf bufs[19] = { ape0, ape1, cdst_a0, cdst_a1, cdst_b0, cdst_b1, cw0a, cw0b, cw1a, cw1b, kv_raw, qr, qw0, qw1,
                                  state0_kv, state0_sc, state1_kv, state1_sc, x };
    ds4_tl_dispatch(enc, pso, bufs, 19, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

/* HC mixer producer: mix[24] = hc_fn . rms_norm(residual_hc[16384]) (one dispatch),
 * then the split / collapse / norm epilogue (tl_hc_split_weighted_sum_norm4).
 * Replaces kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm (two dispatches
 * instead of its last-arriving-threadgroup fusion). */
static int ds4_tl_encode_hc_rms_norm_mix(id<MTLComputeCommandEncoder> enc, ds4_tl_buf mix, ds4_tl_buf w, ds4_tl_buf x,
                                         uint32_t n, uint32_t mix_dim, float eps) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HC_RMS_NORM_MIX_F16_16384);
    if (n != 16384 || mix_dim == 0 || mix_dim > 4096 || (mix.off % 4) || (w.off % 8) || !ds4_tl_aligned(x.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_hc_rms_norm_mix_f16_16384_args_t a;
    memset(&a, 0, sizeof(a));
    a.eps[0] = eps; a.out_dim[0] = (int32_t)mix_dim;
    uint32_t grid[3];
    tl_hc_rms_norm_mix_f16_16384_grid(grid, eps, a.out_dim[0]);
    const ds4_tl_buf bufs[3] = { mix, w, x };   /* buffer order: out, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- compressor per-token chain (unaligned prefill tail chunks, ratio-128 decode emits) ---- */

static int ds4_tl_encode_compressor_store_one(id<MTLComputeCommandEncoder> enc, ds4_tl_buf kv, ds4_tl_buf score, ds4_tl_buf ape, int ape_f16,
                                              ds4_tl_buf state_kv, ds4_tl_buf state_score, uint32_t width, uint32_t ratio, uint32_t pos) {
    static ds4_tl_kernel k16 = DS4_TL_KERNEL(TL_COMPRESSOR_STORE_ONE_F32_APE16);
    static ds4_tl_kernel k32 = DS4_TL_KERNEL(TL_COMPRESSOR_STORE_ONE_F32_APE32);
    ds4_tl_kernel *k = ape_f16 ? &k16 : &k32;
    if (width == 0 || (width % 4) || ratio == 0 || width > (uint32_t)INT32_MAX / 4 || pos > (uint32_t)INT32_MAX || ratio > (uint32_t)INT32_MAX ||
        !ds4_tl_aligned(kv.off) || !ds4_tl_aligned(score.off) || (ape.off % (ape_f16 ? 8 : 16)) ||
        !ds4_tl_aligned(state_kv.off) || !ds4_tl_aligned(state_score.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    tl_compressor_store_one_f32_ape16_args_t a;   /* same layout for both APE types */
    memset(&a, 0, sizeof(a));
    a.pos[0] = (int32_t)pos; a.ratio[0] = (int32_t)ratio; a.width4[0] = (int32_t)(width / 4);
    uint32_t grid[3];
    tl_compressor_store_one_f32_ape16_grid(grid, a.pos[0], a.ratio[0], a.width4[0]);
    const ds4_tl_buf bufs[5] = { ape, kv, score, state_kv, state_score };   /* buffer order: ape, kv, score, state_kv, state_score */
    ds4_tl_dispatch(enc, pso, bufs, 5, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    return 1;
}

static int ds4_tl_encode_compressor_shift_ratio4(id<MTLComputeCommandEncoder> enc, ds4_tl_buf state_kv, ds4_tl_buf state_score, uint32_t width) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_COMPRESSOR_SHIFT_RATIO4_F32);
    if (width == 0 || (width % 4) || width > (uint32_t)INT32_MAX / 8 || !ds4_tl_aligned(state_kv.off) || !ds4_tl_aligned(state_score.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_compressor_shift_ratio4_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.n4[0] = (int32_t)width;   /* 4 rows * width floats = width float4s */
    uint32_t grid[3];
    tl_compressor_shift_ratio4_f32_grid(grid, a.n4[0]);
    const ds4_tl_buf bufs[2] = { state_kv, state_score };   /* buffer order: state_kv, state_score */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- indexed mixed attention (long context: raw window ++ indexer top-k rows) ---- */

/* q16 is scratch for the f16 copy of q (n_tokens * n_head * 512 halves).
 * n_splits == 1: one attention dispatch writing heads. n_splits > 1 (decode):
 * the split partials go to out_part / stats and tl_flash_decode_reduce
 * merges them with the sink. */
static int ds4_tl_encode_indexed_attention(id<MTLComputeCommandEncoder> enc,
                                           ds4_tl_buf heads, ds4_tl_buf q, ds4_tl_buf q16, ds4_tl_buf raw_kv, ds4_tl_buf comp_kv, int comp_f16, ds4_tl_buf topk,
                                           ds4_tl_buf sinks, ds4_tl_buf out_part, ds4_tl_buf stats,
                                           uint32_t n_tokens, uint32_t n_head, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
                                           uint32_t n_comp, uint32_t top_k, uint32_t pos0, uint32_t window, uint32_t ratio,
                                           int sorted_topk, uint32_t n_splits, float scale) {
    static ds4_tl_kernel k16 = DS4_TL_KERNEL(TL_INDEXED_ATTENTION_F16_512_COMP16);
    static ds4_tl_kernel k32 = DS4_TL_KERNEL(TL_INDEXED_ATTENTION_F16_512_COMP32);
    static ds4_tl_kernel k_red = DS4_TL_KERNEL(TL_FLASH_DECODE_REDUCE_F16_512);
    ds4_tl_kernel *k = comp_f16 ? &k16 : &k32;
    if (n_tokens == 0 || n_head == 0 || (n_head % DS4_TL_INDEXED_BM) || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap || ratio == 0 ||
        top_k == 0 || n_comp == 0 || n_splits == 0 || n_splits > 64 || (n_splits > 1 && n_tokens != 1) ||
        (uint64_t)n_tokens * n_head > INT32_MAX / 512 || (uint64_t)n_tokens * top_k > INT32_MAX || (uint64_t)pos0 + n_tokens > INT32_MAX ||
        (uint64_t)n_comp * 512 > INT32_MAX || (uint64_t)raw_cap * 512 > INT32_MAX ||
        !ds4_tl_aligned(heads.off) || !ds4_tl_aligned(q.off) || !ds4_tl_aligned(q16.off) || !ds4_tl_aligned(raw_kv.off) || !ds4_tl_aligned(comp_kv.off) ||
        (topk.off % 4) || (sinks.off % 4) || !ds4_tl_aligned(out_part.off) || (stats.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    id<MTLComputePipelineState> p_red = n_splits > 1 ? ds4_tl_pipeline(&k_red) : nil;
    if (!pso || (n_splits > 1 && !p_red)) return 0;
    if (!ds4_tl_encode_cpy_f32_f16(enc, q, q16, (uint64_t)n_tokens * n_head * 512)) return 0;
    uint32_t grid[3];
    {
        tl_indexed_attention_f16_512_comp16_args_t a;   /* same layout for both comp types */
        memset(&a, 0, sizeof(a));
        a.n_comp[0] = (int32_t)n_comp; a.n_head[0] = (int32_t)n_head; a.n_raw[0] = (int32_t)n_raw; a.n_splits[0] = (int32_t)n_splits;
        a.n_tokens[0] = (int32_t)n_tokens; a.pos0[0] = (int32_t)pos0; a.ratio[0] = (int32_t)ratio; a.raw_cap[0] = (int32_t)raw_cap;
        a.raw_start[0] = (int32_t)raw_start; a.scale[0] = scale; a.sorted_topk[0] = sorted_topk ? 1 : 0; a.top_k[0] = (int32_t)top_k;
        a.window[0] = (int32_t)window;
        tl_indexed_attention_f16_512_comp16_grid(grid, a.n_comp[0], a.n_head[0], a.n_raw[0], a.n_splits[0], a.n_tokens[0], a.pos0[0],
                                                 a.ratio[0], a.raw_cap[0], a.raw_start[0], scale, a.sorted_topk[0], a.top_k[0], a.window[0]);
        /* buffer order: comp_kv, heads, out_part, q, raw_kv, sinks, stats, topk */
        const ds4_tl_buf bufs[8] = { comp_kv, heads, n_splits > 1 ? out_part : heads, q16, raw_kv, sinks, n_splits > 1 ? stats : heads, topk };
        ds4_tl_dispatch(enc, pso, bufs, 8, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    }
    if (n_splits > 1) {
        tl_flash_decode_reduce_f16_512_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_head[0] = (int32_t)n_head; a.nsplit[0] = (int32_t)n_splits;
        tl_flash_decode_reduce_f16_512_grid(grid, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0.0f, 0, a.n_head[0], a.nsplit[0], 0);
        const ds4_tl_buf bufs[4] = { heads, out_part, sinks, stats };   /* buffer order: heads, out_part, sinks, stats */
        ds4_tl_dispatch(enc, p_red, bufs, 4, &a, sizeof(a), grid, k_red.tx, k_red.ty, k_red.tz);
    }
    return 1;
}

/* ---- lightning indexer scores (long context) ---- */

/* scores[t][c] = scale * sum_h relu(q[t,h] . k[c]) * w[t,h] for c < visible(t), else -inf.
 * Decode (no causal mask): pass pos0 so that every row is visible. */
static int ds4_tl_encode_indexer_scores(id<MTLComputeCommandEncoder> enc, ds4_tl_buf scores, ds4_tl_buf q, ds4_tl_buf w, ds4_tl_buf k,
                                        uint32_t n_tokens, uint32_t n_comp, uint32_t n_head, uint32_t head_dim, uint32_t pos0, uint32_t ratio, float scale) {
    static ds4_tl_kernel kk = DS4_TL_KERNEL(TL_INDEXER_SCORES_F32_64_128);
    if (n_head != 64 || head_dim != 128 || n_tokens == 0 || n_comp == 0 || ratio == 0 || (uint64_t)pos0 + n_tokens > INT32_MAX ||
        (uint64_t)n_tokens * n_comp > INT32_MAX || (uint64_t)n_comp * 128 > INT32_MAX ||
        (scores.off % 4) || !ds4_tl_aligned(q.off) || (w.off % 4) || !ds4_tl_aligned(k.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&kk);
    if (!pso) return 0;
    tl_indexer_scores_f32_64_128_args_t a;
    memset(&a, 0, sizeof(a));
    a.comp_len[0] = (int32_t)n_comp;   /* scores row stride */
    a.n_comp[0] = (int32_t)n_comp; a.n_tokens[0] = (int32_t)n_tokens; a.pos0[0] = (int32_t)pos0; a.ratio[0] = (int32_t)ratio; a.scale[0] = scale;
    uint32_t grid[3];
    tl_indexer_scores_f32_64_128_grid(grid, a.comp_len[0], a.n_comp[0], a.n_tokens[0], a.pos0[0], a.ratio[0], scale);
    const ds4_tl_buf bufs[4] = { k, q, scores, w };   /* buffer order: k, q, scores, w */
    ds4_tl_dispatch(enc, pso, bufs, 4, &a, sizeof(a), grid, kk.tx, kk.ty, kk.tz);
    return 1;
}

/* top-k indices per row of scores[n_rows][width] (descending), radix select, one threadgroup per row */
static int ds4_tl_encode_topk_radix_select(id<MTLComputeCommandEncoder> enc, ds4_tl_buf scores, ds4_tl_buf out, uint32_t width, uint32_t n_rows, uint32_t top_k) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_TOPK_RADIX_SELECT_F32);
    if (width == 0 || n_rows == 0 || top_k == 0 || top_k > width || (uint64_t)width * n_rows > INT32_MAX || (uint64_t)top_k * n_rows > INT32_MAX ||
        (scores.off % 4) || (out.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_topk_radix_select_f32_args_t a;
    memset(&a, 0, sizeof(a));
    a.n_rows[0] = (int32_t)n_rows; a.top_k[0] = (int32_t)top_k; a.width[0] = (int32_t)width;
    uint32_t grid[3];
    tl_topk_radix_select_f32_grid(grid, a.n_rows[0], a.top_k[0], a.width[0]);
    const ds4_tl_buf bufs[2] = { out, scores };   /* buffer order: out, scores */
    ds4_tl_dispatch(enc, pso, bufs, 2, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- routed MoE prefill: per-expert packed GEMMs (>= 512 tokens) ---- */

/* quant 1: IQ2_XXS gate/up (K = in_dim), 2: Q2_K down (K = mid_dim). dst is the
 * [pair_rows = n_used * n_tok][ne0] f32 gate/up/experts buffer, w the expert
 * blob, rhs the packed f16 rows (header skipped), offsets/counts/ids/work
 * ds4's map0 + packed_offsets outputs. */
static int ds4_tl_encode_moe_packed_gemm(id<MTLComputeCommandEncoder> enc, uint32_t quant,
                                         ds4_tl_buf dst, ds4_tl_buf w, ds4_tl_buf rhs, ds4_tl_buf offsets, ds4_tl_buf counts, ds4_tl_buf ids, ds4_tl_buf work,
                                         uint32_t in_dim, uint32_t ne0, uint32_t n_expert, uint32_t n_used, uint32_t n_tok, uint32_t work_cap) {
    static ds4_tl_kernel k_iq2 = DS4_TL_KERNEL(TL_MOE_PACKED_GEMM_IQ2_XXS_4096);
    static ds4_tl_kernel k_q2k = DS4_TL_KERNEL(TL_MOE_PACKED_GEMM_Q2_K_2048);
    ds4_tl_kernel *k = quant == 1u && in_dim == 4096 ? &k_iq2 : quant == 2u && in_dim == 2048 ? &k_q2k : NULL;
    const uint64_t pair_rows = (uint64_t)n_used * n_tok;
    if (!k || ne0 == 0 || (ne0 % 64) || n_expert == 0 || n_tok == 0 || work_cap == 0 ||
        pair_rows * ne0 > INT32_MAX || (pair_rows + 32) * in_dim > INT32_MAX || (uint64_t)n_expert * n_tok > INT32_MAX ||
        (uint64_t)n_expert * ne0 * (in_dim / 256) * (quant == 1u ? 33 : 21) > INT32_MAX ||
        !ds4_tl_aligned(dst.off) || (w.off % (quant == 1u ? 2u : 4u)) || !ds4_tl_aligned(rhs.off) || (offsets.off % 4) || (counts.off % 4) ||
        (ids.off % 4) || (work.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(k);
    if (!pso) return 0;
    uint32_t grid[3];
    if (quant == 1u) {
        id<MTLBuffer> tables = ds4_tl_iq2_tables_buffer();
        if (!tables) return 0;
        tl_moe_packed_gemm_iq2_xxs_4096_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_tok_[0] = (int32_t)n_tok; a.ne0[0] = (int32_t)ne0; a.rhs_rows_[0] = (int32_t)(pair_rows + 32); a.work_cap[0] = (int32_t)work_cap;
        tl_moe_packed_gemm_iq2_xxs_4096_grid(grid, a.n_tok_[0], a.ne0[0], a.rhs_rows_[0], a.work_cap[0]);
        /* buffer order: counts, dst, grid, ids, ksigns, offsets, rhs, w, work */
        const ds4_tl_buf bufs[9] = { counts, dst, ds4_tl_mk(tables, DS4_TL_IQ2_GRID_WORDS_OFFSET), ids, ds4_tl_mk(tables, DS4_TL_IQ2_KSIGNS_OFFSET), offsets, rhs, w, work };
        ds4_tl_dispatch(enc, pso, bufs, 9, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    } else {
        tl_moe_packed_gemm_q2_k_2048_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_tok_[0] = (int32_t)n_tok; a.ne0[0] = (int32_t)ne0; a.rhs_rows_[0] = (int32_t)(pair_rows + 32); a.work_cap[0] = (int32_t)work_cap;
        tl_moe_packed_gemm_q2_k_2048_grid(grid, a.n_tok_[0], a.ne0[0], a.rhs_rows_[0], a.work_cap[0]);
        /* buffer order: counts, dst, ids, offsets, rhs, w, work */
        const ds4_tl_buf bufs[7] = { counts, dst, ids, offsets, rhs, w, work };
        ds4_tl_dispatch(enc, pso, bufs, 7, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    }
    return 1;
}

/* map0: per-expert pair-row lists, counts, 32-row work items from ids [n_tok][6] (contiguous int32) */
static int ds4_tl_encode_moe_map0(id<MTLComputeCommandEncoder> enc, ds4_tl_buf ids, ds4_tl_buf counts, ds4_tl_buf hids, ds4_tl_buf work,
                                  uint32_t n_tok, uint32_t n_used, uint32_t n_expert, uint32_t work_cap) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_MOE_MAP0_6);
    if (n_used != 6 || n_tok == 0 || n_expert == 0 || n_expert > 256 || (uint64_t)n_tok * 6 > INT32_MAX || (uint64_t)n_expert * n_tok > INT32_MAX ||
        (ids.off % 4) || (counts.off % 4) || (hids.off % 4) || (work.off % 4)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_moe_map0_6_args_t a;
    memset(&a, 0, sizeof(a));
    a.n_expert[0] = (int32_t)n_expert; a.n_tok[0] = (int32_t)n_tok;
    (void)work_cap;
    uint32_t grid[3];
    tl_moe_map0_6_grid(grid, a.n_expert[0], a.n_tok[0]);
    const ds4_tl_buf bufs[4] = { counts, hids, ids, work };   /* buffer order: counts, hids, ids, work */
    ds4_tl_dispatch(enc, pso, bufs, 4, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* offsets = exclusive prefix sum of counts; then pack the pair rows' activations as f16 rows */
static int ds4_tl_encode_moe_pack_rhs(id<MTLComputeCommandEncoder> enc, int src_f16, ds4_tl_buf src, uint64_t src_tok_stride_bytes, uint64_t src_slot_stride_bytes,
                                      ds4_tl_buf offsets, ds4_tl_buf counts, ds4_tl_buf hids, ds4_tl_buf work, ds4_tl_buf dst,
                                      uint32_t k_dim, uint32_t n_tok, uint32_t n_used, uint32_t n_expert, uint32_t work_cap, uint32_t dst_rows) {
    static ds4_tl_kernel k_off = DS4_TL_KERNEL(TL_MOE_PACKED_OFFSETS);
    static ds4_tl_kernel k32 = DS4_TL_KERNEL(TL_MOE_PACK_RHS_F32);
    static ds4_tl_kernel k16 = DS4_TL_KERNEL(TL_MOE_PACK_RHS_F16);
    ds4_tl_kernel *k = src_f16 ? &k16 : &k32;
    const uint32_t esz = src_f16 ? 2u : 4u;
    if (k_dim == 0 || (k_dim % 4) || n_tok == 0 || n_expert == 0 || n_expert > 256 || n_used == 0 || work_cap == 0 ||
        (src_tok_stride_bytes % (4 * esz)) || (src_slot_stride_bytes % (4 * esz)) || src_tok_stride_bytes / (4 * esz) > INT32_MAX / 4 ||
        (uint64_t)dst_rows * k_dim > INT32_MAX || (uint64_t)n_tok * src_tok_stride_bytes / esz > INT32_MAX ||
        !ds4_tl_aligned(src.off) || (offsets.off % 4) || (counts.off % 4) || (hids.off % 4) || (work.off % 4) || !ds4_tl_aligned(dst.off)) return 0;
    id<MTLComputePipelineState> p_off = ds4_tl_pipeline(&k_off), pso = ds4_tl_pipeline(k);
    if (!p_off || !pso) return 0;
    uint32_t grid[3];
    {
        tl_moe_packed_offsets_args_t a;
        memset(&a, 0, sizeof(a));
        a.n_expert[0] = (int32_t)n_expert;
        tl_moe_packed_offsets_grid(grid, a.n_expert[0]);
        const ds4_tl_buf bufs[2] = { counts, offsets };   /* buffer order: counts, offsets */
        ds4_tl_dispatch(enc, p_off, bufs, 2, &a, sizeof(a), grid, k_off.tx, k_off.ty, k_off.tz);
    }
    {
        tl_moe_pack_rhs_f32_args_t a;   /* same layout for both source types */
        memset(&a, 0, sizeof(a));
        a.k_dim[0] = (int32_t)k_dim; a.n_tok_[0] = (int32_t)n_tok; a.n_used[0] = (int32_t)n_used;
        a.src_slot_stride4[0] = (int32_t)(src_slot_stride_bytes / (4 * esz)); a.src_tok_stride4[0] = (int32_t)(src_tok_stride_bytes / (4 * esz));
        a.work_cap[0] = (int32_t)work_cap;
        (void)dst_rows;
        tl_moe_pack_rhs_f32_grid(grid, a.k_dim[0], a.n_tok_[0], a.n_used[0], a.src_slot_stride4[0], a.src_tok_stride4[0], a.work_cap[0]);
        const ds4_tl_buf bufs[6] = { counts, dst, hids, offsets, src, work };   /* buffer order: counts, dst, hids, offsets, src, work */
        ds4_tl_dispatch(enc, pso, bufs, 6, &a, sizeof(a), grid, k->tx, k->ty, k->tz);
    }
    return 1;
}

/* ---- prefill HC mixer GEMM with fused |x|^2 (mode 0 of ds4_gpu_hc_rms_scale_project_f16_tensor) ---- */
static int ds4_tl_encode_hc_mix_gemm(id<MTLComputeCommandEncoder> enc, ds4_tl_buf mix, ds4_tl_buf w, ds4_tl_buf x,
                                     uint32_t in_dim, uint32_t mix_dim, uint32_t n_rows, float eps) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HC_MIX_GEMM_F16_16384);
    if (in_dim != 16384 || mix_dim != 24 || n_rows == 0 || (uint64_t)n_rows * in_dim > INT32_MAX ||
        (mix.off % 4) || (w.off % 8) || !ds4_tl_aligned(x.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_hc_mix_gemm_f16_16384_args_t a;
    memset(&a, 0, sizeof(a));
    a.eps[0] = eps; a.n_rows[0] = (int32_t)n_rows;
    uint32_t grid[3];
    tl_hc_mix_gemm_f16_16384_grid(grid, eps, a.n_rows[0]);
    const ds4_tl_buf bufs[3] = { mix, w, x };   /* buffer order: mix, w, x */
    ds4_tl_dispatch(enc, pso, bufs, 3, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* ---- decode shared-expert down GEMV + routed add + HC expand (non-concurrent FFN schedule) ---- */
static int ds4_tl_encode_q8_0_gemv_shared_down_hc_expand4(id<MTLComputeCommandEncoder> enc,
        ds4_tl_buf out_hc, ds4_tl_buf shared_out, ds4_tl_buf w, ds4_tl_buf x, ds4_tl_buf routed, ds4_tl_buf residual, ds4_tl_buf split,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_hc) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_Q8_0_GEMV_SHARED_DOWN_HC_EXPAND4_F32_2048);
    if (in_dim != 2048 || n_hc != 4 || out_dim == 0 || out_dim > INT32_MAX || (out_dim % DS4_TL_Q8_0_GEMV_F32_ROWS) != 0 ||
        (out_hc.off % 4u) || (shared_out.off % 4u) || (x.off % 4u) || (routed.off % 4u) || (residual.off % 4u) || (split.off % 4u) || (w.off % 2u)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_q8_0_gemv_shared_down_hc_expand4_f32_2048_args_t a;
    memset(&a, 0, sizeof(a));
    a.out_dim[0] = (int32_t)out_dim;
    uint32_t grid[3];
    tl_q8_0_gemv_shared_down_hc_expand4_f32_2048_grid(grid, a.out_dim[0]);
    /* buffer order: block_out, out_hc, residual, routed, split, w, x */
    const ds4_tl_buf bufs[7] = { shared_out, out_hc, residual, routed, split, w, x };
    ds4_tl_dispatch(enc, pso, bufs, 7, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

/* HC split + pre-gated collapse without the norm (unfused batch path) */
static int ds4_tl_encode_hc_split_weighted_sum4(id<MTLComputeCommandEncoder> enc,
                                                ds4_tl_buf mixes, ds4_tl_buf scale, ds4_tl_buf base, ds4_tl_buf x, ds4_tl_buf split, ds4_tl_buf dst,
                                                uint32_t n_embd, uint32_t rows, uint32_t sinkhorn_iters, float eps) {
    static ds4_tl_kernel k = DS4_TL_KERNEL(TL_HC_SPLIT_WEIGHTED_SUM4_F32_4096);
    if (n_embd != 4096 || rows == 0 || rows > (uint32_t)INT32_MAX / (4u * 4096u) || sinkhorn_iters == 0 || sinkhorn_iters > (uint32_t)INT32_MAX ||
        !ds4_tl_aligned(mixes.off) || (scale.off % 4) || !ds4_tl_aligned(base.off) || !ds4_tl_aligned(x.off) ||
        !ds4_tl_aligned(split.off) || !ds4_tl_aligned(dst.off)) return 0;
    id<MTLComputePipelineState> pso = ds4_tl_pipeline(&k);
    if (!pso) return 0;
    tl_hc_split_weighted_sum4_f32_4096_args_t a;
    memset(&a, 0, sizeof(a));
    a.eps[0] = eps; a.rows[0] = (int32_t)rows; a.sinkhorn_iters[0] = (int32_t)sinkhorn_iters;
    uint32_t grid[3];
    tl_hc_split_weighted_sum4_f32_4096_grid(grid, eps, a.rows[0], a.sinkhorn_iters[0]);
    const ds4_tl_buf bufs[6] = { base, dst, mixes, scale, split, x };   /* buffer order: base, dst, mixes, scale, split, x */
    ds4_tl_dispatch(enc, pso, bufs, 6, &a, sizeof(a), grid, k.tx, k.ty, k.tz);
    return 1;
}

#endif /* DS4_TL_METAL_H */
