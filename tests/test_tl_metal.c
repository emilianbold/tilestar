/* test_tl_metal: TileLang-generated Metal kernels vs ds4's hand-written ones.
 *
 * Model-free. Random inputs, a page-aligned fake model map for the weights,
 * both paths dispatched on the same buffers, compared against a double
 * precision host reference and against each other. Then a throughput
 * measurement: N back-to-back dispatches inside one command batch, wall
 * clock per dispatch (encode overlaps the GPU, so this is GPU time for
 * anything but the smallest shapes).
 *
 *   make test-tl-metal
 *   DS4_TL_BENCH_ITERS=1000 ./tests/test_tl_metal
 */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 0x9e3779b9u;
static float frand(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 65537u) - 32768) / 16384.0f;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int g_failures;

typedef struct {
    double max_abs, max_rel;
} diff_t;

static diff_t diff_rows(const float *a, const float *b, size_t n) {
    diff_t d = {0, 0};
    for (size_t i = 0; i < n; i++) {
        const double ad = fabs((double)a[i] - (double)b[i]);
        const double rel = ad / (fabs((double)b[i]) + 1e-6);
        if (ad > d.max_abs) d.max_abs = ad;
        if (rel > d.max_rel) d.max_rel = rel;
    }
    return d;
}

static void ref_rms_norm(float *out, const float *x, const float *w,
                         uint32_t n, uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * n;
        double ss = 0;
        for (uint32_t i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        const float scale = (float)(1.0 / sqrt(ss / n + eps));
        for (uint32_t i = 0; i < n; i++) {
            out[(size_t)r * n + i] = xr[i] * scale * (w ? w[i] : 1.0f);
        }
    }
}

/* Weight lives at `woff` bytes into a page-aligned map so the wrapped
 * MTLBuffer offset is exercised the way a real GGUF tensor would be. */
static int check_rms_norm(uint32_t n, uint32_t rows, uint64_t woff, int weighted, int bench_iters) {
    const size_t elems = (size_t)n * rows;
    const size_t map_bytes = (size_t)getpagesize() * 8;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    CHECK(woff + (uint64_t)n * sizeof(float) <= map_bytes);
    memset(map, 0, map_bytes);
    float *w = (float *)(map + woff);
    for (uint32_t i = 0; i < n; i++) w[i] = 1.0f + 0.25f * frand();
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    float *x = malloc(elems * sizeof(float));
    float *ref = malloc(elems * sizeof(float));
    float *got_ref = malloc(elems * sizeof(float));
    float *got_tl = malloc(elems * sizeof(float));
    CHECK(x && ref && got_ref && got_tl);
    for (size_t i = 0; i < elems; i++) x[i] = frand() * 4.0f;
    const float eps = 1e-6f;
    ref_rms_norm(ref, x, weighted ? w : NULL, n, rows, eps);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(elems * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(elems * sizeof(float));
    CHECK(tx && tout);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, elems * sizeof(float)));

    int ok_hand, ok_tl;
    memset(got_ref, 0, elems * sizeof(float));
    CHECK(ds4_gpu_tensor_write(tout, 0, got_ref, elems * sizeof(float)));
    if (weighted) {
        ok_hand = ds4_gpu_rms_norm_weight_rows_tensor(tout, tx, map, map_bytes, woff, n, rows, eps);
    } else {
        ok_hand = ds4_gpu_rms_norm_plain_rows_tensor(tout, tx, n, rows, eps);
    }
    CHECK(ok_hand);
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(tout, 0, got_ref, elems * sizeof(float)));

    memset(got_tl, 0, elems * sizeof(float));
    CHECK(ds4_gpu_tensor_write(tout, 0, got_tl, elems * sizeof(float)));
    if (weighted) {
        ok_tl = ds4_gpu_tl_rms_norm_weight_rows_tensor(tout, tx, map, map_bytes, woff, n, rows, eps);
    } else {
        ok_tl = ds4_gpu_tl_rms_norm_plain_rows_tensor(tout, tx, n, rows, eps);
    }
    const char *name = weighted ? "rms_norm_weight" : "rms_norm_plain";
    if (!ok_tl) {
        /* Expected for unsupported shapes / misaligned offsets; must not have
         * written anything. */
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout, 0, got_tl, elems * sizeof(float)));
        int untouched = 1;
        for (size_t i = 0; i < elems; i++) untouched &= got_tl[i] == 0.0f;
        printf("%-16s n=%-5u rows=%-3u woff=%-5llu tl: declined (fallback)%s\n",
               name, n, rows, (unsigned long long)woff, untouched ? "" : " BUT WROTE OUTPUT");
        if (!untouched) g_failures++;
        goto done;
    }
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(tout, 0, got_tl, elems * sizeof(float)));

    {
        const diff_t d_hand = diff_rows(got_ref, ref, elems);
        const diff_t d_tl = diff_rows(got_tl, ref, elems);
        const diff_t d_x = diff_rows(got_tl, got_ref, elems);
        /* Reduction order differs (simd_sum tree vs TileLang butterfly), so
         * the two GPU paths are not bit-identical; both must sit at float
         * rounding distance of the double reference. */
        const int pass = d_tl.max_rel < 1e-5 && d_hand.max_rel < 1e-5;
        printf("%-16s n=%-5u rows=%-3u woff=%-5llu hand|ref %.2e  tl|ref %.2e  tl|hand %.2e (abs %.2e)  %s\n",
               name, n, rows, (unsigned long long)woff,
               d_hand.max_rel, d_tl.max_rel, d_x.max_rel, d_x.max_abs, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }

    if (bench_iters > 0) {
        double t_hand = 0, t_tl = 0;
        for (int which = 0; which < 2; which++) {
            /* warm-up batch, then timed batch */
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    int ok;
                    if (which == 0) {
                        ok = weighted
                            ? ds4_gpu_rms_norm_weight_rows_tensor(tout, tx, map, map_bytes, woff, n, rows, eps)
                            : ds4_gpu_rms_norm_plain_rows_tensor(tout, tx, n, rows, eps);
                    } else {
                        ok = weighted
                            ? ds4_gpu_tl_rms_norm_weight_rows_tensor(tout, tx, map, map_bytes, woff, n, rows, eps)
                            : ds4_gpu_tl_rms_norm_plain_rows_tensor(tout, tx, n, rows, eps);
                    }
                    CHECK(ok);
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) { if (which == 0) t_hand = dt; else t_tl = dt; }
            }
        }
        const double bytes = (double)elems * sizeof(float) * 2 + (weighted ? n * sizeof(float) : 0);
        printf("%-16s n=%-5u rows=%-3u bench x%d: hand %7.2f us (%6.1f GB/s)  tl %7.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               name, n, rows, bench_iters,
               t_hand, bytes / t_hand / 1e3, t_tl, bytes / t_tl / 1e3, t_tl / t_hand);
    }

done:
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    free(x); free(ref); free(got_ref); free(got_tl); free(map);
    return 1;
}

/* f32 -> f16 bits (round to nearest even), enough for the test weights. */
static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;                 /* flush tiny to zero */
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

/* -inf by bit pattern: the INFINITY macro warns under -ffast-math */
static float neg_inf_f32(void) { union { uint32_t u; float f; } v = { 0xff800000u }; return v.f; }

static float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) { x = sign; }
        else { float v = ldexpf((float)mant, -24); memcpy(&x, &v, 4); x |= sign; }
    } else {
        x = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

/* Q8_0 weight x f32 activation matvec: hand-written ds4_gpu_matmul_q8_0_tensor
 * (n_tok = 1) vs ds4_gpu_tl_matmul_q8_0_tensor on host-built GGUF blocks. */
static int check_q8_0_gemv(uint32_t in_dim, uint32_t out_dim, int bench_iters) {
    const uint32_t blocks = in_dim / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * out_dim;
    const uint64_t woff = 32 * 5;   /* GGUF alignment, but not 16-byte aligned rows */
    const size_t page = (size_t)getpagesize();
    /* The bench rotates through enough distinct copies of the weight to
     * defeat the GPU's system-level cache; a single 35 MB matrix re-read
     * back-to-back measures cache bandwidth, not what decode sees (every
     * layer's weights are cold). */
    const size_t cold_target = (size_t)768 << 20;
    const size_t ncopies = bench_iters > 0 && w_bytes < cold_target ? (cold_target + w_bytes - 1) / w_bytes : 1;
    const size_t copy_stride = (w_bytes + 4095) / 4096 * 4096 + 64;   /* keep 2-byte-aligned odd offsets */
    const size_t map_bytes = (copy_stride * ncopies + woff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *w = map + woff;
    float *x = malloc(in_dim * sizeof(float));
    double *ref = malloc(out_dim * sizeof(double));
    float *got_hand = malloc(out_dim * sizeof(float));
    float *got_tl = malloc(out_dim * sizeof(float));
    CHECK(x && ref && got_hand && got_tl);
    for (uint32_t i = 0; i < in_dim; i++) x[i] = frand();
    for (uint32_t r = 0; r < out_dim; r++) {
        double acc = 0;
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (size_t)r * row_bytes + (size_t)b * 34;
            const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
            memcpy(blk, &dbits, 2);
            const double d = f16_bits_to_f32(dbits);
            for (uint32_t k = 0; k < 32; k++) {
                const int8_t q = (int8_t)((int)(frand() * 63.0f));
                blk[2 + k] = (uint8_t)q;
                acc += d * q * x[b * 32 + k];
            }
        }
        ref[r] = acc;
    }
    for (size_t c = 1; c < ncopies; c++) memcpy(w + c * copy_stride, w, w_bytes);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(in_dim * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(out_dim * sizeof(float));
    CHECK(tx && tout);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, in_dim * sizeof(float)));

    memset(got_hand, 0, out_dim * sizeof(float));
    CHECK(ds4_gpu_tensor_write(tout, 0, got_hand, out_dim * sizeof(float)));
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, 1));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(tout, 0, got_hand, out_dim * sizeof(float)));

    memset(got_tl, 0, out_dim * sizeof(float));
    CHECK(ds4_gpu_tensor_write(tout, 0, got_tl, out_dim * sizeof(float)));
    const int ok_tl = ds4_gpu_tl_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx);
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(tout, 0, got_tl, out_dim * sizeof(float)));
    if (!ok_tl) {
        int untouched = 1;
        for (uint32_t i = 0; i < out_dim; i++) untouched &= got_tl[i] == 0.0f;
        printf("q8_0_gemv        in=%-5u out=%-6u tl: declined (fallback)%s\n", in_dim, out_dim,
               untouched ? "" : " BUT WROTE OUTPUT");
        if (!untouched) g_failures++;
        goto done;
    }
    {
        double ref_max = 0, e_hand = 0, e_tl = 0, e_x = 0;
        for (uint32_t i = 0; i < out_dim; i++) {
            if (fabs(ref[i]) > ref_max) ref_max = fabs(ref[i]);
            if (fabs(got_hand[i] - ref[i]) > e_hand) e_hand = fabs(got_hand[i] - ref[i]);
            if (fabs(got_tl[i] - ref[i]) > e_tl) e_tl = fabs(got_tl[i] - ref[i]);
            if (fabs((double)got_tl[i] - got_hand[i]) > e_x) e_x = fabs((double)got_tl[i] - got_hand[i]);
        }
        const int pass = e_tl / ref_max < 1e-5 && e_hand / ref_max < 1e-5;
        printf("q8_0_gemv        in=%-5u out=%-6u hand|ref %.2e  tl|ref %.2e  tl|hand %.2e (rel to max |ref| %.2g)  %s\n",
               in_dim, out_dim, e_hand / ref_max, e_tl / ref_max, e_x / ref_max, ref_max, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t_hand = 0, t_tl = 0;
        for (int which = 0; which < 2; which++) {
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    const uint64_t off = woff + (uint64_t)(it % ncopies) * copy_stride;
                    const int ok = which == 0
                        ? ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, off, in_dim, out_dim, tx, 1)
                        : ds4_gpu_tl_matmul_q8_0_tensor(tout, map, map_bytes, off, in_dim, out_dim, tx);
                    CHECK(ok);
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) { if (which == 0) t_hand = dt; else t_tl = dt; }
            }
        }
        printf("q8_0_gemv        in=%-5u out=%-6u bench x%d (%zu cold copies): hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               in_dim, out_dim, bench_iters, ncopies, t_hand, w_bytes / t_hand / 1e3, t_tl, w_bytes / t_tl / 1e3, t_tl / t_hand);
    }
done:
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tout);
    free(x); free(ref); free(got_hand); free(got_tl); free(map);
    return 1;
}

/* attn_output_b fused with the HC expand: ds4_gpu_matmul_q8_0_hc_expand_tensor
 * vs ds4_gpu_tl_matmul_q8_0_hc_expand_tensor. */
static int check_q8_0_hc_expand(uint32_t in_dim, uint32_t out_dim, int bench_iters) {
    enum { N_HC = 4, MIX = 2 * N_HC + N_HC * N_HC };
    const uint32_t blocks = in_dim / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * out_dim;
    const uint64_t woff = 32 * 5;
    const size_t page = (size_t)getpagesize();
    const size_t cold_target = (size_t)768 << 20;
    const size_t ncopies = bench_iters > 0 && w_bytes < cold_target ? (cold_target + w_bytes - 1) / w_bytes : 1;
    const size_t copy_stride = (w_bytes + 4095) / 4096 * 4096 + 64;
    const size_t map_bytes = (copy_stride * ncopies + woff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *w = map + woff;
    float *x = malloc(in_dim * sizeof(float));
    float *res = malloc((size_t)N_HC * out_dim * sizeof(float));
    float split[MIX];
    double *ref_block = malloc(out_dim * sizeof(double));
    double *ref_hc = malloc((size_t)N_HC * out_dim * sizeof(double));
    float *hand_block = malloc(out_dim * sizeof(float));
    float *hand_hc = malloc((size_t)N_HC * out_dim * sizeof(float));
    float *tl_block = malloc(out_dim * sizeof(float));
    float *tl_hc = malloc((size_t)N_HC * out_dim * sizeof(float));
    CHECK(x && res && ref_block && ref_hc && hand_block && hand_hc && tl_block && tl_hc);
    for (uint32_t i = 0; i < in_dim; i++) x[i] = frand();
    for (size_t i = 0; i < (size_t)N_HC * out_dim; i++) res[i] = frand();
    for (int i = 0; i < MIX; i++) split[i] = frand() * 0.5f;
    for (uint32_t r = 0; r < out_dim; r++) {
        double acc = 0;
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (size_t)r * row_bytes + (size_t)b * 34;
            const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
            memcpy(blk, &dbits, 2);
            const double d = f16_bits_to_f32(dbits);
            for (uint32_t k = 0; k < 32; k++) {
                const int8_t q = (int8_t)((int)(frand() * 63.0f));
                blk[2 + k] = (uint8_t)q;
                acc += d * q * x[b * 32 + k];
            }
        }
        ref_block[r] = acc;
        for (int h = 0; h < N_HC; h++) {
            double v = acc * split[N_HC + h];
            for (int j = 0; j < N_HC; j++) v += (double)split[2 * N_HC + h + j * N_HC] * res[(size_t)j * out_dim + r];
            ref_hc[(size_t)h * out_dim + r] = v;
        }
    }
    for (size_t c = 1; c < ncopies; c++) memcpy(w + c * copy_stride, w, w_bytes);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(in_dim * sizeof(float));
    ds4_gpu_tensor *tres = ds4_gpu_tensor_alloc_managed((size_t)N_HC * out_dim * sizeof(float));
    ds4_gpu_tensor *tsplit = ds4_gpu_tensor_alloc_managed(sizeof(split));
    ds4_gpu_tensor *tblock = ds4_gpu_tensor_alloc_managed(out_dim * sizeof(float));
    ds4_gpu_tensor *thc = ds4_gpu_tensor_alloc_managed((size_t)N_HC * out_dim * sizeof(float));
    CHECK(tx && tres && tsplit && tblock && thc);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, in_dim * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tres, 0, res, (size_t)N_HC * out_dim * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tsplit, 0, split, sizeof(split)));

    for (int which = 0; which < 2; which++) {
        float *blk = which ? tl_block : hand_block;
        float *hc = which ? tl_hc : hand_hc;
        memset(blk, 0, out_dim * sizeof(float));
        memset(hc, 0, (size_t)N_HC * out_dim * sizeof(float));
        CHECK(ds4_gpu_tensor_write(tblock, 0, blk, out_dim * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(thc, 0, hc, (size_t)N_HC * out_dim * sizeof(float)));
        const int ok = which == 0
            ? ds4_gpu_matmul_q8_0_hc_expand_tensor(thc, tblock, map, map_bytes, woff, in_dim, out_dim, tx, tres, tsplit, out_dim, N_HC)
            : ds4_gpu_tl_matmul_q8_0_hc_expand_tensor(thc, tblock, map, map_bytes, woff, in_dim, out_dim, tx, tres, tsplit, out_dim, N_HC);
        CHECK(ok);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tblock, 0, blk, out_dim * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(thc, 0, hc, (size_t)N_HC * out_dim * sizeof(float)));
    }
    {
        double ref_max = 0, eb_hand = 0, eb_tl = 0, eh_hand = 0, eh_tl = 0, hc_max = 0;
        for (uint32_t i = 0; i < out_dim; i++) {
            if (fabs(ref_block[i]) > ref_max) ref_max = fabs(ref_block[i]);
            eb_hand = fmax(eb_hand, fabs(hand_block[i] - ref_block[i]));
            eb_tl = fmax(eb_tl, fabs(tl_block[i] - ref_block[i]));
        }
        for (size_t i = 0; i < (size_t)N_HC * out_dim; i++) {
            if (fabs(ref_hc[i]) > hc_max) hc_max = fabs(ref_hc[i]);
            eh_hand = fmax(eh_hand, fabs(hand_hc[i] - ref_hc[i]));
            eh_tl = fmax(eh_tl, fabs(tl_hc[i] - ref_hc[i]));
        }
        const int pass = eb_tl / ref_max < 1e-5 && eh_tl / hc_max < 1e-5 && eb_hand / ref_max < 1e-5;
        printf("q8_0_hc_expand4  in=%-5u out=%-6u block: hand|ref %.2e tl|ref %.2e  out_hc: hand|ref %.2e tl|ref %.2e  %s\n",
               in_dim, out_dim, eb_hand / ref_max, eb_tl / ref_max, eh_hand / hc_max, eh_tl / hc_max, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t_hand = 0, t_tl = 0;
        for (int which = 0; which < 2; which++) {
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    const uint64_t off = woff + (uint64_t)(it % ncopies) * copy_stride;
                    const int ok = which == 0
                        ? ds4_gpu_matmul_q8_0_hc_expand_tensor(thc, tblock, map, map_bytes, off, in_dim, out_dim, tx, tres, tsplit, out_dim, N_HC)
                        : ds4_gpu_tl_matmul_q8_0_hc_expand_tensor(thc, tblock, map, map_bytes, off, in_dim, out_dim, tx, tres, tsplit, out_dim, N_HC);
                    CHECK(ok);
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) { if (which == 0) t_hand = dt; else t_tl = dt; }
            }
        }
        printf("q8_0_hc_expand4  in=%-5u out=%-6u bench x%d (%zu cold copies): hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               in_dim, out_dim, bench_iters, ncopies, t_hand, w_bytes / t_hand / 1e3, t_tl, w_bytes / t_tl / 1e3, t_tl / t_hand);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tres); ds4_gpu_tensor_free(tsplit);
    ds4_gpu_tensor_free(tblock); ds4_gpu_tensor_free(thc);
    free(x); free(res); free(ref_block); free(ref_hc); free(hand_block); free(hand_hc); free(tl_block); free(tl_hc); free(map);
    return 1;
}

/* attn_output_a: n_groups independent Q8_0 GEMVs (group_dim -> rank). */
static int check_q8_0_grouped(uint32_t group_dim, uint32_t rank, uint32_t n_groups, int bench_iters) {
    const uint32_t blocks = group_dim / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * rank * n_groups;
    const uint64_t woff = 32 * 5;
    const size_t page = (size_t)getpagesize();
    const size_t cold_target = (size_t)768 << 20;
    const size_t ncopies = bench_iters > 0 && w_bytes < cold_target ? (cold_target + w_bytes - 1) / w_bytes : 1;
    const size_t copy_stride = (w_bytes + 4095) / 4096 * 4096 + 64;
    const size_t map_bytes = (copy_stride * ncopies + woff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *w = map + woff;
    const size_t n_x = (size_t)n_groups * group_dim, n_out = (size_t)n_groups * rank;
    float *x = malloc(n_x * sizeof(float));
    double *ref = malloc(n_out * sizeof(double));
    float *got_hand = malloc(n_out * sizeof(float));
    float *got_tl = malloc(n_out * sizeof(float));
    CHECK(x && ref && got_hand && got_tl);
    for (size_t i = 0; i < n_x; i++) x[i] = frand();
    for (uint32_t g = 0; g < n_groups; g++) {
        for (uint32_t r = 0; r < rank; r++) {
            double acc = 0;
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *blk = w + ((size_t)g * rank + r) * row_bytes + (size_t)b * 34;
                const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
                memcpy(blk, &dbits, 2);
                const double d = f16_bits_to_f32(dbits);
                for (uint32_t k = 0; k < 32; k++) {
                    const int8_t q = (int8_t)((int)(frand() * 63.0f));
                    blk[2 + k] = (uint8_t)q;
                    acc += d * q * x[(size_t)g * group_dim + b * 32 + k];
                }
            }
            ref[(size_t)g * rank + r] = acc;
        }
    }
    for (size_t c = 1; c < ncopies; c++) memcpy(w + c * copy_stride, w, w_bytes);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n_x * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    CHECK(tx && tout);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, n_x * sizeof(float)));
    for (int which = 0; which < 2; which++) {
        float *got = which ? got_tl : got_hand;
        memset(got, 0, n_out * sizeof(float));
        CHECK(ds4_gpu_tensor_write(tout, 0, got, n_out * sizeof(float)));
        const int ok = which == 0
            ? ds4_gpu_attention_output_low_q8_tensor(tout, map, map_bytes, woff, group_dim, rank, n_groups, tx)
            : ds4_gpu_tl_attention_output_low_q8_tensor(tout, map, map_bytes, woff, group_dim, rank, n_groups, tx);
        CHECK(ok);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout, 0, got, n_out * sizeof(float)));
    }
    {
        double ref_max = 0, e_hand = 0, e_tl = 0;
        for (size_t i = 0; i < n_out; i++) {
            if (fabs(ref[i]) > ref_max) ref_max = fabs(ref[i]);
            e_hand = fmax(e_hand, fabs(got_hand[i] - ref[i]));
            e_tl = fmax(e_tl, fabs(got_tl[i] - ref[i]));
        }
        const int pass = e_tl / ref_max < 1e-5 && e_hand / ref_max < 1e-5;
        printf("q8_0_grouped     gd=%-4u rank=%-5u g=%u hand|ref %.2e  tl|ref %.2e  %s\n",
               group_dim, rank, n_groups, e_hand / ref_max, e_tl / ref_max, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t_hand = 0, t_tl = 0;
        for (int which = 0; which < 2; which++) {
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    const uint64_t off = woff + (uint64_t)(it % ncopies) * copy_stride;
                    const int ok = which == 0
                        ? ds4_gpu_attention_output_low_q8_tensor(tout, map, map_bytes, off, group_dim, rank, n_groups, tx)
                        : ds4_gpu_tl_attention_output_low_q8_tensor(tout, map, map_bytes, off, group_dim, rank, n_groups, tx);
                    CHECK(ok);
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) { if (which == 0) t_hand = dt; else t_tl = dt; }
            }
        }
        printf("q8_0_grouped     gd=%-4u rank=%-5u g=%u bench x%d (%zu cold copies): hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               group_dim, rank, n_groups, bench_iters, ncopies, t_hand, w_bytes / t_hand / 1e3, t_tl, w_bytes / t_tl / 1e3, t_tl / t_hand);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tout);
    free(x); free(ref); free(got_hand); free(got_tl); free(map);
    return 1;
}

#include "ds4_iq2_tables.h"

/* Routed MoE gate/up (IQ2_XXS) + SwiGLU * route weight, hand-written vs
 * TileLang, on host-built experts: random codebook indices, sign indices and
 * 4-bit scales (every IQ2_XXS bit pattern is valid). */
static int check_moe_gate_up_iq2(uint32_t in_dim, uint32_t mid_dim, uint32_t n_total, uint32_t n_used, int bench_iters) {
    const uint32_t blocks = in_dim / 256;
    const size_t row_bytes = (size_t)blocks * 66;
    const size_t expert_bytes = row_bytes * mid_dim;
    const size_t mat_bytes = expert_bytes * n_total;        /* one of gate / up */
    const uint64_t goff = 32 * 3;
    const size_t page = (size_t)getpagesize();
    const size_t cold_target = (size_t)768 << 20;
    /* Each dispatch touches n_used experts of gate and up; rotate expert ids
     * across copies so consecutive dispatches read different experts. */
    const size_t touched = 2 * expert_bytes * n_used;
    const size_t ncopies = bench_iters > 0 && touched < cold_target ? (cold_target + touched - 1) / touched : 1;
    const size_t copy_stride = (2 * mat_bytes + 4095) / 4096 * 4096 + 64;
    const size_t map_bytes = (copy_stride * ncopies + goff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *wg = map + goff;
    uint8_t *wu = wg + mat_bytes;
    const uint64_t uoff = goff + mat_bytes;

    float *x = malloc(in_dim * sizeof(float));
    int32_t ids[8]; float route[8];
    const float clamp = 7.0f;
    CHECK(n_used <= 8);
    for (uint32_t i = 0; i < in_dim; i++) x[i] = frand();
    for (uint32_t s = 0; s < n_used; s++) { ids[s] = (int32_t)((s * 37 + 11) % n_total); route[s] = 0.1f + 0.3f * (s + 1); }
    /* Random IQ2_XXS blocks everywhere (only the selected experts are read,
     * but the bench rotates through copies). Fill one expert pattern and
     * replicate; different experts get the same bytes which is fine for
     * timing and the reference reads whatever is there. */
    uint8_t *pattern = malloc(expert_bytes);
    for (size_t i = 0; i < expert_bytes; i += 66) {
        const uint16_t d = f32_to_f16_bits(0.002f + 0.004f * (frand() + 2.0f));
        memcpy(pattern + i, &d, 2);
        for (int k = 2; k < 66; k++) pattern[i + k] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 24);
    }
    for (uint32_t e = 0; e < n_total; e++) {
        memcpy(wg + (size_t)e * expert_bytes, pattern, expert_bytes);
        /* perturb up so gate != up */
        memcpy(wu + (size_t)e * expert_bytes, pattern, expert_bytes);
        for (size_t i = 2; i < expert_bytes; i += 66) wu[(size_t)e * expert_bytes + i + 5] ^= (uint8_t)(e + 1);
    }
    free(pattern);

    /* double reference */
    double *ref_g = malloc((size_t)n_used * mid_dim * sizeof(double));
    double *ref_u = malloc((size_t)n_used * mid_dim * sizeof(double));
    double *ref_m = malloc((size_t)n_used * mid_dim * sizeof(double));
    CHECK(x && ref_g && ref_u && ref_m);
    for (uint32_t s = 0; s < n_used; s++) {
        for (uint32_t r = 0; r < mid_dim; r++) {
            for (int which = 0; which < 2; which++) {
                const uint8_t *row = (which ? wu : wg) + (size_t)ids[s] * expert_bytes + (size_t)r * row_bytes;
                double acc = 0;
                for (uint32_t b = 0; b < blocks; b++) {
                    uint16_t dbits; memcpy(&dbits, row + b * 66, 2);
                    const double d = f16_bits_to_f32(dbits);
                    const uint16_t *qs = (const uint16_t *)(row + b * 66 + 2);
                    for (int ib = 0; ib < 8; ib++) {
                        const uint16_t *q = qs + 4 * ib;
                        const uint32_t aux32 = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
                        const uint8_t aux8[4] = { (uint8_t)q[0], (uint8_t)(q[0] >> 8), (uint8_t)q[1], (uint8_t)(q[1] >> 8) };
                        const double dl = d * (0.5 + (aux32 >> 28)) * 0.25;
                        double sum = 0;
                        for (int l = 0; l < 4; l++) {
                            const uint8_t *grid = (const uint8_t *)(ds4_iq2xxs_grid + aux8[l]);
                            const uint8_t signs = ds4_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                            for (int j = 0; j < 8; j++) {
                                const double v = (signs & (1u << j)) ? -(double)grid[j] : (double)grid[j];
                                sum += v * x[b * 256 + ib * 32 + l * 8 + j];
                            }
                        }
                        acc += dl * sum;
                    }
                }
                (which ? ref_u : ref_g)[(size_t)s * mid_dim + r] = acc;
            }
            const double g = ref_g[(size_t)s * mid_dim + r], u = ref_u[(size_t)s * mid_dim + r];
            const double gc = g < clamp ? g : clamp;
            const double uc = u < -clamp ? -clamp : (u > clamp ? clamp : u);
            ref_m[(size_t)s * mid_dim + r] = gc / (1.0 + exp(-gc)) * uc * route[s];
        }
    }
    for (size_t c = 1; c < ncopies; c++) memcpy(wg + c * copy_stride, wg, 2 * mat_bytes);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    const size_t n_out = (size_t)n_used * mid_dim;
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(in_dim * sizeof(float));
    ds4_gpu_tensor *tids = ds4_gpu_tensor_alloc_managed(sizeof(ids));
    ds4_gpu_tensor *trw = ds4_gpu_tensor_alloc_managed(sizeof(route));
    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    ds4_gpu_tensor *tu = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    ds4_gpu_tensor *tm = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    CHECK(tx && tids && trw && tg && tu && tm);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, in_dim * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tids, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_tensor_write(trw, 0, route, sizeof(route)));
    float *got[2][3];
    for (int which = 0; which < 2; which++) {
        for (int k = 0; k < 3; k++) { got[which][k] = calloc(n_out, sizeof(float)); CHECK(got[which][k]); }
        CHECK(ds4_gpu_tensor_write(tg, 0, got[which][0], n_out * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(tu, 0, got[which][1], n_out * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(tm, 0, got[which][2], n_out * sizeof(float)));
        CHECK(ds4_gpu_moe_gate_up_iq2_xxs_swiglu_tensor(tg, tu, tm, map, map_bytes, goff, uoff, expert_bytes, row_bytes,
                                                        in_dim, mid_dim, n_total, tids, trw, n_used, clamp, tx, which));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tg, 0, got[which][0], n_out * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(tu, 0, got[which][1], n_out * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(tm, 0, got[which][2], n_out * sizeof(float)));
    }
    {
        const double *refs[3] = { ref_g, ref_u, ref_m };
        const char *nm[3] = { "gate", "up", "mid" };
        int pass = 1;
        char line[512]; size_t n = 0;
        n += (size_t)snprintf(line + n, sizeof(line) - n, "moe_gate_up_iq2  in=%-5u mid=%-5u used=%u", in_dim, mid_dim, n_used);
        for (int k = 0; k < 3; k++) {
            double rmax = 0, eh = 0, et = 0;
            for (size_t i = 0; i < n_out; i++) {
                rmax = fmax(rmax, fabs(refs[k][i]));
                eh = fmax(eh, fabs(got[0][k][i] - refs[k][i]));
                et = fmax(et, fabs(got[1][k][i] - refs[k][i]));
            }
            /* The hand column is informational only: this branch carries no
             * hand-written MoE kernels (their entry points dispatch no-op
             * stand-ins), so only TileLang against the reference gates. */
            n += (size_t)snprintf(line + n, sizeof(line) - n, "  %s hand %.1e tl %.1e", nm[k], eh / rmax, et / rmax);
            pass = pass && et / rmax < 1e-5;
        }
        printf("%s  %s\n", line, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    const uint64_t off = (uint64_t)(it % ncopies) * copy_stride;
                    CHECK(ds4_gpu_moe_gate_up_iq2_xxs_swiglu_tensor(tg, tu, tm, map, map_bytes, goff + off, uoff + off,
                            expert_bytes, row_bytes, in_dim, mid_dim, n_total, tids, trw, n_used, clamp, tx, which));
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) t[which] = dt;
            }
        }
        printf("moe_gate_up_iq2  in=%-5u mid=%-5u used=%u bench x%d (%zu cold copies, %.1f MB/dispatch): hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               in_dim, mid_dim, n_used, bench_iters, ncopies, touched / 1e6,
               t[0], touched / t[0] / 1e3, t[1], touched / t[1] / 1e3, t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tids); ds4_gpu_tensor_free(trw);
    ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tu); ds4_gpu_tensor_free(tm);
    for (int w = 0; w < 2; w++) for (int k = 0; k < 3; k++) free(got[w][k]);
    free(x); free(ref_g); free(ref_u); free(ref_m); free(map);
    return 1;
}

/* Routed MoE down projection, Q2_K, summed over the selected experts. */
static int check_moe_down_q2k(uint32_t mid_dim, uint32_t out_dim, uint32_t n_total, uint32_t n_used, int bench_iters) {
    const uint32_t blocks = mid_dim / 256;
    const size_t row_bytes = (size_t)blocks * 84;
    const size_t expert_bytes = row_bytes * out_dim;
    const size_t mat_bytes = expert_bytes * n_total;
    const uint64_t doff = 32 * 3;
    const size_t page = (size_t)getpagesize();
    const size_t cold_target = (size_t)768 << 20;
    const size_t touched = expert_bytes * n_used;
    const size_t ncopies = bench_iters > 0 && touched < cold_target ? (cold_target + touched - 1) / touched : 1;
    const size_t copy_stride = (mat_bytes + 4095) / 4096 * 4096 + 64;
    const size_t map_bytes = (copy_stride * ncopies + doff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *wd = map + doff;
    const size_t n_mid = (size_t)n_used * mid_dim;
    float *mid = malloc(n_mid * sizeof(float));
    int32_t ids[8];
    CHECK(mid && n_used <= 8);
    for (size_t i = 0; i < n_mid; i++) mid[i] = frand();
    for (uint32_t s = 0; s < n_used; s++) ids[s] = (int32_t)((s * 29 + 5) % n_total);
    /* random Q2_K blocks: scales/mins nibbles, codes, d/dmin */
    for (size_t i = 0; i < expert_bytes; i += 84) {
        uint8_t *blk = wd + i;
        for (int k = 0; k < 80; k++) blk[k] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 24);
        const uint16_t d = f32_to_f16_bits(0.01f + 0.02f * (frand() + 2.0f));
        const uint16_t dm = f32_to_f16_bits(0.005f * (frand() + 2.0f));
        memcpy(blk + 80, &d, 2);
        memcpy(blk + 82, &dm, 2);
    }
    for (uint32_t e = 1; e < n_total; e++) {
        memcpy(wd + (size_t)e * expert_bytes, wd, expert_bytes);
        for (size_t i = 0; i < expert_bytes; i += 84) wd[(size_t)e * expert_bytes + i + 17] ^= (uint8_t)(e * 7 + 1);
    }
    double *ref = malloc(out_dim * sizeof(double));
    CHECK(ref);
    for (uint32_t r = 0; r < out_dim; r++) {
        double acc = 0;
        for (uint32_t s = 0; s < n_used; s++) {
            const uint8_t *row = wd + (size_t)ids[s] * expert_bytes + (size_t)r * row_bytes;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *blk = row + b * 84;
                uint16_t dbits, mbits;
                memcpy(&dbits, blk + 80, 2); memcpy(&mbits, blk + 82, 2);
                const double d = f16_bits_to_f32(dbits), dmin = f16_bits_to_f32(mbits);
                for (int w = 0; w < 256; w++) {
                    const int is = w / 16;
                    const int q = (blk[16 + (w / 128) * 32 + (w % 32)] >> (2 * ((w % 128) / 32))) & 3;
                    const double v = d * (blk[is] & 0xF) * q - dmin * (blk[is] >> 4);
                    acc += v * mid[(size_t)s * mid_dim + b * 256 + w];
                }
            }
        }
        ref[r] = acc;
    }
    for (size_t c = 1; c < ncopies; c++) memcpy(wd + c * copy_stride, wd, mat_bytes);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    ds4_gpu_tensor *tmid = ds4_gpu_tensor_alloc_managed(n_mid * sizeof(float));
    ds4_gpu_tensor *tids = ds4_gpu_tensor_alloc_managed(sizeof(ids));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(out_dim * sizeof(float));
    CHECK(tmid && tids && tout);
    CHECK(ds4_gpu_tensor_write(tmid, 0, mid, n_mid * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tids, 0, ids, sizeof(ids)));
    float *got[2];
    for (int which = 0; which < 2; which++) {
        got[which] = calloc(out_dim, sizeof(float)); CHECK(got[which]);
        CHECK(ds4_gpu_tensor_write(tout, 0, got[which], out_dim * sizeof(float)));
        CHECK(ds4_gpu_moe_down_q2_k_sum_tensor(tout, map, map_bytes, doff, expert_bytes, row_bytes, mid_dim, out_dim,
                                               n_total, tids, tmid, n_used, which));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout, 0, got[which], out_dim * sizeof(float)));
    }
    {
        double rmax = 0, eh = 0, et = 0;
        for (uint32_t i = 0; i < out_dim; i++) {
            rmax = fmax(rmax, fabs(ref[i]));
            eh = fmax(eh, fabs(got[0][i] - ref[i]));
            et = fmax(et, fabs(got[1][i] - ref[i]));
        }
        const int pass = et / rmax < 1e-5;  /* hand column informational: no hand kernel here */
        printf("moe_down_q2k     mid=%-5u out=%-5u used=%u hand|ref %.1e  tl|ref %.1e  %s\n",
               mid_dim, out_dim, n_used, eh / rmax, et / rmax, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    const uint64_t off = doff + (uint64_t)(it % ncopies) * copy_stride;
                    CHECK(ds4_gpu_moe_down_q2_k_sum_tensor(tout, map, map_bytes, off, expert_bytes, row_bytes, mid_dim, out_dim,
                                                           n_total, tids, tmid, n_used, which));
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) t[which] = dt;
            }
        }
        printf("moe_down_q2k     mid=%-5u out=%-5u used=%u bench x%d (%zu cold copies, %.1f MB/dispatch): hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               mid_dim, out_dim, n_used, bench_iters, ncopies, touched / 1e6,
               t[0], touched / t[0] / 1e3, t[1], touched / t[1] / 1e3, t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tmid); ds4_gpu_tensor_free(tids); ds4_gpu_tensor_free(tout);
    free(got[0]); free(got[1]); free(mid); free(ref); free(map);
    return 1;
}

/* Shared expert gate/up (Q8_0) + SwiGLU: ds4_gpu_shared_gate_up_swiglu_q8_0_tensor
 * vs ds4_gpu_tl_shared_gate_up_swiglu_q8_0_tensor. */
static int check_q8_0_gate_up_swiglu(uint32_t in_dim, uint32_t out_dim, int bench_iters) {
    const uint32_t blocks = in_dim / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * out_dim;          /* one matrix */
    const uint64_t goff = 32 * 5;
    const size_t page = (size_t)getpagesize();
    const size_t cold_target = (size_t)768 << 20;
    const size_t touched = 2 * w_bytes;
    const size_t ncopies = bench_iters > 0 && touched < cold_target ? (cold_target + touched - 1) / touched : 1;
    const size_t copy_stride = (touched + 4095) / 4096 * 4096 + 64;
    const size_t map_bytes = (copy_stride * ncopies + goff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *wg = map + goff;
    uint8_t *wu = wg + w_bytes;
    const uint64_t uoff = goff + w_bytes;
    const float clamp = 7.0f;
    float *x = malloc(in_dim * sizeof(float));
    double *ref[3];
    for (int k = 0; k < 3; k++) { ref[k] = malloc(out_dim * sizeof(double)); CHECK(ref[k]); }
    CHECK(x);
    for (uint32_t i = 0; i < in_dim; i++) x[i] = frand();
    for (uint32_t r = 0; r < out_dim; r++) {
        for (int which = 0; which < 2; which++) {
            uint8_t *row = (which ? wu : wg) + (size_t)r * row_bytes;
            double acc = 0;
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *blk = row + (size_t)b * 34;
                const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
                memcpy(blk, &dbits, 2);
                const double d = f16_bits_to_f32(dbits);
                for (uint32_t k = 0; k < 32; k++) {
                    const int8_t q = (int8_t)((int)(frand() * 63.0f));
                    blk[2 + k] = (uint8_t)q;
                    acc += d * q * x[b * 32 + k];
                }
            }
            ref[which][r] = acc;
        }
        const double g = ref[0][r], u = ref[1][r];
        const double gc = g < clamp ? g : clamp;
        const double uc = u < -clamp ? -clamp : (u > clamp ? clamp : u);
        ref[2][r] = gc / (1.0 + exp(-gc)) * uc;
    }
    for (size_t c = 1; c < ncopies; c++) memcpy(wg + c * copy_stride, wg, touched);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(in_dim * sizeof(float));
    ds4_gpu_tensor *t[3];
    for (int k = 0; k < 3; k++) { t[k] = ds4_gpu_tensor_alloc_managed(out_dim * sizeof(float)); CHECK(t[k]); }
    CHECK(tx && ds4_gpu_tensor_write(tx, 0, x, in_dim * sizeof(float)));
    float *got[2][3];
    for (int which = 0; which < 2; which++) {
        for (int k = 0; k < 3; k++) {
            got[which][k] = calloc(out_dim, sizeof(float)); CHECK(got[which][k]);
            CHECK(ds4_gpu_tensor_write(t[k], 0, got[which][k], out_dim * sizeof(float)));
        }
        const int ok = which == 0
            ? ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(t[0], t[1], t[2], map, map_bytes, goff, uoff, in_dim, out_dim, tx, clamp)
            : ds4_gpu_tl_shared_gate_up_swiglu_q8_0_tensor(t[0], t[1], t[2], map, map_bytes, goff, uoff, in_dim, out_dim, tx, clamp);
        CHECK(ok);
        CHECK(ds4_gpu_synchronize());
        for (int k = 0; k < 3; k++) CHECK(ds4_gpu_tensor_read(t[k], 0, got[which][k], out_dim * sizeof(float)));
    }
    {
        const char *nm[3] = { "gate", "up", "mid" };
        int pass = 1;
        char line[512]; size_t n = 0;
        n += (size_t)snprintf(line + n, sizeof(line) - n, "q8_0_gate_up_swi in=%-5u out=%-5u", in_dim, out_dim);
        for (int k = 0; k < 3; k++) {
            double rmax = 0, eh = 0, et = 0;
            for (uint32_t i = 0; i < out_dim; i++) {
                rmax = fmax(rmax, fabs(ref[k][i]));
                eh = fmax(eh, fabs(got[0][k][i] - ref[k][i]));
                et = fmax(et, fabs(got[1][k][i] - ref[k][i]));
            }
            n += (size_t)snprintf(line + n, sizeof(line) - n, "  %s hand %.1e tl %.1e", nm[k], eh / rmax, et / rmax);
            pass = pass && et / rmax < 1e-5 && eh / rmax < 1e-5;
        }
        printf("%s  %s\n", line, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double tt[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            for (int pass = 0; pass < 2; pass++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    const uint64_t off = (uint64_t)(it % ncopies) * copy_stride;
                    const int ok = which == 0
                        ? ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(t[0], t[1], t[2], map, map_bytes, goff + off, uoff + off, in_dim, out_dim, tx, clamp)
                        : ds4_gpu_tl_shared_gate_up_swiglu_q8_0_tensor(t[0], t[1], t[2], map, map_bytes, goff + off, uoff + off, in_dim, out_dim, tx, clamp);
                    CHECK(ok);
                }
                CHECK(ds4_gpu_end_commands());
                const double dt = (now_us() - t0) / bench_iters;
                if (pass == 1) tt[which] = dt;
            }
        }
        printf("q8_0_gate_up_swi in=%-5u out=%-5u bench x%d (%zu cold copies): hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               in_dim, out_dim, bench_iters, ncopies, tt[0], touched / tt[0] / 1e3, tt[1], touched / tt[1] / 1e3, tt[1] / tt[0]);
    }
    ds4_gpu_tensor_free(tx);
    for (int k = 0; k < 3; k++) { ds4_gpu_tensor_free(t[k]); free(ref[k]); free(got[0][k]); free(got[1][k]); }
    free(x); free(map);
    return 1;
}

/* Prefill-tier Q8_0 GEMM through ds4's own entry (hand-written only for now):
 * out[n_tok][out_dim] = x[n_tok][in_dim] . W^T, timing + TFLOPS. */
static int bench_q8_0_gemm_hand(uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, int iters) {
    const uint32_t blocks = in_dim / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * out_dim;
    const uint64_t woff = 32 * 5;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (w_bytes + woff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *w = map + woff;
    for (size_t i = 0; i < w_bytes; i += 34) {
        const uint16_t d = f32_to_f16_bits(0.01f);
        memcpy(w + i, &d, 2);
        for (int k = 2; k < 34; k++) w[i + k] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 25);
    }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    const size_t n_x = (size_t)n_tok * in_dim, n_out = (size_t)n_tok * out_dim;
    float *x = malloc(n_x * sizeof(float));
    CHECK(x);
    for (size_t i = 0; i < n_x; i++) x[i] = frand();
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n_x * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    CHECK(tx && tout && ds4_gpu_tensor_write(tx, 0, x, n_x * sizeof(float)));
    CHECK(ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok));
    CHECK(ds4_gpu_synchronize());
    double t = 0;
    for (int pass = 0; pass < 2; pass++) {
        const double t0 = now_us();
        CHECK(ds4_gpu_begin_commands());
        for (int it = 0; it < iters; it++) {
            CHECK(ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok));
        }
        CHECK(ds4_gpu_end_commands());
        t = (now_us() - t0) / iters;
    }
    double tt = 0;
    if (ds4_gpu_tl_matmul_q8_0_gemm_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok)) {
        CHECK(ds4_gpu_synchronize());
        for (int pass = 0; pass < 2; pass++) {
            const double t0 = now_us();
            CHECK(ds4_gpu_begin_commands());
            for (int it = 0; it < iters; it++) {
                CHECK(ds4_gpu_tl_matmul_q8_0_gemm_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok));
            }
            CHECK(ds4_gpu_end_commands());
            tt = (now_us() - t0) / iters;
        }
    }
    const double flops = 2.0 * n_tok * in_dim * out_dim;
    printf("q8_0_gemm        in=%-5u out=%-6u tok=%-5u hand %9.1f us (%5.1f TFLOPS)  tl %9.1f us (%5.1f TFLOPS)  tl/hand %.2fx\n",
           in_dim, out_dim, n_tok, t, flops / t / 1e6, tt, tt > 0 ? flops / tt / 1e6 : 0.0, tt > 0 ? tt / t : 0.0);
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tout); free(x); free(map);
    return 1;
}

/* Prefill Q8_0 GEMM: ds4's own tier vs the TileLang cooperative-tensor tier,
 * both against a double reference (f16 operands on both GPU paths, so the
 * tolerance is 1e-2 of max |ref|), then timing. */
static int check_q8_0_gemm(uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, int bench_iters) {
    const uint32_t blocks = in_dim / 32;
    const size_t row_bytes = (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * out_dim;
    const uint64_t woff = 32 * 5;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (w_bytes + woff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *w = map + woff;
    const size_t n_x = (size_t)n_tok * in_dim, n_out = (size_t)n_tok * out_dim;
    float *x = malloc(n_x * sizeof(float));
    float *wf = malloc((size_t)out_dim * in_dim * sizeof(float));
    double *ref = malloc(n_out * sizeof(double));
    float *got[2] = { malloc(n_out * sizeof(float)), malloc(n_out * sizeof(float)) };
    CHECK(x && wf && ref && got[0] && got[1]);
    for (size_t i = 0; i < n_x; i++) x[i] = frand();
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (size_t)r * row_bytes + (size_t)b * 34;
            const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
            memcpy(blk, &dbits, 2);
            const float d = f16_bits_to_f32(dbits);
            for (uint32_t k = 0; k < 32; k++) {
                const int8_t q = (int8_t)((int)(frand() * 63.0f));
                blk[2 + k] = (uint8_t)q;
                wf[(size_t)r * in_dim + b * 32 + k] = d * q;
            }
        }
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            double acc = 0;
            const float *xr = x + (size_t)t * in_dim, *wr = wf + (size_t)r * in_dim;
            for (uint32_t k = 0; k < in_dim; k++) acc += (double)xr[k] * wr[k];
            ref[(size_t)t * out_dim + r] = acc;
        }
    }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n_x * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    CHECK(tx && tout && ds4_gpu_tensor_write(tx, 0, x, n_x * sizeof(float)));
    int tl_ok = 1;
    for (int which = 0; which < 2; which++) {
        memset(got[which], 0, n_out * sizeof(float));
        CHECK(ds4_gpu_tensor_write(tout, 0, got[which], n_out * sizeof(float)));
        const int ok = which == 0
            ? ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok)
            : ds4_gpu_tl_matmul_q8_0_gemm_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok);
        if (which == 1 && !ok) { tl_ok = 0; break; }
        CHECK(ok);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout, 0, got[which], n_out * sizeof(float)));
    }
    if (!tl_ok) {
        printf("q8_0_gemm        in=%-5u out=%-6u tok=%-5u tl: declined (fallback)\n", in_dim, out_dim, n_tok);
    } else {
        double rmax = 0, eh = 0, et = 0;
        for (size_t i = 0; i < n_out; i++) {
            rmax = fmax(rmax, fabs(ref[i]));
            eh = fmax(eh, fabs(got[0][i] - ref[i]));
            et = fmax(et, fabs(got[1][i] - ref[i]));
        }
        const int pass = et / rmax < 1e-2 && eh / rmax < 1e-2;
        printf("q8_0_gemm        in=%-5u out=%-6u tok=%-5u hand|ref %.1e  tl|ref %.1e  %s\n",
               in_dim, out_dim, n_tok, eh / rmax, et / rmax, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
        if (bench_iters > 0) {
            double t[2] = { 0, 0 };
            for (int which = 0; which < 2; which++) {
                for (int pass2 = 0; pass2 < 2; pass2++) {
                    const double t0 = now_us();
                    CHECK(ds4_gpu_begin_commands());
                    for (int it = 0; it < bench_iters; it++) {
                        CHECK(which == 0
                            ? ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok)
                            : ds4_gpu_tl_matmul_q8_0_gemm_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok));
                    }
                    CHECK(ds4_gpu_end_commands());
                    if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
                }
            }
            const double flops = 2.0 * n_tok * in_dim * out_dim;
            printf("q8_0_gemm        in=%-5u out=%-6u tok=%-5u bench x%d: hand %9.1f us (%5.1f TFLOPS)  tl %9.1f us (%5.1f TFLOPS)  tl/hand %.2fx\n",
                   in_dim, out_dim, n_tok, bench_iters, t[0], flops / t[0] / 1e6, t[1], flops / t[1] / 1e6, t[1] / t[0]);
        }
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tout);
    free(x); free(wf); free(ref); free(got[0]); free(got[1]); free(map);
    return 1;
}

/* Prefill attention output: out_a (grouped, n_groups x (group_dim -> rank))
 * then out_b (n_groups*rank -> out_dim); hand vs TileLang-low-stage.
 * `low` is checked against a double reference, `out` hand vs TL. */
static int check_attn_out_batch(uint32_t group_dim, uint32_t rank, uint32_t n_groups, uint32_t out_dim, uint32_t n_tok, int bench_iters, int check) {
    const uint32_t blocks_a = group_dim / 32, blocks_b = (n_groups * rank) / 32;
    const size_t row_a = (size_t)blocks_a * 34, row_b = (size_t)blocks_b * 34;
    const size_t a_bytes = row_a * rank * n_groups, b_bytes = row_b * out_dim;
    const uint64_t aoff = 32 * 3, boff = aoff + ((a_bytes + 63) / 64) * 64;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (boff + b_bytes + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *wa = map + aoff, *wb = map + boff;
    const size_t n_heads = (size_t)n_tok * n_groups * group_dim, n_low = (size_t)n_tok * n_groups * rank, n_out = (size_t)n_tok * out_dim;
    float *heads = malloc(n_heads * sizeof(float));
    float *wa_f = check ? malloc((size_t)n_groups * rank * group_dim * sizeof(float)) : NULL;
    double *ref_low = check ? malloc(n_low * sizeof(double)) : NULL;
    CHECK(heads && (!check || (wa_f && ref_low)));
    for (size_t i = 0; i < n_heads; i++) heads[i] = frand();
    for (size_t r = 0; r < (size_t)n_groups * rank; r++) {
        for (uint32_t b = 0; b < blocks_a; b++) {
            uint8_t *blk = wa + r * row_a + (size_t)b * 34;
            const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
            memcpy(blk, &dbits, 2);
            const float d = f16_bits_to_f32(dbits);
            for (uint32_t k = 0; k < 32; k++) {
                const int8_t q = (int8_t)((int)(frand() * 63.0f));
                blk[2 + k] = (uint8_t)q;
                if (check) wa_f[r * group_dim + b * 32 + k] = d * q;
            }
        }
    }
    for (size_t i = 0; i < b_bytes; i += 34) {
        const uint16_t d = f32_to_f16_bits(0.01f);
        memcpy(wb + i, &d, 2);
        for (int k = 2; k < 34; k++) wb[i + k] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 25);
    }
    if (check) {
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t g = 0; g < n_groups; g++)
                for (uint32_t r = 0; r < rank; r++) {
                    double acc = 0;
                    const float *h = heads + ((size_t)t * n_groups + g) * group_dim;
                    const float *w = wa_f + ((size_t)g * rank + r) * group_dim;
                    for (uint32_t k = 0; k < group_dim; k++) acc += (double)h[k] * w[k];
                    ref_low[((size_t)t * n_groups + g) * rank + r] = acc;
                }
    }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    ds4_gpu_tensor *th = ds4_gpu_tensor_alloc_managed(n_heads * sizeof(float));
    ds4_gpu_tensor *tlow = ds4_gpu_tensor_alloc_managed(n_low * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    ds4_gpu_tensor *tgt = ds4_gpu_tensor_alloc_managed(n_low * sizeof(float));
    ds4_gpu_tensor *tlt = ds4_gpu_tensor_alloc_managed(n_low * sizeof(float));
    CHECK(th && tlow && tout && tgt && tlt && ds4_gpu_tensor_write(th, 0, heads, n_heads * sizeof(float)));
    float *low[2] = { malloc(n_low * sizeof(float)), malloc(n_low * sizeof(float)) };
    float *outv[2] = { malloc(n_out * sizeof(float)), malloc(n_out * sizeof(float)) };
    CHECK(low[0] && low[1] && outv[0] && outv[1]);
    int tl_ok = 1;
    for (int which = 0; which < 2; which++) {
        memset(low[which], 0, n_low * sizeof(float));
        CHECK(ds4_gpu_tensor_write(tlow, 0, low[which], n_low * sizeof(float)));
        int ok;
        if (n_tok < 64) {
            /* small batch: the grouped GEMV with token tiles (family Q8_GEMV) */
            CHECK(ds4_gpu_tl_force("Q8_GEMV", which) && ds4_gpu_tl_force("GEMM", which));
            ok = ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, tgt, tlt, map, map_bytes, aoff, boff, group_dim, rank, n_groups, out_dim, th, n_tok);
            CHECK(ds4_gpu_tl_force("Q8_GEMV", 0) && ds4_gpu_tl_force("GEMM", 0));
        } else {
            ok = which == 0
                ? ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, tgt, tlt, map, map_bytes, aoff, boff, group_dim, rank, n_groups, out_dim, th, n_tok)
                : ds4_gpu_tl_attention_output_q8_batch_tensor(tout, tlow, tgt, tlt, map, map_bytes, aoff, boff, group_dim, rank, n_groups, out_dim, th, n_tok);
        }
        if (which == 1 && !ok) { tl_ok = 0; break; }
        CHECK(ok);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tlow, 0, low[which], n_low * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(tout, 0, outv[which], n_out * sizeof(float)));
    }
    if (!tl_ok) {
        printf("attn_out_batch   gd=%u rank=%u g=%u out=%u tok=%-5u tl: declined (fallback)\n", group_dim, rank, n_groups, out_dim, n_tok);
    } else if (check) {
        double rmax = 0, eh = 0, et = 0, omax = 0, eo = 0;
        for (size_t i = 0; i < n_low; i++) {
            rmax = fmax(rmax, fabs(ref_low[i]));
            eh = fmax(eh, fabs(low[0][i] - ref_low[i]));
            et = fmax(et, fabs(low[1][i] - ref_low[i]));
        }
        for (size_t i = 0; i < n_out; i++) { omax = fmax(omax, fabs(outv[0][i])); eo = fmax(eo, fabs(outv[0][i] - outv[1][i])); }
        const int pass = et / rmax < 1e-2 && eh / rmax < 1e-2 && eo / omax < 2e-2;
        printf("attn_out_batch   gd=%u rank=%u g=%u out=%u tok=%-5u low: hand|ref %.1e tl|ref %.1e  out: tl|hand %.1e  %s\n",
               group_dim, rank, n_groups, out_dim, n_tok, eh / rmax, et / rmax, eo / omax, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (tl_ok && bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                if (n_tok < 64) CHECK(ds4_gpu_tl_force("Q8_GEMV", which) && ds4_gpu_tl_force("GEMM", which));
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(which == 0 || n_tok < 64
                        ? ds4_gpu_attention_output_q8_batch_tensor(tout, tlow, tgt, tlt, map, map_bytes, aoff, boff, group_dim, rank, n_groups, out_dim, th, n_tok)
                        : ds4_gpu_tl_attention_output_q8_batch_tensor(tout, tlow, tgt, tlt, map, map_bytes, aoff, boff, group_dim, rank, n_groups, out_dim, th, n_tok));
                }
                if (n_tok < 64) CHECK(ds4_gpu_tl_force("Q8_GEMV", 0) && ds4_gpu_tl_force("GEMM", 0));
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
        }
        const double flops = 2.0 * n_tok * ((double)n_groups * group_dim * rank + (double)n_groups * rank * out_dim);
        printf("attn_out_batch   gd=%u rank=%u g=%u out=%u tok=%-5u bench x%d: hand %9.1f us (%5.1f TFLOPS)  tl %9.1f us (%5.1f TFLOPS)  tl/hand %.2fx\n",
               group_dim, rank, n_groups, out_dim, n_tok, bench_iters, t[0], flops / t[0] / 1e6, t[1], flops / t[1] / 1e6, t[1] / t[0]);
    }
    ds4_gpu_tensor_free(th); ds4_gpu_tensor_free(tlow); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tgt); ds4_gpu_tensor_free(tlt);
    free(heads); free(wa_f); free(ref_low); free(low[0]); free(low[1]); free(outv[0]); free(outv[1]); free(map);
    return 1;
}

/* YaRN RoPE reference (ggml semantics, mode 0) in double. */
static void ref_rope_pair(double theta_extrap, int r, int n_dims, uint32_t n_ctx_orig, double freq_base,
                          double freq_scale, double ext_factor, double attn_factor, double beta_fast,
                          double beta_slow, int inverse, double *c, double *s) {
    const double corr_f = n_dims * log((double)n_ctx_orig / (beta_fast * 2 * M_PI)) / (2 * log(freq_base));
    const double corr_s = n_dims * log((double)n_ctx_orig / (beta_slow * 2 * M_PI)) / (2 * log(freq_base));
    const double low = fmax(0.0, floor(corr_f)), high = fmin(n_dims - 1.0, ceil(corr_s));
    double theta = freq_scale * theta_extrap, mscale = attn_factor;
    if (ext_factor != 0.0) {
        const double y = ((r / 2) - low) / fmax(0.001, high - low);
        const double ramp = (1.0 - fmin(1.0, fmax(0.0, y))) * ext_factor;
        theta = freq_scale * theta_extrap * (1 - ramp) + theta_extrap * ramp;
        mscale *= 1.0 + 0.1 * log(1.0 / freq_scale);
    }
    *c = cos(theta) * mscale;
    *s = (inverse ? -1.0 : 1.0) * sin(theta) * mscale;
}

static int check_rope(uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, int fused_norm, int yarn, int bench_iters) {
    const size_t n = (size_t)n_tok * n_head * head_dim;
    float *x = malloc(n * sizeof(float));
    double *ref = malloc(n * sizeof(double));
    float *got[2] = { malloc(n * sizeof(float)), malloc(n * sizeof(float)) };
    CHECK(x && ref && got[0] && got[1]);
    for (size_t i = 0; i < n; i++) x[i] = frand();
    const uint32_t pos0 = 1234, n_ctx_orig = yarn ? 4096 : 0;
    const float freq_base = 10000.0f, freq_scale = yarn ? 0.25f : 1.0f, ext_factor = yarn ? 1.0f : 0.0f;
    const float attn_factor = yarn ? 1.1f : 1.0f, beta_fast = 32.0f, beta_slow = 1.0f, eps = 1e-6f;
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t h = 0; h < n_head; h++) {
            const float *xr = x + ((size_t)t * n_head + h) * head_dim;
            double *rr = ref + ((size_t)t * n_head + h) * head_dim;
            double scale = 1.0;
            if (fused_norm) {
                double ss = 0;
                for (uint32_t j = 0; j < head_dim; j++) ss += (double)xr[j] * xr[j];
                scale = 1.0 / sqrt(ss / head_dim + eps);
            }
            for (uint32_t j = 0; j < n_nope; j++) rr[j] = xr[j] * scale;
            for (uint32_t r = 0; r < n_rot; r += 2) {
                const double theta = (double)(pos0 + t) * pow((double)freq_base, -(double)r / n_rot);
                double c, s;
                ref_rope_pair(theta, (int)r, (int)n_rot, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, 0, &c, &s);
                const double x0 = xr[n_nope + r] * scale, x1 = xr[n_nope + r + 1] * scale;
                rr[n_nope + r] = x0 * c - x1 * s;
                rr[n_nope + r + 1] = x0 * s + x1 * c;
            }
        }
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n * sizeof(float));
    CHECK(tx);
    for (int which = 0; which < 2; which++) {
        CHECK(ds4_gpu_tensor_write(tx, 0, x, n * sizeof(float)));
        CHECK(ds4_gpu_tl_force("ROPE", which));
        const int ok = fused_norm
            ? ds4_gpu_head_rms_norm_rope_tail_tensor(tx, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps)
            : ds4_gpu_rope_tail_tensor(tx, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        CHECK(ds4_gpu_tl_force("ROPE", 0));
        CHECK(ok);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tx, 0, got[which], n * sizeof(float)));
    }
    {
        double rmax = 0, eh = 0, et = 0, ex = 0;
        for (size_t i = 0; i < n; i++) {
            rmax = fmax(rmax, fabs(ref[i]));
            eh = fmax(eh, fabs(got[0][i] - ref[i]));
            et = fmax(et, fabs(got[1][i] - ref[i]));
            ex = fmax(ex, fabs((double)got[0][i] - got[1][i]));
        }
        /* f32 cos/sin at theta ~ pos * O(1): both GPU paths sit ~1e-4 off the
         * double reference; they should agree with each other more closely. */
        const int pass = et / rmax < 3e-4 && eh / rmax < 3e-4 && ex / rmax < 3e-5;
        printf("%-16s tok=%-4u heads=%-2u hd=%u yarn=%d hand|ref %.1e  tl|ref %.1e  tl|hand %.1e  %s\n",
               fused_norm ? "head_norm_rope" : "rope_tail", n_tok, n_head, head_dim, yarn, eh / rmax, et / rmax, ex / rmax, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ROPE", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(fused_norm
                        ? ds4_gpu_head_rms_norm_rope_tail_tensor(tx, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps)
                        : ds4_gpu_rope_tail_tensor(tx, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("ROPE", 0));
        }
        printf("%-16s tok=%-4u heads=%-2u hd=%u bench x%d: hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n",
               fused_norm ? "head_norm_rope" : "rope_tail", n_tok, n_head, head_dim, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tx); free(x); free(ref); free(got[0]); free(got[1]);
    return 1;
}

/* HC expand (split form: post/comb inside the per-token 24-float split) and
 * HC weighted sum (split form: pre inside the split), n_tokens derived by ds4
 * from the tensor sizes. */
static int check_hc(uint32_t n_tok, uint32_t n_embd, int bench_iters) {
    enum { HC = 4, MIX = 2 * HC + HC * HC };
    const size_t n_hc_elems = (size_t)n_tok * HC * n_embd, n_elems = (size_t)n_tok * n_embd;
    float *block = malloc(n_elems * sizeof(float));
    float *res = malloc(n_hc_elems * sizeof(float));
    float *split = malloc((size_t)n_tok * MIX * sizeof(float));
    double *ref_exp = malloc(n_hc_elems * sizeof(double));
    double *ref_sum = malloc(n_elems * sizeof(double));
    float *got_exp[2] = { malloc(n_hc_elems * sizeof(float)), malloc(n_hc_elems * sizeof(float)) };
    float *got_sum[2] = { malloc(n_elems * sizeof(float)), malloc(n_elems * sizeof(float)) };
    CHECK(block && res && split && ref_exp && ref_sum && got_exp[0] && got_exp[1] && got_sum[0] && got_sum[1]);
    for (size_t i = 0; i < n_elems; i++) block[i] = frand();
    for (size_t i = 0; i < n_hc_elems; i++) res[i] = frand();
    for (size_t i = 0; i < (size_t)n_tok * MIX; i++) split[i] = frand() * 0.5f;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *sp = split + (size_t)t * MIX;
        for (uint32_t d = 0; d < n_embd; d++) {
            const double v = block[(size_t)t * n_embd + d];
            double acc_sum = 0;
            for (int h = 0; h < HC; h++) {
                double acc = v * sp[HC + h];
                for (int j = 0; j < HC; j++) acc += (double)sp[2 * HC + j * HC + h] * res[((size_t)t * HC + j) * n_embd + d];
                ref_exp[((size_t)t * HC + h) * n_embd + d] = acc;
                acc_sum += (double)res[((size_t)t * HC + h) * n_embd + d] * sp[h];
            }
            ref_sum[(size_t)t * n_embd + d] = acc_sum;
        }
    }
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc_managed(n_elems * sizeof(float));
    ds4_gpu_tensor *tr = ds4_gpu_tensor_alloc_managed(n_hc_elems * sizeof(float));
    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc_managed((size_t)n_tok * MIX * sizeof(float));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc_managed(n_hc_elems * sizeof(float));
    ds4_gpu_tensor *tsum = ds4_gpu_tensor_alloc_managed(n_elems * sizeof(float));
    CHECK(tb && tr && ts && to && tsum);
    CHECK(ds4_gpu_tensor_write(tb, 0, block, n_elems * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tr, 0, res, n_hc_elems * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(ts, 0, split, (size_t)n_tok * MIX * sizeof(float)));
    for (int which = 0; which < 2; which++) {
        memset(got_exp[which], 0, n_hc_elems * sizeof(float));
        memset(got_sum[which], 0, n_elems * sizeof(float));
        CHECK(ds4_gpu_tensor_write(to, 0, got_exp[which], n_hc_elems * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(tsum, 0, got_sum[which], n_elems * sizeof(float)));
        CHECK(ds4_gpu_tl_force("HC", which));
        CHECK(ds4_gpu_hc_expand_split_tensor(to, tb, tr, ts, n_embd, HC));
        CHECK(ds4_gpu_hc_weighted_sum_split_tensor(tsum, tr, ts, n_embd, HC));
        CHECK(ds4_gpu_tl_force("HC", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(to, 0, got_exp[which], n_hc_elems * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(tsum, 0, got_sum[which], n_elems * sizeof(float)));
    }
    {
        double m1 = 0, eh1 = 0, et1 = 0, m2 = 0, eh2 = 0, et2 = 0;
        for (size_t i = 0; i < n_hc_elems; i++) { m1 = fmax(m1, fabs(ref_exp[i])); eh1 = fmax(eh1, fabs(got_exp[0][i] - ref_exp[i])); et1 = fmax(et1, fabs(got_exp[1][i] - ref_exp[i])); }
        for (size_t i = 0; i < n_elems; i++) { m2 = fmax(m2, fabs(ref_sum[i])); eh2 = fmax(eh2, fabs(got_sum[0][i] - ref_sum[i])); et2 = fmax(et2, fabs(got_sum[1][i] - ref_sum[i])); }
        const int pass = et1 / m1 < 1e-5 && eh1 / m1 < 1e-5 && et2 / m2 < 1e-5 && eh2 / m2 < 1e-5;
        printf("hc_expand/wsum   tok=%-4u n=%u expand: hand|ref %.1e tl|ref %.1e  wsum: hand|ref %.1e tl|ref %.1e  %s\n",
               n_tok, n_embd, eh1 / m1, et1 / m1, eh2 / m2, et2 / m2, pass ? "OK" : "FAIL");
        if (!pass) g_failures++;
    }
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("HC", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_hc_expand_split_tensor(to, tb, tr, ts, n_embd, HC));
                    CHECK(ds4_gpu_hc_weighted_sum_split_tensor(tsum, tr, ts, n_embd, HC));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("HC", 0));
        }
        printf("hc_expand/wsum   tok=%-4u n=%u bench x%d (both): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n",
               n_tok, n_embd, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tb); ds4_gpu_tensor_free(tr); ds4_gpu_tensor_free(ts); ds4_gpu_tensor_free(to); ds4_gpu_tensor_free(tsum);
    free(block); free(res); free(split); free(ref_exp); free(ref_sum);
    for (int w = 0; w < 2; w++) { free(got_exp[w]); free(got_sum[w]); }
    return 1;
}

/* Elementwise family: swiglu (gate/up -> mid with clamp + weight) and add,
 * through the public entries. */
static int check_elementwise(uint32_t n, int bench_iters) {
    float *g = malloc(n * sizeof(float)), *u = malloc(n * sizeof(float));
    float *got[2] = { malloc(n * sizeof(float)), malloc(n * sizeof(float)) };
    float *gota[2] = { malloc(n * sizeof(float)), malloc(n * sizeof(float)) };
    CHECK(g && u && got[0] && got[1] && gota[0] && gota[1]);
    for (uint32_t i = 0; i < n; i++) { g[i] = frand() * 8.0f; u[i] = frand() * 8.0f; }
    const float clamp = 7.0f, weight = 0.75f;
    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc_managed(n * sizeof(float));
    ds4_gpu_tensor *tu = ds4_gpu_tensor_alloc_managed(n * sizeof(float));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc_managed(n * sizeof(float));
    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc_managed(n * sizeof(float));
    CHECK(tg && tu && to && ta && ds4_gpu_tensor_write(tg, 0, g, n * sizeof(float)) && ds4_gpu_tensor_write(tu, 0, u, n * sizeof(float)));
    for (int which = 0; which < 2; which++) {
        CHECK(ds4_gpu_tl_force("ELEMENTWISE", which));
        CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, n, clamp, weight));
        CHECK(ds4_gpu_add_tensor(ta, tg, tu, n));
        CHECK(ds4_gpu_tl_force("ELEMENTWISE", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(to, 0, got[which], n * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(ta, 0, gota[which], n * sizeof(float)));
    }
    double m = 0, eh = 0, et = 0, ma = 0, ea = 0;
    for (uint32_t i = 0; i < n; i++) {
        const double gc = fmin((double)g[i], clamp), uc = fmin(fmax((double)u[i], -clamp), clamp);
        const double ref = gc / (1.0 + exp(-gc)) * uc * weight;
        m = fmax(m, fabs(ref)); eh = fmax(eh, fabs(got[0][i] - ref)); et = fmax(et, fabs(got[1][i] - ref));
        const double refa = (double)g[i] + u[i];
        ma = fmax(ma, fabs(refa)); ea = fmax(ea, fabs(gota[1][i] - refa));
    }
    const int pass = et / m < 1e-5 && eh / m < 1e-5 && ea / ma < 1e-6;
    printf("swiglu/add       n=%-7u swiglu: hand|ref %.1e tl|ref %.1e  add tl|ref %.1e  %s\n", n, eh / m, et / m, ea / ma, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ELEMENTWISE", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_swiglu_tensor(to, tg, tu, n, clamp, weight));
                    CHECK(ds4_gpu_add_tensor(ta, tg, tu, n));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("ELEMENTWISE", 0));
        }
        printf("swiglu/add       n=%-7u bench x%d (both): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", n, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tu); ds4_gpu_tensor_free(to); ds4_gpu_tensor_free(ta);
    free(g); free(u); free(got[0]); free(got[1]); free(gota[0]); free(gota[1]);
    return 1;
}

/* Utility family through the public entries: f32->f16 copy, HC repeat,
 * slot-row pack (strided 3-D copy), raw KV store (f16 round trip + set_rows)
 * and the token embedding gather (get_rows f16 + repeat). Each is run with the
 * hand-written and the TileLang kernels; both must match the CPU reference
 * exactly (these are copies). */
static int check_utility(uint32_t n_embd, uint32_t n_tokens, int bench_iters) {
    const uint32_t n_hc = 4, n_vocab = 300, head_dim = 512, raw_cap = 64, n_slots = 3, slot_cap = 9, slot_rows = 5;
    const size_t map_bytes = ((size_t)n_vocab * n_embd * 2 + 4096 + getpagesize()) / getpagesize() * getpagesize();
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    memset(map, 0, map_bytes);
    const uint64_t woff = 4096;
    uint16_t *emb = (uint16_t *)(map + woff);
    for (size_t i = 0; i < (size_t)n_vocab * n_embd; i++) emb[i] = f32_to_f16_bits(frand());
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    const size_t n_flat = (size_t)n_tokens * n_embd;
    float *x = malloc(n_flat * sizeof(float));
    int32_t *tok = malloc(n_tokens * sizeof(int32_t));
    float *slots = malloc((size_t)n_slots * slot_cap * n_embd * sizeof(float));
    float *kvrow = malloc(head_dim * sizeof(float));
    CHECK(x && tok && slots && kvrow);
    for (size_t i = 0; i < n_flat; i++) x[i] = frand() * 3.0f;
    for (uint32_t t = 0; t < n_tokens; t++) tok[t] = (int32_t)((t * 7919u + 13u) % n_vocab);
    for (size_t i = 0; i < (size_t)n_slots * slot_cap * n_embd; i++) slots[i] = frand();
    for (uint32_t i = 0; i < head_dim; i++) kvrow[i] = frand() * 5.0f;

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n_flat * sizeof(float));
    ds4_gpu_tensor *t16 = ds4_gpu_tensor_alloc_managed(n_flat * sizeof(uint16_t) + 64);
    ds4_gpu_tensor *thc = ds4_gpu_tensor_alloc_managed(n_flat * n_hc * sizeof(float));
    ds4_gpu_tensor *ttok = ds4_gpu_tensor_alloc_managed(n_tokens * sizeof(int32_t));
    ds4_gpu_tensor *temb = ds4_gpu_tensor_alloc_managed(n_flat * n_hc * sizeof(float));
    ds4_gpu_tensor *tslots = ds4_gpu_tensor_alloc_managed((size_t)n_slots * slot_cap * n_embd * sizeof(float));
    ds4_gpu_tensor *tpack = ds4_gpu_tensor_alloc_managed((size_t)n_slots * slot_rows * n_embd * sizeof(float));
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc_managed(head_dim * sizeof(float));
    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc_managed((size_t)raw_cap * head_dim * sizeof(float));
    CHECK(tx && t16 && thc && ttok && temb && tslots && tpack && tkv && traw);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, n_flat * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(ttok, 0, tok, n_tokens * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(tslots, 0, slots, (size_t)n_slots * slot_cap * n_embd * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tkv, 0, kvrow, head_dim * sizeof(float)));

    uint16_t *g16[2]; float *ghc[2], *gemb[2], *gpack[2], *graw[2];
    for (int w = 0; w < 2; w++) {
        g16[w] = malloc(n_flat * sizeof(uint16_t)); ghc[w] = malloc(n_flat * n_hc * sizeof(float));
        gemb[w] = malloc(n_flat * n_hc * sizeof(float)); gpack[w] = malloc((size_t)n_slots * slot_rows * n_embd * sizeof(float));
        graw[w] = malloc((size_t)raw_cap * head_dim * sizeof(float));
        CHECK(g16[w] && ghc[w] && gemb[w] && gpack[w] && graw[w]);
    }
    for (int which = 0; which < 2; which++) {
        float *z = calloc((size_t)raw_cap * head_dim, sizeof(float));
        CHECK(z && ds4_gpu_tensor_write(traw, 0, z, (size_t)raw_cap * head_dim * sizeof(float)));
        free(z);
        CHECK(ds4_gpu_tl_force("UTILITY", which));
        CHECK(ds4_gpu_tensor_copy_f32_to_f16(t16, 32, tx, 0, n_flat));            /* 32-byte dst offset */
        CHECK(ds4_gpu_repeat_hc_rows_tensor(thc, tx, n_tokens, n_embd, n_hc));
        CHECK(ds4_gpu_embed_tokens_hc_tensor(temb, ttok, map, map_bytes, woff, n_vocab, n_tokens, n_embd, n_hc));
        CHECK(ds4_gpu_pack_slot_rows_f32_tensor(tpack, tslots, slot_rows, n_embd, n_slots, slot_cap));
        CHECK(ds4_gpu_store_raw_kv_tensor(traw, tkv, raw_cap, 17, head_dim));
        CHECK(ds4_gpu_tl_force("UTILITY", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(t16, 32, g16[which], n_flat * sizeof(uint16_t)));
        CHECK(ds4_gpu_tensor_read(thc, 0, ghc[which], n_flat * n_hc * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(temb, 0, gemb[which], n_flat * n_hc * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(tpack, 0, gpack[which], (size_t)n_slots * slot_rows * n_embd * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(traw, 0, graw[which], (size_t)raw_cap * head_dim * sizeof(float)));
    }
    size_t bad16 = 0, badhc = 0, bademb = 0, badpack = 0, badraw = 0;
    for (size_t i = 0; i < n_flat; i++) if (g16[1][i] != f32_to_f16_bits(x[i])) bad16++;
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t h = 0; h < n_hc; h++)
            for (uint32_t d = 0; d < n_embd; d++) {
                const size_t o = ((size_t)t * n_hc + h) * n_embd + d;
                if (ghc[1][o] != x[(size_t)t * n_embd + d]) badhc++;
                if (gemb[1][o] != f16_bits_to_f32(emb[(size_t)tok[t] * n_embd + d])) bademb++;
            }
    for (uint32_t r = 0; r < slot_rows; r++)       /* out[r][slot][:] = slots[slot][r][:] */
        for (uint32_t sl = 0; sl < n_slots; sl++)
            for (uint32_t d = 0; d < n_embd; d++)
                if (gpack[1][((size_t)r * n_slots + sl) * n_embd + d] != slots[((size_t)sl * slot_cap + r) * n_embd + d]) badpack++;
    for (uint32_t r = 0; r < raw_cap; r++)
        for (uint32_t d = 0; d < head_dim; d++) {
            const float want = r == 17 ? f16_bits_to_f32(f32_to_f16_bits(kvrow[d])) : 0.0f;
            if (graw[1][(size_t)r * head_dim + d] != want) badraw++;
        }
    /* hand-written vs TileLang must also agree bit for bit */
    size_t diff = 0;
    diff += memcmp(g16[0], g16[1], n_flat * sizeof(uint16_t)) != 0;
    diff += memcmp(ghc[0], ghc[1], n_flat * n_hc * sizeof(float)) != 0;
    diff += memcmp(gemb[0], gemb[1], n_flat * n_hc * sizeof(float)) != 0;
    diff += memcmp(gpack[0], gpack[1], (size_t)n_slots * slot_rows * n_embd * sizeof(float)) != 0;
    diff += memcmp(graw[0], graw[1], (size_t)raw_cap * head_dim * sizeof(float)) != 0;
    const int pass = !bad16 && !badhc && !bademb && !badpack && !badraw && !diff;
    printf("utility          n_embd=%-5u tok=%-4u cpy_f16 %zu repeat %zu embed %zu pack %zu raw_kv %zu bad, hand!=tl %zu  %s\n",
           n_embd, n_tokens, bad16, badhc, bademb, badpack, badraw, diff, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("UTILITY", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_tensor_copy_f32_to_f16(t16, 32, tx, 0, n_flat));
                    CHECK(ds4_gpu_repeat_hc_rows_tensor(thc, tx, n_tokens, n_embd, n_hc));
                    CHECK(ds4_gpu_embed_tokens_hc_tensor(temb, ttok, map, map_bytes, woff, n_vocab, n_tokens, n_embd, n_hc));
                    CHECK(ds4_gpu_pack_slot_rows_f32_tensor(tpack, tslots, slot_rows, n_embd, n_slots, slot_cap));
                    CHECK(ds4_gpu_store_raw_kv_tensor(traw, tkv, raw_cap, 17, head_dim));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("UTILITY", 0));
        }
        printf("utility          n_embd=%-5u tok=%-4u bench x%d (5 ops): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", n_embd, n_tokens, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(t16); ds4_gpu_tensor_free(thc); ds4_gpu_tensor_free(ttok); ds4_gpu_tensor_free(temb);
    ds4_gpu_tensor_free(tslots); ds4_gpu_tensor_free(tpack); ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(traw);
    for (int w = 0; w < 2; w++) { free(g16[w]); free(ghc[w]); free(gemb[w]); free(gpack[w]); free(graw[w]); }
    free(x); free(tok); free(slots); free(kvrow); free(map);
    return 1;
}

/* CPU E4M3FN round-to-nearest-even on |y| <= 448 (double in, exact grid). */
static double ref_e4m3fn(double y) {
    const double ay = fabs(y);
    double q;
    if (ay < 0.015625) {            /* subnormal grid 2^-9 */
        q = nearbyint(ay * 512.0) / 512.0;
    } else {
        int e; (void)frexp(ay, &e);   /* ay = m * 2^e, m in [0.5, 1) */
        const double step = ldexp(1.0, e - 4);    /* 3 mantissa bits below the leading 1 at 2^(e-1) */
        q = nearbyint(ay / step) * step;          /* nearbyint: ties to even */
    }
    return y < 0 ? -q : q;
}

/* DSv4 fused small kernels: q/kv rms norm (1024 + 512 rows per token), the
 * FP8 E4M3FN KV fake-quant, and the HC split (sinkhorn) + collapse + norm.
 * TileLang vs hand-written vs a double CPU reference. */
static int check_dsv4_small(uint32_t rows, int bench_iters) {
    const uint32_t q_n = 1024, kv_n = 512, head_dim = 512, n_rot = 64, n_embd = 4096, n_hc = 4, iters = 20;
    const float eps = 1e-6f;
    const size_t map_bytes = (size_t)getpagesize() * 8;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    memset(map, 0, map_bytes);
    const uint64_t qw_off = 0, kvw_off = q_n * 4, scale_off = kvw_off + kv_n * 4, base_off = scale_off + 32, nw_off = base_off + 24 * 4 + 32;
    float *qw = (float *)(map + qw_off), *kvw = (float *)(map + kvw_off), *scale = (float *)(map + scale_off), *base = (float *)(map + base_off), *nw = (float *)(map + nw_off);
    for (uint32_t i = 0; i < q_n; i++) qw[i] = 1.0f + 0.25f * frand();
    for (uint32_t i = 0; i < kv_n; i++) kvw[i] = 1.0f + 0.25f * frand();
    for (uint32_t i = 0; i < n_embd; i++) nw[i] = 1.0f + 0.25f * frand();
    scale[0] = 0.7f; scale[1] = 0.9f; scale[2] = 1.3f;
    for (int i = 0; i < 24; i++) base[i] = 0.5f * frand();
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    float *q = malloc((size_t)rows * q_n * 4), *kv = malloc((size_t)rows * kv_n * 4), *fp = malloc((size_t)rows * head_dim * 4);
    float *mix = malloc((size_t)rows * 24 * 4), *res = malloc((size_t)rows * n_hc * n_embd * 4);
    CHECK(q && kv && fp && mix && res);
    for (size_t i = 0; i < (size_t)rows * q_n; i++) q[i] = frand() * 3.0f;
    for (size_t i = 0; i < (size_t)rows * kv_n; i++) kv[i] = frand() * 3.0f;
    for (size_t i = 0; i < (size_t)rows * head_dim; i++) fp[i] = frand() * (i % 7 == 0 ? 40.0f : 2.0f);
    for (size_t i = 0; i < (size_t)rows * 24; i++) mix[i] = frand() * 2.0f;
    for (size_t i = 0; i < (size_t)rows * n_hc * n_embd; i++) res[i] = frand();

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc_managed((size_t)rows * q_n * 4), *tqo = ds4_gpu_tensor_alloc_managed((size_t)rows * q_n * 4);
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc_managed((size_t)rows * kv_n * 4), *tkvo = ds4_gpu_tensor_alloc_managed((size_t)rows * kv_n * 4);
    ds4_gpu_tensor *tfp = ds4_gpu_tensor_alloc_managed((size_t)rows * head_dim * 4);
    ds4_gpu_tensor *tmix = ds4_gpu_tensor_alloc_managed((size_t)rows * 24 * 4), *tres = ds4_gpu_tensor_alloc_managed((size_t)rows * n_hc * n_embd * 4);
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed((size_t)rows * n_embd * 4), *tnorm = ds4_gpu_tensor_alloc_managed((size_t)rows * n_embd * 4);
    ds4_gpu_tensor *tsplit = ds4_gpu_tensor_alloc_managed((size_t)rows * 24 * 4);
    CHECK(tq && tqo && tkv && tkvo && tfp && tmix && tres && tout && tnorm && tsplit);
    CHECK(ds4_gpu_tensor_write(tq, 0, q, (size_t)rows * q_n * 4) && ds4_gpu_tensor_write(tkv, 0, kv, (size_t)rows * kv_n * 4));
    CHECK(ds4_gpu_tensor_write(tmix, 0, mix, (size_t)rows * 24 * 4) && ds4_gpu_tensor_write(tres, 0, res, (size_t)rows * n_hc * n_embd * 4));

    float *gq[2], *gkv[2], *gfp[2], *gout[2], *gnorm[2], *gsplit[2];
    for (int w = 0; w < 2; w++) {
        gq[w] = malloc((size_t)rows * q_n * 4); gkv[w] = malloc((size_t)rows * kv_n * 4); gfp[w] = malloc((size_t)rows * head_dim * 4);
        gout[w] = malloc((size_t)rows * n_embd * 4); gnorm[w] = malloc((size_t)rows * n_embd * 4); gsplit[w] = malloc((size_t)rows * 24 * 4);
        CHECK(gq[w] && gkv[w] && gfp[w] && gout[w] && gnorm[w] && gsplit[w]);
    }
    for (int which = 0; which < 2; which++) {
        CHECK(ds4_gpu_tensor_write(tfp, 0, fp, (size_t)rows * head_dim * 4));
        CHECK(ds4_gpu_tl_force("RMS_NORM", which) && ds4_gpu_tl_force("KV", which) && ds4_gpu_tl_force("HC", which));
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(tqo, tq, map, map_bytes, qw_off, q_n, tkvo, tkv, kvw_off, kv_n, rows, eps));
        CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tfp, rows, head_dim, n_rot));
        CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(tout, tnorm, tsplit, tmix, tres, map, map_bytes, scale_off, base_off, nw_off, n_embd, n_hc, iters, eps, eps));
        CHECK(ds4_gpu_tl_force("RMS_NORM", 0) && ds4_gpu_tl_force("KV", 0) && ds4_gpu_tl_force("HC", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tqo, 0, gq[which], (size_t)rows * q_n * 4) && ds4_gpu_tensor_read(tkvo, 0, gkv[which], (size_t)rows * kv_n * 4));
        CHECK(ds4_gpu_tensor_read(tfp, 0, gfp[which], (size_t)rows * head_dim * 4));
        CHECK(ds4_gpu_tensor_read(tout, 0, gout[which], (size_t)rows * n_embd * 4) && ds4_gpu_tensor_read(tnorm, 0, gnorm[which], (size_t)rows * n_embd * 4));
        CHECK(ds4_gpu_tensor_read(tsplit, 0, gsplit[which], (size_t)rows * 24 * 4));
    }
    /* references */
    double e_q[2] = {0, 0}, e_kv[2] = {0, 0}, e_out[2] = {0, 0}, e_norm[2] = {0, 0}, e_split[2] = {0, 0};
    size_t bad_fp[2] = {0, 0};
    for (uint32_t r = 0; r < rows; r++) {
        double ss = 0; for (uint32_t i = 0; i < q_n; i++) ss += (double)q[r * q_n + i] * q[r * q_n + i];
        double sc = 1.0 / sqrt(ss / q_n + eps);
        for (uint32_t i = 0; i < q_n; i++) for (int w = 0; w < 2; w++) e_q[w] = fmax(e_q[w], fabs(gq[w][r * q_n + i] - q[r * q_n + i] * sc * qw[i]));
        ss = 0; for (uint32_t i = 0; i < kv_n; i++) ss += (double)kv[r * kv_n + i] * kv[r * kv_n + i];
        sc = 1.0 / sqrt(ss / kv_n + eps);
        for (uint32_t i = 0; i < kv_n; i++) for (int w = 0; w < 2; w++) e_kv[w] = fmax(e_kv[w], fabs(gkv[w][r * kv_n + i] - kv[r * kv_n + i] * sc * kvw[i]));
        /* fp8: per 64-block power-of-two scale, RNE; compare bit-exact (grid values are exact floats) */
        for (uint32_t b = 0; b < (head_dim - n_rot) / 64; b++) {
            float amax = 0; for (int i = 0; i < 64; i++) amax = fmaxf(amax, fabsf(fp[r * head_dim + b * 64 + i]));
            amax = fmaxf(amax, 1e-4f);
            const float s2 = exp2f(ceilf(log2f(amax / 448.0f)));
            for (int i = 0; i < 64; i++) {
                const float y = fminf(fmaxf(fp[r * head_dim + b * 64 + i] / s2, -448.0f), 448.0f);
                const float want = (float)ref_e4m3fn((double)y) * s2;
                for (int w = 0; w < 2; w++) if (gfp[w][r * head_dim + b * 64 + i] != want) bad_fp[w]++;
            }
        }
        for (uint32_t i = head_dim - n_rot; i < head_dim; i++) for (int w = 0; w < 2; w++) if (gfp[w][r * head_dim + i] != fp[r * head_dim + i]) bad_fp[w]++;
        /* hc split */
        double sp[24], R[16];
        for (int h = 0; h < 4; h++) sp[h] = 1.0 / (1.0 + exp(-((double)mix[r * 24 + h] * scale[0] + base[h]))) + eps;
        for (int h = 0; h < 4; h++) sp[4 + h] = 2.0 / (1.0 + exp(-((double)mix[r * 24 + 4 + h] * scale[1] + base[4 + h])));
        for (int k = 0; k < 16; k++) R[k] = (double)mix[r * 24 + 8 + k] * scale[2] + base[8 + k];
        for (int i = 0; i < 4; i++) {
            double m = R[i * 4]; for (int j = 1; j < 4; j++) m = fmax(m, R[i * 4 + j]);
            double sum = 0; for (int j = 0; j < 4; j++) { R[i * 4 + j] = exp(R[i * 4 + j] - m); sum += R[i * 4 + j]; }
            for (int j = 0; j < 4; j++) R[i * 4 + j] = R[i * 4 + j] / sum + eps;
        }
        for (int j = 0; j < 4; j++) { double c = R[j] + R[4 + j] + R[8 + j] + R[12 + j] + eps; for (int i = 0; i < 4; i++) R[i * 4 + j] /= c; }
        for (uint32_t it = 1; it < iters; it++) {
            for (int i = 0; i < 4; i++) { double rs = R[i * 4] + R[i * 4 + 1] + R[i * 4 + 2] + R[i * 4 + 3] + eps; for (int j = 0; j < 4; j++) R[i * 4 + j] /= rs; }
            for (int j = 0; j < 4; j++) { double c = R[j] + R[4 + j] + R[8 + j] + R[12 + j] + eps; for (int i = 0; i < 4; i++) R[i * 4 + j] /= c; }
        }
        for (int k = 0; k < 16; k++) sp[8 + k] = R[k];
        for (int k = 0; k < 24; k++) for (int w = 0; w < 2; w++) e_split[w] = fmax(e_split[w], fabs(gsplit[w][r * 24 + k] - sp[k]));
        ss = 0;
        double *v = malloc(n_embd * sizeof(double));
        for (uint32_t d = 0; d < n_embd; d++) {
            v[d] = 0; for (int h = 0; h < 4; h++) v[d] += (double)res[((size_t)r * 4 + h) * n_embd + d] * sp[h];
            ss += v[d] * v[d];
        }
        const double ns = 1.0 / sqrt(ss / n_embd + eps);
        for (uint32_t d = 0; d < n_embd; d++) for (int w = 0; w < 2; w++) {
            e_out[w] = fmax(e_out[w], fabs(gout[w][(size_t)r * n_embd + d] - v[d]));
            e_norm[w] = fmax(e_norm[w], fabs(gnorm[w][(size_t)r * n_embd + d] - v[d] * ns * nw[d]));
        }
        free(v);
    }
    const double tol = 2e-5;
    const int pass = e_q[1] < tol && e_kv[1] < tol && bad_fp[1] == 0 && e_split[1] < tol && e_out[1] < tol && e_norm[1] < 1e-4 &&
                     e_q[0] < tol && e_kv[0] < tol && bad_fp[0] == 0 && e_split[0] < tol && e_out[0] < tol && e_norm[0] < 1e-4;
    printf("dsv4_small       rows=%-4u qkv_norm hand %.1e tl %.1e | fp8 bad hand %zu tl %zu | split hand %.1e tl %.1e | sum %.1e norm %.1e  %s\n",
           rows, fmax(e_q[0], e_kv[0]), fmax(e_q[1], e_kv[1]), bad_fp[0], bad_fp[1], e_split[0], e_split[1], e_out[1], e_norm[1], pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("RMS_NORM", which) && ds4_gpu_tl_force("KV", which) && ds4_gpu_tl_force("HC", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(tqo, tq, map, map_bytes, qw_off, q_n, tkvo, tkv, kvw_off, kv_n, rows, eps));
                    CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(tfp, rows, head_dim, n_rot));
                    CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(tout, tnorm, tsplit, tmix, tres, map, map_bytes, scale_off, base_off, nw_off, n_embd, n_hc, iters, eps, eps));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("RMS_NORM", 0) && ds4_gpu_tl_force("KV", 0) && ds4_gpu_tl_force("HC", 0));
        }
        printf("dsv4_small       rows=%-4u bench x%d (3 ops): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", rows, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tqo); ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tkvo); ds4_gpu_tensor_free(tfp);
    ds4_gpu_tensor_free(tmix); ds4_gpu_tensor_free(tres); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tnorm); ds4_gpu_tensor_free(tsplit);
    for (int w = 0; w < 2; w++) { free(gq[w]); free(gkv[w]); free(gfp[w]); free(gout[w]); free(gnorm[w]); free(gsplit[w]); }
    free(q); free(kv); free(fp); free(mix); free(res); free(map);
    return 1;
}

/* Router family: decode project+select (f16 GEMV + finalize), single-token
 * select in bias and hash modes, and the batch top-k chain. Selected ids
 * must match the hand-written kernels exactly (same bitonic network);
 * probs / weights to float rounding. */
static int check_router(int bench_iters) {
    const uint32_t n_embd = 4096, n_expert = 256, n_used = 6, n_tokens = 37, hash_rows = 1000;
    const size_t w_bytes = (size_t)n_expert * n_embd * 2;
    const size_t map_bytes = (w_bytes + 4096 * 4 + getpagesize()) / getpagesize() * getpagesize();
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    memset(map, 0, map_bytes);
    const uint64_t w_off = 0, bias_off = w_bytes, hash_off = bias_off + n_expert * 4;
    uint16_t *w = (uint16_t *)(map + w_off); float *bias = (float *)(map + bias_off); int32_t *hash = (int32_t *)(map + hash_off);
    for (size_t i = 0; i < (size_t)n_expert * n_embd; i++) w[i] = f32_to_f16_bits(frand() * 0.05f);
    for (uint32_t e = 0; e < n_expert; e++) bias[e] = frand() * 0.5f;
    CHECK(hash_off + (size_t)hash_rows * n_used * 4 <= map_bytes);
    for (uint32_t r = 0; r < hash_rows; r++) for (uint32_t k = 0; k < n_used; k++) hash[r * n_used + k] = (int32_t)((r * 31 + k * 17) % n_expert);
    CHECK(ds4_gpu_set_model_map(map, map_bytes));

    float *x = malloc(n_embd * 4), *logits = malloc((size_t)n_tokens * n_expert * 4);
    int32_t *tok = malloc(n_tokens * 4);
    CHECK(x && logits && tok);
    for (uint32_t i = 0; i < n_embd; i++) x[i] = frand();
    for (size_t i = 0; i < (size_t)n_tokens * n_expert; i++) logits[i] = frand() * 4.0f;
    /* a few exact ties in the batch keys, to exercise the network order */
    for (uint32_t t = 0; t < n_tokens; t += 5) { logits[t * n_expert + 7] = 3.5f; logits[t * n_expert + 200] = 3.5f; logits[t * n_expert + 13] = 3.5f; }
    for (uint32_t t = 0; t < n_tokens; t++) tok[t] = (int32_t)((t * 97) % hash_rows);

    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n_embd * 4);
    ds4_gpu_tensor *tlog = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_expert * 4);
    ds4_gpu_tensor *tprobs = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_expert * 4);
    ds4_gpu_tensor *tsel = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_used * 4);
    ds4_gpu_tensor *twt = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_used * 4);
    ds4_gpu_tensor *ttok = ds4_gpu_tensor_alloc_managed(n_tokens * 4);
    CHECK(tx && tlog && tprobs && tsel && twt && ttok);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, n_embd * 4) && ds4_gpu_tensor_write(ttok, 0, tok, n_tokens * 4));

    enum { M_FUSED = 0, M_ONE_BIAS, M_ONE_HASH, M_BATCH, M_COUNT };
    float *glog[2], *gprobs[2][M_COUNT], *gwt[2][M_COUNT]; int32_t *gsel[2][M_COUNT];
    for (int wch = 0; wch < 2; wch++) {
        glog[wch] = malloc(n_expert * 4);
        for (int m = 0; m < M_COUNT; m++) { gprobs[wch][m] = malloc((size_t)n_tokens * n_expert * 4); gwt[wch][m] = malloc((size_t)n_tokens * n_used * 4); gsel[wch][m] = malloc((size_t)n_tokens * n_used * 4); CHECK(gprobs[wch][m] && gwt[wch][m] && gsel[wch][m]); }
    }
    for (int which = 0; which < 2; which++) {
        CHECK(ds4_gpu_tl_force("ROUTER", which));
        /* fused decode: logits from the f16 GEMV, bias selection */
        CHECK(ds4_gpu_router_project_select_fused_tensor(tlog, tprobs, tsel, twt, map, map_bytes, w_off, bias_off, true, tx) == 1);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tlog, 0, glog[which], n_expert * 4));
        CHECK(ds4_gpu_tensor_read(tprobs, 0, gprobs[which][M_FUSED], n_expert * 4) && ds4_gpu_tensor_read(tsel, 0, gsel[which][M_FUSED], n_used * 4) && ds4_gpu_tensor_read(twt, 0, gwt[which][M_FUSED], n_used * 4));
        /* single token from given logits: bias mode, then hash mode */
        CHECK(ds4_gpu_tensor_write(tlog, 0, logits, n_expert * 4));
        CHECK(ds4_gpu_router_select_tensor(tsel, twt, tprobs, map, map_bytes, bias_off, hash_off, hash_rows, 5, n_expert, n_used, 1.5f, 0, 0, true, false, tlog));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tprobs, 0, gprobs[which][M_ONE_BIAS], n_expert * 4) && ds4_gpu_tensor_read(tsel, 0, gsel[which][M_ONE_BIAS], n_used * 4) && ds4_gpu_tensor_read(twt, 0, gwt[which][M_ONE_BIAS], n_used * 4));
        CHECK(ds4_gpu_router_select_tensor(tsel, twt, tprobs, map, map_bytes, bias_off, hash_off, hash_rows, 123, n_expert, n_used, 1.5f, 0, 0, false, true, tlog));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tprobs, 0, gprobs[which][M_ONE_HASH], n_expert * 4) && ds4_gpu_tensor_read(tsel, 0, gsel[which][M_ONE_HASH], n_used * 4) && ds4_gpu_tensor_read(twt, 0, gwt[which][M_ONE_HASH], n_used * 4));
        /* batch */
        CHECK(ds4_gpu_tensor_write(tlog, 0, logits, (size_t)n_tokens * n_expert * 4));
        CHECK(ds4_gpu_router_select_batch_tensor(tsel, twt, tprobs, map, map_bytes, bias_off, hash_off, hash_rows, 0, 0, true, false, tlog, ttok, n_expert, n_used, 1.5f, n_tokens));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tprobs, 0, gprobs[which][M_BATCH], (size_t)n_tokens * n_expert * 4) && ds4_gpu_tensor_read(tsel, 0, gsel[which][M_BATCH], (size_t)n_tokens * n_used * 4) && ds4_gpu_tensor_read(twt, 0, gwt[which][M_BATCH], (size_t)n_tokens * n_used * 4));
        CHECK(ds4_gpu_tl_force("ROUTER", 0));
    }
    /* CPU reference for the fused logits */
    double e_log = 0, m_log = 0;
    for (uint32_t e = 0; e < n_expert; e++) {
        double acc = 0; for (uint32_t k = 0; k < n_embd; k++) acc += (double)f16_bits_to_f32(w[(size_t)e * n_embd + k]) * x[k];
        m_log = fmax(m_log, fabs(acc)); e_log = fmax(e_log, fabs(glog[1][e] - acc));
    }
    const size_t rows_of[M_COUNT] = { 1, 1, 1, n_tokens };
    size_t sel_bad[M_COUNT] = {0}; double e_p[M_COUNT] = {0}, e_w[M_COUNT] = {0};
    int sum_ok = 1;
    for (int m = 0; m < M_COUNT; m++) {
        for (size_t i = 0; i < rows_of[m] * n_used; i++) if (gsel[0][m][i] != gsel[1][m][i]) sel_bad[m]++;
        for (size_t i = 0; i < rows_of[m] * n_expert; i++) e_p[m] = fmax(e_p[m], fabs(gprobs[0][m][i] - gprobs[1][m][i]));
        for (size_t i = 0; i < rows_of[m] * n_used; i++) e_w[m] = fmax(e_w[m], fabs(gwt[0][m][i] - gwt[1][m][i]));
        for (size_t r = 0; r < rows_of[m]; r++) { double sw = 0; for (uint32_t k = 0; k < n_used; k++) sw += gwt[1][m][r * n_used + k]; if (fabs(sw - 1.5) > 1e-5) sum_ok = 0; }
    }
    /* the fused path's selection must differ from the bias-mode select only through the logits source */
    const int pass = e_log / m_log < 1e-5 && !sel_bad[0] && !sel_bad[1] && !sel_bad[2] && !sel_bad[3] &&
                     e_p[0] < 1e-5 && e_p[1] < 1e-6 && e_p[2] < 1e-6 && e_p[3] < 1e-6 &&
                     e_w[0] < 1e-5 && e_w[1] < 1e-6 && e_w[2] < 1e-6 && e_w[3] < 1e-6 && sum_ok;
    printf("router           logits tl|ref %.1e | sel mismatches fused %zu one %zu hash %zu batch(%u) %zu | probs %.1e %.1e %.1e %.1e | weights %.1e %.1e %.1e %.1e sum1.5 %d  %s\n",
           e_log / m_log, sel_bad[0], sel_bad[1], sel_bad[2], n_tokens, sel_bad[3], e_p[0], e_p[1], e_p[2], e_p[3], e_w[0], e_w[1], e_w[2], e_w[3], sum_ok, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ROUTER", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_router_project_select_fused_tensor(tlog, tprobs, tsel, twt, map, map_bytes, w_off, bias_off, true, tx) == 1);
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("ROUTER", 0));
        }
        printf("router           fused project+select bench x%d: hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tlog); ds4_gpu_tensor_free(tprobs); ds4_gpu_tensor_free(tsel); ds4_gpu_tensor_free(twt); ds4_gpu_tensor_free(ttok);
    for (int wch = 0; wch < 2; wch++) { free(glog[wch]); for (int m = 0; m < M_COUNT; m++) { free(gprobs[wch][m]); free(gwt[wch][m]); free(gsel[wch][m]); } }
    free(x); free(logits); free(tok); free(map);
    return 1;
}

/* Compressor / indexer small kernels: Hadamard + E2M1 fake quant and the
 * ascending int32 row sort, TileLang vs hand-written (bit-exact). The
 * ratio-4 pool and the row finalize are pinned by ds4_test --metal-kernels
 * (compressor pair state-store exactness) with DS4_TL_KV=1. */
static int check_compressor_small(uint32_t rows, int bench_iters) {
    const uint32_t hd = 128, top_k = 512;
    float *x = malloc((size_t)rows * hd * 4);
    int32_t *ids = malloc((size_t)rows * top_k * 4);
    CHECK(x && ids);
    for (size_t i = 0; i < (size_t)rows * hd; i++) x[i] = frand() * (i % 11 == 0 ? 30.0f : 1.5f);
    for (uint32_t r = 0; r < rows; r++) {          /* a permutation of 0..4095 restricted to top_k unique ids */
        for (uint32_t k = 0; k < top_k; k++) ids[r * top_k + k] = (int32_t)((k * 2654435761u + r * 97u) % 4096u);
        for (uint32_t k = 0; k < top_k; k++) for (uint32_t m = 0; m < k; m++) if (ids[r * top_k + m] == ids[r * top_k + k]) ids[r * top_k + k] = (ids[r * top_k + k] + 1) % 4096;
    }
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed((size_t)rows * hd * 4);
    ds4_gpu_tensor *tids = ds4_gpu_tensor_alloc_managed((size_t)rows * top_k * 4);
    ds4_gpu_tensor *tsorted = ds4_gpu_tensor_alloc_managed((size_t)rows * top_k * 4);
    CHECK(tx && tids && tsorted && ds4_gpu_tensor_write(tids, 0, ids, (size_t)rows * top_k * 4));
    float *gx[2]; int32_t *gs[2];
    for (int w = 0; w < 2; w++) { gx[w] = malloc((size_t)rows * hd * 4); gs[w] = malloc((size_t)rows * top_k * 4); CHECK(gx[w] && gs[w]); }
    for (int which = 0; which < 2; which++) {
        CHECK(ds4_gpu_tensor_write(tx, 0, x, (size_t)rows * hd * 4));
        CHECK(ds4_gpu_tl_force("KV", which));
        CHECK(ds4_gpu_dsv4_indexer_qat_tensor(tx, rows, hd));
        CHECK(ds4_gpu_sort_i32_rows_asc_tensor(tsorted, tids, top_k, rows));
        CHECK(ds4_gpu_tl_force("KV", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tx, 0, gx[which], (size_t)rows * hd * 4) && ds4_gpu_tensor_read(tsorted, 0, gs[which], (size_t)rows * top_k * 4));
    }
    size_t bad_h = 0, bad_s = 0, bad_sorted = 0;
    for (size_t i = 0; i < (size_t)rows * hd; i++) if (gx[0][i] != gx[1][i]) bad_h++;
    for (size_t i = 0; i < (size_t)rows * top_k; i++) if (gs[0][i] != gs[1][i]) bad_s++;
    for (uint32_t r = 0; r < rows; r++) for (uint32_t k = 1; k < top_k; k++) if (gs[1][r * top_k + k] <= gs[1][r * top_k + k - 1]) bad_sorted++;
    /* the E2M1 grid: every output / its block scale must be one of {0, .5, 1, 1.5, 2, 3, 4, 6} */
    size_t bad_grid = 0;
    for (uint32_t r = 0; r < rows; r++) for (int b = 0; b < 4; b++) {
        float amax = 0; for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(gx[1][r * hd + b * 32 + i]));
        if (amax == 0) continue;
        int e; frexpf(amax, &e); const float sc = ldexpf(1.0f, e - 3);   /* amax in [4,6]*scale -> scale = 2^(e-3) */
        for (int i = 0; i < 32; i++) { const float q = fabsf(gx[1][r * hd + b * 32 + i]) / sc; if (!(q == 0 || q == 0.5f || q == 1 || q == 1.5f || q == 2 || q == 3 || q == 4 || q == 6)) bad_grid++; }
    }
    const int pass = !bad_h && !bad_s && !bad_sorted && !bad_grid;
    printf("compressor_small rows=%-4u hadamard+fp4 hand!=tl %zu (off-grid %zu) | sort_i32 hand!=tl %zu unsorted %zu  %s\n", rows, bad_h, bad_grid, bad_s, bad_sorted, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("KV", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_dsv4_indexer_qat_tensor(tx, rows, hd));
                    CHECK(ds4_gpu_sort_i32_rows_asc_tensor(tsorted, tids, top_k, rows));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("KV", 0));
        }
        printf("compressor_small rows=%-4u bench x%d (2 ops): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", rows, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tids); ds4_gpu_tensor_free(tsorted);
    for (int w = 0; w < 2; w++) { free(gx[w]); free(gs[w]); }
    free(x); free(ids);
    return 1;
}

/* Small-batch GEMV tier (2..31 tokens) for Q8_0 and f16 weights, and the
 * single-token f16 GEMV, through the public matmul entries with the family
 * forced off / on. Compared with a double reference. */
static int check_batch_gemv(int is_f16, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, int bench_iters) {
    const uint32_t blocks = in_dim / 32;
    const size_t row_bytes = is_f16 ? (size_t)in_dim * 2 : (size_t)blocks * 34;
    const size_t w_bytes = row_bytes * out_dim;
    const uint64_t woff = 32 * 5;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (w_bytes + woff + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    uint8_t *w = map + woff;
    const size_t n_x = (size_t)n_tok * in_dim, n_out = (size_t)n_tok * out_dim;
    float *x = malloc(n_x * sizeof(float));
    float *wf = malloc((size_t)out_dim * in_dim * sizeof(float));
    double *ref = malloc(n_out * sizeof(double));
    float *got[2] = { malloc(n_out * sizeof(float)), malloc(n_out * sizeof(float)) };
    CHECK(x && wf && ref && got[0] && got[1]);
    for (size_t i = 0; i < n_x; i++) x[i] = frand();
    for (uint32_t r = 0; r < out_dim; r++) {
        if (is_f16) {
            uint16_t *wr = (uint16_t *)(w + (size_t)r * row_bytes);
            for (uint32_t k = 0; k < in_dim; k++) { wr[k] = f32_to_f16_bits(frand() * 0.1f); wf[(size_t)r * in_dim + k] = f16_bits_to_f32(wr[k]); }
        } else {
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *blk = w + (size_t)r * row_bytes + (size_t)b * 34;
                const uint16_t dbits = f32_to_f16_bits(0.001f + 0.02f * (frand() + 2.0f) / 4.0f);
                memcpy(blk, &dbits, 2);
                const float d = f16_bits_to_f32(dbits);
                for (uint32_t k = 0; k < 32; k++) { const int8_t q = (int8_t)((int)(frand() * 63.0f)); blk[2 + k] = (uint8_t)q; wf[(size_t)r * in_dim + b * 32 + k] = d * q; }
            }
        }
    }
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t r = 0; r < out_dim; r++) {
            double acc = 0; const float *xr = x + (size_t)t * in_dim, *wr = wf + (size_t)r * in_dim;
            for (uint32_t k = 0; k < in_dim; k++) acc += (double)xr[k] * wr[k];
            ref[(size_t)t * out_dim + r] = acc;
        }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n_x * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed(n_out * sizeof(float));
    CHECK(tx && tout && ds4_gpu_tensor_write(tx, 0, x, n_x * sizeof(float)));
    const char *fam = is_f16 ? "F16_GEMV" : "Q8_GEMV";
    for (int which = 0; which < 2; which++) {
        memset(got[which], 0, n_out * sizeof(float));
        CHECK(ds4_gpu_tensor_write(tout, 0, got[which], n_out * sizeof(float)));
        CHECK(ds4_gpu_tl_force(fam, which) && ds4_gpu_tl_force("GEMM", which));
        CHECK(is_f16 ? ds4_gpu_matmul_f16_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok)
                     : ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok));
        CHECK(ds4_gpu_tl_force(fam, 0) && ds4_gpu_tl_force("GEMM", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout, 0, got[which], n_out * sizeof(float)));
    }
    double rmax = 0, eh = 0, et = 0;
    for (size_t i = 0; i < n_out; i++) { rmax = fmax(rmax, fabs(ref[i])); eh = fmax(eh, fabs(got[0][i] - ref[i])); et = fmax(et, fabs(got[1][i] - ref[i])); }
    /* 17..31 tokens with a wide output take the f16-activation GEMM tier (3e-4 like ds4's own GEMM);
     * f16 weights: >= 17 tokens with out_dim % 64 == 0 */
    const int gemm_tier = is_f16 ? (n_tok >= 17 && out_dim % 64 == 0) : (n_tok >= 17 && out_dim >= 4096);
    const int pass = et / rmax < (gemm_tier ? 1e-3 : 2e-5) && eh / rmax < 1e-2;
    printf("%s_batch  in=%-5u out=%-6u tok=%-3u hand|ref %.1e tl|ref %.1e  %s\n", is_f16 ? "f16" : "q8_0", in_dim, out_dim, n_tok, eh / rmax, et / rmax, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force(fam, which) && ds4_gpu_tl_force("GEMM", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++)
                    CHECK(is_f16 ? ds4_gpu_matmul_f16_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok)
                                 : ds4_gpu_matmul_q8_0_tensor(tout, map, map_bytes, woff, in_dim, out_dim, tx, n_tok));
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force(fam, 0) && ds4_gpu_tl_force("GEMM", 0));
        }
        const double gb = (double)w_bytes / 1e9;
        printf("%s_batch  in=%-5u out=%-6u tok=%-3u bench x%d: hand %8.2f us (%6.1f GB/s)  tl %8.2f us (%6.1f GB/s)  tl/hand %.2fx\n",
               is_f16 ? "f16" : "q8_0", in_dim, out_dim, n_tok, bench_iters, t[0], gb / (t[0] * 1e-6), t[1], gb / (t[1] * 1e-6), t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tout);
    free(x); free(wf); free(ref); free(got[0]); free(got[1]); free(map);
    return 1;
}

extern void ds4_gpu_set_decode_attn_rope_fuse(uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
                                             float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
extern int ds4_gpu_decode_attn_rope_fuse_used(void);

/* Decode attention: 64 heads x 512 over [n_raw raw ring rows (f32)] ++
 * [n_comp compressed rows (f16)] with sinks, optionally with ds4's deferred
 * inverse RoPE. TileLang (stage + partials + merge) vs hand-written vs a
 * double reference that mirrors the GPU rounding (Q and raw rows in f16). */
static int check_decode_attention(uint32_t n_raw, uint32_t n_comp, int with_rope, int bench_iters) {
    const uint32_t n_head = 64, hd = 512, raw_cap = 128, raw_start = (n_raw * 37u) % raw_cap, n_keys = n_raw + n_comp, pos = 1000;
    const size_t map_bytes = (size_t)getpagesize() * 2;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    memset(map, 0, map_bytes);
    float *sinks = (float *)(map + 256);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = frand() * 2.0f;
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *q = malloc((size_t)n_head * hd * 4), *raw = malloc((size_t)raw_cap * hd * 4);
    uint16_t *comp = malloc((size_t)(n_comp ? n_comp : 1) * hd * 2);
    CHECK(q && raw && comp);
    for (size_t i = 0; i < (size_t)n_head * hd; i++) q[i] = frand() * 0.5f;
    for (size_t i = 0; i < (size_t)raw_cap * hd; i++) raw[i] = f16_bits_to_f32(f32_to_f16_bits(frand() * 0.5f));
    for (size_t i = 0; i < (size_t)n_comp * hd; i++) comp[i] = f32_to_f16_bits(frand() * 0.5f);
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc_managed((size_t)n_head * hd * 4);
    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc_managed((size_t)raw_cap * hd * 4);
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc_managed((size_t)(n_comp ? n_comp : 1) * hd * 2);
    ds4_gpu_tensor *theads = ds4_gpu_tensor_alloc_managed((size_t)n_head * hd * 4);
    CHECK(tq && traw && tcomp && theads);
    CHECK(ds4_gpu_tensor_write(tq, 0, q, (size_t)n_head * hd * 4) && ds4_gpu_tensor_write(traw, 0, raw, (size_t)raw_cap * hd * 4));
    if (n_comp) CHECK(ds4_gpu_tensor_write(tcomp, 0, comp, (size_t)n_comp * hd * 2));
    const float freq_base = 10000.0f, freq_scale = 1.0f / 16.0f, ext_factor = 1.0f, attn_factor = 1.0f, beta_fast = 32.0f, beta_slow = 1.0f;
    const uint32_t n_ctx_orig = 65536;
    float *got[2]; int fused[2] = {0, 0};
    /* TileLang first: the hand-written path stages K/V into the same scratch
     * buffer, and a TL kernel reading stale-but-correct staging would pass. */
    for (int which = 1; which >= 0; which--) {
        got[which] = malloc((size_t)n_head * hd * 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("ATTN", which));
        if (with_rope) ds4_gpu_set_decode_attn_rope_fuse(hd, 64, pos, n_ctx_orig, true, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        CHECK(ds4_gpu_attention_decode_heads_tensor(theads, map, map_bytes, 256, tq, traw, n_raw, raw_cap, raw_start,
                                                    n_comp ? tcomp : NULL, 1, n_comp, NULL, 0, n_head, hd));
        fused[which] = with_rope ? ds4_gpu_decode_attn_rope_fuse_used() : 0;
        if (with_rope && !fused[which])   /* the raw-only hand path leaves the rotation to ds4.c */
            CHECK(ds4_gpu_rope_tail_tensor(theads, 1, n_head, hd, 64, pos, n_ctx_orig, true, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
        CHECK(ds4_gpu_tl_force("ATTN", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(theads, 0, got[which], (size_t)n_head * hd * 4));
    }
    /* reference (no rope): f16-rounded q, f32 keys as stored */
    double rmax = 0, eh = 0, et = 0, eht = 0;
    float *key = malloc((size_t)n_keys * hd * 4);
    for (uint32_t k = 0; k < n_keys; k++)
        for (uint32_t d = 0; d < hd; d++)
            key[(size_t)k * hd + d] = k < n_raw ? raw[((size_t)((raw_start + k) % raw_cap)) * hd + d] : f16_bits_to_f32(comp[(size_t)(k - n_raw) * hd + d]);
    double *sc = malloc(n_keys * sizeof(double)), *o = malloc(hd * sizeof(double));
    for (uint32_t h = 0; h < n_head && !with_rope; h++) {
        double m = sinks[h];
        for (uint32_t k = 0; k < n_keys; k++) {
            double acc = 0;
            for (uint32_t d = 0; d < hd; d++) acc += (double)f16_bits_to_f32(f32_to_f16_bits(q[(size_t)h * hd + d])) * key[(size_t)k * hd + d];
            sc[k] = acc / sqrt((double)hd); m = fmax(m, sc[k]);
        }
        double l = exp(sinks[h] - m);
        for (uint32_t d = 0; d < hd; d++) o[d] = 0;
        for (uint32_t k = 0; k < n_keys; k++) { const double p = exp(sc[k] - m); l += p; for (uint32_t d = 0; d < hd; d++) o[d] += p * key[(size_t)k * hd + d]; }
        for (uint32_t d = 0; d < hd; d++) {
            const double ref = o[d] / l;
            rmax = fmax(rmax, fabs(ref));
            eh = fmax(eh, fabs(got[0][(size_t)h * hd + d] - ref));
            et = fmax(et, fabs(got[1][(size_t)h * hd + d] - ref));
        }
    }
    double hmax = 0;
    for (size_t i = 0; i < (size_t)n_head * hd; i++) { hmax = fmax(hmax, fabs(got[0][i])); eht = fmax(eht, fabs(got[0][i] - got[1][i])); }
    const int pass = with_rope ? (eht / hmax < 2e-5) : (et / rmax < 2e-5 && eht / hmax < 2e-5);
    if (with_rope) printf("decode_attn      raw=%-3u comp=%-4u rope: tl|hand %.1e (hand fused %d, tl fused %d)  %s\n", n_raw, n_comp, eht / hmax, fused[0], fused[1], pass ? "OK" : "FAIL");
    else printf("decode_attn      raw=%-3u comp=%-4u hand|ref %.1e tl|ref %.1e tl|hand %.1e  %s\n", n_raw, n_comp, eh / rmax, et / rmax, eht / hmax, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double t[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ATTN", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    if (with_rope) ds4_gpu_set_decode_attn_rope_fuse(hd, 64, pos, n_ctx_orig, true, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                    CHECK(ds4_gpu_attention_decode_heads_tensor(theads, map, map_bytes, 256, tq, traw, n_raw, raw_cap, raw_start,
                                                                n_comp ? tcomp : NULL, 1, n_comp, NULL, 0, n_head, hd));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) t[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("ATTN", 0));
        }
        printf("decode_attn      raw=%-3u comp=%-4u bench x%d: hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", n_raw, n_comp, bench_iters, t[0], t[1], t[1] / t[0]);
    }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(traw); ds4_gpu_tensor_free(tcomp); ds4_gpu_tensor_free(theads);
    free(q); free(raw); free(comp); free(got[0]); free(got[1]); free(key); free(sc); free(o); free(map);
    return 1;
}

/* Prefill attention (zero-prefix chunk): raw rows with the sliding window
 * and, with ratio > 0, the compressed rows visible by position; TileLang vs
 * hand-written vs a double reference with the GPU's f16 roundings. */
static int check_prefill_attention(uint32_t n_tok, uint32_t ratio, int bench_iters) {
    const uint32_t n_head = 64, hd = 512, window = 128, n_comp = ratio ? n_tok / ratio : 0;
    const size_t map_bytes = (size_t)getpagesize() * 2;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    memset(map, 0, map_bytes);
    float *sinks = (float *)(map + 256);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = frand() * 2.0f;
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    const size_t nq = (size_t)n_tok * n_head * hd;
    float *q = malloc(nq * 4), *kv = malloc((size_t)n_tok * hd * 4);
    uint16_t *comp = malloc((size_t)(n_comp ? n_comp : 1) * hd * 2);
    CHECK(q && kv && comp);
    for (size_t i = 0; i < nq; i++) q[i] = frand() * 0.5f;
    for (size_t i = 0; i < (size_t)n_tok * hd; i++) kv[i] = f16_bits_to_f32(f32_to_f16_bits(frand() * 0.5f));
    for (size_t i = 0; i < (size_t)n_comp * hd; i++) comp[i] = f32_to_f16_bits(frand() * 0.5f);
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc_managed(nq * 4), *tkv = ds4_gpu_tensor_alloc_managed((size_t)n_tok * hd * 4);
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc_managed((size_t)(n_comp ? n_comp : 1) * hd * 2), *theads = ds4_gpu_tensor_alloc_managed(nq * 4);
    CHECK(tq && tkv && tcomp && theads && ds4_gpu_tensor_write(tq, 0, q, nq * 4) && ds4_gpu_tensor_write(tkv, 0, kv, (size_t)n_tok * hd * 4));
    if (n_comp) CHECK(ds4_gpu_tensor_write(tcomp, 0, comp, (size_t)n_comp * hd * 2));
    float *got[2];
    /* TileLang first: the hand-written path stages K/V into the same scratch
     * buffer, and a TL kernel reading stale-but-correct staging would pass. */
    for (int which = 1; which >= 0; which--) {
        got[which] = malloc(nq * 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("ATTN", which));
        if (n_comp) CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(theads, map, map_bytes, 256, tq, tkv, tcomp, 1, n_tok, n_comp, window, ratio, n_head, hd));
        else CHECK(ds4_gpu_attention_prefill_raw_heads_tensor(theads, map, map_bytes, 256, tq, tkv, n_tok, window, n_head, hd));
        CHECK(ds4_gpu_tl_force("ATTN", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(theads, 0, got[which], nq * 4));
    }
    /* reference on a subset of (token, head) rows */
    double rmax = 0, eh = 0, et = 0, eht = 0;
    double *sc = malloc((size_t)(n_tok + n_comp) * sizeof(double)), *o = malloc(hd * sizeof(double));
    for (uint32_t t = 0; t < n_tok; t += (n_tok > 40 ? 7 : 1)) {
        for (uint32_t h = 0; h < n_head; h += 9) {
            const float *qr = q + ((size_t)t * n_head + h) * hd;
            double m = sinks[h];
            for (uint32_t k = 0; k < n_tok + n_comp; k++) {
                int vis;
                if (k < n_tok) vis = k <= t && t - k < window;
                else vis = (k - n_tok) < (t + 1) / ratio;
                if (!vis) { sc[k] = neg_inf_f32(); continue; }
                double acc = 0;
                for (uint32_t d = 0; d < hd; d++) {
                    const double kd = k < n_tok ? kv[(size_t)k * hd + d] : f16_bits_to_f32(comp[(size_t)(k - n_tok) * hd + d]);
                    acc += (double)f16_bits_to_f32(f32_to_f16_bits(qr[d])) * kd;
                }
                sc[k] = acc / sqrt((double)hd); m = fmax(m, sc[k]);
            }
            double l = exp(sinks[h] - m);
            for (uint32_t d = 0; d < hd; d++) o[d] = 0;
            for (uint32_t k = 0; k < n_tok + n_comp; k++) {
                if (sc[k] == neg_inf_f32()) continue;
                const double p = exp(sc[k] - m); l += p;
                for (uint32_t d = 0; d < hd; d++) o[d] += p * (k < n_tok ? kv[(size_t)k * hd + d] : f16_bits_to_f32(comp[(size_t)(k - n_tok) * hd + d]));
            }
            for (uint32_t d = 0; d < hd; d++) {
                const double ref = o[d] / l; const size_t idx = ((size_t)t * n_head + h) * hd + d;
                rmax = fmax(rmax, fabs(ref)); eh = fmax(eh, fabs(got[0][idx] - ref)); et = fmax(et, fabs(got[1][idx] - ref));
            }
        }
    }
    double hmax = 0;
    for (size_t i = 0; i < nq; i++) { hmax = fmax(hmax, fabs(got[0][i])); eht = fmax(eht, fabs(got[0][i] - got[1][i])); }
    const int pass = et / rmax < 5e-5 && eht / hmax < 5e-5;
    printf("prefill_attn     tok=%-4u ratio=%-3u comp=%-4u hand|ref %.1e tl|ref %.1e tl|hand %.1e  %s\n", n_tok, ratio, n_comp, eh / rmax, et / rmax, eht / hmax, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ATTN", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    if (n_comp) CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(theads, map, map_bytes, 256, tq, tkv, tcomp, 1, n_tok, n_comp, window, ratio, n_head, hd));
                    else CHECK(ds4_gpu_attention_prefill_raw_heads_tensor(theads, map, map_bytes, 256, tq, tkv, n_tok, window, n_head, hd));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("ATTN", 0));
        }
        printf("prefill_attn     tok=%-4u ratio=%-3u bench x%d: hand %8.1f us  tl %8.1f us  tl/hand %.2fx\n", n_tok, ratio, bench_iters, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tcomp); ds4_gpu_tensor_free(theads);
    free(q); free(kv); free(comp); free(got[0]); free(got[1]); free(sc); free(o); free(map);
    return 1;
}

/* Routed MoE batch path (2..31 tokens): IQ2_XXS gate/up + SwiGLU*route,
 * Q2_K down summed over the 6 experts. TileLang vs hand-written (the batch
 * entry with the family forced off / on); both must agree with the hand-
 * written single-token path run per token. Synthetic random blocks. */
static int check_moe_batch(uint32_t n_tok, int bench_iters) {
    const uint32_t in_dim = 4096, mid_dim = 2048, out_dim = 4096, n_total = 16, n_used = 6;
    const size_t g_row = (size_t)(in_dim / 256) * 66, g_exp = g_row * mid_dim;
    const size_t d_row = (size_t)(mid_dim / 256) * 84, d_exp = d_row * out_dim;
    const uint64_t goff = 64, uoff = goff + g_exp * n_total, doff = uoff + g_exp * n_total;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (doff + d_exp * n_total + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    for (size_t i = 0; i < map_bytes; i++) map[i] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 24);
    /* sane block scales */
    for (uint32_t e = 0; e < n_total; e++) {
        for (size_t i = 0; i < g_exp; i += 66) { const uint16_t d = f32_to_f16_bits(0.002f + 0.004f * (frand() + 2.0f)); memcpy(map + goff + e * g_exp + i, &d, 2); memcpy(map + uoff + e * g_exp + i, &d, 2); }
        for (size_t i = 0; i < d_exp; i += 84) { const uint16_t d = f32_to_f16_bits(0.001f + 0.002f * (frand() + 2.0f)), dm = f32_to_f16_bits(0.0005f * (frand() + 2.0f)); memcpy(map + doff + e * d_exp + i + 80, &d, 2); memcpy(map + doff + e * d_exp + i + 82, &dm, 2); }
    }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *x = malloc((size_t)n_tok * in_dim * 4), *w = malloc((size_t)n_tok * n_used * 4);
    int32_t *ids = malloc((size_t)n_tok * n_used * 4);
    CHECK(x && w && ids);
    for (size_t i = 0; i < (size_t)n_tok * in_dim; i++) x[i] = frand();
    for (uint32_t t = 0; t < n_tok; t++) for (uint32_t s = 0; s < n_used; s++) { ids[t * n_used + s] = (int32_t)((t * 5 + s * 3 + 1) % n_total); w[t * n_used + s] = 0.1f + 0.2f * s; }
    const float clamp = 10.0f;
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed((size_t)n_tok * in_dim * 4);
    ds4_gpu_tensor *tids = ds4_gpu_tensor_alloc_managed((size_t)n_tok * n_used * 4), *tw = ds4_gpu_tensor_alloc_managed((size_t)n_tok * n_used * 4);
    ds4_gpu_tensor *tgate = ds4_gpu_tensor_alloc_managed((size_t)n_tok * n_used * mid_dim * 4), *tup = ds4_gpu_tensor_alloc_managed((size_t)n_tok * n_used * mid_dim * 4);
    ds4_gpu_tensor *tmid = ds4_gpu_tensor_alloc_managed((size_t)n_tok * n_used * mid_dim * 4), *texp = ds4_gpu_tensor_alloc_managed((size_t)n_tok * n_used * out_dim * 4);
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc_managed((size_t)n_tok * out_dim * 4);
    CHECK(tx && tids && tw && tgate && tup && tmid && texp && tout);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, (size_t)n_tok * in_dim * 4) && ds4_gpu_tensor_write(tids, 0, ids, (size_t)n_tok * n_used * 4) && ds4_gpu_tensor_write(tw, 0, w, (size_t)n_tok * n_used * 4));
    float *got[2], *ref = malloc((size_t)n_tok * out_dim * 4);
    CHECK(ref);
    for (int which = 1; which >= 0; which--) {
        got[which] = malloc((size_t)n_tok * out_dim * 4); CHECK(got[which]);
        bool mid_f16 = false;
        CHECK(ds4_gpu_tl_force("MOE", which));
        CHECK(ds4_gpu_routed_moe_batch_tensor(tout, tgate, tup, tmid, texp, map, map_bytes, goff, uoff, doff, 16, 10, g_exp, g_row, d_exp, d_row,
                                              in_dim, mid_dim, out_dim, tids, tw, n_total, n_used, clamp, tx, 0, n_tok, &mid_f16, true));
        CHECK(ds4_gpu_tl_force("MOE", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout, 0, got[which], (size_t)n_tok * out_dim * 4));
    }
    /* per-token hand-written reference */
    ds4_gpu_tensor *tx1 = ds4_gpu_tensor_alloc_managed(in_dim * 4), *tout1 = ds4_gpu_tensor_alloc_managed(out_dim * 4);
    ds4_gpu_tensor *tids1 = ds4_gpu_tensor_alloc_managed(n_used * 4), *tw1 = ds4_gpu_tensor_alloc_managed(n_used * 4);
    CHECK(tx1 && tout1 && tids1 && tw1);
    for (uint32_t t = 0; t < n_tok; t++) {
        CHECK(ds4_gpu_tensor_write(tx1, 0, x + (size_t)t * in_dim, in_dim * 4) && ds4_gpu_tensor_write(tids1, 0, ids + t * n_used, n_used * 4) && ds4_gpu_tensor_write(tw1, 0, w + t * n_used, n_used * 4));
        CHECK(ds4_gpu_routed_moe_one_tensor(tout1, tgate, tup, tmid, texp, map, map_bytes, goff, uoff, doff, 16, 10, g_exp, g_row, d_exp, d_row,
                                            in_dim, mid_dim, out_dim, tids1, tw1, n_total, n_used, clamp, tx1, NULL, true, 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tout1, 0, ref + (size_t)t * out_dim, out_dim * 4));
    }
    double rmax = 0, eh = 0, et = 0, eth = 0;
    for (size_t i = 0; i < (size_t)n_tok * out_dim; i++) { rmax = fmax(rmax, fabs(ref[i])); eh = fmax(eh, fabs(got[0][i] - ref[i])); et = fmax(et, fabs(got[1][i] - ref[i])); eth = fmax(eth, fabs(got[1][i] - got[0][i])); }
    /* >= 32 tokens: ds4's mm_id tiers (and the packed TileLang GEMMs) use f16 activations and an f16 mid, 1e-3 of the f32 per-token path */
    const double tol = n_tok >= 32 ? 5e-3 : 2e-5;
    const int pass = et / rmax < tol && eh / rmax < tol && (n_tok < 32 || eth / rmax < 2e-3);
    printf("moe_batch        tok=%-4u hand|one %.1e tl|one %.1e tl|hand %.1e  %s\n", n_tok, eh / rmax, et / rmax, eth / rmax, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("MOE", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    bool mid_f16 = false;
                    CHECK(ds4_gpu_routed_moe_batch_tensor(tout, tgate, tup, tmid, texp, map, map_bytes, goff, uoff, doff, 16, 10, g_exp, g_row, d_exp, d_row,
                                                          in_dim, mid_dim, out_dim, tids, tw, n_total, n_used, clamp, tx, 0, n_tok, &mid_f16, true));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("MOE", 0));
        }
        printf("moe_batch        tok=%-3u bench x%d: hand %8.1f us  tl %8.1f us  tl/hand %.2fx\n", n_tok, bench_iters, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tids); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tgate); ds4_gpu_tensor_free(tup); ds4_gpu_tensor_free(tmid); ds4_gpu_tensor_free(texp); ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tx1); ds4_gpu_tensor_free(tout1); ds4_gpu_tensor_free(tids1); ds4_gpu_tensor_free(tw1);
    free(x); free(w); free(ids); free(got[0]); free(got[1]); free(ref); free(map);
    return 1;
}

/* The two data-parallel fused decode kernels: (a) q_a + kv Q8_0 GEMVs with
 * the attention / indexer compressor f16 GEMV pairs and the state append;
 * (b) q/kv rms norm + KV RoPE + E4M3FN round trip + raw ring store.
 * TileLang vs hand-written (tolerance: different reduction trees). */
static int check_fused_decode(uint32_t ratio, int bench_iters) {
    const uint32_t in_dim = 4096, q_dim = 1024, kv_dim = 512, w0 = ratio == 4 ? 1024 : 512, w1 = ratio == 4 ? 256 : 0;
    const uint32_t pos = 1000 + 7, raw_cap = 128, raw_row = pos % raw_cap, n_rot = 64;
    const size_t q8_row = (in_dim / 32) * 34;
    const size_t qa_bytes = q8_row * q_dim, kv_bytes = q8_row * kv_dim, c0_bytes = (size_t)w0 * in_dim * 2, c1_bytes = (size_t)(w1 ? w1 : 1) * in_dim * 2;
    const size_t ape0_bytes = (size_t)ratio * w0 * 2, ape1_bytes = (size_t)ratio * (w1 ? w1 : 1) * 2, norm_bytes = (q_dim + kv_dim) * 4;
    const uint64_t off_qa = 64, off_kv = off_qa + qa_bytes, off_c0a = off_kv + kv_bytes, off_c0b = off_c0a + c0_bytes, off_c1a = off_c0b + c0_bytes,
                   off_c1b = off_c1a + c1_bytes, off_ape0 = off_c1b + c1_bytes, off_ape1 = off_ape0 + ape0_bytes, off_norm = off_ape1 + ape1_bytes;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (off_norm + norm_bytes + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    for (size_t i = 0; i < map_bytes; i++) map[i] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 24);
    for (size_t i = 0; i < qa_bytes + kv_bytes; i += 34) { const uint16_t d = f32_to_f16_bits(0.001f + 0.002f * (frand() + 2.0f)); memcpy(map + off_qa + i, &d, 2); }
    for (size_t i = 0; i < 2 * c0_bytes + 2 * c1_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand() * 0.05f); memcpy(map + off_c0a + i, &h, 2); }
    for (size_t i = 0; i < ape0_bytes + ape1_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand()); memcpy(map + off_ape0 + i, &h, 2); }
    float *norms = (float *)(map + off_norm);
    for (uint32_t i = 0; i < q_dim + kv_dim; i++) norms[i] = 1.0f + 0.25f * frand();
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *x = malloc(in_dim * 4);
    for (uint32_t i = 0; i < in_dim; i++) x[i] = frand();
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(in_dim * 4), *tqr = ds4_gpu_tensor_alloc_managed(q_dim * 4), *tkvr = ds4_gpu_tensor_alloc_managed(kv_dim * 4);
    ds4_gpu_tensor *tc0a = ds4_gpu_tensor_alloc_managed(w0 * 4), *tc0b = ds4_gpu_tensor_alloc_managed(w0 * 4);
    ds4_gpu_tensor *tc1a = ds4_gpu_tensor_alloc_managed((w1 ? w1 : 4) * 4), *tc1b = ds4_gpu_tensor_alloc_managed((w1 ? w1 : 4) * 4);
    const uint32_t st_rows = ratio == 4 ? 8 : 128;
    ds4_gpu_tensor *ts0k = ds4_gpu_tensor_alloc_managed((size_t)st_rows * w0 * 4), *ts0s = ds4_gpu_tensor_alloc_managed((size_t)st_rows * w0 * 4);
    ds4_gpu_tensor *ts1k = ds4_gpu_tensor_alloc_managed((size_t)st_rows * (w1 ? w1 : 4) * 4), *ts1s = ds4_gpu_tensor_alloc_managed((size_t)st_rows * (w1 ? w1 : 4) * 4);
    ds4_gpu_tensor *tqo = ds4_gpu_tensor_alloc_managed(q_dim * 4), *tkvo = ds4_gpu_tensor_alloc_managed(kv_dim * 4), *traw = ds4_gpu_tensor_alloc_managed((size_t)raw_cap * kv_dim * 4);
    CHECK(tx && tqr && tkvr && tc0a && tc0b && tc1a && tc1b && ts0k && ts0s && ts1k && ts1s && tqo && tkvo && traw);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, in_dim * 4));
    const size_t n_all = q_dim + kv_dim + 2 * w0 + 2 * w1 + 2 * w0 + 2 * w1 + q_dim + kv_dim + kv_dim;   /* qr, kv, c0a, c0b, c1a, c1b, state0 rows, state1 rows, q_out, kv_out, raw row */
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc(n_all, 4); CHECK(got[which]);
        float *z = calloc((size_t)st_rows * w0, 4);
        CHECK(z && ds4_gpu_tensor_write(ts0k, 0, z, (size_t)st_rows * w0 * 4) && ds4_gpu_tensor_write(ts0s, 0, z, (size_t)st_rows * w0 * 4));
        if (w1) CHECK(ds4_gpu_tensor_write(ts1k, 0, z, (size_t)st_rows * w1 * 4) && ds4_gpu_tensor_write(ts1s, 0, z, (size_t)st_rows * w1 * 4));
        free(z);
        CHECK(ds4_gpu_tl_force("Q8_GEMV", which) && ds4_gpu_tl_force("KV", which));
        CHECK(ds4_gpu_qkv_pair_quad_compressor_store_tensor(tqr, tkvr, tc0a, tc0b, w1 ? tc1a : tc0a, w1 ? tc1b : tc0b, ts0k, ts0s, w1 ? ts1k : ts0k, w1 ? ts1s : ts0s,
                                                            map, map_bytes, off_qa, off_kv, off_c0a, off_c0b, w1 ? off_c1a : off_c0a, w1 ? off_c1b : off_c0b,
                                                            off_ape0, 1, w1 ? off_ape1 : off_ape0, 1, in_dim, q_dim, kv_dim, w0, w1, tx, ratio, pos) == 1);
        CHECK(ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(tqo, tqr, map, map_bytes, off_norm, q_dim, tkvo, tkvr, off_norm + q_dim * 4, kv_dim, traw, raw_cap, raw_row,
                                                                 n_rot, pos, 65536, 10000.0f, 1.0f / 16, 1.0f, 1.0f, 32.0f, 1.0f, 1e-6f));
        CHECK(ds4_gpu_tl_force("Q8_GEMV", 0) && ds4_gpu_tl_force("KV", 0));
        CHECK(ds4_gpu_synchronize());
        float *g = got[which]; const uint32_t srow = ratio == 4 ? 4 + pos % 4 : pos % ratio;
        CHECK(ds4_gpu_tensor_read(tqr, 0, g, q_dim * 4)); g += q_dim;
        CHECK(ds4_gpu_tensor_read(tkvr, 0, g, kv_dim * 4)); g += kv_dim;
        CHECK(ds4_gpu_tensor_read(tc0a, 0, g, w0 * 4)); g += w0; CHECK(ds4_gpu_tensor_read(tc0b, 0, g, w0 * 4)); g += w0;
        if (w1) { CHECK(ds4_gpu_tensor_read(tc1a, 0, g, w1 * 4)); g += w1; CHECK(ds4_gpu_tensor_read(tc1b, 0, g, w1 * 4)); g += w1; }
        CHECK(ds4_gpu_tensor_read(ts0k, (size_t)srow * w0 * 4, g, w0 * 4)); g += w0; CHECK(ds4_gpu_tensor_read(ts0s, (size_t)srow * w0 * 4, g, w0 * 4)); g += w0;
        if (w1) { CHECK(ds4_gpu_tensor_read(ts1k, (size_t)srow * w1 * 4, g, w1 * 4)); g += w1; CHECK(ds4_gpu_tensor_read(ts1s, (size_t)srow * w1 * 4, g, w1 * 4)); g += w1; }
        CHECK(ds4_gpu_tensor_read(tqo, 0, g, q_dim * 4)); g += q_dim;
        CHECK(ds4_gpu_tensor_read(tkvo, 0, g, kv_dim * 4)); g += kv_dim;
        CHECK(ds4_gpu_tensor_read(traw, (size_t)raw_row * kv_dim * 4, g, kv_dim * 4)); g += kv_dim;
    }
    double emax = 0, vmax = 0; size_t flips = 0;
    for (size_t i = 0; i < n_all; i++) {
        const double d = fabs(got[0][i] - got[1][i]), m = fmax(fabs(got[0][i]), fabs(got[1][i]));
        vmax = fmax(vmax, m);
        if (d > 1e-3 * fmax(m, 1e-3)) flips++; else emax = fmax(emax, d);
    }
    const int pass = emax / vmax < 2e-5 && flips <= 4;   /* a few E4M3 rounding-boundary flips from ULP-level norm differences are expected */
    printf("fused_decode     ratio=%-3u tl|hand %.1e (values %zu, fp8 boundary flips %zu)  %s\n", ratio, emax / vmax, n_all, flips, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("Q8_GEMV", which) && ds4_gpu_tl_force("KV", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_qkv_pair_quad_compressor_store_tensor(tqr, tkvr, tc0a, tc0b, w1 ? tc1a : tc0a, w1 ? tc1b : tc0b, ts0k, ts0s, w1 ? ts1k : ts0k, w1 ? ts1s : ts0s,
                                                                        map, map_bytes, off_qa, off_kv, off_c0a, off_c0b, w1 ? off_c1a : off_c0a, w1 ? off_c1b : off_c0b,
                                                                        off_ape0, 1, w1 ? off_ape1 : off_ape0, 1, in_dim, q_dim, kv_dim, w0, w1, tx, ratio, pos) == 1);
                    CHECK(ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(tqo, tqr, map, map_bytes, off_norm, q_dim, tkvo, tkvr, off_norm + q_dim * 4, kv_dim, traw, raw_cap, raw_row,
                                                                             n_rot, pos, 65536, 10000.0f, 1.0f / 16, 1.0f, 1.0f, 32.0f, 1.0f, 1e-6f));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("Q8_GEMV", 0) && ds4_gpu_tl_force("KV", 0));
        }
        printf("fused_decode     ratio=%-3u bench x%d (2 kernels): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", ratio, bench_iters, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tqr); ds4_gpu_tensor_free(tkvr); ds4_gpu_tensor_free(tc0a); ds4_gpu_tensor_free(tc0b); ds4_gpu_tensor_free(tc1a); ds4_gpu_tensor_free(tc1b);
    ds4_gpu_tensor_free(ts0k); ds4_gpu_tensor_free(ts0s); ds4_gpu_tensor_free(ts1k); ds4_gpu_tensor_free(ts1s); ds4_gpu_tensor_free(tqo); ds4_gpu_tensor_free(tkvo); ds4_gpu_tensor_free(traw);
    free(x); free(got[0]); free(got[1]); free(map);
    return 1;
}

/* The multi-session batch decode schedule (ds4_sessions_eval_batch): the
 * attention front runs the Q8_0 projections row-exactly for all sessions,
 * then each session's phase uses the unfused KV finalizer and the f16-only
 * pair / quad compressor stores; the batched FFN uses the row-exact shared
 * gate/up + SwiGLU. TileLang: single-row GEMVs per row, the quad kernel with
 * empty roles, and tl_kv_fp8_store. */
static int check_batch_schedule(uint32_t ratio, uint32_t n_rows, int bench_iters) {
    const uint32_t in_dim = 4096, q_dim = 1024, kv_dim = 512, w0 = ratio == 4 ? 1024 : 512, w1 = ratio == 4 ? 256 : 0;
    const uint32_t pos = 1000 + 7, raw_cap = 128, raw_row = pos % raw_cap, n_rot = 64;
    const size_t q8_row = (in_dim / 32) * 34;
    const size_t qa_bytes = q8_row * q_dim, kv_bytes = q8_row * kv_dim, c0_bytes = (size_t)w0 * in_dim * 2, c1_bytes = (size_t)(w1 ? w1 : 1) * in_dim * 2;
    const size_t ape0_bytes = (size_t)ratio * w0 * 2, ape1_bytes = (size_t)ratio * (w1 ? w1 : 1) * 2;
    const uint64_t off_qa = 64, off_kv = off_qa + qa_bytes, off_c0a = off_kv + kv_bytes, off_c0b = off_c0a + c0_bytes, off_c1a = off_c0b + c0_bytes,
                   off_c1b = off_c1a + c1_bytes, off_ape0 = off_c1b + c1_bytes, off_ape1 = off_ape0 + ape0_bytes, off_end = off_ape1 + ape1_bytes;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (off_end + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    for (size_t i = 0; i < map_bytes; i++) map[i] = (uint8_t)(seed = seed * 1664525u + 1013904223u, seed >> 24);
    for (size_t i = 0; i < qa_bytes + kv_bytes; i += 34) { const uint16_t d = f32_to_f16_bits(0.001f + 0.002f * (frand() + 2.0f)); memcpy(map + off_qa + i, &d, 2); }
    for (size_t i = 0; i < 2 * c0_bytes + 2 * c1_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand() * 0.05f); memcpy(map + off_c0a + i, &h, 2); }
    for (size_t i = 0; i < ape0_bytes + ape1_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand()); memcpy(map + off_ape0 + i, &h, 2); }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *x = malloc((size_t)n_rows * in_dim * 4), *kvrow = malloc(kv_dim * 4);
    CHECK(x && kvrow);
    for (size_t i = 0; i < (size_t)n_rows * in_dim; i++) x[i] = frand();
    for (uint32_t i = 0; i < kv_dim; i++) kvrow[i] = frand() * (i % 7 == 0 ? 40.0f : 2.0f);
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed((size_t)n_rows * in_dim * 4);
    ds4_gpu_tensor *tqr = ds4_gpu_tensor_alloc_managed((size_t)n_rows * q_dim * 4), *tkvr = ds4_gpu_tensor_alloc_managed((size_t)n_rows * kv_dim * 4);
    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc_managed((size_t)n_rows * q_dim * 4), *tu = ds4_gpu_tensor_alloc_managed((size_t)n_rows * q_dim * 4), *tm = ds4_gpu_tensor_alloc_managed((size_t)n_rows * q_dim * 4);
    ds4_gpu_tensor *tc0a = ds4_gpu_tensor_alloc_managed(w0 * 4), *tc0b = ds4_gpu_tensor_alloc_managed(w0 * 4);
    ds4_gpu_tensor *tc1a = ds4_gpu_tensor_alloc_managed((w1 ? w1 : 4) * 4), *tc1b = ds4_gpu_tensor_alloc_managed((w1 ? w1 : 4) * 4);
    const uint32_t st_rows = ratio == 4 ? 8 : 128;
    ds4_gpu_tensor *ts0k = ds4_gpu_tensor_alloc_managed((size_t)st_rows * w0 * 4), *ts0s = ds4_gpu_tensor_alloc_managed((size_t)st_rows * w0 * 4);
    ds4_gpu_tensor *ts1k = ds4_gpu_tensor_alloc_managed((size_t)st_rows * (w1 ? w1 : 4) * 4), *ts1s = ds4_gpu_tensor_alloc_managed((size_t)st_rows * (w1 ? w1 : 4) * 4);
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc_managed(kv_dim * 4), *traw = ds4_gpu_tensor_alloc_managed((size_t)raw_cap * kv_dim * 4);
    CHECK(tx && tqr && tkvr && tg && tu && tm && tc0a && tc0b && tc1a && tc1b && ts0k && ts0s && ts1k && ts1s && tkv && traw);
    CHECK(ds4_gpu_tensor_write(tx, 0, x, (size_t)n_rows * in_dim * 4));
    const size_t n_all = (size_t)n_rows * (q_dim + kv_dim + 3 * q_dim) + 6 * w0 + 4 * w1 + 2 * kv_dim;   /* projections, gate/up/mid, pair + quad stores, kv + raw row */
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc(n_all, 4); CHECK(got[which]);
        float *z = calloc((size_t)st_rows * w0, 4);
        CHECK(z && ds4_gpu_tensor_write(ts0k, 0, z, (size_t)st_rows * w0 * 4) && ds4_gpu_tensor_write(ts0s, 0, z, (size_t)st_rows * w0 * 4));
        if (w1) CHECK(ds4_gpu_tensor_write(ts1k, 0, z, (size_t)st_rows * w1 * 4) && ds4_gpu_tensor_write(ts1s, 0, z, (size_t)st_rows * w1 * 4));
        free(z);
        CHECK(ds4_gpu_tensor_write(tkv, 0, kvrow, kv_dim * 4));
        CHECK(ds4_gpu_tl_force("Q8_GEMV", which) && ds4_gpu_tl_force("KV", which) && ds4_gpu_tl_force("F16_GEMV", which));
        float *g = got[which]; const uint32_t srow = ratio == 4 ? 4 + pos % 4 : pos % ratio;
        /* 1. row-exact Q8_0 projections for all rows */
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tqr, map, map_bytes, off_qa, in_dim, q_dim, tx, n_rows));
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tkvr, map, map_bytes, off_kv, in_dim, kv_dim, tx, n_rows));
        /* 2. row-exact shared gate/up + SwiGLU (gate == up weights here) */
        CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(tg, tu, tm, map, map_bytes, off_qa, off_qa, in_dim, q_dim, tx, n_rows, 7.0f));
        /* 3. the f16-only pair store (first row's activation) */
        CHECK(ds4_gpu_matmul_f16_pair_compressor_store_tensor(tc0a, tc0b, ts0k, ts0s, map, map_bytes, off_c0a, off_c0b, off_ape0, 1, in_dim, w0, tx, ratio, pos) == 1);
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(ts0k, (size_t)srow * w0 * 4, g, w0 * 4)); g += w0; CHECK(ds4_gpu_tensor_read(ts0s, (size_t)srow * w0 * 4, g, w0 * 4)); g += w0;
        /* 4. the f16-only quad store (ratio 4 only) */
        if (w1) {
            CHECK(ds4_gpu_matmul_f16_quad_compressor_store_tensor(tc0a, tc0b, tc1a, tc1b, ts0k, ts0s, ts1k, ts1s, map, map_bytes, off_c0a, off_c0b, off_c1a, off_c1b,
                                                                  off_ape0, 1, off_ape1, 1, in_dim, w0, w1, tx, ratio, pos) == 1);
            CHECK(ds4_gpu_synchronize());
            CHECK(ds4_gpu_tensor_read(ts1k, (size_t)srow * w1 * 4, g, w1 * 4)); g += w1; CHECK(ds4_gpu_tensor_read(ts1s, (size_t)srow * w1 * 4, g, w1 * 4)); g += w1;
            CHECK(ds4_gpu_tensor_read(tc1a, 0, g, w1 * 4)); g += w1; CHECK(ds4_gpu_tensor_read(tc1b, 0, g, w1 * 4)); g += w1;
        }
        /* 5. the unfused KV finalizer */
        CHECK(ds4_gpu_kv_fp8_store_raw_tensor(tkv, traw, raw_cap, raw_row, kv_dim, n_rot));
        CHECK(ds4_gpu_tl_force("Q8_GEMV", 0) && ds4_gpu_tl_force("KV", 0) && ds4_gpu_tl_force("F16_GEMV", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tqr, 0, g, (size_t)n_rows * q_dim * 4)); g += (size_t)n_rows * q_dim;
        CHECK(ds4_gpu_tensor_read(tkvr, 0, g, (size_t)n_rows * kv_dim * 4)); g += (size_t)n_rows * kv_dim;
        CHECK(ds4_gpu_tensor_read(tg, 0, g, (size_t)n_rows * q_dim * 4)); g += (size_t)n_rows * q_dim;
        CHECK(ds4_gpu_tensor_read(tu, 0, g, (size_t)n_rows * q_dim * 4)); g += (size_t)n_rows * q_dim;
        CHECK(ds4_gpu_tensor_read(tm, 0, g, (size_t)n_rows * q_dim * 4)); g += (size_t)n_rows * q_dim;
        CHECK(ds4_gpu_tensor_read(tc0a, 0, g, w0 * 4)); g += w0; CHECK(ds4_gpu_tensor_read(tc0b, 0, g, w0 * 4)); g += w0;
        CHECK(ds4_gpu_tensor_read(ts0k, (size_t)srow * w0 * 4, g, w0 * 4)); g += w0; CHECK(ds4_gpu_tensor_read(ts0s, (size_t)srow * w0 * 4, g, w0 * 4)); g += w0;
        CHECK(ds4_gpu_tensor_read(tkv, 0, g, kv_dim * 4)); g += kv_dim;
        CHECK(ds4_gpu_tensor_read(traw, (size_t)raw_row * kv_dim * 4, g, kv_dim * 4)); g += kv_dim;
        CHECK((size_t)(g - got[which]) == n_all);
    }
    double emax = 0, vmax = 0; size_t flips = 0;
    for (size_t i = 0; i < n_all; i++) {
        const double d = fabs(got[0][i] - got[1][i]), m = fmax(fabs(got[0][i]), fabs(got[1][i]));
        vmax = fmax(vmax, m);
        /* reduction-order noise on near-cancelling GEMV outputs (|v| ~ 1e-4) is not a flip */
        if (d > 1e-3 * fmax(m, 1e-2)) { flips++; if (getenv("DS4_TL_DEBUG_FLIPS")) fprintf(stderr, "flip at %zu: hand %g tl %g\n", i, got[0][i], got[1][i]); }
        else emax = fmax(emax, d);
    }
    const int pass = emax / vmax < 2e-5 && flips == 0;
    printf("batch_schedule   ratio=%-3u rows=%u tl|hand %.1e (values %zu, flips %zu)  %s\n", ratio, n_rows, emax / vmax, n_all, flips, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tmv[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("Q8_GEMV", which) && ds4_gpu_tl_force("KV", which) && ds4_gpu_tl_force("F16_GEMV", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++) {
                    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(tqr, map, map_bytes, off_qa, in_dim, q_dim, tx, n_rows));
                    CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(tg, tu, tm, map, map_bytes, off_qa, off_qa, in_dim, q_dim, tx, n_rows, 7.0f));
                    CHECK(ds4_gpu_matmul_f16_pair_compressor_store_tensor(tc0a, tc0b, ts0k, ts0s, map, map_bytes, off_c0a, off_c0b, off_ape0, 1, in_dim, w0, tx, ratio, pos) == 1);
                    CHECK(ds4_gpu_kv_fp8_store_raw_tensor(tkv, traw, raw_cap, raw_row, kv_dim, n_rot));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tmv[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("Q8_GEMV", 0) && ds4_gpu_tl_force("KV", 0) && ds4_gpu_tl_force("F16_GEMV", 0));
        }
        printf("batch_schedule   ratio=%-3u rows=%u bench x%d (4 ops): hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", ratio, n_rows, bench_iters, tmv[0], tmv[1], tmv[1] / tmv[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tqr); ds4_gpu_tensor_free(tkvr); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tu); ds4_gpu_tensor_free(tm);
    ds4_gpu_tensor_free(tc0a); ds4_gpu_tensor_free(tc0b); ds4_gpu_tensor_free(tc1a); ds4_gpu_tensor_free(tc1b);
    ds4_gpu_tensor_free(ts0k); ds4_gpu_tensor_free(ts0s); ds4_gpu_tensor_free(ts1k); ds4_gpu_tensor_free(ts1s); ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(traw);
    free(x); free(kvrow); free(got[0]); free(got[1]); free(map);
    return 1;
}

/* HC mixer: mix = hc_fn . rms_norm(residual_hc) (f16 weights, 16384 wide),
 * then sinkhorn split / collapse / attention norm. The hand kernel fuses
 * both halves via a last-arriving-threadgroup handoff; TileLang uses two
 * dispatches. Reduction trees differ: tolerance check. */
static int check_hc_rms_norm_mix(int bench_iters) {
    const uint32_t n = 16384, mix_dim = 24, n_embd = 4096, n_hc = 4, sinkhorn_iters = 20;
    const size_t w_bytes = (size_t)n * mix_dim * 2;
    const uint64_t off_w = 64, off_scale = off_w + w_bytes, off_base = off_scale + 16, off_norm = off_base + mix_dim * 4;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (off_norm + n_embd * 4 + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    for (size_t i = 0; i < w_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand() * 0.02f); memcpy(map + off_w + i, &h, 2); }
    float *scale = (float *)(map + off_scale), *base = (float *)(map + off_base), *norm_w = (float *)(map + off_norm);
    scale[0] = 0.5f; scale[1] = 0.5f; scale[2] = 0.5f;
    for (uint32_t i = 0; i < mix_dim; i++) base[i] = 0.25f * frand() + (i >= 4 && i < 8 && (i - 4) == (i - 4) ? (i == 4 + ((i - 4) % 4) ? 1.0f : 0.0f) : 0.0f);
    for (uint32_t i = 0; i < n_embd; i++) norm_w[i] = 1.0f + 0.25f * frand();
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *x = malloc(n * 4);
    for (uint32_t i = 0; i < n; i++) x[i] = frand();
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed(n * 4), *tmix = ds4_gpu_tensor_alloc_managed(mix_dim * 4), *tout = ds4_gpu_tensor_alloc_managed(n_embd * 4);
    ds4_gpu_tensor *tnorm = ds4_gpu_tensor_alloc_managed(n_embd * 4), *tsplit = ds4_gpu_tensor_alloc_managed((size_t)n_hc * n_hc * 4 + 4 * n_hc * 4 + 64 * 4);
    CHECK(tx && tmix && tout && tnorm && tsplit && ds4_gpu_tensor_write(tx, 0, x, n * 4));
    const size_t n_all = mix_dim + 2 * n_embd;
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc(n_all, 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("HC", which));
        CHECK(ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(tmix, tout, tnorm, tsplit, tx, map, map_bytes, off_w, off_scale, off_base, off_norm,
                                                             n, mix_dim, n_embd, n_hc, sinkhorn_iters, 1e-6f, 1e-6f, 1e-6f) == 1);
        CHECK(ds4_gpu_tl_force("HC", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tmix, 0, got[which], mix_dim * 4));
        CHECK(ds4_gpu_tensor_read(tout, 0, got[which] + mix_dim, n_embd * 4));
        CHECK(ds4_gpu_tensor_read(tnorm, 0, got[which] + mix_dim + n_embd, n_embd * 4));
    }
    double emax = 0, vmax = 0;
    for (size_t i = 0; i < n_all; i++) { emax = fmax(emax, fabs(got[0][i] - got[1][i])); vmax = fmax(vmax, fabs(got[0][i])); }
    const int pass = emax / vmax < 2e-5;
    if (getenv("DS4_TL_DEBUG_HC")) for (uint32_t i = 0; i < mix_dim + 4; i++) printf("  [%u] hand %g tl %g\n", i, got[0][i], got[1][i]);
    printf("hc_rms_norm_mix  tl|hand %.1e (mix %u, out %u, norm %u)  %s\n", emax / vmax, mix_dim, n_embd, n_embd, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("HC", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < bench_iters; it++)
                    CHECK(ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(tmix, tout, tnorm, tsplit, tx, map, map_bytes, off_w, off_scale, off_base, off_norm,
                                                                         n, mix_dim, n_embd, n_hc, sinkhorn_iters, 1e-6f, 1e-6f, 1e-6f) == 1);
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / bench_iters;
            }
            CHECK(ds4_gpu_tl_force("HC", 0));
        }
        printf("hc_rms_norm_mix  bench x%d: hand %7.2f us  tl %7.2f us (2 dispatches)  tl/hand %.2fx\n", bench_iters, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tmix); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tnorm); ds4_gpu_tensor_free(tsplit);
    free(x); free(got[0]); free(got[1]); free(map);
    return 1;
}

/* Compressor per-token chain (unaligned prefill tail chunks and ratio-128
 * emits): store_one + one-comp pool + norm + rope (+ ratio-4 shift) through
 * ds4_gpu_compressor_update_tensor, prefill mode. TileLang vs hand. */
static int check_compressor_chain(uint32_t ratio, uint32_t head_dim, int bench_iters) {
    const uint32_t coff = ratio == 4 ? 2 : 1, width = coff * head_dim, state_rows = coff * ratio;
    const uint32_t n_tok = ratio == 4 ? 12 : 128 + 5;   /* 3 emits + 0 tail | 1 emit + 5 tail tokens */
    const size_t ape_bytes = (size_t)ratio * width * 2, norm_bytes = head_dim * 4;
    const uint64_t off_ape = 64, off_norm = off_ape + ape_bytes;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (off_norm + norm_bytes + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    for (size_t i = 0; i < ape_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand()); memcpy(map + off_ape + i, &h, 2); }
    float *norm_w = (float *)(map + off_norm);
    for (uint32_t i = 0; i < head_dim; i++) norm_w[i] = 1.0f + 0.25f * frand();
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *kv = malloc((size_t)n_tok * width * 4), *sc = malloc((size_t)n_tok * width * 4);
    for (size_t i = 0; i < (size_t)n_tok * width; i++) { kv[i] = frand(); sc[i] = 2.0f * frand(); }
    ds4_gpu_tensor *tkv = ds4_gpu_tensor_alloc_managed((size_t)n_tok * width * 4), *tsc = ds4_gpu_tensor_alloc_managed((size_t)n_tok * width * 4);
    ds4_gpu_tensor *tskv = ds4_gpu_tensor_alloc_managed((size_t)state_rows * width * 4), *tssc = ds4_gpu_tensor_alloc_managed((size_t)state_rows * width * 4);
    const uint32_t n_comp = n_tok / ratio;
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc_managed((size_t)(n_comp + 1) * head_dim * 4);
    CHECK(tkv && tsc && tskv && tssc && tcomp && ds4_gpu_tensor_write(tkv, 0, kv, (size_t)n_tok * width * 4) && ds4_gpu_tensor_write(tsc, 0, sc, (size_t)n_tok * width * 4));
    const size_t n_all = (size_t)n_comp * head_dim + 2 * (size_t)state_rows * width;
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc(n_all, 4); CHECK(got[which]);
        float *z = calloc((size_t)state_rows * width, 4); CHECK(z);
        CHECK(ds4_gpu_tensor_write(tskv, 0, z, (size_t)state_rows * width * 4) && ds4_gpu_tensor_write(tssc, 0, z, (size_t)state_rows * width * 4));
        free(z);
        CHECK(ds4_gpu_tl_force("KV", which) && ds4_gpu_tl_force("RMS_NORM", which) && ds4_gpu_tl_force("ROPE", which));
        for (uint32_t t = 0; t < n_tok; t++) {
            ds4_gpu_tensor *vkv = ds4_gpu_tensor_view(tkv, (size_t)t * width * 4, width * 4), *vsc = ds4_gpu_tensor_view(tsc, (size_t)t * width * 4, width * 4);
            CHECK(vkv && vsc);
            CHECK(ds4_gpu_compressor_update_tensor(vkv, vsc, tskv, tssc, tcomp, map, map_bytes, off_ape, 1, off_norm, 0, head_dim, ratio, t, t / ratio,
                                                   64, 65536, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f, false, false, false));
            ds4_gpu_tensor_free(vkv); ds4_gpu_tensor_free(vsc);
        }
        CHECK(ds4_gpu_tl_force("KV", 0) && ds4_gpu_tl_force("RMS_NORM", 0) && ds4_gpu_tl_force("ROPE", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tcomp, 0, got[which], (size_t)n_comp * head_dim * 4));
        CHECK(ds4_gpu_tensor_read(tskv, 0, got[which] + (size_t)n_comp * head_dim, (size_t)state_rows * width * 4));
        CHECK(ds4_gpu_tensor_read(tssc, 0, got[which] + (size_t)n_comp * head_dim + (size_t)state_rows * width, (size_t)state_rows * width * 4));
    }
    double emax = 0, vmax = 0, smax = 0;
    for (size_t i = 0; i < n_all; i++) {
        const double d = fabs(got[0][i] - got[1][i]);
        if (i < (size_t)n_comp * head_dim) { emax = fmax(emax, d); vmax = fmax(vmax, fabs(got[0][i])); } else smax = fmax(smax, d);
    }
    const int pass = emax / vmax < 2e-5 && smax == 0;
    printf("compressor_chain ratio=%-3u head=%-3u tl|hand comp %.1e state %.1e (%u tokens, %u emits)  %s\n", ratio, head_dim, emax / vmax, smax, n_tok, n_comp, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        const int reps = bench_iters / (int)n_tok + 1;
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("KV", which) && ds4_gpu_tl_force("RMS_NORM", which) && ds4_gpu_tl_force("ROPE", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < reps; it++) for (uint32_t t = 0; t < n_tok; t++) {
                    ds4_gpu_tensor *vkv = ds4_gpu_tensor_view(tkv, (size_t)t * width * 4, width * 4), *vsc = ds4_gpu_tensor_view(tsc, (size_t)t * width * 4, width * 4);
                    CHECK(ds4_gpu_compressor_update_tensor(vkv, vsc, tskv, tssc, tcomp, map, map_bytes, off_ape, 1, off_norm, 0, head_dim, ratio, t, t / ratio,
                                                           64, 65536, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-6f, false, false, false));
                    ds4_gpu_tensor_free(vkv); ds4_gpu_tensor_free(vsc);
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / (reps * n_tok);
            }
            CHECK(ds4_gpu_tl_force("KV", 0) && ds4_gpu_tl_force("RMS_NORM", 0) && ds4_gpu_tl_force("ROPE", 0));
        }
        printf("compressor_chain ratio=%-3u head=%-3u bench x%d per token: hand %7.2f us  tl %7.2f us  tl/hand %.2fx\n", ratio, head_dim, reps * n_tok, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc); ds4_gpu_tensor_free(tskv); ds4_gpu_tensor_free(tssc); ds4_gpu_tensor_free(tcomp);
    free(kv); free(sc); free(got[0]); free(got[1]); free(map);
    return 1;
}

/* Indexed mixed attention (long context): per token the raw window rows ++
 * the indexer's top-k compressed rows (sorted ascending for prefill, score
 * order for decode; -1 holes and invisible indices included) with sinks.
 * TileLang (one dispatch; decode: 12 splits + reduce) vs hand-written
 * (heads16_dual / heads8_split). Also a double reference on a few rows. */
static int check_indexed_attention(uint32_t n_tokens, uint32_t pos0, uint32_t n_comp, int bench_iters) {
    const uint32_t n_head = 64, hd = 512, top_k = 512, window = 128, ratio = 4, raw_cap = 4352;
    const uint32_t last_pos = pos0 + n_tokens - 1;
    const uint32_t n_raw = n_tokens + window < raw_cap ? n_tokens + window : raw_cap;
    const uint32_t raw_start = (pos0 * 7u) % raw_cap;
    const uint32_t first_raw_pos = last_pos + 1 - n_raw;
    const size_t map_bytes = (size_t)getpagesize() * 2;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), map_bytes) == 0);
    memset(map, 0, map_bytes);
    float *sinks = (float *)(map + 256);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = frand() * 2.0f;
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *q = malloc((size_t)n_tokens * n_head * hd * 4), *raw = malloc((size_t)raw_cap * hd * 4);
    uint16_t *comp = malloc((size_t)n_comp * hd * 2);
    int32_t *topk = malloc((size_t)n_tokens * top_k * 4);
    CHECK(q && raw && comp && topk);
    for (size_t i = 0; i < (size_t)n_tokens * n_head * hd; i++) q[i] = frand() * 0.5f;
    for (size_t i = 0; i < (size_t)raw_cap * hd; i++) raw[i] = frand();
    for (size_t i = 0; i < (size_t)n_comp * hd; i++) comp[i] = f32_to_f16_bits(frand());
    /* top-k rows: distinct indices in [0, n_comp) in score order, a few -1 holes; prefill
     * rows get sorted ascending by the host (the kernel relies on it). */
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t *row = topk + (size_t)t * top_k;
        uint32_t used = 0;
        while (used < top_k) {
            const int32_t cand = (int32_t)((seed = seed * 1664525u + 1013904223u) % n_comp);
            int dup = 0;
            for (uint32_t j = 0; j < used && !dup; j++) dup = row[j] == cand;
            if (!dup) row[used++] = cand;
        }
        for (uint32_t j = 0; j < top_k; j += 97) row[j] = -1;
    }
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_head * hd * 4), *traw = ds4_gpu_tensor_alloc_managed((size_t)raw_cap * hd * 4);
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc_managed((size_t)n_comp * hd * 2), *ttopk = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * top_k * 4);
    ds4_gpu_tensor *theads = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_head * hd * 4);
    CHECK(tq && traw && tcomp && ttopk && theads);
    CHECK(ds4_gpu_tensor_write(tq, 0, q, (size_t)n_tokens * n_head * hd * 4) && ds4_gpu_tensor_write(traw, 0, raw, (size_t)raw_cap * hd * 4) &&
          ds4_gpu_tensor_write(tcomp, 0, comp, (size_t)n_comp * hd * 2) && ds4_gpu_tensor_write(ttopk, 0, topk, (size_t)n_tokens * top_k * 4));
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc((size_t)n_tokens * n_head * hd, 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("ATTN", which) && ds4_gpu_tl_force("KV", which));
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(theads, map, map_bytes, 256, tq, traw, tcomp, 1, ttopk, n_tokens, pos0, n_raw, raw_cap, raw_start,
                                                                 n_comp, top_k, window, ratio, n_head, hd));
        CHECK(ds4_gpu_tl_force("ATTN", 0) && ds4_gpu_tl_force("KV", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(theads, 0, got[which], (size_t)n_tokens * n_head * hd * 4));
    }
    /* double reference, f16-rounded Q and rows like the GPU, for a few (token, head) rows */
    double rmax = 0;
    for (uint32_t t = 0; t < n_tokens; t += (n_tokens > 8 ? n_tokens / 8 : 1)) {
        const uint32_t qpos = pos0 + t;
        const uint32_t first = first_raw_pos > qpos + 1 - window ? first_raw_pos : qpos + 1 - window;
        const uint32_t visible = (qpos + 1) / ratio < n_comp ? (qpos + 1) / ratio : n_comp;
        for (uint32_t h = 0; h < n_head; h += 21) {
            double m = -1e300, l = 0; double *acc = calloc(hd, sizeof(double)); CHECK(acc);
            const float *qr = q + ((size_t)t * n_head + h) * hd;
            float qh[512];
            for (uint32_t d = 0; d < hd; d++) qh[d] = f16_bits_to_f32(f32_to_f16_bits(qr[d]));
            #define ATTEND_ROW(getter) do { double sdot = 0; float rowv[512]; for (uint32_t d = 0; d < hd; d++) { rowv[d] = getter; sdot += (double)qh[d] * rowv[d]; } \
                sdot /= sqrt((double)hd); const double mn = sdot > m ? sdot : m; const double a = exp(m - mn), e = exp(sdot - mn); \
                for (uint32_t d = 0; d < hd; d++) acc[d] = acc[d] * a + e * rowv[d]; l = l * a + e; m = mn; } while (0)
            for (uint32_t pos = first; pos <= qpos; pos++) {
                const uint32_t rr = (raw_start + pos - first_raw_pos) % raw_cap;
                ATTEND_ROW(f16_bits_to_f32(f32_to_f16_bits(raw[(size_t)rr * hd + d])));
            }
            for (uint32_t j = 0; j < top_k; j++) {
                const int32_t idx = topk[(size_t)t * top_k + j];
                if (idx < 0 || (uint32_t)idx >= visible) continue;
                ATTEND_ROW(f16_bits_to_f32(comp[(size_t)idx * hd + d]));
            }
            #undef ATTEND_ROW
            { const double mn = sinks[h] > m ? sinks[h] : m; const double a = exp(m - mn); for (uint32_t d = 0; d < hd; d++) acc[d] *= a; l = l * a + exp(sinks[h] - mn); }
            for (uint32_t d = 0; d < hd; d++) {
                const double ref = acc[d] / l;
                rmax = fmax(rmax, fabs(ref - got[1][((size_t)t * n_head + h) * hd + d]) / (1.0 + fabs(ref)));
            }
            free(acc);
        }
    }
    double emax = 0, vmax = 0;
    for (size_t i = 0; i < (size_t)n_tokens * n_head * hd; i++) { emax = fmax(emax, fabs(got[0][i] - got[1][i])); vmax = fmax(vmax, fabs(got[0][i])); }
    const int pass = emax / vmax < 2e-4 && rmax < 1e-3;
    if (!pass) for (int i = 0; i < 6; i++) printf("  [%d] hand %g tl %g   [tail %d] hand %g tl %g\n", i, got[0][i], got[1][i], i, got[0][(size_t)n_tokens * n_head * hd - 1 - i], got[1][(size_t)n_tokens * n_head * hd - 1 - i]);
    printf("indexed_attn     n_tok=%-4u pos0=%-5u n_comp=%-5u tl|hand %.1e  tl|ref %.1e  %s\n", n_tokens, pos0, n_comp, emax / vmax, rmax, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        const int reps = n_tokens > 1 ? bench_iters / 20 + 1 : bench_iters;
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ATTN", which) && ds4_gpu_tl_force("KV", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < reps; it++)
                    CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(theads, map, map_bytes, 256, tq, traw, tcomp, 1, ttopk, n_tokens, pos0, n_raw, raw_cap, raw_start,
                                                                             n_comp, top_k, window, ratio, n_head, hd));
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / reps;
            }
            CHECK(ds4_gpu_tl_force("ATTN", 0) && ds4_gpu_tl_force("KV", 0));
        }
        printf("indexed_attn     n_tok=%-4u bench x%d: hand %9.1f us  tl %9.1f us  tl/hand %.2fx\n", n_tokens, reps, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(traw); ds4_gpu_tensor_free(tcomp); ds4_gpu_tensor_free(ttopk); ds4_gpu_tensor_free(theads);
    free(q); free(raw); free(comp); free(topk); free(got[0]); free(got[1]); free(map);
    return 1;
}

/* Lightning indexer scores: scale * sum_h relu(q[t,h].k[c]) * w[t,h] with the
 * causal -inf mask (prefill: pos0 chunk; decode: one token, no mask).
 * TileLang vs hand (NAX / tiled / score_one_direct) vs a double reference
 * with f16-rounded q and k (the GPU kernels' operand precision). */
static int check_indexer_scores(uint32_t n_tokens, uint32_t pos0, uint32_t n_comp, int bench_iters) {
    const uint32_t n_head = 64, hd = 128, ratio = 4;
    const float scale = 1.0f / sqrtf((float)(hd * n_head));
    float *q = malloc((size_t)n_tokens * n_head * hd * 4), *w = malloc((size_t)n_tokens * n_head * 4), *k = malloc((size_t)n_comp * hd * 4);
    CHECK(q && w && k);
    for (size_t i = 0; i < (size_t)n_tokens * n_head * hd; i++) q[i] = frand();
    for (size_t i = 0; i < (size_t)n_tokens * n_head; i++) w[i] = frand() + 1.5f;
    for (size_t i = 0; i < (size_t)n_comp * hd; i++) k[i] = frand();
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_head * hd * 4), *tw = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_head * 4);
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc_managed((size_t)n_comp * hd * 4), *ts = ds4_gpu_tensor_alloc_managed((size_t)n_tokens * n_comp * 4);
    CHECK(tq && tw && tk && ts && ds4_gpu_tensor_write(tq, 0, q, (size_t)n_tokens * n_head * hd * 4) && ds4_gpu_tensor_write(tw, 0, w, (size_t)n_tokens * n_head * 4) &&
          ds4_gpu_tensor_write(tk, 0, k, (size_t)n_comp * hd * 4));
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc((size_t)n_tokens * n_comp, 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("ATTN", which));
        if (n_tokens == 1) CHECK(ds4_gpu_indexer_score_one_tensor(ts, tq, tw, tk, n_comp, n_head, hd, scale));
        else CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(ts, tq, tw, tk, n_comp, n_tokens, pos0, n_head, hd, ratio, scale));
        CHECK(ds4_gpu_tl_force("ATTN", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(ts, 0, got[which], (size_t)n_tokens * n_comp * 4));
    }
    double emax = 0, vmax = 0, rmax = 0; size_t mask_mismatch = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t visible = n_tokens == 1 ? n_comp : ((pos0 + t + 1) / ratio < n_comp ? (pos0 + t + 1) / ratio : n_comp);
        for (uint32_t c = 0; c < n_comp; c++) {
            const float a = got[0][(size_t)t * n_comp + c], b = got[1][(size_t)t * n_comp + c];
            if (c >= visible) { if (!(a < -1e37f) || !(b < -1e37f)) mask_mismatch++; continue; }
            emax = fmax(emax, fabs(a - b)); vmax = fmax(vmax, fabs(a));
            if ((t % 37) == 0 && (c % 13) == 0) {
                double acc = 0;
                for (uint32_t h = 0; h < n_head; h++) {
                    double d = 0;
                    for (uint32_t e = 0; e < hd; e++) d += (double)f16_bits_to_f32(f32_to_f16_bits(q[((size_t)t * n_head + h) * hd + e])) * f16_bits_to_f32(f32_to_f16_bits(k[(size_t)c * hd + e]));
                    acc += (d > 0 ? d : 0) * w[(size_t)t * n_head + h];
                }
                acc *= scale;
                rmax = fmax(rmax, fabs(acc - b) / (1.0 + fabs(acc)));
            }
        }
    }
    const int pass = emax / vmax < (n_tokens == 1 ? 5e-3 : 1e-4) && rmax < 1e-3 && mask_mismatch == 0;
    printf("indexer_scores   n_tok=%-4u pos0=%-5u n_comp=%-5u tl|hand %.1e  tl|ref %.1e  mask mismatches %zu  %s\n", n_tokens, pos0, n_comp, emax / vmax, rmax, mask_mismatch, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        const int reps = n_tokens > 1 ? bench_iters / 10 + 1 : bench_iters;
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ATTN", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < reps; it++) {
                    if (n_tokens == 1) CHECK(ds4_gpu_indexer_score_one_tensor(ts, tq, tw, tk, n_comp, n_head, hd, scale));
                    else CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(ts, tq, tw, tk, n_comp, n_tokens, pos0, n_head, hd, ratio, scale));
                }
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / reps;
            }
            CHECK(ds4_gpu_tl_force("ATTN", 0));
        }
        printf("indexer_scores   n_tok=%-4u bench x%d: hand %9.1f us  tl %9.1f us  tl/hand %.2fx\n", n_tokens, reps, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(ts);
    free(q); free(w); free(k); free(got[0]); free(got[1]);
    return 1;
}

static int cmp_i32(const void *a, const void *b) { const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b; return (x > y) - (x < y); }
static int cmp_f32_desc(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x < y) - (x > y); }

/* Indexer top-k: TileLang radix select vs the hand block sort + merge. Rows
 * mimic indexer scores: finite values with a causal -inf tail (and ties).
 * Compared as sets of indices plus the multiset of selected values. */
static int check_indexer_topk(uint32_t n_rows, uint32_t width, uint32_t top_k, int bench_iters) {
    float *sc = malloc((size_t)n_rows * width * 4); CHECK(sc);
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint32_t visible = r % 3 == 0 ? width : (uint32_t)(width * 0.3) + r % 500;
        for (uint32_t c = 0; c < width; c++) sc[(size_t)r * width + c] = c < visible ? (r % 5 == 0 ? (float)(c % 7) : frand() * 3.0f) : neg_inf_f32();
    }
    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc_managed((size_t)n_rows * width * 4), *tsel = ds4_gpu_tensor_alloc_managed((size_t)n_rows * top_k * 4);
    CHECK(ts && tsel && ds4_gpu_tensor_write(ts, 0, sc, (size_t)n_rows * width * 4));
    int32_t *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc((size_t)n_rows * top_k, 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("ATTN", which));
        CHECK(ds4_gpu_indexer_topk_tensor(tsel, ts, width, n_rows, top_k));
        CHECK(ds4_gpu_tl_force("ATTN", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tsel, 0, got[which], (size_t)n_rows * top_k * 4));
    }
    size_t bad_rows = 0, dup_rows = 0;
    float *va = malloc(top_k * 4), *vb = malloc(top_k * 4); int32_t *ia = malloc(top_k * 4); CHECK(va && vb && ia);
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t j = 0; j < top_k; j++) {
            const int32_t a = got[0][(size_t)r * top_k + j], b = got[1][(size_t)r * top_k + j];
            if (a < 0 || (uint32_t)a >= width || b < 0 || (uint32_t)b >= width) { bad_rows++; break; }
            va[j] = sc[(size_t)r * width + a]; vb[j] = sc[(size_t)r * width + b]; ia[j] = b;
        }
        qsort(va, top_k, 4, cmp_f32_desc); qsort(vb, top_k, 4, cmp_f32_desc); qsort(ia, top_k, 4, cmp_i32);
        int same = 1;
        for (uint32_t j = 0; j < top_k && same; j++) same = va[j] == vb[j] || (isinf(va[j]) && isinf(vb[j]));
        if (!same) bad_rows++;
        for (uint32_t j = 1; j < top_k; j++) if (ia[j] == ia[j - 1]) { dup_rows++; break; }
    }
    const int pass = bad_rows == 0 && dup_rows == 0;
    printf("indexer_topk     rows=%-4u width=%-5u k=%-3u value-set mismatches %zu, duplicate rows %zu  %s\n", n_rows, width, top_k, bad_rows, dup_rows, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        const int reps = bench_iters / 4 + 1;
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("ATTN", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < reps; it++) CHECK(ds4_gpu_indexer_topk_tensor(tsel, ts, width, n_rows, top_k));
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / reps;
            }
            CHECK(ds4_gpu_tl_force("ATTN", 0));
        }
        printf("indexer_topk     rows=%-4u width=%-5u bench x%d: hand %9.1f us  tl %9.1f us  tl/hand %.2fx\n", n_rows, width, reps, tm[0], tm[1], tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(ts); ds4_gpu_tensor_free(tsel); free(sc); free(got[0]); free(got[1]); free(va); free(vb); free(ia);
    return 1;
}

/* Prefill HC mixer: mix[t][24] = rms_norm(x_t) . fn^T over 16384 inputs.
 * Hand: rms_norm pass + f16 GEMV batch; TileLang: one GEMM with the row
 * |x|^2 fused. Compared to each other and to a double reference with the
 * f16 operand rounding of the GEMM path. */
static int check_hc_mix_gemm(uint32_t n_rows, int bench_iters) {
    const uint32_t in_dim = 16384, mix_dim = 24;
    const size_t w_bytes = (size_t)in_dim * mix_dim * 2;
    const size_t page = (size_t)getpagesize();
    const size_t map_bytes = (64 + w_bytes + page - 1) / page * page;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, page, map_bytes) == 0);
    memset(map, 0, map_bytes);
    for (size_t i = 0; i < w_bytes; i += 2) { const uint16_t h = f32_to_f16_bits(frand() * 0.02f); memcpy(map + 64 + i, &h, 2); }
    CHECK(ds4_gpu_set_model_map(map, map_bytes));
    float *x = malloc((size_t)n_rows * in_dim * 4); CHECK(x);
    for (size_t i = 0; i < (size_t)n_rows * in_dim; i++) x[i] = frand() * (1.0f + (float)(i % 7));
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_managed((size_t)n_rows * in_dim * 4), *tscr = ds4_gpu_tensor_alloc_managed((size_t)n_rows * in_dim * 4);
    ds4_gpu_tensor *tmix = ds4_gpu_tensor_alloc_managed((size_t)n_rows * mix_dim * 4);
    CHECK(tx && tscr && tmix && ds4_gpu_tensor_write(tx, 0, x, (size_t)n_rows * in_dim * 4));
    float *got[2];
    for (int which = 1; which >= 0; which--) {
        got[which] = calloc((size_t)n_rows * mix_dim, 4); CHECK(got[which]);
        CHECK(ds4_gpu_tl_force("HC", which) && ds4_gpu_tl_force("GEMM", which) && ds4_gpu_tl_force("RMS_NORM", which) && ds4_gpu_tl_force("F16_GEMV", which));
        CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(tmix, tscr, map, map_bytes, 64, in_dim, mix_dim, tx, n_rows, 1e-6f));
        CHECK(ds4_gpu_tl_force("HC", 0) && ds4_gpu_tl_force("GEMM", 0) && ds4_gpu_tl_force("RMS_NORM", 0) && ds4_gpu_tl_force("F16_GEMV", 0));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(tmix, 0, got[which], (size_t)n_rows * mix_dim * 4));
    }
    double emax = 0, vmax = 0, rmax = 0;
    for (uint32_t t = 0; t < n_rows; t += (n_rows > 16 ? n_rows / 16 : 1)) {
        double ss = 0;
        for (uint32_t k = 0; k < in_dim; k++) ss += (double)x[(size_t)t * in_dim + k] * x[(size_t)t * in_dim + k];
        const double sc = 1.0 / sqrt(ss / in_dim + 1e-6);
        for (uint32_t r = 0; r < mix_dim; r++) {
            double d = 0;
            for (uint32_t k = 0; k < in_dim; k++) {
                uint16_t h; memcpy(&h, map + 64 + ((size_t)r * in_dim + k) * 2, 2);
                d += (double)f16_bits_to_f32(h) * f16_bits_to_f32(f32_to_f16_bits(x[(size_t)t * in_dim + k] * (float)sc));
            }
            rmax = fmax(rmax, fabs(d - got[1][(size_t)t * mix_dim + r]) / (1.0 + fabs(d)));
        }
    }
    for (size_t i = 0; i < (size_t)n_rows * mix_dim; i++) { emax = fmax(emax, fabs(got[0][i] - got[1][i])); vmax = fmax(vmax, fabs(got[0][i])); }
    const int pass = emax / vmax < 5e-3 && rmax < 5e-3;   /* f16 operand rounding: hand rounds rms_norm(x), TL rounds x and scales after */
    printf("hc_mix_gemm      rows=%-5u tl|hand %.1e  tl|ref %.1e  %s\n", n_rows, emax / vmax, rmax, pass ? "OK" : "FAIL");
    if (!pass) g_failures++;
    if (bench_iters > 0) {
        double tm[2] = { 0, 0 };
        const int reps = bench_iters / 20 + 1;
        for (int which = 0; which < 2; which++) {
            CHECK(ds4_gpu_tl_force("HC", which) && ds4_gpu_tl_force("GEMM", which) && ds4_gpu_tl_force("RMS_NORM", which) && ds4_gpu_tl_force("F16_GEMV", which));
            for (int pass2 = 0; pass2 < 2; pass2++) {
                const double t0 = now_us();
                CHECK(ds4_gpu_begin_commands());
                for (int it = 0; it < reps; it++) CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(tmix, tscr, map, map_bytes, 64, in_dim, mix_dim, tx, n_rows, 1e-6f));
                CHECK(ds4_gpu_end_commands());
                if (pass2 == 1) tm[which] = (now_us() - t0) / reps;
            }
            CHECK(ds4_gpu_tl_force("HC", 0) && ds4_gpu_tl_force("GEMM", 0) && ds4_gpu_tl_force("RMS_NORM", 0) && ds4_gpu_tl_force("F16_GEMV", 0));
        }
        const double bytes = (double)n_rows * in_dim * 4;
        printf("hc_mix_gemm      rows=%-5u bench x%d: hand %9.1f us (%5.0f GB/s of x)  tl %9.1f us (%5.0f GB/s)  tl/hand %.2fx\n", n_rows, reps, tm[0], bytes / tm[0] / 1e3, tm[1], bytes / tm[1] / 1e3, tm[1] / tm[0]);
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tscr); ds4_gpu_tensor_free(tmix); free(x); free(got[0]); free(got[1]); free(map);
    return 1;
}

int main(void) {
    const char *iters_env = getenv("DS4_TL_BENCH_ITERS");
    const int iters = iters_env ? atoi(iters_env) : 200;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "test_tl_metal: no Metal device\n");
        return 1;
    }
    if (getenv("DS4_TL_ONLY_MOE")) {   /* quick iteration on the packed MoE GEMMs */
        if (!check_moe_batch(600, iters / 100 + 1)) return 1;
        if (!check_moe_batch(1011, iters / 100 + 1)) return 1;
        return g_failures ? 1 : 0;
    }
    if (getenv("DS4_TL_ONLY_INDEXED")) {   /* quick iteration on the indexed attention kernel */
        if (!check_indexed_attention(256, 4096, 1100, iters)) return 1;
        if (!check_indexed_attention(1, 5000, 1250, iters)) return 1;
        if (!check_indexer_scores(1011, 4096, 1276, iters)) return 1;
        if (!check_indexer_scores(1, 0, 1250, iters)) return 1;
        if (!check_indexer_topk(1011, 1276, 512, iters)) return 1;
        if (!check_indexer_topk(1, 1250, 512, iters)) return 1;
        return g_failures ? 1 : 0;
    }

    const uint32_t widths[] = {16384, 4096, 2048, 512, 128};
    const uint32_t row_counts[] = {1, 7, 64, 512};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        for (size_t ri = 0; ri < sizeof(row_counts) / sizeof(*row_counts); ri++) {
            const uint32_t n = widths[wi], rows = row_counts[ri];
            if (n == 16384 && rows == 512) continue; /* 32 MB x 2; nothing new */
            /* 32-byte offset: GGUF tensor alignment. */
            if (!check_rms_norm(n, rows, 32 * 7, 1, rows == 1 || rows == 64 ? iters : 0)) return 1;
            if (!check_rms_norm(n, rows, 0, 0, rows == 1 || rows == 64 ? iters : 0)) return 1;
        }
    }
    /* Alignment and specialisation fallbacks. */
    if (!check_rms_norm(4096, 3, 32 * 7 + 4, 1, 0)) return 1;   /* 4-byte offset: must decline */
    if (!check_rms_norm(4096, 3, 32 * 7 + 16, 1, 0)) return 1;  /* 16-byte: fine */
    if (!check_rms_norm(1024, 3, 32 * 7, 1, 0)) return 1;       /* no 1024 variant: must decline */

    /* Q8_0 decode matvecs of DeepSeek V4 Flash: attn_q_b (1024 -> 32768, per
     * layer), lm_head (4096 -> 129280), plus a small and an odd shape. */
    if (!check_q8_0_gemv(1024, 32768, iters / 5)) return 1;
    if (!check_q8_0_gemv(2048, 4096, iters / 5)) return 1;       /* shared expert down */
    if (!check_q8_0_gate_up_swiglu(4096, 2048, iters / 5)) return 1;
    if (!check_q8_0_gemv(4096, 4096, iters / 5)) return 1;
    if (!check_q8_0_gemv(4096, 129280, iters / 25 > 0 ? iters / 25 : 1)) return 1;
    if (!check_q8_0_gemv(1024, 24, 0)) return 1;
    if (!check_q8_0_gemv(4096, 6, 0)) return 1;      /* out_dim % 4 != 0: must decline */
    if (!check_q8_0_gemv(2048, 64, 0)) return 1;     /* no 2048 variant: must decline */

    /* routed MoE gate/up: 6 of 16 experts here (the model has 256; the
     * kernels index experts through ids, so 16 keeps the host fill cheap). */
    if (!check_moe_gate_up_iq2(4096, 2048, 16, 6, iters / 5)) return 1;
    if (!check_moe_down_q2k(2048, 4096, 16, 6, iters / 5)) return 1;
    /* attn_output_a: 8 head groups x (4096 -> 1024). */
    if (!check_q8_0_grouped(4096, 1024, 8, iters / 5)) return 1;
    /* attn_output_b + HC expand (8192 -> 4096, n_hc 4). */
    if (!check_q8_0_hc_expand(8192, 4096, iters / 5)) return 1;

    /* elementwise family */
    if (!check_elementwise(2048, iters)) return 1;
    if (!check_elementwise(2048 * 37 + 5, iters / 5)) return 1;
    if (!check_utility(4096, 1, iters)) return 1;
    if (!check_utility(4096, 37, iters / 5)) return 1;
    if (!check_dsv4_small(1, iters)) return 1;
    if (!check_dsv4_small(37, iters / 5)) return 1;
    if (!check_router(iters)) return 1;
    if (!check_compressor_small(1, iters)) return 1;
    if (!check_compressor_small(64, iters / 5)) return 1;
    /* small-batch tier: ds4's ext kernels (2..16) and the legacy f16 GEMM (9..31, out % 64 != 0) */
    if (!check_batch_gemv(0, 4096, 1024, 4, iters / 5)) return 1;
    if (!check_batch_gemv(0, 1024, 32768, 8, iters / 10)) return 1;
    if (!check_batch_gemv(0, 4096, 2048, 13, iters / 5)) return 1;
    if (!check_batch_gemv(0, 2048, 4096, 16, iters / 5)) return 1;
    if (!check_batch_gemv(0, 4096, 1024, 31, iters / 5)) return 1;
    if (!check_batch_gemv(0, 8192, 4096, 24, iters / 10)) return 1;
    if (!check_batch_gemv(0, 4096, 1024, 24, iters / 10)) return 1;
    if (!check_batch_gemv(0, 1024, 32768, 20, iters / 10)) return 1;
    if (!check_batch_gemv(1, 16384, 24, 1, iters / 5)) return 1;
    if (!check_batch_gemv(1, 16384, 24, 12, iters / 5)) return 1;
    if (!check_batch_gemv(1, 4096, 512, 1, iters / 5)) return 1;
    if (!check_batch_gemv(1, 4096, 1024, 6, iters / 5)) return 1;
    if (!check_batch_gemv(1, 1024, 8192, 16, iters / 5)) return 1;
    if (!check_batch_gemv(1, 4096, 24, 31, iters / 5)) return 1;
    if (!check_batch_gemv(1, 4096, 1024, 1011, iters / 20)) return 1;   /* f16 GEMM tier: compressor projections, tail chunk */
    if (!check_batch_gemv(1, 4096, 256, 1011, iters / 20)) return 1;    /* router / indexer compressor */
    if (!check_batch_gemv(1, 1024, 8192, 1011, iters / 20)) return 1;   /* indexer q_b */
    if (!check_batch_gemv(1, 4096, 64, 1011, iters / 20)) return 1;     /* indexer proj */
    if (!check_batch_gemv(1, 4096, 1024, 4096, iters / 40)) return 1;   /* full chunk (hand: mpp direct rhs n128) */
    if (!check_batch_gemv(1, 1024, 8192, 4096, iters / 40)) return 1;
    if (!check_decode_attention(37, 0, 0, iters)) return 1;        /* raw-only layer */
    if (!check_decode_attention(128, 0, 0, iters)) return 1;
    if (!check_decode_attention(128, 50, 0, iters)) return 1;      /* gathered, n_keys % 32 != 0 */
    if (!check_decode_attention(128, 896, 0, iters / 2)) return 1; /* 1024 keys: packed kernel limit */
    if (!check_decode_attention(128, 2000, 0, iters / 5)) return 1;
    if (!check_decode_attention(128, 50, 1, iters)) return 1;      /* with the deferred inverse RoPE */
    if (!check_decode_attention(37, 0, 1, 0)) return 1;
    if (!check_prefill_attention(24, 0, iters / 5)) return 1;      /* vec path (< 20 keys goes vec; 24 -> non-vec) */
    if (!check_prefill_attention(12, 4, iters / 5)) return 1;      /* vec path, mixed */
    if (!check_prefill_attention(100, 4, iters / 10)) return 1;    /* non-vec, mixed, n_q % 8 != 0 */
    if (!check_prefill_attention(256, 128, iters / 20)) return 1;
    if (!check_prefill_attention(700, 4, iters / 50 > 0 ? iters / 50 : 1)) return 1;
    if (!check_moe_batch(3, iters / 10)) return 1;
    if (!check_moe_batch(8, iters / 10)) return 1;
    if (!check_moe_batch(24, iters / 20)) return 1;
    if (!check_moe_batch(600, iters / 100 + 1)) return 1;    /* packed GEMM tier (>= 512 tokens) */
    if (!check_moe_batch(1011, iters / 100 + 1)) return 1;
    if (!check_fused_decode(4, iters)) return 1;
    if (!check_fused_decode(128, iters)) return 1;
    if (!check_batch_schedule(4, 3, iters)) return 1;
    if (!check_batch_schedule(128, 5, iters)) return 1;
    if (!check_hc_rms_norm_mix(iters)) return 1;
    if (!check_hc_mix_gemm(1011, iters)) return 1;
    if (!check_hc_mix_gemm(4096, iters)) return 1;
    if (!check_compressor_chain(4, 512, iters)) return 1;
    if (!check_compressor_chain(4, 128, iters)) return 1;
    if (!check_compressor_chain(128, 512, iters)) return 1;
    if (!check_compressor_chain(128, 128, iters)) return 1;
    if (!check_indexed_attention(256, 4096, 1100, iters)) return 1;
    if (!check_indexed_attention(1011, 4096, 1276, iters)) return 1;
    if (!check_indexed_attention(1, 5000, 1250, iters)) return 1;
    if (!check_indexer_scores(1011, 4096, 1276, iters)) return 1;
    if (!check_indexer_scores(4096, 0, 1024, iters)) return 1;
    if (!check_indexer_scores(1, 0, 1250, iters)) return 1;
    if (!check_indexer_topk(1011, 1276, 512, iters)) return 1;
    if (!check_indexer_topk(4096, 1024, 512, iters)) return 1;
    if (!check_indexer_topk(64, 8194, 512, iters)) return 1;
    if (!check_indexer_topk(1, 1250, 512, iters)) return 1;
    /* HC family */
    if (!check_hc(1, 4096, iters)) return 1;
    if (!check_hc(37, 4096, iters / 5)) return 1;
    /* RoPE family: decode (1 token) and prefill shapes, with and without YaRN */
    if (!check_rope(1, 64, 512, 64, 0, 1, iters)) return 1;
    if (!check_rope(1, 1, 512, 64, 0, 0, iters)) return 1;
    if (!check_rope(1, 64, 128, 64, 0, 1, iters)) return 1;
    if (!check_rope(37, 64, 512, 64, 0, 1, iters / 5)) return 1;
    if (!check_rope(1, 64, 512, 64, 1, 1, iters)) return 1;
    if (!check_rope(37, 64, 512, 64, 1, 0, iters / 5)) return 1;
    /* prefill GEMM tier: aligned and unaligned token counts, small shapes
     * for the correctness pass (the reference is O(n_tok*in*out) on the CPU) */
    if (!check_q8_0_gemm(4096, 256, 128, 0)) return 1;
    if (!check_q8_0_gemm(1024, 512, 200, 0)) return 1;      /* padded tail */
    if (!check_q8_0_gemm(2048, 128, 64, 0)) return 1;
    if (!check_q8_0_gemm(8192, 128, 100, 0)) return 1;
    if (!check_q8_0_gemm(4096, 128, 32, 0)) return 1;       /* below the tier threshold: declines */
    /* prefill attention output: 2 groups keep the host reference cheap */
    if (!check_attn_out_batch(4096, 128, 2, 256, 100, 0, 1)) return 1;
    if (!check_attn_out_batch(4096, 128, 2, 256, 13, 0, 1)) return 1;    /* small batch: grouped GEMV with token tiles */
    if (!check_attn_out_batch(4096, 1024, 8, 4096, 24, iters / 20, 0)) return 1;
    if (getenv("DS4_TL_GEMM_BENCH")) {
        const uint32_t ntok = (uint32_t)atoi(getenv("DS4_TL_GEMM_BENCH"));
        /* timing only (the double reference would take minutes at these sizes) */
        if (!bench_q8_0_gemm_hand(4096, 1024, ntok, 10)) return 1;     /* attn_q_a */
        if (!bench_q8_0_gemm_hand(1024, 32768, ntok, 10)) return 1;    /* attn_q_b */
        if (!bench_q8_0_gemm_hand(4096, 2048, ntok, 10)) return 1;     /* shared gate / up */
        if (!bench_q8_0_gemm_hand(2048, 4096, ntok, 10)) return 1;     /* shared down */
        if (!bench_q8_0_gemm_hand(8192, 4096, ntok, 10)) return 1;     /* attn_output_b */
        if (!check_attn_out_batch(4096, 1024, 8, 4096, ntok, 10, 0)) return 1;  /* attn_output_a + b, timing only */
    }

    if (g_failures) {
        printf("test_tl_metal: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("test_tl_metal: all passed\n");
    return 0;
}
