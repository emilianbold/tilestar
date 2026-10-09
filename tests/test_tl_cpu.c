/* Model-free check of TileLang-generated CPU kernels (gen/cpu/) against the
 * hand-written reference implementations in ds4.c.
 *
 * The generated kernels are the same TileLang sources that produce the Metal
 * (and later CUDA/HIP) kernels, lowered to plain C++ with extern "C" entry
 * points. Passing here validates kernel semantics before any GPU is involved.
 *
 * Build: make test-tl-cpu */
#include "../ds4.c"
#include "../gen/cpu/tl_launch_meta.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return ((float)(rng_state >> 40) / (float)(1ull << 24)) * 2.0f - 1.0f;
}

static void fill(float *p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = frand() * scale;
}

static int compare(const char *what, const float *got, const float *want, size_t n, float tol) {
    double max_abs = 0.0, max_rel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        double r = d / (fabs((double)want[i]) + 1e-6);
        if (d > max_abs) { max_abs = d; worst = i; }
        if (r > max_rel) max_rel = r;
    }
    int ok = max_rel <= tol;
    printf("%-40s n=%zu max_abs=%.3g max_rel=%.3g %s\n", what, n, max_abs, max_rel, ok ? "OK" : "FAIL");
    if (!ok) printf("  worst at %zu: got %.9g want %.9g\n", worst, got[worst], want[worst]);
    return ok;
}

typedef int32_t (*rms_norm_weight_fn)(float *out, float *w, float *x, float eps, int32_t rows);
typedef int32_t (*rms_norm_plain_fn)(float *out, float *x, float eps, int32_t rows);

static int test_rms_norm_weight(const char *name, rms_norm_weight_fn fn, uint32_t n, uint32_t rows) {
    const float eps = 1e-6f;
    float *x = malloc(sizeof(float) * n * rows);
    float *w = malloc(sizeof(float) * n);
    float *got = malloc(sizeof(float) * n * rows);
    float *want = malloc(sizeof(float) * n * rows);
    assert(x && w && got && want);
    fill(x, (size_t)n * rows, 3.0f);
    fill(w, n, 1.0f);
    for (uint32_t r = 0; r < rows; r++)
        rms_norm_weight(want + (size_t)r * n, x + (size_t)r * n, w, n, eps);
    int rc = fn(got, w, x, eps, (int32_t)rows);
    int ok = rc == 0 && compare(name, got, want, (size_t)n * rows, 2e-5f);
    if (rc != 0) printf("%s: kernel returned %d\n", name, rc);
    free(x); free(w); free(got); free(want);
    return ok;
}

static int test_rms_norm_plain(const char *name, rms_norm_plain_fn fn, uint32_t n, uint32_t rows) {
    const float eps = 1e-6f;
    float *x = malloc(sizeof(float) * n * rows);
    float *got = malloc(sizeof(float) * n * rows);
    float *want = malloc(sizeof(float) * n * rows);
    assert(x && got && want);
    fill(x, (size_t)n * rows, 3.0f);
    for (uint32_t r = 0; r < rows; r++)
        rms_norm_no_weight(want + (size_t)r * n, x + (size_t)r * n, n, eps);
    int rc = fn(got, x, eps, (int32_t)rows);
    int ok = rc == 0 && compare(name, got, want, (size_t)n * rows, 2e-5f);
    if (rc != 0) printf("%s: kernel returned %d\n", name, rc);
    free(x); free(got); free(want);
    return ok;
}

/* Random Q8_0 rows in the on-disk layout: f16 d, int8 qs[32] per 32 inputs. */
static uint8_t *make_q8_0_rows(uint32_t out_dim, uint32_t in_dim) {
    const uint32_t blocks = in_dim / 32;
    uint8_t *w = malloc((size_t)out_dim * blocks * 34);
    assert(w);
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + ((size_t)r * blocks + b) * 34;
            uint16_t d = f32_to_f16(0.01f + 0.02f * (frand() + 1.0f));
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) ((int8_t *)(blk + 2))[i] = (int8_t)(frand() * 127.0f);
        }
    }
    return w;
}

typedef int32_t (*q8_gemv_fn)(float *out, uint8_t *w, int8_t *xq, float *xscale, int32_t out_dim);

static int test_q8_0_gemv(const char *name, q8_gemv_fn fn, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32;
    uint8_t *w = make_q8_0_rows(out_dim, in_dim);
    float *x = malloc(sizeof(float) * in_dim);
    int8_t *xq = malloc(blocks * 32);
    float *xscale = malloc(sizeof(float) * blocks);
    float *got = malloc(sizeof(float) * out_dim);
    float *want = malloc(sizeof(float) * out_dim);
    assert(x && xq && xscale && got && want);
    fill(x, in_dim, 2.0f);
    quantize_q8_0_activation(x, xq, xscale, in_dim);
    for (uint32_t r = 0; r < out_dim; r++)
        want[r] = dot_q8_0_row(w + (size_t)r * blocks * 34, xq, xscale, in_dim, blocks);
    int rc = fn(got, w, xq, xscale, (int32_t)out_dim);
    int ok = rc == 0 && compare(name, got, want, out_dim, 1e-4f);
    if (rc != 0) printf("%s: kernel returned %d\n", name, rc);
    free(w); free(x); free(xq); free(xscale); free(got); free(want);
    return ok;
}

typedef int32_t (*q8_gemm_fn)(float *out, uint8_t *w, int8_t *xq, float *xscale, int32_t n_tok, int32_t out_dim);

static int test_q8_0_gemm(const char *name, q8_gemm_fn fn, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t blocks = in_dim / 32;
    uint8_t *w = make_q8_0_rows(out_dim, in_dim);
    float *x = malloc(sizeof(float) * in_dim * n_tok);
    int8_t *xq = malloc((size_t)blocks * 32 * n_tok);
    float *xscale = malloc(sizeof(float) * blocks * n_tok);
    float *got = malloc(sizeof(float) * out_dim * n_tok);
    float *want = malloc(sizeof(float) * out_dim * n_tok);
    assert(x && xq && xscale && got && want);
    fill(x, (size_t)in_dim * n_tok, 2.0f);
    for (uint32_t t = 0; t < n_tok; t++) {
        quantize_q8_0_activation(x + (size_t)t * in_dim, xq + (size_t)t * blocks * 32, xscale + (size_t)t * blocks, in_dim);
        for (uint32_t r = 0; r < out_dim; r++)
            want[(size_t)t * out_dim + r] = dot_q8_0_row(w + (size_t)r * blocks * 34, xq + (size_t)t * blocks * 32,
                                                         xscale + (size_t)t * blocks, in_dim, blocks);
    }
    int rc = fn(got, w, xq, xscale, (int32_t)n_tok, (int32_t)out_dim);
    int ok = rc == 0 && compare(name, got, want, (size_t)out_dim * n_tok, 1e-4f);
    if (rc != 0) printf("%s: kernel returned %d\n", name, rc);
    free(w); free(x); free(xq); free(xscale); free(got); free(want);
    return ok;
}

int main(void) {
    int ok = 1;
#define W(n, rows) ok &= test_rms_norm_weight(TL_RMS_NORM_WEIGHT_F32_##n##_SYMBOL, tl_rms_norm_weight_f32_##n##_kernel, n, rows)
#define P(n, rows) ok &= test_rms_norm_plain(TL_RMS_NORM_PLAIN_F32_##n##_SYMBOL, tl_rms_norm_plain_f32_##n##_kernel, n, rows)
    W(4096, 7);  W(512, 64);  W(2048, 3);
    P(4096, 7);  P(512, 64);  P(2048, 3);
#undef W
#undef P
    ok &= test_q8_0_gemv(TL_Q8_0_GEMV_PREQ_4096_SYMBOL, tl_q8_0_gemv_preq_4096_kernel, 4096, 64);
    ok &= test_q8_0_gemv(TL_Q8_0_GEMV_PREQ_1024_SYMBOL, tl_q8_0_gemv_preq_1024_kernel, 1024, 40);
    ok &= test_q8_0_gemv(TL_Q8_0_GEMV_PREQ_LANES_4096_SYMBOL, tl_q8_0_gemv_preq_lanes_4096_kernel, 4096, 64);
    ok &= test_q8_0_gemv(TL_Q8_0_GEMV_PREQ_LANES_1024_SYMBOL, tl_q8_0_gemv_preq_lanes_1024_kernel, 1024, 40);
    ok &= test_q8_0_gemv(TL_Q8_0_GEMV_PREQ_SIMD_4096_SYMBOL, tl_q8_0_gemv_preq_simd_4096_kernel, 4096, 64);
    ok &= test_q8_0_gemv(TL_Q8_0_GEMV_PREQ_SIMD_1024_SYMBOL, tl_q8_0_gemv_preq_simd_1024_kernel, 1024, 40);
    ok &= test_q8_0_gemm(TL_Q8_0_GEMM_PREQ_SIMD_4096_SYMBOL, tl_q8_0_gemm_preq_simd_4096_kernel, 4096, 64, 7);
    ok &= test_q8_0_gemm(TL_Q8_0_GEMM_PREQ_SIMD_1024_SYMBOL, tl_q8_0_gemm_preq_simd_1024_kernel, 1024, 40, 4);
    printf(ok ? "test_tl_cpu: all passed\n" : "test_tl_cpu: FAILED\n");
    return ok ? 0 : 1;
}
