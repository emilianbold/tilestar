"""Q8_0 block primitives for CPU targets.

tl_q8_0_block_acc(acc, w, xq, d):
    acc[0..7] += d * (sum of int8 products of the 32-element block), kept as
    8 int32 lane partials so the only horizontal reduction is one per row
    (the caller reduces acc at the end). `w` points at the 32 quants of a
    Q8_0 block (byte offset 2 inside the 34-byte block, only 2-byte aligned),
    `xq` at the matching 32 prequantised activations.

This mirrors llama.cpp's AVX2 vec_dot_q8_0_q8_0 inner step (vpsignb trick so
vpmaddubsw can be used on signed x signed, vpmaddwd to int32, cvt, fmadd)
and the NEON path of ds4's own dot_q8_0_row (two vdotq_s32 per block).
"""

BLOCK_ACC_FUNC = "tl_q8_0_block_acc"

BLOCK_ACC_SRC = r"""
#ifndef TL_Q8_0_BLOCK_ACC_DEFINED
#define TL_Q8_0_BLOCK_ACC_DEFINED
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
static inline void tl_q8_0_block_acc(float *acc, const uint8_t *w, const int8_t *xq, float d) {
    const __m256i qw = _mm256_loadu_si256((const __m256i *)w);
    const __m256i qx = _mm256_loadu_si256((const __m256i *)xq);
    const __m256i aw = _mm256_sign_epi8(qw, qw);           /* |w| */
    const __m256i sx = _mm256_sign_epi8(qx, qw);           /* x * sign(w) */
    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);      /* 16 x int16 pair sums */
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1)); /* 8 x int32 */
    __m256 a = _mm256_loadu_ps(acc);
    a = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), a);
    _mm256_storeu_ps(acc, a);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
static inline void tl_q8_0_block_acc(float *acc, const uint8_t *w, const int8_t *xq, float d) {
    const int8x16_t w0 = vld1q_s8((const int8_t *)w);
    const int8x16_t w1 = vld1q_s8((const int8_t *)w + 16);
    const int8x16_t x0 = vld1q_s8(xq);
    const int8x16_t x1 = vld1q_s8(xq + 16);
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), w0, x0);
    int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), w1, x1);
#else
    const int16x8_t m0 = vmlal_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)), vget_high_s8(w0), vget_high_s8(x0));
    const int16x8_t m1 = vmlal_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(x1)), vget_high_s8(w1), vget_high_s8(x1));
    int32x4_t d0 = vpaddlq_s16(m0);
    int32x4_t d1 = vpaddlq_s16(m1);
#endif
    float32x4_t a0 = vld1q_f32(acc);
    float32x4_t a1 = vld1q_f32(acc + 4);
    a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(d0), d);
    a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(d1), d);
    vst1q_f32(acc, a0);
    vst1q_f32(acc + 4, a1);
}
#else
static inline void tl_q8_0_block_acc(float *acc, const uint8_t *w, const int8_t *xq, float d) {
    for (int j = 0; j < 8; j++) {
        int32_t s = 0;
        for (int t = 0; t < 4; t++) s += (int32_t)(int8_t)w[j * 4 + t] * (int32_t)xq[j * 4 + t];
        acc[j] += d * (float)s;
    }
}
#endif
#endif /* TL_Q8_0_BLOCK_ACC_DEFINED */
"""
