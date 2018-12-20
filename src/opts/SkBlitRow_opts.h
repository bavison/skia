/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkBlitRow_opts_DEFINED
#define SkBlitRow_opts_DEFINED

#include "Sk4px.h"
#include "SkColorData.h"
#include "SkMSAN.h"

#if SK_CPU_SSE_LEVEL >= SK_CPU_SSE_LEVEL_SSE2
    #include "SkColor_opts_SSE2.h"
    #include <immintrin.h>
#endif

namespace SK_OPTS_NS {

// Color32 uses the blend_256_round_alt algorithm from tests/BlendTest.cpp.
// It's not quite perfect, but it's never wrong in the interesting edge cases,
// and it's quite a bit faster than blend_perfect.
//
// blend_256_round_alt is our currently blessed algorithm.  Please use it or an analogous one.
static inline
void blit_row_color32(SkPMColor* dst, const SkPMColor* src, int count, SkPMColor color) {
    unsigned invA = 255 - SkGetPackedA32(color);
    invA += invA >> 7;
    SkASSERT(invA < 256);  // We've should have already handled alpha == 0 externally.

    Sk16h colorHighAndRound = Sk4px::DupPMColor(color).widenHi() + Sk16h(128);
    Sk16b invA_16x(invA);

    Sk4px::MapSrc(count, dst, src, [&](const Sk4px& src4) -> Sk4px {
        return (src4 * invA_16x).addNarrowHi(colorHighAndRound);
    });
}

#if defined(SK_ARM_HAS_NEON)
#ifdef __ARM_64BIT_STATE
// No attempt has been made to adapt the inline assembly version for AArch64
// so fall back to the less performant version that uses intrinsics instead

// Return a uint8x8_t value, r, computed as r[i] = SkMulDiv255Round(x[i], y[i]), where r[i], x[i],
// y[i] are the i-th lanes of the corresponding NEON vectors.
static inline uint8x8_t SkMulDiv255Round_neon8(uint8x8_t x, uint8x8_t y) {
    uint16x8_t prod = vmull_u8(x, y);
    return vraddhn_u16(prod, vrshrq_n_u16(prod, 8));
}

// The implementations of SkPMSrcOver below perform alpha blending consistently with
// SkMulDiv255Round. They compute the color components (numbers in the interval [0, 255]) as:
//
//   result_i = src_i + rint(g(src_alpha, dst_i))
//
// where g(x, y) = ((255.0 - x) * y) / 255.0 and rint rounds to the nearest integer.

// In this variant of SkPMSrcOver each NEON register, dst.val[i], src.val[i], contains the value
// of the same color component for 8 consecutive pixels. The result of this function follows the
// same convention.
static inline uint8x8x4_t SkPMSrcOver_neon8(uint8x8x4_t dst, uint8x8x4_t src) {
    uint8x8_t nalphas = vmvn_u8(src.val[3]);
    uint8x8x4_t result;
    result.val[0] = vadd_u8(src.val[0], SkMulDiv255Round_neon8(nalphas,  dst.val[0]));
    result.val[1] = vadd_u8(src.val[1], SkMulDiv255Round_neon8(nalphas,  dst.val[1]));
    result.val[2] = vadd_u8(src.val[2], SkMulDiv255Round_neon8(nalphas,  dst.val[2]));
    result.val[3] = vadd_u8(src.val[3], SkMulDiv255Round_neon8(nalphas,  dst.val[3]));
    return result;
}

// In this variant of SkPMSrcOver dst and src contain the color components of two consecutive
// pixels. The return value follows the same convention.
static inline uint8x8_t SkPMSrcOver_neon2(uint8x8_t dst, uint8x8_t src) {
    const uint8x8_t alpha_indices = vcreate_u8(0x0707070703030303);
    uint8x8_t nalphas = vmvn_u8(vtbl1_u8(src, alpha_indices));
    return vadd_u8(src, SkMulDiv255Round_neon8(nalphas, dst));
}

#else // __ARM_64BIT_STATE
// Inline ARM AArch32 assembly version

// Macros to specify instructions to only include if targeting ARM or Thumb instruction sets
#ifdef __thumb__
#define A(x)
#define T(x) x
#else
#define A(x) x
#define T(x)
#endif

// These macros permit optionally-included features to be switched using a parameter to another macro
#define YES(x) x
#define NO(x)

// How far ahead (pixels) to preload (undefine to disable prefetch) - determined empirically
#define PREFETCH_DISTANCE "24"

#ifdef PREFETCH_DISTANCE
#define IF_PRELOAD YES
#else
#define IF_PRELOAD NO
#endif

/// Macro to load or store 1..7 pixels in growing powers-of-2 in size - suitable for leading pixels
#define S32A_LOADSTORE_LEADING_7(ls, r0, r1, r2, r3, base, opt)                                       \
                "tst         %[group_size], #1                                                \n\t"   \
                opt"                                                                          \n\t"   \
                "beq         1f                                                               \n\t"   \
                "v"#ls"4.8   {"#r0"[1],"#r1"[1],"#r2"[1],"#r3"[1]}, [%["#base"]:32]!          \n\t"   \
                "1:                                                                           \n\t"   \
                "lsls        %[tmp], %[group_size], #30                                       \n\t"   \
                "add         %[tmp], %["#base"], #4                                           \n\t"   \
                "bpl         1f                                                               \n\t"   \
                "v"#ls"4.8   {"#r0"[2],"#r1"[2],"#r2"[2],"#r3"[2]}, [%["#base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[3],"#r1"[3],"#r2"[3],"#r3"[3]}, [%[tmp]:32], %[eight]     \n\t"   \
                "1:                                                                           \n\t"   \
                "bcc         1f                                                               \n\t"   \
                "v"#ls"4.8   {"#r0"[4],"#r1"[4],"#r2"[4],"#r3"[4]}, [%["#base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[5],"#r1"[5],"#r2"[5],"#r3"[5]}, [%[tmp]:32], %[eight]     \n\t"   \
                "v"#ls"4.8   {"#r0"[6],"#r1"[6],"#r2"[6],"#r3"[6]}, [%["#base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[7],"#r1"[7],"#r2"[7],"#r3"[7]}, [%[tmp]:32], %[eight]     \n\t"   \
                "1:                                                                           \n\t"   \

/// Macro to load or store 1..7 pixels in shrink powers-of-2 in size - suitable for trailing pixels
#define S32A_LOADSTORE_TRAILING_7(ls, r0, r1, r2, r3, base, opt)                                      \
                "lsls        %[tmp], %[group_size], #30                                       \n\t"   \
                "add         %[tmp], %["#base"], #4                                           \n\t"   \
                opt"                                                                          \n\t"   \
                "bcc         1f                                                               \n\t"   \
                "v"#ls"4.8   {"#r0"[0],"#r1"[0],"#r2"[0],"#r3"[0]}, [%["#base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[1],"#r1"[1],"#r2"[1],"#r3"[1]}, [%[tmp]:32], %[eight]     \n\t"   \
                "v"#ls"4.8   {"#r0"[2],"#r1"[2],"#r2"[2],"#r3"[2]}, [%["#base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[3],"#r1"[3],"#r2"[3],"#r3"[3]}, [%[tmp]:32], %[eight]     \n\t"   \
                "1:                                                                           \n\t"   \
                "bpl         1f                                                               \n\t"   \
                "v"#ls"4.8   {"#r0"[4],"#r1"[4],"#r2"[4],"#r3"[4]}, [%["#base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[5],"#r1"[5],"#r2"[5],"#r3"[5]}, [%[tmp]:32], %[eight]     \n\t"   \
                "1:                                                                           \n\t"   \
                "tst         %[group_size], #1                                                \n\t"   \
                "beq         1f                                                               \n\t"   \
                "v"#ls"4.8   {"#r0"[6],"#r1"[6],"#r2"[6],"#r3"[6]}, [%["#base"]:32]!          \n\t"   \
                "1:                                                                           \n\t"   \

/// Macro to do testing and "over" compositing of a group of 1..7 32bpp premultiplied ARGB leading or trailing pixels
#define S32A_OPAQUE_7PIX_PROCESS(loadstore_7, size)                                                   \
    do {                                                                                              \
        __asm__ volatile (                                                                            \
                /* Load the leading/trailing source pixels,                                           \
                 * after initialising all the unused indexes from the first pixel                     \
                 * so the all-opaque and all-transparent tests still work */                          \
                loadstore_7(ld, d0, d1, d2, d3, src,                                                  \
                "vld4.8      {d0[],d1[],d2[],d3[]}, [%[src]]")                                        \
                /* Test for all-opaque or all-transparent */                                          \
                "vmov        %[alo], s6                                                       \n\t"   \
                "vmov        %[ahi], s7                                                       \n\t"   \
                "vmvn        d31, d3                                                          \n\t"   \
                "orrs        %[tmp], %[alo], %[ahi]                                           \n\t"   \
                "it          eq                                                               \n\t"   \
                "addeq       %[dst], %[group_size], lsl #2                                    \n\t"   \
                "beq         9f                                                               \n\t"   \
                "cmp         %[alo], #-1                                                      \n\t"   \
                "it          eq                                                               \n\t"   \
                "cmpeq       %[ahi], #-1                                                      \n\t"   \
                "beq         5f                                                               \n\t"   \
                /* Translucency used, or a mixture of opaque and transparent */                       \
                loadstore_7(ld, d20, d21, d22, d23, dst, )                                            \
                "sub         %[dst], %[group_size], lsl #2                                    \n\t"   \
                S32A_OPAQUE_8PIX_BLEND(, , q0, q1,, NO, NO, NO)                                       \
                "5:                                                                           \n\t"   \
                loadstore_7(st, d0, d1, d2, d3, dst, )                                                \
                /* Drop through */                                                                    \
                "9:                                                                           \n\t"   \
        : /* Outputs */                                                                               \
                [alo]"=&r"(alo),                                                                      \
                [ahi]"=&r"(ahi),                                                                      \
                [tmp]"=&r"(tmp),                                                                      \
                [src]"+r"(src),                                                                       \
                [dst]"+r"(dst)                                                                        \
        : /* Inputs */                                                                                \
                [group_size]"r"(size),                                                                \
                     [eight]"r"(eight)                                                                \
        : /* Clobbers */                                                                              \
                "cc", "memory"                                                                        \
        );                                                                                            \
    } while (0)

/// Macro to do testing and "over" compositing of an aligned group of 8 32bpp premultiplied ARGB leading or trailing pixels
#define S32A_OPAQUE_8PIX_PROCESS(align, if_load)                                                      \
    do {                                                                                              \
        __asm__ (                                                                                     \
if_load(        "vld4.8      {d0-d3}, [%[src]]!                                               \n\t")  \
                /* Test for all-opaque or all-transparent */                                          \
                "vmov        %[alo], s6                                                       \n\t"   \
                "vmov        %[ahi], s7                                                       \n\t"   \
                "vmvn        d31, d3                                                          \n\t"   \
                "orrs        %[tmp], %[alo], %[ahi]                                           \n\t"   \
                "it          eq                                                               \n\t"   \
                "addeq       %[dst], #8*4                                                     \n\t"   \
                "beq         9f                                                               \n\t"   \
                "cmp         %[alo], #-1                                                      \n\t"   \
                "it          eq                                                               \n\t"   \
                "cmpeq       %[ahi], #-1                                                      \n\t"   \
                "beq         5f                                                               \n\t"   \
                /* Translucency used, or a mixture of opaque and transparent */                       \
                S32A_OPAQUE_8PIX_BLEND(align, , q0, q1, "5:", YES, NO, NO)                            \
                /* Drop through */                                                                    \
                "9:                                                                           \n\t"   \
        :  /* Outputs */                                                                              \
                [alo]"=&r"(alo),                                                                      \
                [ahi]"=&r"(ahi),                                                                      \
                [tmp]"=&r"(tmp),                                                                      \
                [src]"+r"(src),                                                                       \
                [dst]"+r"(dst)                                                                        \
        : /* Inputs */                                                                                \
        : /* Clobbers */                                                                              \
                "cc", "memory"                                                                        \
        );                                                                                            \
    } while (0)

/// Macro to do "over" compositing blending on 8 32bpp premultiplied ARGB pixels
/// which are with either translucent or a mixture of opaque and transparent.
/// Relies on A(x) to determine whether to emit code in ARM state (as opposed to Thumb state).
/// @arg align           bit-alignment specifier on destination loads/stores (optional)
/// @arg other_src_alpha D-register specifier for alpha source in other bank (only IF_OVERLAP)
/// @arg src0            Q-register specifier for blue/green source in this bank
/// @arg src1            Q-register specifier for red/alpha source in this bank
/// @arg opt             optional instruction to emit
/// @arg if_loadstore    YES or NO: whether to do load/store
/// @arg if_overlap      YES or NO: whether to interleave processing of next iteration
/// @arg if_preload      YES or NO: whether to insert prefetch instructions
#define S32A_OPAQUE_8PIX_BLEND(align, other_src_alpha, src0, src1, opt, if_loadstore, if_overlap, if_preload) \
if_loadstore(   "vld4.8      {d20-d23}, [%[dst]"#align"]                                      \n\t")  \
if_preload(     "sub         %[tmp], %[len], #1                                               \n\t")  \
if_overlap(     "vmov        %[alo], %[ahi], "#other_src_alpha"                               \n\t")  \
if_preload(     "cmp         %[tmp], #" PREFETCH_DISTANCE "                                   \n\t")  \
                "vmull.u8    q8, d20, d31                                                     \n\t"   \
if_preload(     "it          cs                                                               \n\t")  \
if_preload(     "movcs       %[tmp], #" PREFETCH_DISTANCE "                                   \n\t")  \
                "vmull.u8    q9, d21, d31                                                     \n\t"   \
                "vmull.u8    q10, d22, d31                                                    \n\t"   \
                "vmull.u8    q11, d23, d31                                                    \n\t"   \
if_preload(     "pld         [%[src], %[tmp], lsl #2]                                         \n\t")  \
                "vrshr.u16   q12, q8, #8                                                      \n\t"   \
if_preload(     "add         %[tmp], #(32+32)/4                                               \n\t")  \
                "vrshr.u16   q13, q9, #8                                                      \n\t"   \
                "vrshr.u16   q14, q10, #8                                                     \n\t"   \
                "vrshr.u16   q15, q11, #8                                                     \n\t"   \
if_preload(     "pld         [%[dst], %[tmp], lsl #2]                                         \n\t")  \
                "vraddhn.u16 d16, q8, q12                                                     \n\t"   \
                "vraddhn.u16 d17, q9, q13                                                     \n\t"   \
                "vraddhn.u16 d18, q10, q14                                                    \n\t"   \
                "vraddhn.u16 d19, q11, q15                                                    \n\t"   \
if_overlap(     "mvn         %[tmp], %[alo]                                                   \n\t")  \
if_overlap(     "vmvn        d31, "#other_src_alpha"                                          \n\t")  \
if_overlap(A(   "orr         %[alo], %[ahi]                                                   \n\t")) \
                "vadd.i8     "#src0", q8                                                      \n\t"   \
if_overlap(A(   "mvn         %[ahi], %[ahi]                                                   \n\t")) \
                "vadd.i8     "#src1", q9                                                      \n\t"   \
                opt"                                                                          \n\t"   \
if_loadstore(   "vst4.8      {"#src0", "#src1"}, [%[dst]"#align"]!                            \n\t")  \

#endif // __ARM_64BIT_STATE
#endif // defined(SK_ARM_HAS_NEON)

/*not static*/ inline
void blit_row_s32a_opaque(SkPMColor* dst, const SkPMColor* src, int len, U8CPU alpha) {
    SkASSERT(alpha == 0xFF);
    sk_msan_assert_initialized(src, src+len);

#if SK_CPU_SSE_LEVEL >= SK_CPU_SSE_LEVEL_SSE41
    while (len >= 16) {
        // Load 16 source pixels.
        auto s0 = _mm_loadu_si128((const __m128i*)(src) + 0),
             s1 = _mm_loadu_si128((const __m128i*)(src) + 1),
             s2 = _mm_loadu_si128((const __m128i*)(src) + 2),
             s3 = _mm_loadu_si128((const __m128i*)(src) + 3);

        const auto alphaMask = _mm_set1_epi32(0xFF000000);

        auto ORed = _mm_or_si128(s3, _mm_or_si128(s2, _mm_or_si128(s1, s0)));
        if (_mm_testz_si128(ORed, alphaMask)) {
            // All 16 source pixels are transparent.  Nothing to do.
            src += 16;
            dst += 16;
            len -= 16;
            continue;
        }

        auto d0 = (__m128i*)(dst) + 0,
             d1 = (__m128i*)(dst) + 1,
             d2 = (__m128i*)(dst) + 2,
             d3 = (__m128i*)(dst) + 3;

        auto ANDed = _mm_and_si128(s3, _mm_and_si128(s2, _mm_and_si128(s1, s0)));
        if (_mm_testc_si128(ANDed, alphaMask)) {
            // All 16 source pixels are opaque.  SrcOver becomes Src.
            _mm_storeu_si128(d0, s0);
            _mm_storeu_si128(d1, s1);
            _mm_storeu_si128(d2, s2);
            _mm_storeu_si128(d3, s3);
            src += 16;
            dst += 16;
            len -= 16;
            continue;
        }

        // TODO: This math is wrong.
        // Do SrcOver.
        _mm_storeu_si128(d0, SkPMSrcOver_SSE2(s0, _mm_loadu_si128(d0)));
        _mm_storeu_si128(d1, SkPMSrcOver_SSE2(s1, _mm_loadu_si128(d1)));
        _mm_storeu_si128(d2, SkPMSrcOver_SSE2(s2, _mm_loadu_si128(d2)));
        _mm_storeu_si128(d3, SkPMSrcOver_SSE2(s3, _mm_loadu_si128(d3)));
        src += 16;
        dst += 16;
        len -= 16;
    }

#elif SK_CPU_SSE_LEVEL >= SK_CPU_SSE_LEVEL_SSE2
    while (len >= 16) {
        // Load 16 source pixels.
        auto s0 = _mm_loadu_si128((const __m128i*)(src) + 0),
             s1 = _mm_loadu_si128((const __m128i*)(src) + 1),
             s2 = _mm_loadu_si128((const __m128i*)(src) + 2),
             s3 = _mm_loadu_si128((const __m128i*)(src) + 3);

        const auto alphaMask = _mm_set1_epi32(0xFF000000);

        auto ORed = _mm_or_si128(s3, _mm_or_si128(s2, _mm_or_si128(s1, s0)));
        if (0xffff == _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(ORed, alphaMask),
                                                       _mm_setzero_si128()))) {
            // All 16 source pixels are transparent.  Nothing to do.
            src += 16;
            dst += 16;
            len -= 16;
            continue;
        }

        auto d0 = (__m128i*)(dst) + 0,
             d1 = (__m128i*)(dst) + 1,
             d2 = (__m128i*)(dst) + 2,
             d3 = (__m128i*)(dst) + 3;

        auto ANDed = _mm_and_si128(s3, _mm_and_si128(s2, _mm_and_si128(s1, s0)));
        if (0xffff == _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(ANDed, alphaMask),
                                                       alphaMask))) {
            // All 16 source pixels are opaque.  SrcOver becomes Src.
            _mm_storeu_si128(d0, s0);
            _mm_storeu_si128(d1, s1);
            _mm_storeu_si128(d2, s2);
            _mm_storeu_si128(d3, s3);
            src += 16;
            dst += 16;
            len -= 16;
            continue;
        }

        // TODO: This math is wrong.
        // Do SrcOver.
        _mm_storeu_si128(d0, SkPMSrcOver_SSE2(s0, _mm_loadu_si128(d0)));
        _mm_storeu_si128(d1, SkPMSrcOver_SSE2(s1, _mm_loadu_si128(d1)));
        _mm_storeu_si128(d2, SkPMSrcOver_SSE2(s2, _mm_loadu_si128(d2)));
        _mm_storeu_si128(d3, SkPMSrcOver_SSE2(s3, _mm_loadu_si128(d3)));

        src += 16;
        dst += 16;
        len -= 16;
    }

#elif defined(SK_ARM_HAS_NEON)
#ifdef __ARM_64BIT_STATE
    // No attempt has been made to adapt the inline assembly version for AArch64
    // so fall back to the less performant version that uses intrinsics instead

    // Do 8-pixels at a time. A 16-pixels at a time version of this code was also tested, but it
    // underperformed on some of the platforms under test for inputs with frequent transitions of
    // alpha (corresponding to changes of the conditions [~]alpha_u64 == 0 below). It may be worth
    // revisiting the situation in the future.
    while (len >= 8) {
        // Load 8 pixels in 4 NEON registers. src_col.val[i] will contain the same color component
        // for 8 consecutive pixels (e.g. src_col.val[3] will contain all alpha components of 8
        // pixels).
        uint8x8x4_t src_col = vld4_u8(reinterpret_cast<const uint8_t*>(src));
        src += 8;
        len -= 8;

        // We now detect 2 special cases: the first occurs when all alphas are zero (the 8 pixels
        // are all transparent), the second when all alphas are fully set (they are all opaque).
        uint8x8_t alphas = src_col.val[3];
        uint64_t alphas_u64 = vget_lane_u64(vreinterpret_u64_u8(alphas), 0);
        if (alphas_u64 == 0) {
            // All pixels transparent.
            dst += 8;
            continue;
        }

        if (~alphas_u64 == 0) {
            // All pixels opaque.
            vst4_u8(reinterpret_cast<uint8_t*>(dst), src_col);
            dst += 8;
            continue;
        }

        uint8x8x4_t dst_col = vld4_u8(reinterpret_cast<uint8_t*>(dst));
        vst4_u8(reinterpret_cast<uint8_t*>(dst), SkPMSrcOver_neon8(dst_col, src_col));
        dst += 8;
    }

    // Deal with leftover pixels.
    for (; len >= 2; len -= 2, src += 2, dst += 2) {
        uint8x8_t src2 = vld1_u8(reinterpret_cast<const uint8_t*>(src));
        uint8x8_t dst2 = vld1_u8(reinterpret_cast<const uint8_t*>(dst));
        vst1_u8(reinterpret_cast<uint8_t*>(dst), SkPMSrcOver_neon2(dst2, src2));
    }

    if (len != 0) {
        uint8x8_t result = SkPMSrcOver_neon2(vcreate_u8(*dst), vcreate_u8(*src));
        vst1_lane_u32(dst, vreinterpret_u32_u8(result), 0);
    }
    return;

#else // __ARM_64BIT_STATE
    // Inline ARM AArch32 assembly version
    uint32_t tmp, alo, ahi;
    const int eight = 8;
    if (len < 15) {
        // Too short to attempt aligned processing
        if (len & 8)
            S32A_OPAQUE_8PIX_PROCESS(, YES);
        if (len & 7)
            S32A_OPAQUE_7PIX_PROCESS(S32A_LOADSTORE_TRAILING_7, len & 7);
    } else {
        // The last 8 - 15 pixels (starting from a 4-pixel boundary) are handled together
        uintptr_t startrup = (uintptr_t) dst / sizeof (*dst) + 3;
        uintptr_t end = (uintptr_t) dst / sizeof (*dst) + len;
        size_t trailing;
        if ((end & 3) == 0)
            // No blocks of <8 pixels used at end in these cases
            trailing = 8;
        else
            // If length (discounting alignment to 4-pixel boundaries) is an odd number of 4-pixels,
            // assign this to trailing end to avoid possibility of a leading run of exactly 4
            trailing = 8 + ((startrup ^ end) & 4) + (end & 3);
        // The inner loop handles an integer number (0+) of 16-pixel blocks at 4-pixel boundaries
        // The 0..15 pixels leading up to this are handled together
        size_t leading8 = (len - trailing) & 8;
        size_t leading7 = (len - trailing) & 7;

        // Do leading pixels
        if (leading7 != 0) {
            len -= leading7;
            S32A_OPAQUE_7PIX_PROCESS(S32A_LOADSTORE_LEADING_7, leading7);
        }
        if (leading8 != 0) {
            len -= 8;
            S32A_OPAQUE_8PIX_PROCESS(:128, YES);
        }

        // Do inner loop
        __asm__ (
                // We enter and leave each iteration of the inner loop with the source
                // pointer 8 pixels ahead and the destination pointer 8 pixels behind
                // in order to permit good pipelining. The count of remaining pixels is
                // reduced by 16 to allow the loop termination test to be combined with
                // the decrementing of the remaining length.
                "sub         %[dst], #8*4                                                     \n\t"
                "vld4.8      {d0-d3}, [%[src]]!                                               \n\t"
                "subs        %[len], #16                                                      \n\t"
                "bcc         49f                                                              \n\t"

                "10:                                                                          \n\t"
                // Move alpha to ARM registers for comparison
                "vmov        %[alo], s6                                                       \n\t"
                "vmov        %[ahi], s7                                                       \n\t"
                // Fetch source data for next iteration
                "vld4.8      {d4-d7}, [%[src]]!                                               \n\t"
                "add         %[dst], #8*4                                                     \n\t"
                // Test if all source pixels are transparent (alpha=0)
                "orrs        %[tmp], %[alo], %[ahi]                                           \n\t"
                "beq         19f                                                              \n\t"
                // Find inverse alpha in case full blending required
                "vmvn        d31, d3                                                          \n\t"
                // Test if all source pixels are opaque (alpha=0xff)
                "cmp         %[alo], #-1                                                      \n\t"
                "it          eq                                                               \n\t"
                "cmpeq       %[ahi], #-1                                                      \n\t"
                "bne         30f                                                              \n\t"
                // Opaque case: copy source to destination
                "vst4.8      {d0-d3}, [%[dst]:128]                                            \n\t"
                // Drop through
                "19:                                                                          \n\t"

                // Move alpha to ARM registers for comparison
                "vmov        %[alo], s14                                                      \n\t"
                "vmov        %[ahi], s15                                                      \n\t"
                // Fetch source data for next iteration
                "vld4.8      {d0-d3}, [%[src]]!                                               \n\t"
                "add         %[dst], #8*4                                                     \n\t"
                // Test if all source pixels are transparent (alpha=0)
                "orrs        %[tmp], %[alo], %[ahi]                                           \n\t"
                "beq         29f                                                              \n\t"
                // Find inverse alpha in case full blending required
                "vmvn        d31, d7                                                          \n\t"
                // Test if all source pixels are opaque (alpha=0xff)
                "cmp         %[alo], #-1                                                      \n\t"
                "it          eq                                                               \n\t"
                "cmpeq       %[ahi], #-1                                                      \n\t"
                "bne         40f                                                              \n\t"
                // Opaque case: copy source to destination
                "vst4.8      {d4-d7}, [%[dst]:128]                                            \n\t"
                // Drop through
                "29:                                                                          \n\t"
                "subs        %[len], #16                                                      \n\t"
                "bcs         10b                                                              \n\t"
                "b           49f                                                              \n\t"

                // Mixed or translucent pixels in d0-d3
                "30:                                                                          \n\t"
                S32A_OPAQUE_8PIX_BLEND(:128, d7, q0, q1,, YES, YES, IF_PRELOAD)
A(              "teq         %[alo], #0                                                       \n\t")
T(              "orrs        %[alo], %[alo], %[ahi]                                           \n\t")
                "vld4.8      {d0-d3}, [%[src]]!                                               \n\t"
                "beq         29b                                                              \n\t"
A(              "orrs        %[tmp], %[tmp], %[ahi]                                           \n\t")
T(              "orns        %[tmp], %[tmp], %[ahi]                                           \n\t")
                "bne         40f                                                              \n\t"
                "vst4.8      {d4-d7}, [%[dst]:128]                                            \n\t"
                "b           29b                                                              \n\t"

                // Mixed or translucent pixels in d4-d7
                "40:                                                                          \n\t"
                S32A_OPAQUE_8PIX_BLEND(:128, d3, q2, q3, \
                "subs        %[len], #16", YES, YES, NO)
                "bcc         50f                                                              \n\t"
A(              "teq         %[alo], #0                                                       \n\t")
T(              "orrs        %[alo], %[alo], %[ahi]                                           \n\t")
                "vld4.8      {d4-d7}, [%[src]]!                                               \n\t"
                "beq         19b                                                              \n\t"
A(              "orrs        %[tmp], %[tmp], %[ahi]                                           \n\t")
T(              "orns        %[tmp], %[tmp], %[ahi]                                           \n\t")
                "bne         30b                                                              \n\t"
                "vst4.8      {d0-d3}, [%[dst]:128]                                            \n\t"
                "b           19b                                                              \n\t"

                "49:                                                                          \n\t"
                "add         %[dst], #8*4                                                     \n\t"
                "50:                                                                          \n\t"
        : // Outputs
                [dst]"+r"(dst),
                [src]"+r"(src),
                [len]"+r"(len),
                [alo]"+r"(alo),
                [ahi]"+r"(ahi),
                [tmp]"+r"(tmp)
        : // Inputs
        : // Clobbers
                "cc", "memory"
        );

        // Do trailing pixels.
        // There will always be more than 8 of these, and the first 8 are already in d0-d3
        S32A_OPAQUE_8PIX_PROCESS(:128, NO);
        if (len & 7)
            S32A_OPAQUE_7PIX_PROCESS(S32A_LOADSTORE_TRAILING_7, len & 7);
    }
    return;

#endif // __ARM_64BIT_STATE
#endif

    while (len-- > 0) {
        // This 0xFF000000 is not semantically necessary, but for compatibility
        // with chromium:611002 we need to keep it until we figure out where
        // the non-premultiplied src values (like 0x00FFFFFF) are coming from.
        // TODO(mtklein): sort this out and assert *src is premul here.
        if (*src & 0xFF000000) {
            *dst = (*src >= 0xFF000000) ? *src : SkPMSrcOver(*src, *dst);
        }
        src++;
        dst++;
    }
}

}  // SK_OPTS_NS

#endif//SkBlitRow_opts_DEFINED
