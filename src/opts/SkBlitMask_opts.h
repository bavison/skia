/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkBlitMask_opts_DEFINED
#define SkBlitMask_opts_DEFINED

#include "Sk4px.h"

namespace SK_OPTS_NS {

#if defined(SK_ARM_HAS_NEON)
    // The Sk4px versions below will work fine with NEON, but we have had many indications
    // that it doesn't perform as well as this NEON-specific code.  TODO(mtklein): why?
    #include "SkColor_opts_neon.h"

    template <bool isColor>
    static void D32_A8_Opaque_Color_neon(void* SK_RESTRICT dst, size_t dstRB,
                                         const void* SK_RESTRICT maskPtr, size_t maskRB,
                                         SkColor color, int width, int height) {
        SkPMColor pmc = SkPreMultiplyColor(color);
        SkPMColor* SK_RESTRICT device = (SkPMColor*)dst;
        const uint8_t* SK_RESTRICT mask = (const uint8_t*)maskPtr;
        uint8x8x4_t vpmc;

        maskRB -= width;
        dstRB -= (width << 2);

        if (width >= 8) {
            vpmc.val[NEON_A] = vdup_n_u8(SkGetPackedA32(pmc));
            vpmc.val[NEON_R] = vdup_n_u8(SkGetPackedR32(pmc));
            vpmc.val[NEON_G] = vdup_n_u8(SkGetPackedG32(pmc));
            vpmc.val[NEON_B] = vdup_n_u8(SkGetPackedB32(pmc));
        }
        do {
            int w = width;
            while (w >= 8) {
                uint8x8_t vmask = vld1_u8(mask);
                uint16x8_t vscale, vmask256 = SkAlpha255To256_neon8(vmask);
                if (isColor) {
                    vscale = vsubw_u8(vdupq_n_u16(256),
                            SkAlphaMul_neon8(vpmc.val[NEON_A], vmask256));
                } else {
                    vscale = vsubw_u8(vdupq_n_u16(256), vmask);
                }
                uint8x8x4_t vdev = vld4_u8((uint8_t*)device);

                vdev.val[NEON_A] =   SkAlphaMul_neon8(vpmc.val[NEON_A], vmask256)
                    + SkAlphaMul_neon8(vdev.val[NEON_A], vscale);
                vdev.val[NEON_R] =   SkAlphaMul_neon8(vpmc.val[NEON_R], vmask256)
                    + SkAlphaMul_neon8(vdev.val[NEON_R], vscale);
                vdev.val[NEON_G] =   SkAlphaMul_neon8(vpmc.val[NEON_G], vmask256)
                    + SkAlphaMul_neon8(vdev.val[NEON_G], vscale);
                vdev.val[NEON_B] =   SkAlphaMul_neon8(vpmc.val[NEON_B], vmask256)
                    + SkAlphaMul_neon8(vdev.val[NEON_B], vscale);

                vst4_u8((uint8_t*)device, vdev);

                mask += 8;
                device += 8;
                w -= 8;
            }

            while (w--) {
                unsigned aa = *mask++;
                if (isColor) {
                    *device = SkBlendARGB32(pmc, *device, aa);
                } else {
                    *device = SkAlphaMulQ(pmc, SkAlpha255To256(aa))
                        + SkAlphaMulQ(*device, SkAlpha255To256(255 - aa));
                }
                device += 1;
            };

            device = (uint32_t*)((char*)device + dstRB);
            mask += maskRB;

        } while (--height != 0);
    }

    static void blit_mask_d32_a8_general(SkPMColor* dst, size_t dstRB,
                                         const SkAlpha* mask, size_t maskRB,
                                         SkColor color, int w, int h) {
        D32_A8_Opaque_Color_neon<true>(dst, dstRB, mask, maskRB, color, w, h);
    }

    // As above, but made slightly simpler by requiring that color is opaque.
    static void blit_mask_d32_a8_opaque(SkPMColor* dst, size_t dstRB,
                                        const SkAlpha* mask, size_t maskRB,
                                        SkColor color, int w, int h) {
        D32_A8_Opaque_Color_neon<false>(dst, dstRB, mask, maskRB, color, w, h);
    }

    // Same as _opaque, but assumes color == SK_ColorBLACK, a very common and even simpler case.
    static void blit_mask_d32_a8_black(SkPMColor* dst, size_t dstRB,
                                       const SkAlpha* maskPtr, size_t maskRB,
                                       int width, int height) {
        SkPMColor* SK_RESTRICT device = (SkPMColor*)dst;
        const uint8_t* SK_RESTRICT mask = (const uint8_t*)maskPtr;

        maskRB -= width;
        dstRB -= (width << 2);
        do {
            int w = width;
            while (w >= 8) {
                uint8x8_t vmask = vld1_u8(mask);
                uint16x8_t vscale = vsubw_u8(vdupq_n_u16(256), vmask);
                uint8x8x4_t vdevice = vld4_u8((uint8_t*)device);

                vdevice = SkAlphaMulQ_neon8(vdevice, vscale);
                vdevice.val[NEON_A] += vmask;

                vst4_u8((uint8_t*)device, vdevice);

                mask += 8;
                device += 8;
                w -= 8;
            }
            while (w-- > 0) {
                unsigned aa = *mask++;
                *device = (aa << SK_A32_SHIFT)
                            + SkAlphaMulQ(*device, SkAlpha255To256(255 - aa));
                device += 1;
            };
            device = (uint32_t*)((char*)device + dstRB);
            mask += maskRB;
        } while (--height != 0);
    }

#else
    static void blit_mask_d32_a8_general(SkPMColor* dst, size_t dstRB,
                                         const SkAlpha* mask, size_t maskRB,
                                         SkColor color, int w, int h) {
        auto s = Sk4px::DupPMColor(SkPreMultiplyColor(color));
        auto fn = [&](const Sk4px& d, const Sk4px& aa) {
            //  = (s + d(1-sa))aa + d(1-aa)
            //  = s*aa + d(1-sa*aa)
            auto left  = s.approxMulDiv255(aa),
                 right = d.approxMulDiv255(left.alphas().inv());
            return left + right;  // This does not overflow (exhaustively checked).
        };
        while (h --> 0) {
            Sk4px::MapDstAlpha(w, dst, mask, fn);
            dst  +=  dstRB / sizeof(*dst);
            mask += maskRB / sizeof(*mask);
        }
    }

    // As above, but made slightly simpler by requiring that color is opaque.
    static void blit_mask_d32_a8_opaque(SkPMColor* dst, size_t dstRB,
                                        const SkAlpha* mask, size_t maskRB,
                                        SkColor color, int w, int h) {
        SkASSERT(SkColorGetA(color) == 0xFF);
        auto s = Sk4px::DupPMColor(SkPreMultiplyColor(color));
        auto fn = [&](const Sk4px& d, const Sk4px& aa) {
            //  = (s + d(1-sa))aa + d(1-aa)
            //  = s*aa + d(1-sa*aa)
            //   ~~~>
            //  = s*aa + d(1-aa)
            return s.approxMulDiv255(aa) + d.approxMulDiv255(aa.inv());
        };
        while (h --> 0) {
            Sk4px::MapDstAlpha(w, dst, mask, fn);
            dst  +=  dstRB / sizeof(*dst);
            mask += maskRB / sizeof(*mask);
        }
    }

    // Same as _opaque, but assumes color == SK_ColorBLACK, a very common and even simpler case.
    static void blit_mask_d32_a8_black(SkPMColor* dst, size_t dstRB,
                                       const SkAlpha* mask, size_t maskRB,
                                       int w, int h) {
        auto fn = [](const Sk4px& d, const Sk4px& aa) {
            //   = (s + d(1-sa))aa + d(1-aa)
            //   = s*aa + d(1-sa*aa)
            //   ~~~>
            // a = 1*aa + d(1-1*aa) = aa + d(1-aa)
            // c = 0*aa + d(1-1*aa) =      d(1-aa)
            return aa.zeroColors() + d.approxMulDiv255(aa.inv());
        };
        while (h --> 0) {
            Sk4px::MapDstAlpha(w, dst, mask, fn);
            dst  +=  dstRB / sizeof(*dst);
            mask += maskRB / sizeof(*mask);
        }
    }
#endif

/*not static*/ inline void blit_mask_d32_a8(SkPMColor* dst, size_t dstRB,
                                            const SkAlpha* mask, size_t maskRB,
                                            SkColor color, int w, int h) {
    if (color == SK_ColorBLACK) {
        blit_mask_d32_a8_black(dst, dstRB, mask, maskRB, w, h);
    } else if (SkColorGetA(color) == 0xFF) {
        blit_mask_d32_a8_opaque(dst, dstRB, mask, maskRB, color, w, h);
    } else {
        blit_mask_d32_a8_general(dst, dstRB, mask, maskRB, color, w, h);
    }
}

#if defined(SK_ARM_HAS_NEON) && !defined(__ARM_64BIT_STATE)

// These macros permit optionally-included features to be switched using a parameter to another macro
#define YES(x) x
#define NO(x)

// How far ahead (pixels) to preload (undefine to disable prefetch) - determined empirically
#define PREFETCH_DISTANCE "52"

#ifdef PREFETCH_DISTANCE
#define IF_PRELOAD YES
#else
#define IF_PRELOAD NO
#endif

/// Macro to load 1..7 source and mask pixels in growing powers-of-2 in size - suitable for leading pixels
#define S32A_A8_LOAD_SM_LEADING_7(r0, r1, r2, r3, s_base, r4, m_base, opt1, opt2)                       \
                opt1"                                                                           \n\t"   \
                "tst         %[group_size], #1                                                  \n\t"   \
                opt2"                                                                           \n\t"   \
                "beq         1f                                                                 \n\t"   \
                "vld1.8      {"#r4"[1]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[1],"#r1"[1],"#r2"[1],"#r3"[1]}, [%["#s_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \
                "lsls        %[tmp], %[group_size], #30                                         \n\t"   \
                "bpl         1f                                                                 \n\t"   \
                "vld1.8      {"#r4"[2]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[2],"#r1"[2],"#r2"[2],"#r3"[2]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[3]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[3],"#r1"[3],"#r2"[3],"#r3"[3]}, [%["#s_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \
                "bcc         1f                                                                 \n\t"   \
                "vld1.8      {"#r4"[4]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[4],"#r1"[4],"#r2"[4],"#r3"[4]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[5]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[5],"#r1"[5],"#r2"[5],"#r3"[5]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[6]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[6],"#r1"[6],"#r2"[6],"#r3"[6]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[7]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[7],"#r1"[7],"#r2"[7],"#r3"[7]}, [%["#s_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \

/// Macro to load or store 1..7 destination pixels in growing powers-of-2 in size - suitable for leading pixels
#define S32A_A8_LOADSTORE_D_LEADING_7(ls, r0, r1, r2, r3, d_base)                                       \
                "tst         %[group_size], #1                                                  \n\t"   \
                "beq         1f                                                                 \n\t"   \
                "v"#ls"4.8   {"#r0"[1],"#r1"[1],"#r2"[1],"#r3"[1]}, [%["#d_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \
                "lsls        %[tmp], %[group_size], #30                                         \n\t"   \
                "add         %[tmp], %["#d_base"], #4                                           \n\t"   \
                "bpl         1f                                                                 \n\t"   \
                "v"#ls"4.8   {"#r0"[2],"#r1"[2],"#r2"[2],"#r3"[2]}, [%["#d_base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[3],"#r1"[3],"#r2"[3],"#r3"[3]}, [%[tmp]:32], %[eight]       \n\t"   \
                "1:                                                                             \n\t"   \
                "bcc         1f                                                                 \n\t"   \
                "v"#ls"4.8   {"#r0"[4],"#r1"[4],"#r2"[4],"#r3"[4]}, [%["#d_base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[5],"#r1"[5],"#r2"[5],"#r3"[5]}, [%[tmp]:32], %[eight]       \n\t"   \
                "v"#ls"4.8   {"#r0"[6],"#r1"[6],"#r2"[6],"#r3"[6]}, [%["#d_base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[7],"#r1"[7],"#r2"[7],"#r3"[7]}, [%[tmp]:32], %[eight]       \n\t"   \
                "1:                                                                             \n\t"   \

/// Macro to load 1..7 source and mask pixels in shrinking powers-of-2 in size - suitable for trailing pixels
#define S32A_A8_LOAD_SM_TRAILING_7(r0, r1, r2, r3, s_base, r4, m_base, opt1, opt2)                      \
                opt1"                                                                           \n\t"   \
                "lsls        %[tmp], %[group_size], #30                                         \n\t"   \
                opt2"                                                                           \n\t"   \
                "bcc         1f                                                                 \n\t"   \
                "vld1.8      {"#r4"[0]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[0],"#r1"[0],"#r2"[0],"#r3"[0]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[1]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[1],"#r1"[1],"#r2"[1],"#r3"[1]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[2]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[2],"#r1"[2],"#r2"[2],"#r3"[2]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[3]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[3],"#r1"[3],"#r2"[3],"#r3"[3]}, [%["#s_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \
                "bpl         1f                                                                 \n\t"   \
                "vld1.8      {"#r4"[4]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[4],"#r1"[4],"#r2"[4],"#r3"[4]}, [%["#s_base"]:32]!          \n\t"   \
                "vld1.8      {"#r4"[5]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[5],"#r1"[5],"#r2"[5],"#r3"[5]}, [%["#s_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \
                "tst         %[group_size], #1                                                  \n\t"   \
                "beq         1f                                                                 \n\t"   \
                "vld1.8      {"#r4"[6]}, [%["#m_base"]]!                                        \n\t"   \
                "vld4.8      {"#r0"[6],"#r1"[6],"#r2"[6],"#r3"[6]}, [%["#s_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \

/// Macro to load or store 1..7 destination pixels in shrinking powers-of-2 in size - suitable for trailing pixels
#define S32A_A8_LOADSTORE_D_TRAILING_7(ls, r0, r1, r2, r3, d_base)                                      \
                "lsls        %[tmp], %[group_size], #30                                         \n\t"   \
                "add         %[tmp], %["#d_base"], #4                                           \n\t"   \
                "bcc         1f                                                                 \n\t"   \
                "v"#ls"4.8   {"#r0"[0],"#r1"[0],"#r2"[0],"#r3"[0]}, [%["#d_base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[1],"#r1"[1],"#r2"[1],"#r3"[1]}, [%[tmp]:32], %[eight]       \n\t"   \
                "v"#ls"4.8   {"#r0"[2],"#r1"[2],"#r2"[2],"#r3"[2]}, [%["#d_base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[3],"#r1"[3],"#r2"[3],"#r3"[3]}, [%[tmp]:32], %[eight]       \n\t"   \
                "1:                                                                             \n\t"   \
                "bpl         1f                                                                 \n\t"   \
                "v"#ls"4.8   {"#r0"[4],"#r1"[4],"#r2"[4],"#r3"[4]}, [%["#d_base"]:32], %[eight] \n\t"   \
                "v"#ls"4.8   {"#r0"[5],"#r1"[5],"#r2"[5],"#r3"[5]}, [%[tmp]:32], %[eight]       \n\t"   \
                "1:                                                                             \n\t"   \
                "tst         %[group_size], #1                                                  \n\t"   \
                "beq         1f                                                                 \n\t"   \
                "v"#ls"4.8   {"#r0"[6],"#r1"[6],"#r2"[6],"#r3"[6]}, [%["#d_base"]:32]!          \n\t"   \
                "1:                                                                             \n\t"   \

/// Macro to do shortcut testing for "over" compositing of 32bpp premultiplied ARGB source and 8-bit alpha mask
#define S32A_A8_TEST(dst_adjust)                                                                        \
                "vmov        %[mlo], %[mhi], d16                                                \n\t"   \
                "vmov        %[alo], s6                                                         \n\t"   \
                "vmov        %[ahi], s7                                                         \n\t"   \
                "and         %[tmp], %[mlo], %[mhi]                                             \n\t"   \
                "orrs        %[mlo], %[mhi]                                                     \n\t"   \
                "it          ne                                                                 \n\t"   \
                "orrsne      %[mlo], %[alo], %[ahi]                                             \n\t"   \
                "it          eq                                                                 \n\t"   \
                "addeq       %[dst], " dst_adjust "                                             \n\t"   \
                "beq         9f                                                                 \n\t"   \
                "and         %[tmp], %[alo]                                                     \n\t"   \
                "and         %[tmp], %[ahi]                                                     \n\t"   \
                "cmp         %[tmp], #-1                                                        \n\t"   \
                "beq         5f                                                                 \n\t"   \

/// Macro to do testing and "over" compositing of a group of 1..7 32bpp premultiplied ARGB source and 1..7 8-bit alpha mask leading or trailing pixels
#define S32A_A8_7PIX_PROCESS(load_sm_7, loadstore_d_7, size)                                            \
    do {                                                                                                \
        __asm__ volatile (                                                                              \
                /* Load the leading/trailing source pixels,                                             \
                 * after initialising all the unused indexes from the first pixel                       \
                 * so the all-opaque and all-transparent tests still work */                            \
                load_sm_7(d0, d1, d2, d3, src, d16, msk,                                                \
                "vld1.8      {d16[]}, [%[msk]]",                                                        \
                "vld4.8      {d0[], d1[], d2[], d3[]}, [%[src]]")                                       \
                S32A_A8_TEST("%[group_size], lsl #2")                                                   \
                /* Translucency used, or a mixture of opaque and transparent */                         \
                loadstore_d_7(ld, d4, d5, d6, d7, dst)                                                  \
                "sub         %[dst], %[group_size], lsl #2                                      \n\t"   \
                S32A_A8_8PIX_BLEND(, NO, NO)                                                      \
                loadstore_d_7(st, d0, d1, d2, d3, dst)                                                  \
                /* Drop through */                                                                      \
                "9:                                                                             \n\t"   \
        : /* Outputs */                                                                                 \
                [mlo]"=&r"(mlo),                                                                        \
                [mhi]"=&r"(mhi),                                                                        \
                [alo]"=&r"(alo),                                                                        \
                [ahi]"=&r"(ahi),                                                                        \
                [tmp]"=&r"(tmp),                                                                        \
                [src]"+r"(src),                                                                         \
                [msk]"+r"(msk),                                                                         \
                [dst]"+r"(dst)                                                                          \
        : /* Inputs */                                                                                  \
                [group_size]"r"(size),                                                                  \
                     [eight]"r"(eight)                                                                  \
        : /* Clobbers */                                                                                \
                "cc", "memory"                                                                          \
        );                                                                                              \
    } while (0)

/// Macro to do "over" compositing blending on 8 32bpp premultiplied ARGB source and 8 8-bit alpha mask pixels
/// which are with either translucent or a mixture of opaque and transparent.
/// Relies on A(x) to determine whether to emit code in ARM state (as opposed to Thumb state).
/// @arg align           bit-alignment specifier on destination loads/stores (optional)
/// @arg if_loadstore    YES or NO: whether to do load/store of destination
/// @arg if_preload      YES or NO: whether to insert prefetch instructions
#define S32A_A8_8PIX_BLEND(align, if_loadstore, if_preload)                                        \
if_loadstore(   "vld4.8      {d4-d7}, [%[dst]"#align"]                                          \n\t")  \
if_preload(     "sub         %[tmp], %[len], #1                                                 \n\t")  \
                "vmull.u8    q9, d3, d16                                                        \n\t"   \
if_preload(     "cmp         %[tmp], #" PREFETCH_DISTANCE "                                     \n\t")  \
                "vmull.u8    q10, d0, d16                                                       \n\t"   \
if_preload(     "it          cs                                                                 \n\t")  \
if_preload(     "movcs       %[tmp], #" PREFETCH_DISTANCE "                                     \n\t")  \
                "vmull.u8    q11, d1, d16                                                       \n\t"   \
                "vmull.u8    q8, d2, d16                                                        \n\t"   \
                "vrshr.u16   q1, q9, #8                                                         \n\t"   \
if_preload(     "pld         [%[msk], %[tmp]]                                                   \n\t")  \
                "vrshr.u16   q0, q10, #8                                                        \n\t"   \
if_preload(     "pld         [%[src], %[tmp], lsl #2]                                           \n\t")  \
                "vraddhn.u16 d3, q9, q1                                                         \n\t"   \
if_preload(     "add         %[tmp], #32/4                                                      \n\t")  \
                "vrshr.u16   q9, q11, #8                                                        \n\t"   \
                "vrshr.u16   q12, q8, #8                                                        \n\t"   \
                "vmvn        d2, d3                                                             \n\t"   \
if_preload(     "pld         [%[dst], %[tmp], lsl #2]                                           \n\t")  \
                "vraddhn.u16 d0, q10, q0                                                        \n\t"   \
                "vmull.u8    q10, d4, d2                                                        \n\t"   \
                "vmull.u8    q13, d5, d2                                                        \n\t"   \
                "vmull.u8    q14, d6, d2                                                        \n\t"   \
                "vmull.u8    q15, d7, d2                                                        \n\t"   \
                "vrshr.u16   q2, q10, #8                                                        \n\t"   \
                "vrshr.u16   q3, q13, #8                                                        \n\t"   \
                "vraddhn.u16 d1, q11, q9                                                        \n\t"   \
                "vrshr.u16   q9, q14, #8                                                        \n\t"   \
                "vrshr.u16   q11, q15, #8                                                       \n\t"   \
                "vraddhn.u16 d4, q10, q2                                                        \n\t"   \
                "vraddhn.u16 d5, q13, q3                                                        \n\t"   \
                "vraddhn.u16 d2, q8, q12                                                        \n\t"   \
                "vraddhn.u16 d6, q14, q9                                                        \n\t"   \
                "vraddhn.u16 d7, q15, q11                                                       \n\t"   \
                "vadd.u8     q0, q2                                                             \n\t"   \
                "vadd.u8     q1, q3                                                             \n\t"   \
                "5:                                                                             \n\t"   \
if_loadstore(   "vst4.8      {d0-d3}, [%[dst]"#align"]!                                         \n\t")  \

#endif

/*not static*/ inline
void blit_row_s32a_a8(SkPMColor* dst, const void* vmask, const SkPMColor* src, int n) {
#if defined(SK_ARM_HAS_NEON) && !defined(__ARM_64BIT_STATE)
    const SkAlpha* msk = static_cast<const SkAlpha*>(vmask);
    uint32_t tmp, mlo, mhi, alo, ahi;
    const int eight = 8;
    if (n < 15) {
        // Too short to attempt aligned processing
        if (n & 8) {
            __asm__ (
                    "vld1.8      {d16}, [%[msk]]!                                                   \n\t"
                    "vld4.8      {d0-d3}, [%[src]]!                                                 \n\t"
                    S32A_A8_TEST("#8*4")
                    /* Translucency used, or a mixture of opaque and transparent */
                    S32A_A8_8PIX_BLEND(, YES, NO)
                    /* Drop through */
                    "9:                                                                             \n\t"
            :  /* Outputs */
                    [mlo]"=&r"(mlo),
                    [mhi]"=&r"(mhi),
                    [alo]"=&r"(alo),
                    [ahi]"=&r"(ahi),
                    [tmp]"=&r"(tmp),
                    [src]"+r"(src),
                    [msk]"+r"(msk),
                    [dst]"+r"(dst)
            : /* Inputs */
            : /* Clobbers */
                    "cc", "memory"
            );
        }
        if (n & 7)
            S32A_A8_7PIX_PROCESS(S32A_A8_LOAD_SM_TRAILING_7, S32A_A8_LOADSTORE_D_TRAILING_7, n & 7);
    } else {
        // The last 0 - 7 pixels (starting from a 4-pixel boundary) are handled together
        uintptr_t startrup = (uintptr_t) dst / sizeof (*dst) + 3;
        uintptr_t end = (uintptr_t) dst / sizeof (*dst) + n;
        size_t trailing;
        if ((end & 3) == 0)
            // No blocks of <8 pixels used at end in these cases
            trailing = 0;
        else
            // If length (discounting alignment to 4-pixel boundaries) is an odd number of 4-pixels,
            // assign 4 pixels to trailing end to avoid possibility of a leading run of exactly 4,
            // otherwise use <4 trailing pixels to maximise central 8-pixel blocks
            trailing = ((startrup ^ end) & 4) + (end & 3);
        // The inner loop handles an integer number (0+) of 8-pixel blocks at 4-pixel boundaries
        // The 0 - 7 pixels leading up to this are handled together
        size_t leading = (n - trailing) & 7;

        // Do leading pixels
        if (leading != 0) {
            n -= leading;
            S32A_A8_7PIX_PROCESS(S32A_A8_LOAD_SM_LEADING_7, S32A_A8_LOADSTORE_D_LEADING_7, leading);
        }

        // Do inner loop
        __asm__ (
                "subs        %[len], #8                                                         \n\t"
                "bcc         50f                                                                \n\t"

                "10:                                                                            \n\t"
                "vld1.8      {d16}, [%[msk]]!                                                   \n\t"
                "vld4.8      {d0-d3}, [%[src]]!                                                 \n\t"
                S32A_A8_TEST("#8*4")
                /* Translucency used, or a mixture of opaque and transparent */
                S32A_A8_8PIX_BLEND(:128, YES, IF_PRELOAD)
                /* Drop through */
                "9:                                                                             \n\t"
                "subs        %[len], #8                                                         \n\t"
                "bcs         10b                                                                \n\t"
                "50:                                                                            \n\t"
        : // Outputs
                [mlo]"=&r"(mlo),
                [mhi]"=&r"(mhi),
                [alo]"=&r"(alo),
                [ahi]"=&r"(ahi),
                [tmp]"=&r"(tmp),
                [src]"+r"(src),
                [msk]"+r"(msk),
                [dst]"+r"(dst),
                [len]"+r"(n)
        : // Inputs
        : // Clobbers
                "cc", "memory"
        );

        // Do trailing pixels.
        if (n & 7)
            S32A_A8_7PIX_PROCESS(S32A_A8_LOAD_SM_TRAILING_7, S32A_A8_LOADSTORE_D_TRAILING_7, n & 7);
    }
#else
    auto mask = (const uint8_t*)vmask;

#ifdef SK_SUPPORT_LEGACY_A8_MASKBLITTER
    for (int i = 0; i < n; ++i) {
        if (mask[i]) {
            dst[i] = SkBlendARGB32(src[i], dst[i], mask[i]);
        }
    }
#else
    Sk4px::MapDstSrcAlpha(n, dst, src, mask, [](const Sk4px& d, const Sk4px& s, const Sk4px& aa) {
        const auto s_aa = s.approxMulDiv255(aa);
        return s_aa + d.approxMulDiv255(s_aa.alphas().inv());
    });
#endif
#endif
}

}  // SK_OPTS_NS

#endif//SkBlitMask_opts_DEFINED
