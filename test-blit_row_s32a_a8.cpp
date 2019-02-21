#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <arm_neon.h>

#define PIXELS 128

#define KILOBYTE (1024)
#define MEGABYTE (1024*1024)

#define L1CACHESIZE (32*KILOBYTE)
#define L2CACHESIZE (512*KILOBYTE)
#define TESTSIZE (200*MEGABYTE)

typedef unsigned U8CPU;
typedef unsigned U16CPU;
typedef uint32_t SkPMColor;
typedef uint8_t SkAlpha;

/* Just used for cancelling out the overheads */
static void control(SkPMColor* dst, const void* msk, const SkPMColor* src, int len)
{
}

namespace neon {

#define SK_RGBA_R32_SHIFT   0
#define SK_RGBA_G32_SHIFT   8
#define SK_RGBA_B32_SHIFT   16
#define SK_RGBA_A32_SHIFT   24
#define SK_A32_SHIFT    24

#define SkGetPackedA32(packed)      ((uint32_t)((packed) << (24 - SK_A32_SHIFT)) >> 24)

static inline unsigned SkAlpha255To256(U8CPU alpha) {
//    SkASSERT(SkToU8(alpha) == alpha);
    // this one assues that blending on top of an opaque dst keeps it that way
    // even though it is less accurate than a+(a>>7) for non-opaque dsts
    return alpha + 1;
}

static inline U16CPU SkAlphaMulInv256(U16CPU value, U16CPU alpha256) {
    unsigned prod = 0xFFFF - value * alpha256;
    return (prod + (prod >> 8)) >> 8;
}

static inline SkPMColor SkBlendARGB32(SkPMColor src, SkPMColor dst, U8CPU aa) {
//    SkASSERT((unsigned)aa <= 255);

    unsigned src_scale = SkAlpha255To256(aa);
    unsigned dst_scale = SkAlphaMulInv256(SkGetPackedA32(src), src_scale);

    const uint32_t mask = 0xFF00FF;

    uint32_t src_rb = (src & mask) * src_scale;
    uint32_t src_ag = ((src >> 8) & mask) * src_scale;

    uint32_t dst_rb = (dst & mask) * dst_scale;
    uint32_t dst_ag = ((dst >> 8) & mask) * dst_scale;

    return (((src_rb + dst_rb) >> 8) & mask) | ((src_ag + dst_ag) & ~mask);
}

/*not static*/ inline
void blit_row_s32a_a8(SkPMColor* dst, const void* maskIn, const SkPMColor* src, int len) {
	const SkAlpha* mask = static_cast<const SkAlpha*>(maskIn);
    for (int i = 0; i < len; ++i) {
        if (mask[i]) {
            dst[i] = SkBlendARGB32(src[i], dst[i], mask[i]);
        }
    }
}

}

// These macros permit optionally-included features to be switched using a parameter to another macro
#define YES(x) x
#define NO(x)

// How far ahead (pixels) to preload (undefine to disable prefetch) - determined empirically
//#define PREFETCH_DISTANCE "24"

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

static __attribute__((noinline)) void new_blit_row_s32a_a8(SkPMColor* dst, const void* msk, const SkPMColor* src, int len)
{
    uint32_t tmp, mlo, mhi, alo, ahi;
    const int eight = 8;
    if (len < 15) {
        // Too short to attempt aligned processing
        if (len & 8) {
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
        if (len & 7)
            S32A_A8_7PIX_PROCESS(S32A_A8_LOAD_SM_TRAILING_7, S32A_A8_LOADSTORE_D_TRAILING_7, len & 7);
    } else {
        // The last 0 - 7 pixels (starting from a 4-pixel boundary) are handled together
        uintptr_t startrup = (uintptr_t) dst / sizeof (*dst) + 3;
        uintptr_t end = (uintptr_t) dst / sizeof (*dst) + len;
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
        size_t leading = (len - trailing) & 7;

        // Do leading pixels
        if (leading != 0) {
            len -= leading;
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
                [len]"+r"(len)
        : // Inputs
        : // Clobbers
                "cc", "memory"
        );

        // Do trailing pixels.
        if (len & 7)
            S32A_A8_7PIX_PROCESS(S32A_A8_LOAD_SM_TRAILING_7, S32A_A8_LOADSTORE_D_TRAILING_7, len & 7);
    }
}

uint32_t rand32(void)
{
    return rand() ^ (rand() << 16);
}


static uint64_t gettime(void)
{
    struct timeval tv;

    gettimeofday (&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

static uint32_t bench(void (*test)(SkPMColor* dst, const void* msk, const SkPMColor* src, int len), SkPMColor *a, SkPMColor *b, SkAlpha *c, size_t len, size_t times)
{
    int i, x = 0;
    for (i = times; i >= 0; i--)
    {
        /* If we cared about CPUs with non-write-allocate caches and operations that
         * don't read from the destination buffer, we'd need to ensure the destination
         * hadn't been evicted from the cache here - but this isn't the case */
        x = (x + 1) & 63;
        test(a + x, c + x, b + 63 - x, len);
    }
    return len * times * sizeof *a;
}

static uint64_t diffword(uint32_t a, uint32_t b)
{
    uint32_t lo, hi;
    __asm__ (
            "vmov        s0, %[a]         \n\t"
            "vmov        s2, %[b]         \n\t"
            "vmov.i8     d2, #2           \n\t"
//            "vabd.u8     d0, d1           \n\t"
            "vsub.i8     d0, d1           \n\t"
            "vabs.s8     d0, d0           \n\t"
            "vcgt.u8     d0, d2           \n\t"
            "vmovl.s8    q0, d0           \n\t"
            "vneg.s8     d0, d0           \n\t"
            "vrev64.16   d0, d0           \n\t"
            "vmov        %[lo], %[hi], d0 \n\t"
    : // Outputs
            [lo]"=r"(lo),
            [hi]"=r"(hi)
    : // Inputs
            [a]"r"(a),
            [b]"r"(b)
    );
    return (uint64_t) hi << 32 | lo;
}

int main(void)
{
#if 1
    uint32_t src[PIXELS] __attribute__((aligned(16))),
            dst0[PIXELS] __attribute__((aligned(16))),
            dst1[PIXELS] __attribute__((aligned(16))),
            dst2[PIXELS] __attribute__((aligned(16)));
    uint8_t msk[PIXELS] __attribute__((aligned(16)));
    srand(0);

    for (size_t len = 0; len <= 100; len++) {
        for (size_t offset = 4; offset < 8; offset++) {
            for (size_t i = 0; i < PIXELS; i++)
            {
                msk[i] = rand();
                src[i] = rand32();
                if ((src[i] & 0xff000000) == 0)
                    src[i] = 0;
                dst2[i] = dst1[i] = dst0[i] = rand32();
                if ((i & 7) == 7)
                {
                    switch (rand() % 3)
                    {
                    case 0:
                        msk[i-7] = 0xff;
                        msk[i-6] = 0xff;
                        msk[i-5] = 0xff;
                        msk[i-4] = 0xff;
                        msk[i-3] = 0xff;
                        msk[i-2] = 0xff;
                        msk[i-1] = 0xff;
                        msk[i-0] = 0xff;
                        break;
                    case 1:
                        msk[i-7] = 0;
                        msk[i-6] = 0;
                        msk[i-5] = 0;
                        msk[i-4] = 0;
                        msk[i-3] = 0;
                        msk[i-2] = 0;
                        msk[i-1] = 0;
                        msk[i-0] = 0;
                        break;
                    default:
                        break;
                    }
                    switch (rand() % 3)
                    {
                    case 0:
                        src[i-7] |= 0xff000000;
                        src[i-6] |= 0xff000000;
                        src[i-5] |= 0xff000000;
                        src[i-4] |= 0xff000000;
                        src[i-3] |= 0xff000000;
                        src[i-2] |= 0xff000000;
                        src[i-1] |= 0xff000000;
                        src[i-0] |= 0xff000000;
                        break;
                    case 1:
                        src[i-7] = 0;
                        src[i-6] = 0;
                        src[i-5] = 0;
                        src[i-4] = 0;
                        src[i-3] = 0;
                        src[i-2] = 0;
                        src[i-1] = 0;
                        src[i-0] = 0;
                        break;
                    default:
                        break;
                    }
                }
            }

            neon::blit_row_s32a_a8(dst1+offset, msk, src, len);
            new_blit_row_s32a_a8(dst2+offset, msk, src, len);
            size_t i;
            for (i = 0; i < PIXELS; i++)
                if (diffword(dst1[i], dst2[i]))
                    break;
            if (i < PIXELS) {
//            if (memcmp(dst1, dst2, sizeof dst1) != 0) {
                printf("offset %d pixels, length %d pixels\n", offset, len);
                printf("src");
                for (size_t i = 0; i < PIXELS; i++)
                    printf(" %08X", src[i]);
                printf("\n");

                printf("msk");
                for (size_t i = 0; i < PIXELS; i++)
                    printf("       %02X", msk[i]);
                printf("\n");

                printf("pre");
                for (size_t i = 0; i < PIXELS; i++)
                    printf(" %08X", dst0[i]);
                printf("\n");

                printf("old");
                for (size_t i = 0; i < PIXELS; i++)
                    printf(" %08X", dst1[i]);
                printf("\n");

                printf("new");
                for (size_t i = 0; i < PIXELS; i++)
                    printf(" %08X", dst2[i]);
                printf("\n");

                printf("   ");
                for (size_t i = 0; i < PIXELS; i++) {
                    uint64_t diff = diffword(dst1[i], dst2[i]);
                    if (diff == 0)
                        printf("         ");
                    else {
                        diff *= '^' - ' ';
                        diff += 0x0101010101010101ll * ' ';
                        printf(" %.8s", (char *) &diff);
                    }
                }
                printf("\n");

                exit(EXIT_FAILURE);
            }
        }
    }
#endif

#if 1

//#define CANDIDATE neon::blit_row_s32a_a8
#define CANDIDATE new_blit_row_s32a_a8

    uint64_t t1, t2, t3;
    uint32_t byte_cnt;
    static SkPMColor bufa[TESTSIZE / sizeof (SkPMColor)] __attribute__((aligned(4096)));
    static SkPMColor bufb[TESTSIZE / sizeof (SkPMColor)] __attribute__((aligned(4096)));
    static SkAlpha bufc[TESTSIZE / sizeof (SkPMColor)] __attribute__((aligned(4096)));
#if 1 // translucent
#define M_VALUE 0x80
#define S_VALUE 0x80
#elif 1 // transparent
#define M_VALUE 0
#define S_VALUE 0
#elif 1 // transparent source, opaque mask
#define M_VALUE 0xff
#define S_VALUE 0
#else // opaque
#define M_VALUE 0xff
#define S_VALUE 0xff
#endif
    memset(bufa, 0, sizeof bufa);
    memset(bufb, S_VALUE, sizeof bufb);
    memset(bufc, M_VALUE, sizeof bufc);

#define BENCH(bytes_per_loop, separator)                                                                       \
    do {                                                                                                       \
        t1 = gettime();                                                                                        \
        bench(control, bufa, bufb, bufc, (bytes_per_loop) / sizeof *bufa, TESTSIZE / (bytes_per_loop));              \
        t2 = gettime();                                                                                        \
        byte_cnt = bench(CANDIDATE, bufa, bufb, bufc, (bytes_per_loop) / sizeof *bufa, TESTSIZE / (bytes_per_loop)); \
        t3 = gettime();                                                                                        \
        printf("%6.2f" separator, ((double)byte_cnt) / ((t3 - t2) - (t2 - t1)));                               \
        fflush(stdout);                                                                                        \
    } while (0)

    BENCH(L1CACHESIZE / 4 - 64 * sizeof (SkPMColor), ", ");
    BENCH(L2CACHESIZE / 4 - 64 * sizeof (SkPMColor), ", ");
    BENCH(TESTSIZE - 64 * sizeof (SkPMColor), "\n");

#endif
}
