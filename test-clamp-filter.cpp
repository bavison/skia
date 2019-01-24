#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

typedef int64_t SkFixed3232;
typedef SkFixed3232 SkFractionalInt;
typedef int32_t SkFixed;
#define SK_Fixed1           (1 << 16)

static inline int SkClampMax(int value, int max) {
    // ensure that max is positive
    static_cast<void>(0);
    if (value < 0) {
        value = 0;
    }
    if (value > max) {
        value = max;
    }
    return value;
}

static inline uint32_t ClampX_ClampY_pack_filter_x_neon(SkFixed f, unsigned max,
                                          SkFixed one ) {
    unsigned i = SkClampMax((f) >> 16, max);
    i = (i << 4) | (((f) >> 12) & 0xF);
    return (i << 14) | (SkClampMax(((f + one)) >> 16, max));
}

static void oldversion(uint32_t xy[], int count, SkFractionalInt fx, SkFractionalInt dx, unsigned maxX)
{
    while (--count >= 0) {
        *xy++ = ClampX_ClampY_pack_filter_x_neon(((SkFixed)((fx) >> 16)), maxX, SK_Fixed1 );
        fx += dx;
    }
}

static __attribute__((noinline)) void newversion(uint32_t xy[], int count, SkFractionalInt fx, SkFractionalInt dx, unsigned maxX)
{
    if (dx >= 0) {
        --count;
        while (count >= 0 && fx < 0) {
            *xy++ = 0;
            fx += dx;
            --count;
        }
        while (count >= 0 && ((uintptr_t) xy & 0xf) && fx < ((SkFractionalInt) maxX << 32)) {
            *xy++ = ((uint32_t)(fx >> 14) & 0xffffc000) + (uint32_t)(fx >> 32) + 1;
            fx += dx;
            --count;
        }
        if ((count -= 8-1) >= 0 && fx + 7*dx < ((SkFractionalInt) maxX << 32)) {
            SkFractionalInt rem = (((SkFractionalInt) maxX << 32) - 7*dx - fx - 1) / 8;
            int32_t rem_hi = rem >> 32;
            uint32_t rem_lo = (uint32_t) rem;
            int32_t fx_hi = fx >> 32;
            uint32_t fx_lo = (uint32_t) fx;
            __asm__ (
                    "vmov        d16, %[fx_lo], %[fx_hi]     \n\t"
                    "vmov        d24, %[dx_lo], %[dx_hi]     \n\t"
                    "vadd.i64    d17, d16, d24               \n\t"
                    "vmov        d25, %[dx_lo], %[dx_hi]     \n\t"
                    "vmvn.i32    q13, #0x3fff                \n\t"
                    "vadd.i64    d18, d17, d24               \n\t"
                    "vmov.i32    q14, #1                     \n\t"
                    "vadd.i64    d19, d18, d24               \n\t"
                    "vshl.i64    q12, #2                     \n\t"
                    "b           2f                          \n\t"
                    "1:                                      \n\t"
                    "vadd.i64    q8, q10, q12                \n\t"
                    "vadd.i64    q9, q11, q12                \n\t"
                    "2:                                      \n\t"
                    "vadd.i64    q10, q8, q12                \n\t"
                    "vadd.i64    q11, q9, q12                \n\t"
                    "vshrn.i64   d16, q8, #14                \n\t"
                    "vshrn.i64   d17, q9, #14                \n\t"
                    "vand        q8, q13                     \n\t"
                    "vorr        q8, q14                     \n\t"
                    "vshrn.i64   d18, q10, #14               \n\t"
                    "vshrn.i64   d19, q11, #14               \n\t"
                    "vand        q9, q13                     \n\t"
                    "subs        %[rem_lo], %[dx_lo]         \n\t"
                    "vorr        q9, q14                     \n\t"
                    "sbcs        %[rem_hi], %[dx_hi]         \n\t"
                    "vsra.u32    q8, #18                     \n\t"
                    "subs        %[count], #8                \n\t"
                    "vsra.u32    q9, #18                     \n\t"
                    "it          pl                          \n\t"
                    "teqpl       %[rem_hi], #0               \n\t"
                    "vst1.32     {q8-q9}, [%[dst]:128]!      \n\t"
                    "bpl         1b                          \n\t"
                    "vadd.i64    d16, d20, d24               \n\t"
                    "vmov        %[fx_lo], %[fx_hi], d16     \n\t"
            : // Outputs
                     [count]"+l"(count),
                       [dst]"+r"(xy),
                    [rem_hi]"+l"(rem_hi),
                    [rem_lo]"+l"(rem_lo),
                     [fx_hi]"+r"(fx_hi),
                     [fx_lo]"+r"(fx_lo)
            : // Inputs
                    [dx_hi]"l"((int32_t) (dx >> 32)),
                    [dx_lo]"l"((uint32_t) dx)
            : // Clobbers
                    "cc", "memory"
            );
            fx = ((SkFractionalInt) fx_hi << 32) | fx_lo;
        }
        count += 8-1;
        while (count >= 0 && fx < ((SkFractionalInt) maxX << 32)) {
            *xy++ = ((uint32_t)(fx >> 14) & 0xffffc000) + (uint32_t)(fx >> 32) + 1;
            fx += dx;
            --count;
        }
        while (count >= 0) {
            *xy++ = (maxX << 18) + maxX;
            --count;
        }
    } else {
        // Reflection case. Don't bother to optimize this as much -
        // not even sure if it's used!
        while (count >= 1 && fx >= ((SkFractionalInt) maxX << 32)) {
            *xy++ = (maxX << 18) + maxX;
            fx += dx;
            --count;
        }
        while (count >= 1 && fx >= 0) {
            *xy++ = ((uint32_t)(fx >> 14) & 0xffffc000) + (uint32_t)(fx >> 32) + 1;
            fx += dx;
            --count;
        }
        while (count >= 1) {
            *xy++ = 0;
            --count;
        }
    }
}

int main(void)
{
#define PIXELS 32
    uint32_t xyold[PIXELS] __attribute__((aligned(16)));
    uint32_t xynew[PIXELS] __attribute__((aligned(16)));
    SkFractionalInt fxdx[][2] = {
            { -0x3ffffffff,  0x1ffffffff },
            { -0x3ffffffff,  0x0ffffffff },
            { -0x3ffffffff,  0x07fffffff },
            { -0x3ffffffff, -0x100000001 },
            {  0x000000001,  0x1ffffffff },
            {  0x000000001,  0x0ffffffff },
            {  0x000000001,  0x07fffffff },
            {  0x000000001, -0x100000001 },
            { 0x1700000001,  0x1ffffffff },
            { 0x1700000001,  0x0ffffffff },
            { 0x1700000001,  0x07fffffff },
            { 0x1700000001, -0x100000001 },
            { 0x1f00000001,  0x1ffffffff },
            { 0x1f00000001,  0x0ffffffff },
            { 0x1f00000001,  0x07fffffff },
            { 0x1f00000001, -0x100000001 },
            { 0x2000000001,  0x1ffffffff },
            { 0x2000000001,  0x0ffffffff },
            { 0x2000000001,  0x07fffffff },
            { 0x2000000001, -0x100000001 },
    };

    for (int offset = 0; offset < PIXELS; offset++)
        for (int count = 0; count <= PIXELS - offset; count++)
            for (int i = 0; i < sizeof fxdx / sizeof *fxdx; i++)
            {
                memset(xyold, 0, sizeof xyold);
                memset(xynew, 0, sizeof xynew);
                oldversion(xyold + offset, count, fxdx[i][0], fxdx[i][1], PIXELS-1);
                for (int fixup = 0; fixup < PIXELS; fixup++) {
                    if ((xyold[fixup] & 0x3fff) == 0)
                        xyold[fixup] = 0;
                    if ((xyold[fixup] >> 18) == (PIXELS-1))
                        xyold[fixup] = ((PIXELS-1) << 18) | (PIXELS-1);
                }
                newversion(xynew + offset, count, fxdx[i][0], fxdx[i][1], PIXELS-1);
                if (memcmp(xyold, xynew, sizeof xyold) != 0)
                {
                    fprintf(stderr, "Fault at offset %d, count %d, fx %16llX, dx %16llX\n", offset, count, fxdx[i][0], fxdx[i][1]);
                    fprintf(stderr, "old");
                    for (int x = 0; x < PIXELS; x++)
                        fprintf(stderr, " %04X.%X.%04X", xyold[x] >> 18, (xyold[x] >> 14) & 0xf, xyold[x] & 0x3fff);
                    fprintf(stderr, "\nnew");
                    for (int x = 0; x < PIXELS; x++)
                        fprintf(stderr, " %04X.%X.%04X", xynew[x] >> 18, (xynew[x] >> 14) & 0xf, xynew[x] & 0x3fff);
                    fprintf(stderr, "\n   ");
                    for (int x = 0; x < PIXELS; x++)
                        fprintf(stderr, xynew[x] == xyold[x] ? "            " : " ^^^^^^^^^^^");
                    fprintf(stderr, "\n");
                    exit(EXIT_FAILURE);
                }
            }
}
