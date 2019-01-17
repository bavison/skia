#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <arm_neon.h>

#define PIXELS 1920
//#define PIXELS 128
//#define PIXELS 8
#define ROWS 1080

#define KILOBYTE (1024)
#define MEGABYTE (1024*1024)

#define L1CACHESIZE (32*KILOBYTE)
#define L2CACHESIZE (512*KILOBYTE)
#define TESTSIZE (200*MEGABYTE)

#define MIN(a,b) ((a)<(b)?(a):(b))

typedef uint32_t SkPMColor;

class SkPixmap {
public:
    SkPixmap(const void* addr, size_t rowBytes) : fPixels(addr), fRowBytes(rowBytes) {}
    size_t rowBytes() const { return fRowBytes; }
    const void* addr() const { return fPixels; }
private:
    const void* fPixels;
    size_t fRowBytes;
};

struct SkBitmapProcInfo {
    SkBitmapProcInfo(const void* addr, size_t rowBytes) : fPixmap(addr, rowBytes) {}
    SkPixmap fPixmap;
};

struct SkBitmapProcState : public SkBitmapProcInfo {
    SkBitmapProcState(const void* addr, size_t rowBytes) : SkBitmapProcInfo(addr, rowBytes) {}
};

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


/* Just used for cancelling out the overheads */
static void control(const SkBitmapProcState& s,
        const uint32_t* __restrict__ xy,
         int count, SkPMColor* __restrict__ colors)
{
}


/*
 * Filter_32_opaque
 *
 * There is no hard-n-fast rule that the filtering must produce
 * exact results for the color components, but if the 4 incoming colors are
 * all opaque, then the output color must also be opaque. Subsequent parts of
 * the drawing pipeline may rely on this (e.g. which blitrow proc to use).
 *
 */
// Chrome on Android uses -Os so we need to force these inline. Otherwise
// calling the function in the inner loops will cause significant overhead on
// some platforms.
static __attribute__((always_inline)) inline void Filter_32_opaque_neon(unsigned x, unsigned y,
                                                   SkPMColor a00, SkPMColor a01,
                                                   SkPMColor a10, SkPMColor a11,
                                                   SkPMColor *dst) {
    uint8x8_t vy, vconst16_8, v16_y, vres;
    uint16x4_t vx, vconst16_16, v16_x, tmp;
    uint32x2_t va0, va1;
    uint16x8_t tmp1, tmp2;

    vy = vdup_n_u8(y); // duplicate y into vy
    vconst16_8 = vmov_n_u8(16); // set up constant in vconst16_8
    v16_y = vsub_u8(vconst16_8, vy); // v16_y = 16-y

    va0 = vdup_n_u32(a00); // duplicate a00
    va1 = vdup_n_u32(a10); // duplicate a10
    va0 = __extension__ ({ uint32_t __s0 = a01; uint32x2_t __s1 = va0; uint32x2_t __ret; __ret = (uint32x2_t) __builtin_neon_vset_lane_i32(__s0, (int8x8_t)__s1, 1); __ret; }); // set top to a01
    va1 = __extension__ ({ uint32_t __s0 = a11; uint32x2_t __s1 = va1; uint32x2_t __ret; __ret = (uint32x2_t) __builtin_neon_vset_lane_i32(__s0, (int8x8_t)__s1, 1); __ret; }); // set top to a11

    tmp1 = vmull_u8(vreinterpret_u8_u32(va0), v16_y); // tmp1 = [a01|a00] * (16-y)
    tmp2 = vmull_u8(vreinterpret_u8_u32(va1), vy); // tmp2 = [a11|a10] * y

    vx = vdup_n_u16(x); // duplicate x into vx
    vconst16_16 = vmov_n_u16(16); // set up constant in vconst16_16
    v16_x = vsub_u16(vconst16_16, vx); // v16_x = 16-x

    tmp = vmul_u16(vget_high_u16(tmp1), vx); // tmp  = a01 * x
    tmp = vmla_u16(tmp, vget_high_u16(tmp2), vx); // tmp += a11 * x
    tmp = vmla_u16(tmp, vget_low_u16(tmp1), v16_x); // tmp += a00 * (16-x)
    tmp = vmla_u16(tmp, vget_low_u16(tmp2), v16_x); // tmp += a10 * (16-x)

    vres = __extension__ ({ uint16x8_t __s0 = vcombine_u16(tmp, vcreate_u16(0)); uint8x8_t __ret; __ret = (uint8x8_t) __builtin_neon_vshrn_n_v((int8x16_t)__s0, 8, 16); __ret; }); // shift down result by 8
    __extension__ ({ uint32x2_t __s1 = vreinterpret_u32_u8(vres); __builtin_neon_vst1_lane_v(dst, (int8x8_t)__s1, 0, 18); }); // store result
}

void S32_opaque_D32_filter_DX_neon(const SkBitmapProcState& s,
                          const uint32_t* __restrict__ xy,
                           int count, SkPMColor* __restrict__ colors) {
    static_cast<void>(0);
    static_cast<void>(0);

    const char* __restrict__ srcAddr = (const char*)s.fPixmap.addr();
    size_t rb = s.fPixmap.rowBytes();
    unsigned subY;
    const SkPMColor* __restrict__ row0;
    const SkPMColor* __restrict__ row1;

    // setup row ptrs and update proc_table
    {
        uint32_t XY = *xy++;
        unsigned y0 = XY >> 14;
        row0 = (const SkPMColor*)(srcAddr + (y0 >> 4) * rb);
        row1 = (const SkPMColor*)(srcAddr + (XY & 0x3FFF) * rb);
        subY = y0 & 0xF;
    }

    do {
        uint32_t XX = *xy++; // x0:14 | 4 | x1:14
        unsigned x0 = XX >> 14;
        unsigned x1 = XX & 0x3FFF;
        unsigned subX = x0 & 0xF;
        x0 >>= 4;

        Filter_32_opaque_neon(subX, subY, row0[x0], row0[x1], row1[x0], row1[x1], colors);
        colors += 1;

    } while (--count != 0);
}

#define S32_OPAQUE_D32_FILTER_DX_1PIX_NEON(opt)              \
            "ldr         %[x], [%[xy]], #4           \n\t"   \
            "uxth        %[tmp2], %[x], ror #16      \n\t"   \
            "lsl         %[tmp3], %[x], #2           \n\t"   \
            "bic         %[tmp2], #3                 \n\t"   \
            "uxth        %[tmp3], %[tmp3]            \n\t"   \
            "add         %[tmp0], %[row0], %[tmp2]   \n\t"   \
            "add         %[tmp1], %[row0], %[tmp3]   \n\t"   \
            "add         %[tmp2], %[row1], %[tmp2]   \n\t"   \
            "add         %[tmp3], %[row1], %[tmp3]   \n\t"   \
            "lsr         %[x], #14                   \n\t"   \
            "vldr        s0, [%[tmp0]]               \n\t"   \
            "and         %[x], #0xf                  \n\t"   \
            "vldr        s1, [%[tmp1]]               \n\t"   \
            "vldr        s2, [%[tmp2]]               \n\t"   \
            "vldr        s3, [%[tmp3]]               \n\t"   \
            "vdup.16     d2, %[x]                    \n\t"   \
            "vsub.i16    d3, d23, d2                 \n\t"   \
            "vmull.u8    q2, d0, d31                 \n\t"   \
            "vmlal.u8    q2, d1, d30                 \n\t"   \
            "vmul.u16    d0, d4, d3                  \n\t"   \
            "vmla.u16    d0, d5, d2                  \n\t"   \
            opt"                                     \n\t"   \
            "vshrn.u16   d0, q0, #8                  \n\t"   \
            "vst1.32     {d0[0]}, [%[dst]:32]!       \n\t"   \

void __attribute__((noinline)) new_S32_opaque_D32_filter_DX_neon(const SkBitmapProcState& s,
                          const uint32_t* __restrict__ xy,
                          int count, SkPMColor* __restrict__ colors) {
    const char* __restrict__ srcAddr = (const char*)s.fPixmap.addr();
    size_t rb = s.fPixmap.rowBytes();
    unsigned subY;
    const SkPMColor* __restrict__ row0;
    const SkPMColor* __restrict__ row1;

    // setup row ptrs and update proc_table
    {
        uint32_t XY = *xy++;
        unsigned y0 = XY >> 14;
        row0 = (const SkPMColor*)(srcAddr + (y0 >> 4) * rb);
        row1 = (const SkPMColor*)(srcAddr + (XY & 0x3FFF) * rb);
        subY = y0 & 0xF;
    }

    uint32_t tmp0, tmp1, tmp2, tmp3, x;
    __asm__ volatile (
            "vpush       {q4}                        \n\t"
            "vmov.i16    d22, #0xf                   \n\t"
            "vmov.i16    d23, #0x10                  \n\t"
            "vmov.i32    q12, #0x3fff                \n\t"
            "vdup.32     q13, %[row0]                \n\t"
            "vdup.32     q14, %[row1]                \n\t"
            "vdup.i8     d30, %[subY]                \n\t"
            "vmov.i8     d31, #16                    \n\t"
            "vshl.i32    q12, #2                     \n\t"
            "tst         %[dst], #0xc                \n\t"
            "vsub.i8     d31, d30                    \n\t"
            "beq         2f                          \n\t"

            "1:                                      \n\t"
            S32_OPAQUE_D32_FILTER_DX_1PIX_NEON(
            "add         %[tmp0], %[dst], #4         \n\t"
            "subs        %[len], #1                  \n\t"
            "it          ne                          \n\t"
            "tstne       %[tmp0], #0xc"
            )
            "bne         1b                          \n\t"

            "2:"
            "subs        %[len], #4                  \n\t"
            "bmi         13f                         \n\t"

            "vld1.32     {q8}, [%[xy]]!              \n\t"
            "vshr.u32    q9, q8, #16                 \n\t"
            "vand        q9, q12                     \n\t"
            "vadd.i32    q1, q13, q9                 \n\t"
            "vshl.i32    q0, q8, #2                  \n\t"
            "vand        q0, q12                     \n\t"
            "vadd.i32    q2, q13, q0                 \n\t"
            "vmov        %[tmp0], s4                 \n\t"
            "vmov        %[tmp1], s5                 \n\t"

            "11:                                     \n\t"
            "vadd.i32    q3, q14, q9                 \n\t"
            "vmov        %[tmp2], %[tmp3], d3        \n\t"
            "vadd.i32    q4, q14, q0                 \n\t"
            "vldr        s4, [%[tmp0]]               \n\t"
            "vmov        %[tmp0], s8                 \n\t"
            "vldr        s5, [%[tmp1]]               \n\t"
            "vmov        %[tmp1], s9                 \n\t"
            "vldr        s6, [%[tmp2]]               \n\t"
            "vmov        %[tmp2], s10                \n\t"
            "vldr        s7, [%[tmp3]]               \n\t"
            "vmov        %[tmp3], s11                \n\t"
            "vldr        s8, [%[tmp0]]               \n\t"
            "vmov        %[tmp0], s12                \n\t"
            "vldr        s9, [%[tmp1]]               \n\t"
            "vmov        %[tmp1], s13                \n\t"
            "vldr        s10, [%[tmp2]]              \n\t"
            "vmov        %[tmp2], s14                \n\t"
            "vldr        s11, [%[tmp3]]              \n\t"
            "vmov        %[tmp3], s15                \n\t"
            "vldr        s12, [%[tmp0]]              \n\t"
            "vmov        %[tmp0], s16                \n\t"
            "vldr        s13, [%[tmp1]]              \n\t"
            "vmov        %[tmp1], s17                \n\t"
            "vldr        s14, [%[tmp2]]              \n\t"
            "vmov        %[tmp2], s18                \n\t"
            "vldr        s15, [%[tmp3]]              \n\t"
            "vmov        %[tmp3], s19                \n\t"
            "vldr        s16, [%[tmp0]]              \n\t"
            "vshrn.i32   d1, q8, #14                 \n\t"
            "vldr        s17, [%[tmp1]]              \n\t"
            "vand        d1, d22                     \n\t"
            "vldr        s18, [%[tmp2]]              \n\t"
            "vsub.i16    d0, d23, d1                 \n\t"
            "vldr        s19, [%[tmp3]]              \n\t"
            "vmull.u8    q10, d2, d31                \n\t"
            "vmlal.u8    q10, d6, d30                \n\t"
            "vmull.u8    q1, d3, d31                 \n\t"
            "vmlal.u8    q1, d7, d30                 \n\t"
            "vmull.u8    q3, d4, d31                 \n\t"
            "subs        %[len], #4                  \n\t"
            "vmlal.u8    q3, d8, d30                 \n\t"
            "bmi         12f                         \n\t"

            "  vld1.32     {q8}, [%[xy]]!            \n\t"
            "vmull.u8    q2, d5, d31                 \n\t"
            "vmlal.u8    q2, d9, d30                 \n\t"
            "vmul.u16    d8, d20, d0[0]              \n\t"
            "  vshr.u32    d18, d16, #16             \n\t"
            "vmul.u16    d9, d21, d0[1]              \n\t"
            "  vshr.u32    d19, d17, #16             \n\t"
            "vmul.u16    d20, d2, d0[2]              \n\t"
            "  vand        d18, d24                  \n\t"
            "vmul.u16    d21, d3, d0[3]              \n\t"
            "  vand        d19, d25                  \n\t"
            "vmla.u16    d8, d6, d1[0]               \n\t"
            "  vadd.i32    d2, d26, d18              \n\t"
            "vmla.u16    d9, d7, d1[1]               \n\t"
            "  vadd.i32    d3, d27, d19              \n\t"
            "vmla.u16    d20, d4, d1[2]              \n\t"
            "  vshl.i32    d0, d16, #2               \n\t"
            "vmla.u16    d21, d5, d1[3]              \n\t"
            "  vshl.i32    d1, d17, #2               \n\t"
            "  vand        q0, q12                   \n\t"
            "  vadd.i32    q2, q13, q0               \n\t"
            "  vmov        %[tmp0], s4               \n\t"
            "  vmov        %[tmp1], s5               \n\t"
            "vshrn.u16   d8, q4, #8                  \n\t"
            "vshrn.u16   d9, q10, #8                 \n\t"
            "vst1.32     {q4}, [%[dst]:128]!         \n\t"
            "b           11b                         \n\t"

            "12:                                     \n\t"
            "vmull.u8    q2, d5, d31                 \n\t"
            "vmlal.u8    q2, d9, d30                 \n\t"
            "vmul.u16    d8, d20, d0[0]              \n\t"
            "vmul.u16    d9, d21, d0[1]              \n\t"
            "vmul.u16    d20, d2, d0[2]              \n\t"
            "vmul.u16    d21, d3, d0[3]              \n\t"
            "vmla.u16    d8, d6, d1[0]               \n\t"
            "vmla.u16    d9, d7, d1[1]               \n\t"
            "vmla.u16    d20, d4, d1[2]              \n\t"
            "vmla.u16    d21, d5, d1[3]              \n\t"
            "vshrn.u16   d8, q4, #8                  \n\t"
            "vshrn.u16   d9, q10, #8                 \n\t"
            "vst1.32     {q4}, [%[dst]:128]!         \n\t"

            "13:                                     \n\t"
            "adds        %[len], #4-1                \n\t"
            "bmi         22f                         \n\t"

            "21:                                     \n\t"
            S32_OPAQUE_D32_FILTER_DX_1PIX_NEON("subs %[len], #1")
            "bpl         21b                         \n\t"

            "22:                                     \n\t"
            "vpop        {q4}                        \n\t"
    : // Outputs
             [dst]"+r"(colors),
              [xy]"+r"(xy),
             [len]"+r"(count),
            [tmp0]"=&r"(tmp0),
            [tmp1]"=&r"(tmp1),
            [tmp2]"=&r"(tmp2),
            [tmp3]"=&r"(tmp3),
               [x]"=&r"(x)
    : // Inputs
            [row0]"r"(row0),
            [row1]"r"(row1),
            [subY]"r"(subY)
    : // Clobbers
            "cc", "memory"
    );
}

static uint32_t bench(void(*test)(const SkBitmapProcState& s, const uint32_t* __restrict__ xy, int count, SkPMColor* __restrict__ colors),
        const SkBitmapProcState& s, uint32_t * __restrict__ xy, SkPMColor* __restrict__ colors, size_t rows)
{
    int i, x = 0, row0 = 0, row1 = 1, len = PIXELS-64,
            times = TESTSIZE / (len * sizeof *colors);
    for (i = times; i >= 0; i--)
    {
        /* If we cared about CPUs with non-write-allocate caches and operations that
         * don't read from the destination buffer, we'd need to ensure the destination
         * hadn't been evicted from the cache here - but this isn't the case */
        *xy = (row0 << 18) | (4 << 14) | (row1 << 0);
        x = (x + 1) & 63;
        test(s, xy, len, colors + row0 * PIXELS + x);
        ++row0;
        ++row1;
        if (row0 == rows) { row0 = 0; row1 = 1; }
    }
    return len * times * sizeof *colors;
}

int main(void)
{
    uint32_t src[2][PIXELS] __attribute__((aligned(16))),
            dst0[PIXELS] __attribute__((aligned(16))),
            dst1[PIXELS] __attribute__((aligned(16))),
            dst2[PIXELS] __attribute__((aligned(16)));
    srand(0);

    /* Random (but reproducible) initial data; same for all destination arrays */
    for (size_t i = 0; i < PIXELS; i++) {
#if 0
        src[0][i] = 0x55aa55aa;
        src[1][i] = 0xaa55aa55;
#else
        src[0][i] = rand32();
        src[1][i] = rand32();
#endif
        dst0[i] = dst1[i] = dst2[i] = rand32();
    }

    /* Create interpolation coefficients: slight horizontal enlargement, quarter way from row 0 to row 1 */
    uint32_t xy[1+PIXELS];
    xy[0] = (0 << 18) | (4 << 14) | (1 << 0);
    uint32_t x14p4 = 0;
    for (size_t i = 0; i < PIXELS; i++) {
        xy[1+i] = ((x14p4 >> 4) << 18) | ((x14p4 & 0xf) << 14) | (MIN((x14p4 >> 4) + 1, PIXELS-1) << 0);
        x14p4 += 16;
    }

    for (size_t offset = 0; offset <= 4; offset += 1) {
        for (size_t len = 1; len <= PIXELS - offset; len += 1) {
            S32_opaque_D32_filter_DX_neon(SkBitmapProcState(src, PIXELS * sizeof(SkPMColor)), xy, len, dst1 + offset);
            new_S32_opaque_D32_filter_DX_neon(SkBitmapProcState(src, PIXELS * sizeof(SkPMColor)), xy, len, dst2 + offset);
            if (memcmp(dst1, dst2, sizeof dst1) != 0) {
                printf("offset %d pixels, length %d pixels\n", offset, len);
                printf("src");
                for (size_t i = 0; i < PIXELS; i++)
                    printf(" %08X", src[0][i]);
                printf("\n");

                printf("   ");
                for (size_t i = 0; i < PIXELS; i++)
                    printf(" %08X", src[1][i]);
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
                for (size_t i = 0; i < PIXELS; i++)
                    printf(dst1[i] == dst2[i] ? "         " : " ^^^^^^^^");
                printf("\n");

                exit(EXIT_FAILURE);
            }
        }
    }

//#define CANDIDATE S32_opaque_D32_filter_DX_neon
#define CANDIDATE new_S32_opaque_D32_filter_DX_neon

    uint64_t t1, t2, t3;
    uint32_t byte_cnt;
    static SkPMColor bufa[ROWS+1][PIXELS];
    static SkPMColor bufb[ROWS][PIXELS];
    SkBitmapProcState s(bufa, PIXELS * sizeof(SkPMColor));
#define VALUE 0x80
    memset(bufa, VALUE, sizeof bufa);
    memset(bufb, VALUE, sizeof bufb);

#define BENCH(rows, separator)                                                   \
    do {                                                                         \
        t1 = gettime();                                                          \
        bench(control, s, xy, bufb[0], rows);                                    \
        t2 = gettime();                                                          \
        byte_cnt = bench(CANDIDATE, s, xy, bufb[0], rows);                       \
        t3 = gettime();                                                          \
        printf("%6.2f" separator, ((double)byte_cnt) / ((t3 - t2) - (t2 - t1))); \
        fflush(stdout);                                                          \
    } while (0)

    BENCH((L1CACHESIZE / 4) / ((PIXELS-64) * sizeof (SkPMColor)), ", ");
    BENCH((L2CACHESIZE / 4) / ((PIXELS-64) * sizeof (SkPMColor)), ", ");
    BENCH(ROWS, "\n");
}
