#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <arm_neon.h>

typedef int32_t SkFixed;

void new_decal_filter_scale_neon(uint32_t dst[], SkFixed fx, SkFixed dx, int count) {
//    SkASSERT(((fx + (count-1) * dx) >> (16 + 14)) == 0);
    fx = (fx << 2) + 1;
    dx <<= 2;
    while (((uintptr_t) dst & 0xf) && --count >= 0) {
        *dst++ = (fx & 0xffffc001) + (fx >> 18);
        fx += dx;
    }
    if ((count -= 4) >= 0) {
        uint32_t tmp;
        __asm__ (
                "adr         %[tmp], 1f                  \n\t"
                "vmvn.i32    q10, #0x3fff                \n\t"
                "vld1.32     {q11}, [%[tmp]]             \n\t"
                "vdup.32     q8, %[fx]                   \n\t"
                "vdup.32     q9, %[dx]                   \n\t"
                "vsra.u32    q10, #31                    \n\t"
                "vmla.u32    q8, q9, q11                 \n\t"
                "vshl.u32    q9, #2                      \n\t"
                "b           2f                          \n\t"
                "1:                                      \n\t"
                ".long       0                           \n\t"
                ".long       1                           \n\t"
                ".long       2                           \n\t"
                ".long       3                           \n\t"
                "2:                                      \n\t"
                "vand        q11, q8, q10                \n\t"
                "vshr.u32    q12, q8, #18                \n\t"
                "vadd.i32    q11, q12                    \n\t"
                "vadd.i32    q8, q9                      \n\t"
                "subs        %[count], #4                \n\t"
                "vst1.32     {q11}, [%[dst]:128]!        \n\t"
                "bpl         2b                          \n\t"
                "vmov.32     %[fx], d16[0]               \n\t"
        : // Outputs
                [count]"+l"(count),
                  [dst]"+r"(dst),
                   [fx]"+r"(fx),
                  [tmp]"=&r"(tmp)
        : // Inputs
                [dx]"r"(dx)
        : // Clobbers
                "cc", "memory"
        );
    }
    if ((count += 4-1) >= 0) {
        do {
            *dst++ = (fx & 0xffffc001) + (fx >> 18);
            fx += dx;
        } while (--count >= 0);
    }
}
void decal_filter_scale_neon(uint32_t dst[], SkFixed fx, SkFixed dx, int count) {
    if (count >= 8) {
        SkFixed dx8 = dx * 8;
        int32x4_t vdx8 = vdupq_n_s32(dx8);

        int32x4_t wide_fx, wide_fx2;
        wide_fx = vdupq_n_s32(fx);
        wide_fx = vsetq_lane_s32(fx + dx, wide_fx, 1);
        wide_fx = vsetq_lane_s32(fx + dx + dx, wide_fx, 2);
        wide_fx = vsetq_lane_s32(fx + dx + dx + dx, wide_fx, 3);

        wide_fx2 = vaddq_s32(wide_fx, vdupq_n_s32(4 * dx));

        while (count >= 8) {
            int32x4_t wide_out;
            int32x4_t wide_out2;

            wide_out = vshlq_n_s32(vshrq_n_s32(wide_fx, 12), 14);
            wide_out = wide_out | (vshrq_n_s32(wide_fx,16) + vdupq_n_s32(1));

            wide_out2 = vshlq_n_s32(vshrq_n_s32(wide_fx2, 12), 14);
            wide_out2 = wide_out2 | (vshrq_n_s32(wide_fx2,16) + vdupq_n_s32(1));

            vst1q_u32(dst, vreinterpretq_u32_s32(wide_out));
            vst1q_u32(dst+4, vreinterpretq_u32_s32(wide_out2));

            dst += 8;
            fx += dx8;
            wide_fx += vdx8;
            wide_fx2 += vdx8;
            count -= 8;
        }
    }

    if (count & 1)
    {
//        SkASSERT((fx >> (16 + 14)) == 0);
        *dst++ = (fx >> 12 << 14) | ((fx >> 16) + 1);
        fx += dx;
    }
    while ((count -= 2) >= 0)
    {
//        SkASSERT((fx >> (16 + 14)) == 0);
        *dst++ = (fx >> 12 << 14) | ((fx >> 16) + 1);
        fx += dx;

        *dst++ = (fx >> 12 << 14) | ((fx >> 16) + 1);
        fx += dx;
    }
}

#define PIXELS 20

int main(void)
{
  uint32_t __attribute__((aligned(16))) xy1[PIXELS], __attribute__((aligned(16))) xy2[PIXELS];
  for (int offset = 0; offset <= 4; offset++) {
    for (int length = 1; length < PIXELS-offset; length++) {
      memset(xy1, 0, sizeof xy1);
      decal_filter_scale_neon(xy1+offset, 0xff0000, 0xffff, length);
      memset(xy2, 0, sizeof xy2);
      new_decal_filter_scale_neon(xy2+offset, 0xff0000, 0xffff, length);
      if (memcmp(xy1, xy2, sizeof xy1) != 0) {
        fprintf(stderr, "Failed with offset %d, length %d\n", offset, length);
        exit(EXIT_FAILURE);
      }
    }
  }
}

