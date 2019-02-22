/*
 * Copyright 2017 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkUtils_opts_DEFINED
#define SkUtils_opts_DEFINED

#include <stdint.h>
#include "SkNx.h"

namespace SK_OPTS_NS {

    template <typename T>
    static void memsetT(T buffer[], T value, int count) {
    #if defined(__AVX__)
        static const int N = 32 / sizeof(T);
    #else
        static const int N = 16 / sizeof(T);
    #endif
        while (count >= N) {
            SkNx<N,T>(value).store(buffer);
            buffer += N;
            count  -= N;
        }
        while (count --> 0) {
            *buffer++ = value;
        }
    }

    /*not static*/ inline void memset16(uint16_t buffer[], uint16_t value, int count) {
        memsetT(buffer, value, count);
    }
    /*not static*/ inline void memset32(uint32_t buffer[], uint32_t value, int count) {
#if defined(SK_ARM_HAS_NEON) && !defined(__ARM_64BIT_STATE)
        uint32_t *p1 = buffer;
        uint32_t off;
        __asm__ volatile (
                "vdup.32     q0, %[p2]                     \n\t"
                "cmp         %[n], #3+16                   \n\t"
                "vdup.32     q1, %[p2]                     \n\t"
                "blo         20f                           \n\t"

                // Long case (at least one 16-byte-aligned 64-byte block)
                "ands        %[off], %[buffer], #12        \n\t"
                "bne         15f                           \n\t"

                // 16-byte aligned. Set up inner loop
                "10:                                       \n\t"
                "mov         %[off], #64                   \n\t"
                "sub         %[n], #16                     \n\t"
                "add         %[p2], %[p1], #32             \n\t"

                // Inner loop
                "11:                                       \n\t"
                "vst1.32     {q0-q1}, [%[p1] :128], %[off] \n\t"
                "subs        %[n], #16                     \n\t"
                "vst1.32     {q0-q1}, [%[p2] :128], %[off] \n\t"
                "bhs         11b                           \n\t"

                // Handle trailing 1..15 words
                "12:                                       \n\t"
                "lsls        %[n], #29                     \n\t"
                "bcc         1f                            \n\t"
                "vst1.32     {q0-q1}, [%[p1]]!             \n\t"
                "1:                                        \n\t"
                "bpl         1f                            \n\t"
                "vst1.32     {q0}, [%[p1]]!                \n\t"
                "1:                                        \n\t"
                "lsls        %[n], #2                      \n\t"
                "it          cs                            \n\t"
                "vstmcs      %[p1]!, {d0}                  \n\t"
                "it          mi                            \n\t"
                "vstmmi      %[p1]!, {s0}                  \n\t"
                "b           90f                           \n\t"

                // Handle first 1..3 words to achieve 16-byte alignment
                "15:                                       \n\t"
                "rsb         %[off], #16                   \n\t"
                "sub         %[n], %[off], lsr #2          \n\t"
                "lsls        %[off], #29                   \n\t"
                "it          mi                            \n\t"
                "vstmmi      %[p1]!, {s0}                  \n\t"
                "it          cs                            \n\t"
                "vstmcs      %[p1]!, {d0}                  \n\t"
                "b           10b                           \n\t"

                // Short case
                "20:                                       \n\t"
                "cmp         %[n], #8                      \n\t"
                "blo         12b                           \n\t"
                "sub         %[n], #8                      \n\t"
                "vst1.8      {q0-q1}, [%[p1]]!             \n\t"
                "b           12b                           \n\t"

                "90:                                       \n\t"
        : // Outputs
                 [p2]"+r"(value),
                  [n]"+r"(count),
                 [p1]"+r"(p1),
                [off]"=&r"(off)
        : // Inputs
                [buffer]"r"(buffer)
        : // Clobbers
                "cc", "memory"
        );
#else
        memsetT(buffer, value, count);
#endif
    }
    /*not static*/ inline void memset64(uint64_t buffer[], uint64_t value, int count) {
        memsetT(buffer, value, count);
    }

}

#endif//SkUtils_opts_DEFINED
