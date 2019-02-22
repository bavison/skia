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

/* Just used for cancelling out the overheads */
static void control(uint32_t* dst, uint32_t value, int len)
{
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

static uint32_t bench_L(void (*test)(uint32_t* dst, uint32_t value, int len), uint32_t *a, size_t len, size_t times)
{
    int i, j, x = 0, q = 0;
    volatile int qx;
    for (i = times; i >= 0; i--)
    {
        /* Ensure the destination is in cache */
        for (j = 0; j < len; j += 32)
            q += a[j];
        q += a[len-1];
        x = (x + 1) & 63;
        test(a + x, 0, len);
    }
    qx = q;
    return len * times * sizeof *a;
}

static uint32_t bench_M(void (*test)(uint32_t* dst, uint32_t value, int len), uint32_t *a, size_t len, size_t times)
{
    int i, x = 0;
    for (i = times; i >= 0; i--)
    {
        /* If we cared about CPUs with non-write-allocate caches and operations that
         * don't read from the destination buffer, we'd need to ensure the destination
         * hadn't been evicted from the cache here - but this isn't the case */
        x = (x + 1) & 63;
        test(a + x, 0, len);
    }
    return len * times * sizeof *a;
}

namespace {

// The default SkNx<N,T> just proxies down to a pair of SkNx<N/2, T>.
template <int N, typename T>
struct SkNx {
    typedef SkNx<N/2, T> Half;

    Half fLo, fHi;

    __attribute__((always_inline)) inline SkNx() = default;
    __attribute__((always_inline)) inline SkNx(const Half& lo, const Half& hi) : fLo(lo), fHi(hi) {}

    __attribute__((always_inline)) inline SkNx(T v) : fLo(v), fHi(v) {}

    __attribute__((always_inline)) inline SkNx(T a, T b) : fLo(a) , fHi(b) { static_assert(N==2, ""); }
    __attribute__((always_inline)) inline SkNx(T a, T b, T c, T d) : fLo(a,b), fHi(c,d) { static_assert(N==4, ""); }
    __attribute__((always_inline)) inline SkNx(T a, T b, T c, T d, T e, T f, T g, T h) : fLo(a,b,c,d), fHi(e,f,g,h) {
        static_assert(N==8, "");
    }
    __attribute__((always_inline)) inline SkNx(T a, T b, T c, T d, T e, T f, T g, T h,
            T i, T j, T k, T l, T m, T n, T o, T p)
        : fLo(a,b,c,d, e,f,g,h), fHi(i,j,k,l, m,n,o,p) {
        static_assert(N==16, "");
    }

    __attribute__((always_inline)) inline T operator[](int k) const {
        static_cast<void>(0);
        return k < N/2 ? fLo[k] : fHi[k-N/2];
    }

    __attribute__((always_inline)) inline static SkNx Load(const void* vptr) {
        auto ptr = (const char*)vptr;
        return { Half::Load(ptr), Half::Load(ptr + N/2*sizeof(T)) };
    }
    __attribute__((always_inline)) inline void store(void* vptr) const {
        auto ptr = (char*)vptr;
        fLo.store(ptr);
        fHi.store(ptr + N/2*sizeof(T));
    }

    __attribute__((always_inline)) inline static void Load4(const void* vptr, SkNx* a, SkNx* b, SkNx* c, SkNx* d) {
        auto ptr = (const char*)vptr;
        Half al, bl, cl, dl,
             ah, bh, ch, dh;
        Half::Load4(ptr , &al, &bl, &cl, &dl);
        Half::Load4(ptr + 4*N/2*sizeof(T), &ah, &bh, &ch, &dh);
        *a = SkNx{al, ah};
        *b = SkNx{bl, bh};
        *c = SkNx{cl, ch};
        *d = SkNx{dl, dh};
    }
    __attribute__((always_inline)) inline static void Load3(const void* vptr, SkNx* a, SkNx* b, SkNx* c) {
        auto ptr = (const char*)vptr;
        Half al, bl, cl,
             ah, bh, ch;
        Half::Load3(ptr , &al, &bl, &cl);
        Half::Load3(ptr + 3*N/2*sizeof(T), &ah, &bh, &ch);
        *a = SkNx{al, ah};
        *b = SkNx{bl, bh};
        *c = SkNx{cl, ch};
    }
    __attribute__((always_inline)) inline static void Load2(const void* vptr, SkNx* a, SkNx* b) {
        auto ptr = (const char*)vptr;
        Half al, bl,
             ah, bh;
        Half::Load2(ptr , &al, &bl);
        Half::Load2(ptr + 2*N/2*sizeof(T), &ah, &bh);
        *a = SkNx{al, ah};
        *b = SkNx{bl, bh};
    }
    __attribute__((always_inline)) inline static void Store4(void* vptr, const SkNx& a, const SkNx& b, const SkNx& c, const SkNx& d) {
        auto ptr = (char*)vptr;
        Half::Store4(ptr, a.fLo, b.fLo, c.fLo, d.fLo);
        Half::Store4(ptr + 4*N/2*sizeof(T), a.fHi, b.fHi, c.fHi, d.fHi);
    }
    __attribute__((always_inline)) inline static void Store3(void* vptr, const SkNx& a, const SkNx& b, const SkNx& c) {
        auto ptr = (char*)vptr;
        Half::Store3(ptr, a.fLo, b.fLo, c.fLo);
        Half::Store3(ptr + 3*N/2*sizeof(T), a.fHi, b.fHi, c.fHi);
    }

    __attribute__((always_inline)) inline bool anyTrue() const { return fLo.anyTrue() || fHi.anyTrue(); }
    __attribute__((always_inline)) inline bool allTrue() const { return fLo.allTrue() && fHi.allTrue(); }

    __attribute__((always_inline)) inline SkNx abs() const { return { fLo. abs(), fHi. abs() }; }
    __attribute__((always_inline)) inline SkNx sqrt() const { return { fLo. sqrt(), fHi. sqrt() }; }
    __attribute__((always_inline)) inline SkNx rsqrt() const { return { fLo. rsqrt(), fHi. rsqrt() }; }
    __attribute__((always_inline)) inline SkNx floor() const { return { fLo. floor(), fHi. floor() }; }
    __attribute__((always_inline)) inline SkNx invert() const { return { fLo.invert(), fHi.invert() }; }

    __attribute__((always_inline)) inline SkNx operator!() const { return { !fLo, !fHi }; }
    __attribute__((always_inline)) inline SkNx operator-() const { return { -fLo, -fHi }; }
    __attribute__((always_inline)) inline SkNx operator~() const { return { ~fLo, ~fHi }; }

    __attribute__((always_inline)) inline SkNx operator<<(int bits) const { return { fLo << bits, fHi << bits }; }
    __attribute__((always_inline)) inline SkNx operator>>(int bits) const { return { fLo >> bits, fHi >> bits }; }

    __attribute__((always_inline)) inline SkNx operator+(const SkNx& y) const { return { fLo + y.fLo, fHi + y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator-(const SkNx& y) const { return { fLo - y.fLo, fHi - y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator*(const SkNx& y) const { return { fLo * y.fLo, fHi * y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator/(const SkNx& y) const { return { fLo / y.fLo, fHi / y.fHi }; }

    __attribute__((always_inline)) inline SkNx operator&(const SkNx& y) const { return { fLo & y.fLo, fHi & y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator|(const SkNx& y) const { return { fLo | y.fLo, fHi | y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator^(const SkNx& y) const { return { fLo ^ y.fLo, fHi ^ y.fHi }; }

    __attribute__((always_inline)) inline SkNx operator==(const SkNx& y) const { return { fLo == y.fLo, fHi == y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator!=(const SkNx& y) const { return { fLo != y.fLo, fHi != y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator<=(const SkNx& y) const { return { fLo <= y.fLo, fHi <= y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator>=(const SkNx& y) const { return { fLo >= y.fLo, fHi >= y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator< (const SkNx& y) const { return { fLo < y.fLo, fHi < y.fHi }; }
    __attribute__((always_inline)) inline SkNx operator> (const SkNx& y) const { return { fLo > y.fLo, fHi > y.fHi }; }

    __attribute__((always_inline)) inline SkNx saturatedAdd(const SkNx& y) const {
        return { fLo.saturatedAdd(y.fLo), fHi.saturatedAdd(y.fHi) };
    }

    __attribute__((always_inline)) inline SkNx mulHi(const SkNx& m) const {
        return { fLo.mulHi(m.fLo), fHi.mulHi(m.fHi) };
    }
    __attribute__((always_inline)) inline SkNx thenElse(const SkNx& t, const SkNx& e) const {
        return { fLo.thenElse(t.fLo, e.fLo), fHi.thenElse(t.fHi, e.fHi) };
    }
    __attribute__((always_inline)) inline static SkNx Min(const SkNx& x, const SkNx& y) {
        return { Half::Min(x.fLo, y.fLo), Half::Min(x.fHi, y.fHi) };
    }
    __attribute__((always_inline)) inline static SkNx Max(const SkNx& x, const SkNx& y) {
        return { Half::Max(x.fLo, y.fLo), Half::Max(x.fHi, y.fHi) };
    }
};

template <>
class SkNx<4, uint32_t> {
public:
    __attribute__((always_inline)) inline SkNx(const uint32x4_t& vec) : fVec(vec) {}

    __attribute__((always_inline)) inline SkNx() {}
    __attribute__((always_inline)) inline SkNx(uint32_t v) {
        fVec = vdupq_n_u32(v);
    }
    __attribute__((always_inline)) inline SkNx(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        fVec = (uint32x4_t){a,b,c,d};
    }
    __attribute__((always_inline)) inline static SkNx Load(const void* ptr) {
        return __extension__ ({ uint32x4_t __ret; __ret = (uint32x4_t) __builtin_neon_vld1q_v((const uint32_t*)ptr, 50); __ret; });
    }
    __attribute__((always_inline)) inline void store(void* ptr) const {
        return __extension__ ({ uint32x4_t __s1 = fVec; __builtin_neon_vst1q_v((uint32_t*)ptr, (int8x16_t)__s1, 50); });
    }
    __attribute__((always_inline)) inline uint32_t operator[](int k) const {
        static_cast<void>(0);
        union { uint32x4_t v; uint32_t us[4]; } pun = {fVec};
        return pun.us[k&3];
    }

    __attribute__((always_inline)) inline SkNx operator + (const SkNx& o) const { return vaddq_u32(fVec, o.fVec); }
    __attribute__((always_inline)) inline SkNx operator - (const SkNx& o) const { return vsubq_u32(fVec, o.fVec); }
    __attribute__((always_inline)) inline SkNx operator * (const SkNx& o) const { return vmulq_u32(fVec, o.fVec); }

    __attribute__((always_inline)) inline SkNx operator & (const SkNx& o) const { return vandq_u32(fVec, o.fVec); }
    __attribute__((always_inline)) inline SkNx operator | (const SkNx& o) const { return vorrq_u32(fVec, o.fVec); }
    __attribute__((always_inline)) inline SkNx operator ^ (const SkNx& o) const { return veorq_u32(fVec, o.fVec); }

    __attribute__((always_inline)) inline SkNx operator << (int bits) const { return fVec << SkNx(bits).fVec; }
    __attribute__((always_inline)) inline SkNx operator >> (int bits) const { return fVec >> SkNx(bits).fVec; }

    __attribute__((always_inline)) inline SkNx operator == (const SkNx& o) const { return vceqq_u32(fVec, o.fVec); }
    __attribute__((always_inline)) inline SkNx operator < (const SkNx& o) const { return vcltq_u32(fVec, o.fVec); }
    __attribute__((always_inline)) inline SkNx operator > (const SkNx& o) const { return vcgtq_u32(fVec, o.fVec); }

    __attribute__((always_inline)) inline static SkNx Min(const SkNx& a, const SkNx& b) { return vminq_u32(a.fVec, b.fVec); }
    // TODO as needed

    __attribute__((always_inline)) inline SkNx mulHi(const SkNx& m) const {
        uint64x2_t hi = vmull_u32(vget_high_u32(fVec), vget_high_u32(m.fVec));
        uint64x2_t lo = vmull_u32( vget_low_u32(fVec), vget_low_u32(m.fVec));

        return { vcombine_u32(__extension__ ({ uint64x2_t __s0 = lo; uint32x2_t __ret; __ret = (uint32x2_t) __builtin_neon_vshrn_n_v((int8x16_t)__s0, 32, 18); __ret; }), __extension__ ({ uint64x2_t __s0 = hi; uint32x2_t __ret; __ret = (uint32x2_t) __builtin_neon_vshrn_n_v((int8x16_t)__s0, 32, 18); __ret; })) };
    }

    __attribute__((always_inline)) inline SkNx thenElse(const SkNx& t, const SkNx& e) const {
        return vbslq_u32(fVec, t.fVec, e.fVec);
    }

    uint32x4_t fVec;
};

}

namespace neon {

    template <typename T>
    static void memsetT(T buffer[], T value, int count) {



        static const int N = 16 / sizeof(T);

        while (count >= N) {
            SkNx<N,T>(value).store(buffer);
            buffer += N;
            count -= N;
        }
        while (count --> 0) {
            *buffer++ = value;
        }
    }

    /*not static*/ inline void memset32(uint32_t buffer[], uint32_t value, int count) {
        memsetT(buffer, value, count);

    }
    /*not static*/ inline void new_memset32(uint32_t buffer[], uint32_t value, int count) {
        uint32_t *p1 = buffer;
        uint32_t off;
        __asm__ volatile (
                "vdup.32     q0, %[p2]                     \n\t"
                "cmp         %[n], #3+16                   \n\t"
                "vdup.32     q1, %[p2]                     \n\t"
                "blo         20f                           \n\t"

                "ands        %[off], %[buffer], #12        \n\t"
                "bne         15f                           \n\t"

                "10:                                       \n\t"
                "mov         %[off], #64                   \n\t"
                "sub         %[n], #16                     \n\t"
                "add         %[p2], %[p1], #32             \n\t"

                "11:                                       \n\t"
                "vst1.32     {q0-q1}, [%[p1] :128], %[off] \n\t"
                "subs        %[n], #16                     \n\t"
                "vst1.32     {q0-q1}, [%[p2] :128], %[off] \n\t"
                "bhs         11b                           \n\t"

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

                "15:                                       \n\t"
                "rsb         %[off], #16                   \n\t"
                "sub         %[n], %[off], lsr #2          \n\t"
                "lsls        %[off], #29                   \n\t"
                "it          mi                            \n\t"
                "vstmmi      %[p1]!, {s0}                  \n\t"
                "it          cs                            \n\t"
                "vstmcs      %[p1]!, {d0}                  \n\t"
                "b           10b                           \n\t"

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
    }
}

int main(void)
{
#if 1
    uint32_t dst0[PIXELS] __attribute__((aligned(16))),
             dst1[PIXELS] __attribute__((aligned(16))),
             dst2[PIXELS] __attribute__((aligned(16)));
    srand(0);

    for (size_t len = 0; len <= 100; len++) {
        for (size_t offset = 4; offset < 8; offset++) {
            for (size_t i = 0; i < PIXELS; i++)
            {
                dst2[i] = dst1[i] = dst0[i] = rand32();
            }

            neon::memset32(dst1+offset, 0x12345678, len);
            neon::new_memset32(dst2+offset, 0x12345678, len);
            if (memcmp(dst1, dst2, sizeof dst1) != 0) {
                printf("offset %d pixels, length %d pixels\n", offset, len);
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
                    if (dst1[i] == dst2[i])
                        printf("         ");
                    else
                        printf(" ^^^^^^^^");
                }
                printf("\n");

                exit(EXIT_FAILURE);
            }
        }
    }
#endif

#if 1

#define CANDIDATE neon::memset32
//#define CANDIDATE neon::new_memset32

    uint64_t t1, t2, t3;
    uint32_t byte_cnt;
    static uint32_t bufa[TESTSIZE / sizeof (uint32_t)] __attribute__((aligned(4096)));
    memset(bufa, 0, sizeof bufa);

#define BENCH(type, bytes_per_loop, separator)                                                                       \
    do {                                                                                                       \
        t1 = gettime();                                                                                        \
        bench_##type(control, bufa, (bytes_per_loop) / sizeof *bufa, TESTSIZE / (bytes_per_loop));              \
        t2 = gettime();                                                                                        \
        byte_cnt = bench_##type(CANDIDATE, bufa, (bytes_per_loop) / sizeof *bufa, TESTSIZE / (bytes_per_loop)); \
        t3 = gettime();                                                                                        \
        printf("%6.2f" separator, ((double)byte_cnt) / ((t3 - t2) - (t2 - t1)));                               \
        fflush(stdout);                                                                                        \
    } while (0)

    BENCH(L, L1CACHESIZE / 2 - 64 * sizeof (uint32_t), ", ");
    BENCH(L, L2CACHESIZE / 2 - 64 * sizeof (uint32_t), ", ");
    BENCH(M, TESTSIZE - 64 * sizeof (uint32_t), "\n");

#endif
}
