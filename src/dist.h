#pragma once

#include <x86intrin.h>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <math.h>

namespace efanna2e {

    class Distance {
    public:
        virtual float compare(const float *a, const float *b, unsigned length) const = 0;

        virtual ~Distance() { }
    };

    class DistanceL2 : public Distance {
    public:
        float compare(const float *a, const float *b, unsigned size) const {
            float result = 0;

#ifdef __GNUC__
#ifdef __AVX__

#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
      tmp1 = _mm256_loadu_ps(addr1);\
      tmp2 = _mm256_loadu_ps(addr2);\
      tmp1 = _mm256_sub_ps(tmp1, tmp2); \
      tmp1 = _mm256_mul_ps(tmp1, tmp1); \
      dest = _mm256_add_ps(dest, tmp1);

            __m256 sum;
            __m256 l0, l1;
            __m256 r0, r1;
            unsigned D = (size + 7) & ~7U;
            unsigned DR = D % 16;
            unsigned DD = D - DR;
            std::cout << "size = " << size << " D = " << D << " DR = " << DR << " DD = " << DD << std::endl;
            const float *l = a;
            const float *r = b;
            const float *e_l = l + DD;
            const float *e_r = r + DD;
            float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

            sum = _mm256_loadu_ps(unpack);
            if(DR){AVX_L2SQR(e_l, e_r, sum, l0, r0);}

            for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
                AVX_L2SQR(l, r, sum, l0, r0);
                AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
            }
            _mm256_storeu_ps(unpack, sum);
            result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm_load_ps(addr1);\
          tmp2 = _mm_load_ps(addr2);\
          tmp1 = _mm_sub_ps(tmp1, tmp2); \
          tmp1 = _mm_mul_ps(tmp1, tmp1); \
          dest = _mm_add_ps(dest, tmp1);

            __m128 sum;
            __m128 l0, l1, l2, l3;
            __m128 r0, r1, r2, r3;
            unsigned D = (size + 3) & ~3U;
            unsigned DR = D % 16;
            unsigned DD = D - DR;
            const float *l = a;
            const float *r = b;
            const float *e_l = l + DD;
            const float *e_r = r + DD;
            float unpack[4] __attribute__ ((aligned (16))) = { 0, 0, 0, 0 };

            sum = _mm_load_ps(unpack);
            switch (DR) {
                case 12:
                SSE_L2SQR(e_l + 8, e_r + 8, sum, l2, r2);
                case 8:
                SSE_L2SQR(e_l + 4, e_r + 4, sum, l1, r1);
                case 4:
                SSE_L2SQR(e_l, e_r, sum, l0, r0);
                default:
                    break;
            }
            for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
                SSE_L2SQR(l, r, sum, l0, r0);
                SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
                SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
                SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
            }
            _mm_storeu_ps(unpack, sum);
            result += unpack[0] + unpack[1] + unpack[2] + unpack[3];

            //normal distance
#else

            float diff0, diff1, diff2, diff3;
            const float* last = a + size;
            const float* unroll_group = last - 3;

            /* Process 4 items with each loop for efficiency. */
            while (a < unroll_group) {
                diff0 = a[0] - b[0];
                diff1 = a[1] - b[1];
                diff2 = a[2] - b[2];
                diff3 = a[3] - b[3];
                result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                a += 4;
                b += 4;
            }
            /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
            while (a < last) {
                diff0 = *a++ - *b++;
                result += diff0 * diff0;
            }
#endif
#endif
#endif

            return result;
        }

    };

}  // namespace efanna2e

float mips_distance(float *p, float *q, unsigned d){
    float result = 0;
    for(unsigned i=0; i<d; i++){
      result += (q[i]) * (p[i]);
    }
    return -result;
}

double vec_norm(float* p, unsigned d) {
    double result = 0.f;
    for (unsigned i = 0; i < d; ++i) result += p[i] * p[i];
    return result;
}

bool L2Normalize(float* p, unsigned d) {
    double norm = vec_norm(p, d);
    if (norm < 1e-10) {
        std::cerr << "Vector is fully zero " << std::endl;
        for (unsigned i = 0; i < d; ++i) {
            std::cerr << p[i] << " ";
        }
        std::cerr << std::endl;
        return false;
    }
    double sqrt_norm = std::sqrt(norm);
    for (unsigned i = 0; i < d; ++i) p[i] /= sqrt_norm;
    return true;
}

float distance(float *p, float *q, unsigned d) {
    #ifdef MIPS_DISTANCE
    return mips_distance(p, q, d);
    #else
    efanna2e::DistanceL2 distfunc;
    return distfunc.compare(p, q, d);
    #endif
}

void Normalize(PointSet& points) {
    for (size_t i = 0; i < points.n; ++i) {
        float* p = points.GetPoint(i);
        if (!L2Normalize(p, points.d)) {
            std::cerr << "Point " << i << " is fully zero --> delete" << std::endl;
        }
    }
    std::cout << "finished normalizing" << std::endl;
}
