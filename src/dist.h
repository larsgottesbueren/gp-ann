#pragma once

#include "../external/hnswlib/hnswlib/space_l2.h"

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

#ifdef __AVX__
            size_t qty = size;
            size_t qty16 = qty >> 4 << 4;
            result = hnswlib::L2SqrSIMD16ExtAVX(a, b, &qty16);
            size_t qty_left = qty - qty16;
            result += hnswlib::L2Sqr(a + qty_left, b + qty_left, &qty_left);
#else
            float diff0, diff1, diff2, diff3;
            const float* last = a + size;
            const float* unroll_group = last - 3;

            /* Process 4 items with each loop for efficiency. */
            size_t dim = 0;
            while (a < unroll_group) {
                diff0 = a[0] - b[0];
                diff1 = a[1] - b[1];
                diff2 = a[2] - b[2];
                diff3 = a[3] - b[3];
                result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                a += 4;
                b += 4;
                dim += 4;
            }
            std::cout << "dim after " << dim;
            /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
            while (a < last) {
                diff0 = *a++ - *b++;
                result += diff0 * diff0;
                dim++;
            }
            std::cout << "dim final " << dim;
#endif

            return result;
        }

        float compare2(float* a, float* b, unsigned size) {
            float result = 0.0f;
            float diff0, diff1, diff2, diff3;
            const float* last = a + size;
            const float* unroll_group = last - 3;

            /* Process 4 items with each loop for efficiency. */
            size_t dim = 0;
            while (a < unroll_group) {
                diff0 = a[0] - b[0];
                diff1 = a[1] - b[1];
                diff2 = a[2] - b[2];
                diff3 = a[3] - b[3];
                result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                a += 4;
                b += 4;
                dim += 4;
            }
            /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
            while (a < last) {
                diff0 = *a++ - *b++;
                result += diff0 * diff0;
                dim++;
            }
            return result;
        }


        float compare3(float* a, float* b, unsigned size) {
            #define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
                              tmp1 = _mm256_loadu_ps(addr1);                  \
                              tmp2 = _mm256_loadu_ps(addr2);                  \
                              tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
                              tmp1 = _mm256_mul_ps(tmp1, tmp1);               \
                              dest = _mm256_add_ps(dest, tmp1);


            __m256 sum;
            __m256 l0, l1;
            __m256 r0, r1;
            size_t qty16 = size >> 4;
            size_t aligned_size = qty16 << 4;
            const float *l = a;
            const float *r = b;

            float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
            sum = _mm256_loadu_ps(unpack);

            for (unsigned i = 0; i < aligned_size; i += 16, l += 16, r += 16) {
                AVX_L2SQR(l, r, sum, l0, r0);
                AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
            }
            _mm256_storeu_ps(unpack, sum);
            float result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
            for (unsigned i = aligned_size; i < size; ++i, ++l, ++r) {
                float diff = *l - *r;
                result += diff * diff;
            }
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
