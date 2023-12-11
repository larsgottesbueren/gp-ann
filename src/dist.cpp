#include "dist.h"

#include "../external/hnswlib/hnswlib/space_l2.h"

#include <iostream>
#include <math.h>


float sqr_l2_dist(const float *a, const float *b, unsigned size) {
    float result = 0;
#ifdef __AVX__
    size_t qty = size;
    size_t qty16 = qty >> 4 << 4;
    result = hnswlib::L2SqrSIMD16ExtAVX(a, b, &qty16);
    size_t qty_left = qty - qty16;
    result += hnswlib::L2Sqr(a + qty16, b + qty16, &qty_left);
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
    return result;
}


float inner_product(float* p, float* q, unsigned d) {
    float result = 0;
    for(unsigned i=0; i<d; i++){
        result += (q[i]) * (p[i]);
    }
    return result;
}

float mips_distance(float *p, float *q, unsigned d){
    return 1.0f - inner_product(p, q, d);
}

float vec_norm(float* p, unsigned d) {
    float result = 0.f;
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
    return sqr_l2_dist(p, q, d);
    #endif
}

float pos_distance(float* p, float* q, unsigned d) {
#ifdef MIPS_DISTANCE
    return distance(p, q, d) + 1.0;
#endif
    return distance(p, q, d);
}
