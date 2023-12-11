#pragma once

float sqr_l2_dist(const float *a, const float *b, unsigned size);

float inner_product(float* p, float* q, unsigned d);

float mips_distance(float *p, float *q, unsigned d);

float vec_norm(float* p, unsigned d);

bool L2Normalize(float* p, unsigned d);

float distance(float *p, float *q, unsigned d);

float pos_distance(float* p, float* q, unsigned d);
