#ifndef BEST_H
#define BEST_H

#include <curand_kernel.h>
#include <math.h>

__device__ inline float gamma_best(float a, curandState *state)
{
    float b = a - 1.0f;
    float c = 3.0f * a - 0.75f;
    float U, V, W, Y, X, Z;
    while (true)
    {
        U = curand_uniform(state);
        V = curand_uniform(state);
        W = U * (1.0f - U);
        Y = sqrtf(c / W) * (U - 0.5f);
        X = b + Y;
        if (X < 0.0f)
            continue;
        Z = 64.0f * W * W * W * V * V * V;
        if (logf(Z) <= 2.0f * (b * logf(X / b) - Y))
            return X;
    }
}
#endif