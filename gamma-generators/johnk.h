#ifndef JOHNK_H
#define JOHNK_H

#include <curand_kernel.h>
#include <math.h>

__device__ inline float gamma_johnk(float a, curandState_t *state)
{
    float X, Y, U, V;
    do
    {
        U = curand_uniform(state);
        V = curand_uniform(state);
        X = powf(U, 1.0f / a);
        Y = powf(V, 1.0f / (1.0f - a));
    } while (X + Y > 1.0f);
    float E = -logf(curand_uniform(state));
    return X * E / (X + Y);
}

#endif