#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cmath>
#include <curand_kernel.h>
#include <stdio.h>
#include "../gamma-generators/johnk.h"

__device__ float Td[5];      // Time to maturity
__device__ float Kd[5];      // Strike price
__device__ float kappad[10]; // Kappa
__device__ float thetad[10]; // Theta
__device__ float sigmad[10]; // Sigma

void testCUDA(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("There is an error in file %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

__global__ void init_curand_state_kernel(curandState *state)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ void VG_MC_parallel_kernel(
    int Nsteps, int Ntraj,
    curandState *state,
    float *results,
    int *valid_paths)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= 5 * 5 * 10 * 10 * 10)
        return;

    curandState localState = state[idx];

    int param_idx = idx;

    int t_idx = param_idx / (5 * 10 * 10 * 10); // K * kappa * theta * sigma
    param_idx -= t_idx * (5 * 10 * 10 * 10);
    int k_idx = param_idx / (10 * 10 * 10); // kappa * theta * sigma
    param_idx -= k_idx * (10 * 10 * 10);
    int kappa_idx = param_idx / (10 * 10); // theta * sigma
    param_idx -= kappa_idx * (10 * 10);
    int theta_idx = param_idx / 10; // sigma
    param_idx -= theta_idx * 10;
    int sigma_idx = param_idx;

    float T = Td[t_idx];
    float K = Kd[k_idx];
    float kappa = kappad[kappa_idx];
    float theta = thetad[theta_idx];
    float sigma = sigmad[sigma_idx];

    float dt = T / Nsteps;
    float w = (1.0f / kappa) * logf(1.0f - theta * kappa - 0.5f * kappa * sigma * sigma);

    float sum_payoff = 0.0f;
    float sum_payoff_sq = 0.0f;
    int num_valid = 0;

    for (int i = 0; i < Ntraj; i++) // On parralélise sur les paramètres et non pas sur les trajectoires
    {
        float S = w * T;

        for (int j = 0; j < Nsteps; ++j)
        {
            float gamma_shape = dt / kappa;
            float gamma_scale = kappa;

            float delta_Si = gamma_johnk(gamma_shape, &localState) * gamma_scale;
            float Ni = curand_normal(&localState);

            float delta_X = sigma * Ni * sqrtf(delta_Si) + theta * delta_Si;
            S += delta_X;
        }

        float YT = expf(S);

        if (YT < 12.0f)
        {
            float payoff = (YT > K) ? (YT - K) : 0.0f;
            sum_payoff += payoff;
            sum_payoff_sq += payoff * payoff;
            num_valid++;
        }
    }

    if (num_valid > 0)
    {
        results[2 * idx] = sum_payoff / num_valid;
        results[2 * idx + 1] = sum_payoff_sq / num_valid;
    }
    else
    {
        results[2 * idx] = 0.0f;
        results[2 * idx + 1] = 0.0f;
    }
    valid_paths[idx] = num_valid;

    state[idx] = localState;
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float T[5] = {0.3f, 0.5f, 1.0f, 2.0f, 3.0f};
    float K[5] = {0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    // Admissible parameters provided by the professor:
    float sigma[10] = {0.1f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f};
    float theta[10] = {-0.34f, -0.3f, -0.27f, -0.24f, -0.21f, -0.25f, -0.26f, -0.35f, -0.4f, -0.45f};
    float kappa[10] = {0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f};

    cudaMemcpyToSymbol(Td, T, 5 * sizeof(float));
    cudaMemcpyToSymbol(Kd, K, 5 * sizeof(float));
    cudaMemcpyToSymbol(kappad, kappa, 10 * sizeof(float));
    cudaMemcpyToSymbol(thetad, theta, 10 * sizeof(float));
    cudaMemcpyToSymbol(sigmad, sigma, 10 * sizeof(float));

    int total_combinations = 5 * 5 * 10 * 10 * 10; // T * K * kappa * theta * sigma
    int threadsPerBlock = 256;
    int blocks = (total_combinations + threadsPerBlock - 1) / threadsPerBlock;

    // Configuration
    int Nsteps = 100;
    int Ntraj = 10000;

    curandState *states;
    float *results;
    int *valid_paths;

    cudaMalloc(&states, total_combinations * sizeof(curandState));
    cudaMallocManaged(&results, 2 * total_combinations * sizeof(float));
    cudaMallocManaged(&valid_paths, total_combinations * sizeof(int));

    init_curand_state_kernel<<<blocks, threadsPerBlock>>>(states);
    testCUDA(cudaDeviceSynchronize());

    cudaEventRecord(start);

    VG_MC_parallel_kernel<<<blocks, threadsPerBlock>>>(
        Nsteps, Ntraj, states, results, valid_paths);
    testCUDA(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Simulation completed in " << milliseconds << " milliseconds" << std::endl;
    std::ofstream execution_time("data/execution_time.md");
    execution_time << "The simulation completed in " << milliseconds << " milliseconds (" << milliseconds / 1000.0f << " seconds).\n\n";
    execution_time << "> Configuration\n\n";
    execution_time << "- Number of steps: " << Nsteps << "\n";
    execution_time << "- Number of trajectories: " << Ntraj << "\n";
    execution_time << "- Total parameter combinations: " << total_combinations << "\n";
    execution_time << "- Threads per block: " << threadsPerBlock << "\n";
    execution_time << "- Number of blocks: " << blocks << "\n";
    execution_time.close();

    std::ofstream fout("data/vg_prices_all.csv");
    fout << "T,K,kappa,theta,sigma,price,error,valid_paths\n";

    for (int t_idx = 0; t_idx < 5; t_idx++)
    {
        for (int k_idx = 0; k_idx < 5; k_idx++)
        {
            for (int kappa_idx = 0; kappa_idx < 10; kappa_idx++)
            {
                for (int theta_idx = 0; theta_idx < 10; theta_idx++)
                {
                    for (int sigma_idx = 0; sigma_idx < 10; sigma_idx++)
                    {
                        int idx = t_idx * (5 * 10 * 10 * 10) + // K * kappa * theta * sigma
                                  k_idx * (10 * 10 * 10) +     // kappa * theta * sigma
                                  kappa_idx * (10 * 10) +      // theta * sigma
                                  theta_idx * 10 +             // sigma
                                  sigma_idx;
                        float price = results[2 * idx];
                        float var = results[2 * idx + 1] - (price * price);
                        float error = 1.96f * sqrtf(var) / sqrtf((float)Ntraj);
                        int num_valid = valid_paths[idx];
                        fout << T[t_idx] << ","
                             << K[k_idx] << ","
                             << kappa[kappa_idx] << ","
                             << theta[theta_idx] << ","
                             << sigma[sigma_idx] << ","
                             << price << ","
                             << error << ","
                             << num_valid << "\n";
                    }
                }
            }
        }
    }

    fout.close();
    std::cout << "Simulation completed. Results saved to vg_prices_all.csv" << std::endl;

    cudaFree(states);
    cudaFree(results);
    cudaFree(valid_paths);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
