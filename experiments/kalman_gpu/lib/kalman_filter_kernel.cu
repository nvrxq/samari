#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "kalman_filter.h"



// extern "C" {
//     __global__ void init_covariance_kernel(...) { ... }
//     __global__ void compute_std_kernel(...) { ... }
//     __global__ void project_innovation_kernel(...) { ... }
// }


__global__ void init_covariance_kernel(float* cov, const float* std, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * dim;
    if (idx >= total) return;
    
    int b = idx / dim;
    int i = idx % dim;
    cov[b*dim*dim + i*dim + i] = std[b*dim + i] * std[b*dim + i];
}

__global__ void compute_std_kernel(
    float* std_pos, float* std_vel, 
    const float* mean, float swp, float swv, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float m3 = mean[idx*8 + 3];
    
    // Fill std_pos
    std_pos[idx*4 + 0] = swp * m3;
    std_pos[idx*4 + 1] = swp * m3;
    std_pos[idx*4 + 2] = 1e-2;
    std_pos[idx*4 + 3] = swp * m3;
    
    // Fill std_vel
    std_vel[idx*4 + 0] = swv * m3;
    std_vel[idx*4 + 1] = swv * m3;
    std_vel[idx*4 + 2] = 1e-5;
    std_vel[idx*4 + 3] = swv * m3;
}

__global__ void project_innovation_kernel(
    float* innov_cov, const float* mean, 
    float swp, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * 4;
    if (idx >= total) return;
    
    int b = idx / 4;
    int i = idx % 4;
    float m3 = mean[b*8 + 3];
    float val = (i == 2) ? 1e-1 : swp * m3;
    innov_cov[b*16 + i*4 + i] = val * val;
}

__host__ void init_covariance_kernel_wrapper(float* cov, const float* std, int batch_size, int dim) {
    dim3 blocks((batch_size*dim + 255)/256);
    init_covariance_kernel<<<blocks, 256>>>(cov, std, batch_size, dim);
}

__host__ void compute_std_kernel_wrapper(float* std_pos, float* std_vel, const float* mean, float swp, float swv, int batch_size) {
    dim3 blocks((batch_size + 255)/256);
    compute_std_kernel<<<blocks, 256>>>(std_pos, std_vel, mean, swp, swv, batch_size);
}

__host__ void project_innovation_kernel_wrapper(float* innov_cov, const float* mean, float swp, int batch_size) {
    dim3 blocks((batch_size*4 + 255)/256);
    project_innovation_kernel<<<blocks, 256>>>(innov_cov, mean, swp, batch_size);
}