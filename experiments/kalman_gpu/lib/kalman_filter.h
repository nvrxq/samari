#pragma once
#include <torch/extension.h>

#ifdef __cplusplus
extern "C" {
#endif

__host__ void init_covariance_kernel_wrapper(float* cov, const float* std, int batch_size, int dim);
__host__ void compute_std_kernel_wrapper(float* std_pos, float* std_vel, const float* mean, float swp, float swv, int batch_size);
__host__ void project_innovation_kernel_wrapper(float* innov_cov, const float* mean, float swp, int batch_size);

#ifdef __cplusplus
}
#endif


class KalmanFilter {
private:
    torch::Tensor motion_mat;
    torch::Tensor update_mat;
    float std_weight_position;
    float std_weight_velocity;
    int ndim;
    float dt;

public:
    KalmanFilter();
    std::pair<torch::Tensor, torch::Tensor> initiate(torch::Tensor measurement);
    std::pair<torch::Tensor, torch::Tensor> predict(torch::Tensor mean, torch::Tensor covariance);
    std::pair<torch::Tensor, torch::Tensor> project(torch::Tensor mean, torch::Tensor covariance);
};

__global__ void init_covariance_kernel(float* cov, const float* std, int batch_size, int dim);
__global__ void compute_std_kernel(float* std_pos, float* std_vel, const float* mean, float swp, float swv, int batch_size);
__global__ void project_innovation_kernel(float* innov_cov, const float* mean, float swp, int batch_size);