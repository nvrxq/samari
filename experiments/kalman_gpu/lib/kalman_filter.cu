// kalman_filter.cu 
#include "kalman_filter.h"
#include <ATen/ATen.h>
#include <cuda_runtime.h>


KalmanFilter::KalmanFilter() : ndim(4), dt(1.0), 
                              std_weight_position(1.0/20), 
                              std_weight_velocity(1.0/160) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available");
    }
    
    motion_mat = torch::eye(8, 8, torch::kCUDA);
    for(int i=0; i<4; ++i) {
        motion_mat[i][4+i] = dt;
    }
    update_mat = torch::eye(4, 8, torch::kCUDA);
}

std::pair<torch::Tensor, torch::Tensor> KalmanFilter::initiate(torch::Tensor measurement) {
    int batch_size = measurement.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    torch::Tensor mean_pos = measurement.clone();
    torch::Tensor mean_vel = torch::zeros({batch_size, 4}, options);
    torch::Tensor mean = torch::cat({mean_pos, mean_vel}, 1);
    
    torch::Tensor std = torch::zeros({batch_size, 8}, options);
    float* std_ptr = std.data_ptr<float>();
    const float* m_ptr = measurement.data_ptr<float>();
    
    compute_std_kernel_wrapper(
        std_ptr, std_ptr + 4*batch_size,
        m_ptr, 
        2*std_weight_position, 
        10*std_weight_velocity,
        batch_size
    );
    
    torch::Tensor covariance = torch::zeros({batch_size, 8, 8}, options);
    init_covariance_kernel_wrapper(
        covariance.data_ptr<float>(),
        std.data_ptr<float>(),
        batch_size,
        8
    );
    
    return {mean, covariance};
}

std::pair<torch::Tensor, torch::Tensor> KalmanFilter::predict(torch::Tensor mean, torch::Tensor covariance) {
    int batch_size = mean.size(0);
    auto options = mean.options();
    
    torch::Tensor std_pos = torch::empty({batch_size, 4}, options);
    torch::Tensor std_vel = torch::empty({batch_size, 4}, options);
    
    compute_std_kernel_wrapper(
        std_pos.data_ptr<float>(),
        std_vel.data_ptr<float>(),
        mean.data_ptr<float>(),
        std_weight_position,
        std_weight_velocity,
        batch_size
    );
    
    torch::Tensor motion_cov = torch::zeros({batch_size, 8, 8}, options);
    torch::Tensor std_combined = torch::cat({std_pos, std_vel}, 1);
    init_covariance_kernel_wrapper(
        motion_cov.data_ptr<float>(),
        std_combined.data_ptr<float>(),
        batch_size,
        8
    );
    
    mean = torch::matmul(mean, motion_mat.transpose(0, 1));
    covariance = torch::matmul(torch::matmul(motion_mat, covariance), motion_mat.transpose(0, 1)) + motion_cov;
    
    return {mean, covariance};
}

std::pair<torch::Tensor, torch::Tensor> KalmanFilter::project(torch::Tensor mean, torch::Tensor covariance) {
    int batch_size = mean.size(0);
    auto options = mean.options();
    
    torch::Tensor innov_cov = torch::zeros({batch_size, 4, 4}, options);
    project_innovation_kernel_wrapper(
        innov_cov.data_ptr<float>(),
        mean.data_ptr<float>(),
        std_weight_position,
        batch_size
    );
    
    torch::Tensor proj_mean = torch::matmul(mean, update_mat.transpose(0, 1));
    torch::Tensor proj_cov = torch::matmul(torch::matmul(update_mat, covariance), update_mat.transpose(0, 1)) + innov_cov;
    
    return {proj_mean, proj_cov};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<>())
        .def("initiate", &KalmanFilter::initiate)
        .def("predict", &KalmanFilter::predict)
        .def("project", &KalmanFilter::project);
}