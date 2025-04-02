from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="kalman_filter",
    ext_modules=[
        CUDAExtension(
            name="kalman_filter",
            sources=[
                "lib/kalman_filter.cu",  # Изменили расширение
                "lib/kalman_filter_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
