from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

setup(
    name='embedding_cuda',
    ext_modules=[
        CUDAExtension(
            name='embedding_cuda',
            sources=['embedding_wrapper.cpp', 'embedding.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_80']  # sm_80 for most cloud GPUs
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
) 