from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='embedding_cuda',
    ext_modules=[
        CUDAExtension(
            name='embedding_cuda',
            sources=['embedding_wrapper.cpp', 'embedding.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_89']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 