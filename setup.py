import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('_torch_unified_gpu', [
        'torch_unified_gpu.cpp',
    ]))

setup(
    name='torch_unified',
    packages=['torch_unified'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)