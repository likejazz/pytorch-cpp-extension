from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='pycudaexpr',
      ext_modules=[
          CUDAExtension('pycudaexpr', [
              'cudaexpr.cu'
          ]),
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
