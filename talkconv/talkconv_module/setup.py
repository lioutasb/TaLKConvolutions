from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='talkconv_module',
    ext_modules=[
        CUDAExtension(
            name='talkconv_cuda',
            sources=[
                'bind.cpp',
                'talk_convolution_interface.cpp',
                'talk_convolution_cuda.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })