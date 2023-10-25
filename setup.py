from setuptools import setup, find_packages
from torch.utils import cpp_extension

# This is needed in order to build the C++ extension
import torch
print(f'>>>>> {torch.__version__} <<<<<')
ext = cpp_extension.CppExtension('zpk_cpp', ['src/zpk_cpp/pk.cpp'], extra_compile_args=['-std=c++17'])
setup_args = dict(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[ext],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)

setup(**setup_args)
