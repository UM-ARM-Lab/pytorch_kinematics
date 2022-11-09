from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(
    name='pytorch_kinematics',
    version='0.4.0',
    packages=['pytorch_kinematics'],
    url='https://github.com/UM-ARM-Lab/pytorch_kinematics',
    license='MIT',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='Robot kinematics implemented in pytorch',
    install_requires=[
        'torch',
        'numpy',
        'transformations',
        'absl-py',
        'lxml',
        'mujoco',
        'dm_control',
        'pyyaml'
    ],
    tests_require=[
        'pytest'
    ],
    ext_modules=[cpp_extension.CppExtension('zpk_cpp', ['pytorch_kinematics/pk.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
