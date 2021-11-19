from setuptools import setup

setup(
    name='pytorch_kinematics',
    version='0.3.0',
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
        'absl-py'
    ],
    tests_require=[
        'pytest'
    ]
)
