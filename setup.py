from setuptools import setup, find_packages

setup(
    name='perceptron-from-scratch',
    version='0.1',
    description='Implementation of the linear classifier',
    author='Jean Reinhold',
    author_email='jeanpaulreinhold@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)