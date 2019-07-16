from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setup(
    name='oinkoink',
    version='1.0.1',
    author='Muff2n',
    description='A Reinforcement Learning agent plays connect4',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(exclude=[
        'tests*',
        '*archive',
        'oinkoink/scripts'
    ]),
    python_requires='>=3.7',
    install_requires=[
        'anytree',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'visdom'
    ],
    extras_require={
        'test': ['coverage'],
    },
    package_data={
        'oinkoink': ['data/*'],
    },
    entry_points={
        'console_scripts' : ['oinkoink = oinkoink:main']
    },
    url='http://github.com/muff2n/connect4'
)
