import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='connect4',
    version='1.0.0',
    author='Muff2n',
    description='A Reinforcement Learning agent plays connect4',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent',
    ],
    packages=['connect4'],
    python_requires='>=3.7',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'pytorch',
        'scipy'
    ],
    extras_require={
        'test': ['coverage'],
    },
    package_data={
        'sample': ['misc/net.pth'],
    },
    url='http://github.com/muff2n/connect4',
    dependency_links=[
        'https://github.com/c0fec0de/anytree/archive/master.zip',
        'https://github.com/c0fec0de/facebookresearch/visdom/master.zip'
    ]
)
