import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='connect4',
      version='1.0.0',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      description='the meanest game engine in the world',
      long_description=long_description,
      url='http://github.com/muff2n/connect4',
      license='MIT',
      packages=['connect4'],
      # packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      install_requires=[
          'anytree',
          'numpy',
          'pandas',
          'pytorch',
          'visdom'
      ]
)
