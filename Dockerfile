FROM nividia/cudate:10.0-base-ubuntu18.04

# Copied from https://github.com/Paperspace/fastai-docker/blob/master/fastai-v3/Dockerfile
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8 LC_ALL=C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# Guessing miniconda from https://hub.docker.com/r/continuumio/miniconda/dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# my additions
RUN mkdir ~/repo && \
    cd ~/repo && \
    git clone https://github.com/Muff2n/connect4.git && \
    cd connect4 && \
    conda install -y --file requirements.txt


