FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


#### System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install libgl1 -y \
        git python3 python3-dev libpython3-dev python3-pip sudo wget nano tmux cmake g++ gcc curl \
        unzip less htop iftop iotop \
        libglib2.0-0 libsm6 libxext6 libxrender1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip && \
    pip install gpustat --no-cache-dir 

#### OPENMPI
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
RUN mkdir -p /build && \
    cd /build && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ~ && \
    rm -rf /build

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

#### Python packages
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir  && pip cache purge
COPY requirements/*.txt .

## Install APEX
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

RUN pip install -r dev.txt && pip install -r prod.txt && pip cache purge

WORKDIR /sttr

## SSH
# COPY .ssh/id_rsa /root/.ssh/id_rsa
# RUN chmod 600 /root/.ssh/id_rsa
# COPY .ssh/authorized_keys /root/.ssh/authorized_keys
# RUN chmod 644 /root/.ssh/authorized_keys
# COPY job/hostfile /job/hostfile
# COPY .ssh/config /root/.ssh/config

# RUN echo 'Port 2222' >> /etc/ssh/sshd_config
# RUN echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config
# EXPOSE 2222