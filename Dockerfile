FROM nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION=3.13.12

# needed to install the CUDA version of PyTorch3D
# see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# see https://github.com/pytorch/extension-cpp/issues/71
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    git \
    g++ \
    gnupg \
    libatomic1 \
    libegl1 \
    libbz2-dev \
    libffi-dev \
    libgl1 \
    libgomp1 \
    liblzma-dev \
    lsb-release \
    libssl-dev \
    libx11-xcb1 \
    make \
    software-properties-common \
    tzdata \
    unzip \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

######################## Python installation #########################

# Download Python source code from official site and build it
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxvf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && make && make install && \
    cd .. && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py

#################### Create Virtual Environment ######################

ENV VIRTUAL_ENV=/workspace/venv
RUN python3.13 -m venv $VIRTUAL_ENV

# by adding the venv to the search path, we avoid activating it in each command
# see https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install pip into venv
RUN python get-pip.py && python -m pip install --upgrade pip

######## Install PyTorch and packages that depend on PyTorch #########

# these packages are dependencies of the other packages but not listed in their requirements
# therefore, they have to be installed manually
RUN python -m pip install tensorboard yapf wheel packaging
RUN python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129
RUN python -m pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
RUN python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN python -m pip install spconv-cu124
RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp313-cp313-linux_x86_64.whl
RUN python -m pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==26.2.*"

# ######################## Open3D build #################################
RUN git clone https://github.com/isl-org/Open3D
RUN git clone https://github.com/isl-org/Open3D-ML.git

# remove sudo from script since it is not available in Docker
RUN sed -i 's/SUDO=${SUDO:=sudo}/SUDO=" "/g' Open3D/util/install_deps_ubuntu.sh && \
    Open3D/util/install_deps_ubuntu.sh assume-yes

# install latest version of cmake
# see https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && apt install kitware-archive-keyring && rm /etc/apt/trusted.gpg.d/kitware.gpg && apt install cmake -y

# first, build CPU version and then build CUDA version
RUN mkdir /workspace/Open3D/build && cd /workspace/Open3D/build && \
    cmake -DPYTHON3_INCLUDE_DIR=$(python -c "import sysconfig && print(sysconfig.get_path('include'))") \
          -DPYTHON3_LIBRARY=$(python -c "import sysconfig && print(sysconfig.get_config_var('LIBDIR'))") \
          -DBUILD_CUDA_MODULE=OFF -DGLIBCXX_USE_CXX11_ABI=ON -DBUILD_PYTORCH_OPS=ON -DBUILD_TENSORFLOW_OPS=OFF \
          -DBUNDLE_OPEN3D_ML=ON -DOPEN3D_ML_ROOT=/workspace/Open3D-ML .. && \
    make -j$(nproc) && \
    cmake -DPYTHON3_INCLUDE_DIR=$(python -c "import sysconfig && print(sysconfig.get_path('include'))") \
          -DPYTHON3_LIBRARY=$(python -c "import sysconfig && print(sysconfig.get_config_var('LIBDIR'))") \
          -DBUILD_CUDA_MODULE=ON -DGLIBCXX_USE_CXX11_ABI=ON -DBUILD_PYTORCH_OPS=ON -DBUILD_TENSORFLOW_OPS=OFF \
          -DBUNDLE_OPEN3D_ML=ON -DOPEN3D_ML_ROOT=/workspace/Open3D-ML .. && \
    make -j$(nproc) && \
    make install-pip-package

# uninstall build dependencies from virtual environment again
RUN python -m pip uninstall -y yapf wheel packaging
    
####################################################################
######################### Target Image #############################
####################################################################

FROM nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION=3.13.12

WORKDIR /workspace

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    git \
    g++ \
    gnupg \
    libatomic1 \
    libegl1 \
    libbz2-dev \
    libffi-dev \
    libgl1 \
    libgomp1 \
    liblzma-dev \
    lsb-release \
    libssl-dev \
    libx11-xcb1 \
    make \
    software-properties-common \
    tzdata \
    unzip \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

######################## Rclone installation ########################

RUN curl https://rclone.org/install.sh | bash

#################### Copy Virtual Environment ######################

ENV VIRTUAL_ENV=/workspace/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder /workspace/venv /workspace/venv

# copy additional libs that are needed by Open3D
COPY --from=builder /usr/lib/x86_64-linux-gnu/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so
COPY --from=builder /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so.1
COPY --from=builder /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0
COPY --from=builder /usr/lib/x86_64-linux-gnu/libGLX.so.0 /usr/lib/x86_64-linux-gnu/libGLX.so.0

# copy python installation to avoid building python twice
COPY --from=builder /usr/local/bin/python3.13 /usr/local/bin/python3.13
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/include/python3.13 /usr/local/include/python3.13

RUN mkdir pointtorch
ADD . pointtorch
RUN python -m pip install ./pointtorch[dev,docs]