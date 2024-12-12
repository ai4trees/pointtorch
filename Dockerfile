FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# needed to install the CUDA version of PyTorch3D
# see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# see https://github.com/pytorch/extension-cpp/issues/71
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends git lsb-release software-properties-common wget

######################## Python installation #########################

RUN apt-get install -y --no-install-recommends python3.11-dev python3.11-distutils python3.11-venv
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN python3.11 get-pip.py && python3.11 -m pip install --upgrade pip

#################### Create Virtual Environment ######################

ENV VIRTUAL_ENV=/workspace/venv
RUN python3.11 -m venv $VIRTUAL_ENV

# by adding the venv to the search path, we avoid activating it in each command
# see https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

######## Install PyTorch and packages that depend on PyTorch #########

# these packages are dependencies of the other packages but not listed in their requirements
# therefore, they have to be installed manually
RUN python -m pip install tensorboard yapf wheel packaging
RUN python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
RUN python -m pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
RUN python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN python -m pip install spconv-cu124
RUN python -m pip install flash-attn --no-build-isolation

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
          -DBUILD_CUDA_MODULE=OFF -DGLIBCXX_USE_CXX11_ABI=OFF -DBUILD_PYTORCH_OPS=ON -DBUILD_TENSORFLOW_OPS=OFF \
          -DBUNDLE_OPEN3D_ML=ON -DOPEN3D_ML_ROOT=/workspace/Open3D-ML .. && \
    make -j$(nproc) && \
    cmake -DPYTHON3_INCLUDE_DIR=$(python -c "import sysconfig && print(sysconfig.get_path('include'))") \
          -DPYTHON3_LIBRARY=$(python -c "import sysconfig && print(sysconfig.get_config_var('LIBDIR'))") \
          -DBUILD_CUDA_MODULE=ON -DGLIBCXX_USE_CXX11_ABI=OFF -DBUILD_PYTORCH_OPS=ON -DBUILD_TENSORFLOW_OPS=OFF \
          -DBUNDLE_OPEN3D_ML=ON -DOPEN3D_ML_ROOT=/workspace/Open3D-ML .. && \
    make -j$(nproc) && \
    make install-pip-package

# uninstall build dependencies from virtual environment again
RUN python -m pip uninstall -y yapf wheel packaging
    
####################################################################
######################### Target Image #############################
####################################################################

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# include packages needed for the rclone installation / as runtime dependencies of Open3D
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git gnupg make unzip libegl1 libgl1 libgomp1 libx11-xcb1 libatomic1

######################## Python installation ########################

RUN apt-get update && apt-get install -y --no-install-recommends wget python3.11-dev python3.11-distutils python3.11-venv
RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN python3.11 get-pip.py && python3.11 -m pip install --upgrade pip

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

RUN mkdir pointtorch
ADD . pointtorch
RUN python -m pip install ./pointtorch[dev,docs]
