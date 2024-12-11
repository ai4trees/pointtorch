FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git

######## Install Python packages that depend on PyTorch #########

RUN python -m pip install --upgrade torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
RUN python -m pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
RUN python -m pip install spconv-cu113

########################## Open3D build ##########################

WORKDIR /workspace

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

# these packages are dependencies of Open3D but not listed in the requirements of the Open3D package setup
# therefore, they have to be installed manually
RUN python -m pip install tensorboard yapf wheel

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

########################## PyTorch3D build ##########################

RUN python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

COPY --from=builder /opt/conda/lib/python3.11/site-packages/pytorch3d /opt/conda/lib/python3.11/site-packages
COPY --from=builder /opt/conda/lib/python3.11/site-packages/pytorch3d-*.dist-info /opt/conda/lib/python3.11/site-packages

RUN apt-get update && apt-get install -y --no-install-recommends curl git gnupg make unzip && \
    curl https://rclone.org/install.sh | bash

RUN python -m pip install \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
RUN mkdir pointtorch
ADD . pointtorch
RUN python -m pip install .[dev,docs]
