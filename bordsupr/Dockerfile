FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
SHELL ["/bin/bash", "-c"]

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Base Ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    tzdata \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    python3-pip \
    git \
    libeigen3-dev \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# Add ROS 2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Humble and ROS tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
 && rosdep init || true \
 && rm -rf /var/lib/apt/lists/*

# Build and install OpenGV
RUN git clone --depth 1 https://github.com/laurentkneip/opengv.git /tmp/opengv \
 && cmake -S /tmp/opengv -B /tmp/opengv/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
 && cmake --build /tmp/opengv/build -j"$(nproc)" \
 && cmake --install /tmp/opengv/build \
 && rm -rf /tmp/opengv

# Help CMake find packages installed in /usr/local
ENV CMAKE_PREFIX_PATH=/usr/local:${CMAKE_PREFIX_PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Install pip packages
RUN python3 -m pip install openai ultralytics "numpy<2" opencv-python psycopg transformers huggingface_hub psycopg2 sentence-transformers timm
RUN python3 -m pip install --upgrade Pillow

# InsightFace + ONNX Runtime GPU (CUDAExecutionProvider) for GPU face embeddings, plus the
# CUDA-12 runtime libs the GPU wheel links against (this image ships CUDA 13 / torch cu13).
# Baked into the image so `docker compose up --build` needs no re-download; the runtime command
# in docker-compose.yaml re-asserts these for plain `docker compose up` on a fresh container.
RUN python3 -m pip install --no-cache-dir insightface onnxruntime-gpu==1.23.2 \
    && python3 -m pip install --no-cache-dir \
       nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cudnn-cu12

# Let the loader find the CUDA-12 nvidia libs container-wide (image level).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib
# setuptools >= 83 imports top-level `packaging` and calls
# canonicalize_version(strip_trailing_zero=...), which the apt-packaged
# packaging 21.3 does not support. Install a modern packaging so colcon
# builds (ament_cmake_python egg_info) do not fail.
RUN python3 -m pip install --upgrade "packaging>=24.2"

CMD ["/bin/bash"]

# docker build -t ros2-humble:latest .
# cd runtime
# docker run -it --rm -v ${PWD}:/workspace/src -w /workspace/ ros2-humble

# Epic Newton