# Use the base CUDA image
FROM nvidia/cuda:11.3.0-base

WORKDIR /app

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install CUDA runtime components
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-11-0 \
    && rm -rf /var/lib/apt/lists/*

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the CUDA_HOME environment variable
ENV CUDA_HOME="/usr/local/cuda"

RUN apt-get install -f
RUN apt-get --fix-broken install

# # Install NVIDIA System Management Interface (nvidia-smi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda git cmake build-essential libboost-all-dev\
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# # Install Python and required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the CUDA_HOME environment variable
ENV CUDA_HOME="/usr/local/cuda"
# Download and extract dlib source code

RUN curl -LO https://github.com/davisking/dlib/archive/v19.24.2.tar.gz \
    && tar xzf v19.24.2.tar.gz \
    && rm v19.24.2.tar.gz

# Build and install dlib
WORKDIR /app/dlib-19.24.2
RUN mkdir build \
    && cd build \
    && cmake .. \
    && cmake --build . --config Release \
    && make install

WORKDIR /app
COPY docker_requirements.txt .
RUN ldconfig && pip3 install -r ./docker_requirements.txt

# Set the default command to run nvidia-smi
CMD ["nvidia-smi"]
