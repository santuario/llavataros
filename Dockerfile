# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libsndfile1 \
    build-essential \
    cmake \
    gcc \
    g++ \
    ffmpeg \
    unzip \
    libopenblas-dev \ 
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda with Python 3.9 as the base environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init bash

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create a Conda environment for MFA (also using Python 3.9)
RUN conda create -n mfa python=3.9 -y \
    && /opt/conda/bin/conda run -n mfa conda install -c conda-forge montreal-forced-aligner -y \
    && /opt/conda/bin/conda run -n mfa mfa version || { echo "MFA installation failed"; exit 1; }

# Download MFA models (dictionary and acoustic model)
RUN mkdir -p /app/mfa_models \
    && wget -O /app/mfa_models/english_us_arpa.dict https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/refs/heads/main/dictionary/english/us_arpa/english_us_arpa.dict \
    && wget -O /app/mfa_models/english_us_arpa.zip https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_us_arpa-v3.0.0/english_us_arpa.zip \
    && unzip /app/mfa_models/english_us_arpa.zip -d /app/mfa_models \
    && rm /app/mfa_models/english_us_arpa.zip
# Install codec2 from source
RUN git clone https://github.com/drowe67/codec2.git \
    && cd codec2 \
    && mkdir build_linux \
    && cd build_linux \
    && cmake .. \
    && make \
    && make install \
    && ldconfig

# Set LD_LIBRARY_PATH for codec2
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy project files
COPY . .

# # Install Python dependencies in the mfa Conda environment
RUN /opt/conda/bin/conda run -n mfa conda install -y pip \
&& /opt/conda/bin/conda run -n mfa pip install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html \
&& /opt/conda/bin/conda run -n mfa pip install --no-cache-dir -r requirements.txt



# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV PATH=/usr/local/cuda-12.6/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Command to run the application
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate mfa && python main.py"]