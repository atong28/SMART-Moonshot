# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl ca-certificates bzip2 git build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ARG MINICONDA=Miniconda3-py312_24.5.0-0-Linux-x86_64.sh
RUN curl -fsSL https://repo.anaconda.com/miniconda/${MINICONDA} -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda config --system --set auto_update_conda false && \
    conda config --system --add channels conda-forge && \
    conda config --system --set channel_priority strict && \
    conda install -n base -y mamba && \
    conda clean -afy

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace
COPY environment.yml /workspace/environment.yml

# Create env & install Python 3.12
RUN mamba create -y -n moonshot python=3.12 && \
    conda activate moonshot && \
    # Install PyTorch CUDA 12.4 wheels
    pip install --no-cache-dir \
      torch==2.5.1+cu124 \
      torchvision==0.20.1+cu124 \
      torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124 && \
    # Update environment from YAML (no prune)
    mamba env update -n moonshot -f /workspace/environment.yml && \
    conda clean -afy

# Make env "always active"
ENV CONDA_DEFAULT_ENV=moonshot \
    CONDA_PREFIX=/opt/conda/envs/moonshot \
    PATH=/opt/conda/envs/moonshot/bin:/opt/conda/bin:$PATH
RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate moonshot' >> /etc/bash.bashrc

CMD ["/bin/bash"]
