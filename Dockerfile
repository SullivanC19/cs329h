FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Sanity check: this line will FAIL the build if gcc is missing
RUN gcc --version && cc --version

# Torch + vLLM + deps ...
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        "torch==2.4.0" \
        --extra-index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir \
        numpy==1.26.4 \
        scikit-learn==1.4.0 \
        matplotlib==3.8.2 \
        seaborn==0.13.2 \
        huggingface-hub==0.34.0 \
        datasets==2.19.0 \
        hdbscan==0.8.40 \
        sentence-transformers==5.1.1 \
        vllm==0.11.0 \
        ipykernel==6.29.4

WORKDIR /experiments
COPY experiments /experiments/

CMD ["/bin/bash"]
