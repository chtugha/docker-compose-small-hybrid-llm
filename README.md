# docker-compose.yml
# Docker Compose v2 format (version key is deprecated and omitted)
# Create a .env file in the same directory with: HF_TOKEN=hf_yourtoken
# Add .env to .gitignore to prevent accidental commit.

services:
    # 1. LMCache Backend Server (Stores offloaded KV caches in CPU/System RAM)
  lmcache-server:
    image: lmcache/standalone:latest  # Use official LMCache image
    container_name: lmcache-server
    ports:
      - "5555:5555"
    command: python3 -m lmcache_server.server --host 0.0.0.0 --port 5555
    restart: unless-stopped

  vllm:
    image: vllm/vllm-openai:latest  # Pin to a specific version for reproducibility
    container_name: vllm-ornith
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    env_file:
      - .env                          # Contains HF_TOKEN=hf_yourtoken; not committed to VCS
    environment:
      # env_file supplies HF_TOKEN; remap to the name vLLM/HF libraries expect:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    restart: unless-stopped
    entrypoint: ["vllm", "serve"]
    command: >
      --model cyankiwi/Ornith-1.0-9B-AWQ-INT4
      --quantization compressed-tensors
      --gpu-memory-utilization 0.85
      --max-model-len 131072
      --max-num-batched-tokens 8192
      --tensor-parallel-size 1
      --max-num-seqs 1
      --enable-prefix-caching
      --enable-chunked-prefill
      --port 8000
      --kv-cache-dtype fp8
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --reasoning-parser qwen3
      --language-model-only
      --kv-offloading-backend lmcache
      --kv-offloading-size 16GiB
      --kv-offloading-config '{"host":"lmcache-server","port":5555}'
    depends_on:
      - lmcache-server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      # /health path is version-dependent. /v1/models is stable across vLLM versions.
      test: ["CMD", "curl", "-f", "--max-time", "5", "http://localhost:8000/health"]
      interval: 60s
      timeout: 10s
      retries: 10        # Increased: model download on first run may exceed 5 retries
      start_period: 901s # Grace period before health checks begin
