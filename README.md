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
    command: sh -c "export PYTHONHASHSEED=0 && python3 -m lmcache.v1.standalone --host 0.0.0.0 --port 5555"
    restart: unless-stopped

  vllm:
    image: lmcache/vllm-openai:latest  # Pin to a specific version for reproducibility
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
      - LMCACHE_PORT=5555
      - LMCACHE_HOST=0.0.0.0
      - LMCACHE_USE_CONTROLLER=False
    restart: unless-stopped
    entrypoint: ["vllm", "serve"]
    command: >
      cyankiwi/Ornith-1.0-9B-AWQ-INT4
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
      --kv-offloading-size 16
      --no-disable-hybrid-kv-cache-manager
      --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"lmcache-server","lmc>
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
