cd /tmp

wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb

dpkg -i cuda-keyring_1.1-1_all.deb

apt update

apt install nvidia-open

apt install nvidia-driver-cuda nvidia-kernel-open-dkms cuda-toolkit

apt install -y python3-venv python3-pip

apt-get install -y linux-cpupower numactl

cpupower frequency-set -g performance

mkdir -p /opt/vllm
cd /opt/vLLM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -U lmcache vllm

nano /etc/systemd/system/lmcache.service

[Unit]
Description=LMCache MP KV Cache Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/vLLM
Environment="PATH=/opt/vLLM/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONHASHSEED=0"
ExecStart=/opt/vLLM/.venv/bin/lmcache server --host 127.0.0.1 --port 5555 --l1-size-gb 24 --eviction-policy LRU --chunk-size 1056
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target



systemctl daemon-reload
systemctl enable --now lmcache.service


cat >/etc/systemd/system/vllm.service <<'EOF'
[Unit]
Description=vLLM Ornith-1.5-9B-AWQ-INT4
After=network-online.target lmcache.service
Wants=network-online.target
Requires=lmcache.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/vLLM

Environment="PATH=/opt/vLLM/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="HF_HOME=/opt/vLLM/huggingface"
Environment="PYTHONHASHSEED=0"
Environment="HF_TOKEN=hf_*****'"
Environment="CUDA_HOME=/usr/local/cuda"
Environment="PATH=/usr/local/cuda/bin:/opt/vLLM/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ExecStart=/opt/vLLM/.venv/bin/vllm serve cyankiwi/Ornith-1.5-9B-AWQ-INT4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --safetensors-load-strategy=prefetch \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --performance-mode interactivity \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'

Restart=on-failure
RestartSec=10
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOF



export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda




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
