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
ExecStart=/opt/vLLM/.venv/bin/lmcache server --host 127.0.0.1 --port 5555 --l1-size-gb 24 --eviction-policy LRU --chunk-size 1056 --worker-reap-timeout-seconds 0 --separate-object-groups
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
Environment="HF_TOKEN=hf_cSkfdDe"
Environment="CUDA_HOME=/usr/local/cuda"
Environment="PATH=/usr/local/cuda/bin:/opt/vLLM/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


ExecStart=/opt/vLLM/.venv/bin/vllm serve cyankiwi/Ornith-1.5-9B-AWQ-INT4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --safetensors-load-strategy prefetch \
    --mm-processor-kwargs '{"cap_pixels_per_frame":true}' \
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



export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda





____________________________________


nano /opt/vLLM/codex_proxy.py

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_URL = "http://127.0.0.1:8000"

# Nur Basis-Tools von Codex durchlassen, alle Plugin-/MCP-Tools für das 9B-Modell herausfiltern
ALLOWED_TOOL_PREFIXES = ("shell", "local_shell", "apply_patch", "read_file", "write_file", "list_dir", "grep", "find")

class CodexFilterProxy(BaseHTTPRequestHandler):
    def do_GET(self):
        req = urllib.request.Request(VLLM_URL + self.path)
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() != "transfer-encoding":
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)

            # 1. Unsupported 'additional_tools' aus input entfernen
            if isinstance(data.get("input"), list):
                data["input"] = [
                    item for item in data["input"]
                    if not (isinstance(item, dict) and item.get("type") == "additional_tools")
                ]

            # 2. Plugin- und MCP-Tools aus tools entfernen (nur Basis-Coding-Tools behalten)
            if isinstance(data.get("tools"), list):
                filtered_tools = []
                for tool in data["tools"]:
                    t_name = tool.get("name") or tool.get("function", {}).get("name", "")
                    t_type = tool.get("type", "")
                    if t_type in ("local_shell", "shell") or t_name.startswith(ALLOWED_TOOL_PREFIXES):
                        filtered_tools.append(tool)
                data["tools"] = filtered_tools

            # 3. OpenAI-spezifische Felder entfernen
            data.pop("service_tier", None)

            body = json.dumps(data).encode("utf-8")
        except Exception:
            pass

        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(VLLM_URL + self.path, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() != "transfer-encoding":
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8001), CodexFilterProxy).serve_forever()


_____________

nano /etc/systemd/system/codex-proxy.service

[Unit]
Description=Codex Filter Proxy for vLLM
After=network.target vllm.service
Wants=vllm.service

[Service]
Type=simple
User=root
ExecStart=/opt/vLLM/.venv/bin/python /opt/vLLM/codex_proxy.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target



____________

sed -i 's/if __name__ == "__main__":/ThreadingHTTPServer.allow_reuse_address = True\nif __name__ == "__main__":/' /opt/vLLM/codex_proxy.py

systemctl daemon-reload
systemctl enable --now codex-proxy.service

