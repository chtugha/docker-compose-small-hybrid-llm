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
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_URL = "http://127.0.0.1:8000"
KNOWN_MODELS = ("cyankiwi/Ornith-1.5-9B-AWQ-INT4", "gpt-5.6-sol")
DROP_TOP_LEVEL_KEYS = ("service_tier", "store", "include", "prompt_cache_key", "safety_identifier", "client_metadata")
BLOCKED_TOOL_PREFIXES = ("mcp__", "browser", "computer", "sky", "pdf", "spreadsheet", "presentation", "document", "visualize", "template", "web_search")
BLOCKED_EXACT_NAMES = ("js", "js_reset")

def extract_tools_from_list(tool_list):
    extracted = []
    for t in tool_list:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type", "")
        if t_type == "namespace" and isinstance(t.get("tools"), list):
            extracted.extend(extract_tools_from_list(t["tools"]))
            continue

        name = t.get("name") or t.get("function", {}).get("name", "")
        low_name = name.lower()
        if not name or low_name in BLOCKED_EXACT_NAMES or low_name.startswith(BLOCKED_TOOL_PREFIXES):
            continue

        desc = t.get("description") or t.get("function", {}).get("description", "")
        params = t.get("parameters") or t.get("function", {}).get("parameters")
        if not isinstance(params, dict):
            params = {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input payload or command for the tool"}
                },
                "required": ["input"]
            }

        extracted.append({
            "type": "function",
            "name": name,
            "description": desc,
            "parameters": params
        })
    return extracted

class CodexFilterProxy(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

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
        except (BrokenPipeError, ConnectionResetError):
            pass
        except urllib.error.HTTPError as e:
            try:
                err = e.read()
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(err)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)

            if data.get("model") not in KNOWN_MODELS:
                data["model"] = "gpt-5.6-sol"

            collected_tools = extract_tools_from_list(data.get("tools", []))

            if isinstance(data.get("input"), list):
                cleaned_input = []
                for item in data["input"]:
                    if isinstance(item, dict) and item.get("type") == "additional_tools":
                        if isinstance(item.get("tools"), list):
                            collected_tools.extend(extract_tools_from_list(item["tools"]))
                        continue
                    if isinstance(item, dict) and item.get("type") == "item_reference":
                        continue
                    cleaned_input.append(item)
                data["input"] = cleaned_input

            if collected_tools:
                unique_tools = {}
                for tool in collected_tools:
                    unique_tools[tool["name"]] = tool
                data["tools"] = list(unique_tools.values())

            for key in DROP_TOP_LEVEL_KEYS:
                data.pop(key, None)

            body = json.dumps(data).encode("utf-8")
        except Exception as ex:
            sys.stderr.write(f"JSON PARSE ERROR: {ex}\n")
            sys.stderr.flush()

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
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except urllib.error.HTTPError as e:
            try:
                err = e.read()
                sys.stderr.write(f"VLLM REJECTED {e.code}: {err.decode('utf-8', 'ignore')}\n")
                sys.stderr.flush()
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(err)
            except (BrokenPipeError, ConnectionResetError):
                pass

ThreadingHTTPServer.allow_reuse_address = True
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



systemctl daemon-reload
systemctl enable --now codex-proxy.service

_____________

On the MAC: !!!! be AWARE OF THE NECESSARY base_url CHANGE !!!! AND THE PATHS !!!

nano ~/.codex/config.toml

sandbox_mode = "workspace-write"
web_search = "live"
service_tier = "priority"
notify = ["/Users/ollama/.codex/computer-use/Codex Computer Use.app/Contents/SharedSupport/SkyComputerUseClient.app/Contents/MacOS/SkyComputerUseClient", "turn-ended"]
model_provider = "dietpi_vllm"
model = "gpt-5.6-sol"
model_reasoning_effort = "low"
model_context_window = 131072
model_auto_compact_token_limit = 95000

[model_providers.dietpi_vllm]
name = "DietPi vLLM (Ornith-1.5-9B)"
base_url = "http://192.168.10"
wire_api = "responses"
requires_openai_auth = false
env_key = "OPENAI_API_KEY"

[profiles.dietpi]
model_provider = "dietpi_vllm"
model = "cyankiwi/Ornith-1.5-9B-AWQ-INT4"
model_reasoning_effort = "low"
model_context_window = 131072
model_auto_compact_token_limit = 95000
web_search = "disabled"
service_tier = "default"

[profiles.dietpi.mcp_servers.node_repl]
enabled = false

[profiles.dietpi.plugins."browser@openai-bundled"]
enabled = false

[profiles.dietpi.plugins."unified-computer-use@openai-bundled"]
enabled = false

[profiles.dietpi.plugins."computer-use@openai-bundled"]
enabled = false

[profiles.dietpi.plugins."visualize@openai-bundled"]
enabled = false

[profiles.dietpi.plugins."documents@openai-primary-runtime"]
enabled = false

[profiles.dietpi.plugins."pdf@openai-primary-runtime"]
enabled = false

[profiles.dietpi.plugins."spreadsheets@openai-primary-runtime"]
enabled = false

[profiles.dietpi.plugins."presentations@openai-primary-runtime"]
enabled = false

[profiles.dietpi.plugins."template-creator@openai-primary-runtime"]
enabled = false

[desktop]
followUpQueueMode = "queue"
conversationDetailMode = "STEPS_COMMANDS"
ambient-suggestions-enabled = false
projectlessWorkspaceRoot = "/Volumes/SSDE/Projectless"
preventSleepWhileRunning = true
composerPlainTextMode = true
show-context-window-usage = true
composerEnterBehavior = "cmdIfMultiline"
agent-usage-reset-enabled = false

[desktop.open-in-target-preferences]
global = "systemDefault"

[marketplaces.openai-bundled]
source_type = "local"
source = "/Users/ollama/.codex/.tmp/bundled-marketplaces/openai-bundled"

[marketplaces.openai-primary-runtime]
source_type = "local"
source = "/Users/ollama/.cache/codex-runtimes/codex-primary-runtime/plugins/openai-primary-runtime"

[plugins."codex-app-tools@openai-bundled"]
enabled = true

[plugins."browser@openai-bundled"]
enabled = true

[plugins."unified-computer-use@openai-bundled"]
enabled = true

[plugins."visualize@openai-bundled"]
enabled = true

[plugins."documents@openai-primary-runtime"]
enabled = true

[plugins."pdf@openai-primary-runtime"]
enabled = true

[plugins."spreadsheets@openai-primary-runtime"]
enabled = true

[plugins."presentations@openai-primary-runtime"]
enabled = true

[plugins."template-creator@openai-primary-runtime"]
enabled = true

[plugins."computer-use@openai-bundled"]
enabled = true

[mcp_servers.node_repl]
args = []
command = "/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node_repl"
env_vars = ["BROWSER_USE_ACCESS_EXECUTOR_ID", "BROWSER_USE_ACCESS_PIPE_PATH", "BROWSER_USE_ACCESS_VERIFYING_IDENTITY"]
startup_timeout_sec = 120

[mcp_servers.node_repl.env]
NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS = "1000"
NODE_REPL_NODE_MODULE_DIRS = "/Applications/ChatGPT.app/Contents/Resources/cua_node/lib/node_modules"
NODE_REPL_NODE_PATH = "/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node"
NODE_REPL_TRUSTED_CODE_PATHS = "/Users/ollama/.codex:/Applications/ChatGPT.app/Contents/Resources/cua_node/lib/node_modules"
CODEX_HOME = "/Users/ollama/.codex"
BROWSER_USE_AVAILABLE_BACKENDS = "chrome,iab,mcpapps"
BROWSER_USE_TINYSKY_ENABLED = "1"
NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER = ""
NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME = ""
NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE = ""
BROWSER_USE_CODEX_APP_BUILD_FLAVOR = "prod"
BROWSER_USE_CODEX_APP_VERSION = "26.1111.21308"
NODE_REPL_TRUSTED_SERVICES = '{"browser":"/Users/ollama/.codex/plugins/cache/openai-bundled/browser/26.1111.21308/scripts/browser-service.mjs","sky":"@oai/sky/service"}'
SKY_CUA_SERVICE_PATH = "/Users/ollama/.codex/computer-use/Codex Computer Use.app"
CODEX_CLI_PATH = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"

[features]
local_thread_store_compression = true

[projects."/Volumes/SSDE/brassclaw"]
trust_level = "trusted"

[projects."/private/tmp"]
trust_level = "trusted"

[projects."/Users/ollama/.codex"]
trust_level = "trusted"

[sandbox_workspace_write]
network_access = true
writable_roots = [
  "/Volumes/SSDE/brassclaw",
  "/Users/ollama/brassclaw-target",
  "/private/tmp",
  "/Volumes/Philips",
]

[shell_environment_policy.set]
CARGO_TARGET_DIR = "/Users/ollama/brassclaw-target"

[apps.connector_openai_chatgpt_pets]
enabled = false

[plugins."codex-app-tools@openai-bundled".mcp_servers.codex_app.tools.send_message_to_thread]
approval_mode = "approve"
