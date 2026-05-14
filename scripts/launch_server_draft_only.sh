#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../inference/configs/idlm_blockN4_draft_only_config.yaml"

python -m sglang.launch_server \
    --model-path yifanyu/I-DLM-8B \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm IDLMBlockNDraftOnly \
    --dllm-algorithm-config "${CONFIG}" \
    --port 30000
