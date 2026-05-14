python -m sglang.launch_server \
    --model-path yifanyu/I-DLM-8B \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm IDLMBlockN \
    --dllm-algorithm-config inference/configs/idlm_blockN4_config.yaml \
    --port 30000