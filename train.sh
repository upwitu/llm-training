#!/bin/bash

# Pick a free port automatically
MASTER_PORT=$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)

echo "Using master port: $MASTER_PORT"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
torchrun \
    --nproc_per_node=${NPROC_PER_NODE:-1} \
    --master_port=$MASTER_PORT \
    sft.py