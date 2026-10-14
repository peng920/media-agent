#!/bin/bash
# ComfyUI 引擎启动（本机约定：共享机上只用 0 号 T4 卡，其余卡归其他租户）
cd /data/ComfyUI
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --cuda-device 0
