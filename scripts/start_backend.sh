#!/bin/bash
# 控制面后端启动（attach 已在 8188 运行的引擎；engine_input_dir 见 backend/.env）
cd "$(dirname "$0")/../backend"
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8090
