#!/bin/bash
# 停引擎(8188)与后端(8090)；注意不要用 pkill -f（会误杀自身）
lsof -ti:8188 | xargs -r kill
lsof -ti:8090 | xargs -r kill
echo "stopped"
