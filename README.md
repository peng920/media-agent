# AIGC 教学创作平台（M1 打样）

控制面（FastAPI，本仓库 `backend/`）+ 执行面（外部 ComfyUI，attach 模式）。
设计文档：`docs/superpowers/specs/2026-10-12-aigc-teaching-platform-design.md`
实施计划：`docs/superpowers/plans/2026-10-12-m1-prototype.md`

## 前置条件

1. 本机 ComfyUI 已运行于 `http://127.0.0.1:8188`（attach 模式，控制面不管理引擎进程；本机启动方式见下文「本机环境约定」）。
2. 文生图模板（三选一在场即可，互不依赖）：
   - **Z-Image Turbo**（推荐，T4/Turing 可用）：`diffusion_models/z_image_turbo_bf16.safetensors`（12.3G）+
     `text_encoders/qwen_3_4b.safetensors`（8.0G，与 klein 共用）+ `vae/ae.safetensors`（0.34G），
     仓库 `Comfy-Org/z_image_turbo`
   - **FLUX.2 klein 4B**：`diffusion_models/flux-2-klein-4b.safetensors`（7.8G）+
     `text_encoders/qwen_3_4b.safetensors` + `vae/flux2-vae.safetensors`（0.34G），
     仓库 `Comfy-Org/flux2-klein-4B`
   - **Qwen-Image 2.1 Turbo**：`diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors` +
     `text_encoders/qwen3vl_8b_w4a8.safetensors`（或 int8_convrot）+ `vae/qwen_image_2.1_vae_bf16.safetensors`。
     **仅限驱动 ≥580（CUDA 13）的机器**——int8_convrot/w4a8 量化走 comfy-kitchen 的 CUDA13 内核，
     驱动 550（如本机 T4）会报 "CUDA driver version is insufficient"
3. 以上模型均可用 `uv run scripts/download_models.py` 从 ModelScope 自动下载到引擎 models/ 对应目录。

## 启动

```bash
cd backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8090
```

种子账号：`admin/admin123`、`student1/stu123`、`student2/stu123`。
`GET /api/templates` 的 `missing_models` 为空即模型齐备，可直接提交任务。

### 本机（共享 4×T4 服务器）环境约定

- 引擎：`/data/ComfyUI`（**pin v0.35.2** + torch 2.7.1+cu126 + comfy-kitchen 0.2.33 + GGUF/KJNodes 节点）。
  启动用 `scripts/start_engine.sh`（**固定只用 0 号 T4**，其余卡归其他租户）；后端 `scripts/start_backend.sh`；停止 `scripts/stop_services.sh`。
- **版本锁死原因**：本机 NVIDIA 驱动 550（CUDA 12.4）。comfy-kitchen 自 0.2.30 起所有内核均为
  CUDA 13 构建（驱动 580+ 才能跑），而 int8_convrot/w4a8 量化必须走 kitchen 内核 →
  **Qwen-Image 2.1 Turbo（int8_convrot DiT + w4a8 编码器）在本机不可用**，文生图改用
  Z-Image Turbo（bf16）与 FLUX.2 klein 4B（蒸馏）模板。
  ComfyUI ≥ v0.36 的代码也依赖新版 kitchen API，故引擎版本锁 v0.35.2，升级前必须先升驱动。
- 后端连引擎配置在 `backend/.env`（`ZIMG_ENGINE_INPUT_DIR=/data/ComfyUI/input`）。
- 模型下载：`uv run scripts/download_models.py --list` 查看清单/缺什么，去掉 `--list` 下载
  （ModelScope 源、断点续传、可 `--only 组名`）。

### 前端

- **单进程模式**（演示用）：`cd frontend && npm run build` 会产出到 `backend/frontend_dist/`，FastAPI 自动挂载，`http://127.0.0.1:8090/` 即完整界面。
- **开发模式**：`cd frontend && npm run dev`，访问 `http://localhost:5173`，`/api`、`/files`、`/ws` 已代理到 8090。

## 测试

```bash
cd backend && uv run pytest -q        # 离线单测（假引擎）
```

## M1 边界

单槽队列、文生图模板（Qwen-Image 2.1 Turbo）、HMAC 认证与多用户隔离、WS 进度事件、产物按用户落盘 `data/users/{uid}/outputs/{job_id}/`。
**M2 进行中**：LTX-2.3 图生视频模板（音视频联合生成，unsloth GGUF 配方）、图片上传（`POST /api/uploads`，落引擎 `input/` 目录）、视频产物预览。
多步 DSL 链、显存 Profile、Agent、加密狗 → M2+（见设计文档 §8）。

### LTX-2.3 图生视频前置（GGUF 路线）

- `diffusion_models/ltx-2.3-22b-dev-Q3_K_S.gguf`（unsloth）
- `text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`
  `https://hf-mirror.com/unsloth/gemma-3-12b-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`
- `text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors`（unsloth）
- `vae/ltx-2.3-22b-dev_video_vae.safetensors`、`vae/ltx-2.3-22b-dev_audio_vae.safetensors`
- 文本编码器经 `DualCLIPLoaderGGUF(type=ltxv)` 加载——需引擎以**较新代码**启动（ComfyUI 核心 v0.39+ 的 `DualCLIPLoader` type 枚举含 ltxv；若引擎启动过早需重启进程）
- 本机 ComfyUI-GGUF 有本地补丁（qwen3vl mmproj 合并支持），已提交到其本地 git 分支，更新该节点仓库时注意保留

### LTX-Video 2B 蒸馏版 前置（低显存档）

- `diffusion_models/ltxv-2b-0.9.6-distilled-04-25.safetensors`（Lightricks/LTX-Video，5.9G）
- `text_encoders/t5-v1_1-xxl-encoder-Q4_K_M.gguf`（city96/t5-v1_1-xxl-encoder-gguf，4.2G）
- `vae/ltxv_vae.safetensors`：**注意** Lightricks 仓库的 `vae/diffusion_pytorch_model.safetensors` 已被替换为 LTX 2.x 新格式（键名带 `.conv.` 嵌套），与 0.9.x 不兼容；旧格式 VAE 需从 `ltx-video-2b-v0.9.5.safetensors` checkpoint 抽取 `vae.*` 键（`scripts/download_models.py` 已自动处理）

### Wan 2.1 1.3B 文生视频 前置（Comfy-Org 重打包）

- `diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors`（2.6G）
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`（6.3G）
- `vae/wan_2.1_vae.safetensors`（0.24G）
- 仓库：`Comfy-Org/Wan_2.1_ComfyUI_repackaged` 的 `split_files/` 对应子目录

### FastHunyuan 文生视频 前置（720p 6步蒸馏）

- `diffusion_models/fast-hunyuan-video-t2v-720p-Q3_K_S.gguf`（city96/FastHunyuan-gguf，5.7G）
- `text_encoders/llava_llama3_fp8_scaled.safetensors` + `text_encoders/clip_l.safetensors`（Comfy-Org/HunyuanVideo_repackaged；双编码器经 `DualCLIPLoader(type=hunyuan_video)`）
- `vae/hunyuan_video_vae_bf16.safetensors`（同上仓库，0.46G）
- 官方推荐参数已内置：6 步 / shift 17 / CFG 7 / 1280×720×125 帧
