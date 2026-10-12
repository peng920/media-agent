# AIGC 教学创作平台（M1 打样）

控制面（FastAPI，本仓库 `backend/`）+ 执行面（外部 ComfyUI，attach 模式）。
设计文档：`docs/superpowers/specs/2026-10-12-aigc-teaching-platform-design.md`
实施计划：`docs/superpowers/plans/2026-10-12-m1-prototype.md`

## 前置条件

1. 本机 ComfyUI 已运行于 `http://127.0.0.1:8188`（打样为 attach 模式，控制面不管理引擎进程）。
2. 模型就位（`/data/github/ComfyUI/models/`）：
   - `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors`
   - `text_encoders/qwen3vl_8b_int8_convrot.safetensors` ← **需手动下载**
     `https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors`
     （ModelScope 镜像：Comfy-Org/Qwen-Image-2.1 同路径）
   - `vae/qwen_image_2.1_vae_bf16.safetensors`
   - `loras/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors`

## 启动

```bash
cd backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8090
```

种子账号：`admin/admin123`、`student1/stu123`、`student2/stu123`。
`GET /api/templates` 的 `missing_models` 为空即模型齐备，可直接提交任务。

### 前端

- **单进程模式**（演示用）：`cd frontend && npm run build` 会产出到 `backend/frontend_dist/`，FastAPI 自动挂载，`http://127.0.0.1:8090/` 即完整界面。
- **开发模式**：`cd frontend && npm run dev`，访问 `http://localhost:5173`，`/api`、`/files`、`/ws` 已代理到 8090。

## 测试

```bash
cd backend && uv run pytest -q        # 离线单测（假引擎）
```

## M1 边界

单槽队列、文生图模板（Qwen-Image 2.1 Turbo）、HMAC 认证与多用户隔离、WS 进度事件、产物按用户落盘 `data/users/{uid}/outputs/{job_id}/`。
多步 DSL 链、显存 Profile、Agent、加密狗 → M2+（见设计文档 §8）。
