# 项目交接文档（2026-10-14）

> 给接手的 AI 助手/工程师：本文是完整上下文，读完即可继续开发。先读 §1-§3，再按 §7 待办推进。

## 1. 项目是什么

教育采购项目（标书序号 12，交互媒体制作）：交付**本地化全栈 AIGC 生成系统（软件平台）**，要求：双架构（Windows/CUDA + Apple MPS）、全离线、教学级多用户（学生/教师/管理员 + 配额 + 队列）、含文生图/图生视频/TTS、内置本地 LLM Agent、加密狗授权。交付硬件为 Lenovo P2（i7-14700K + RTX 3060 12GB）+ MacBook M5。

**核心架构决策（已评审定稿）**：不自研生成管线，**旁挂控制面**（FastAPI，闭源）+ **ComfyUI 作为 headless 执行引擎**（不改其内核，GPL 风险收敛为"独立程序 HTTP 通信"）；控制面队列是唯一队列，ComfyUI 作单槽执行器；所有 GPU 负载过执行面串行；任务 DSL 中间层隔离上游 schema 漂移。完整设计见 `docs/superpowers/specs/2026-10-12-aigc-teaching-platform-design.md`（12 条标书要求→验收物料映射在 §4.9）。

当前处于 **M1 收口 + M2 进行中**：打样在开发机 RTX 4070 12GB / WSL2 上进行。

## 2. 仓库与运行方式

- **仓库**：`git@github.com:peng920/media-agent.git`（SSH 已认证账号 peng920）
- **分支**：全部工作在 `feature/m1-prototype`（main 落后，继续用 feature 分支，M2 完成后再合并）；截至交接 HEAD = `5a7c3bc`
- **后端**：`backend/`，Python 3.12 + uv。启动：`cd backend && uv run uvicorn app.main:app --host 127.0.0.1 --port 8090`；测试：`uv run pytest -q`（当前 **43 passed**，全部离线假引擎）
- **前端**：`frontend/`，React 18 + Vite + TS + antd 6（**注意**：不可升 React 19，antd 6 + 18 是验证过的组合）。开发：`npm run dev`（5173，代理 /api /ws /files → 8090）；构建产物输出到 `backend/frontend_dist/`，FastAPI 自动挂载单进程服务
- **配置**：环境变量前缀 `ZIMG_`（`engine_base_url` 默认 http://127.0.0.1:8188、`engine_input_dir` 默认 /data/github/ComfyUI/input——**新服务器必须改这个**）
- 种子账号：admin/admin123、student1/stu123、student2/stu123

## 3. 引擎（新服务器要重建的部分）

本机现状（不随仓库走）：`/data/github/ComfyUI`（v0.39.0，miniconda python3.13 运行，`--listen 127.0.0.1 --port 8188`），custom nodes 有 **ComfyUI-GGUF（含本地补丁：qwen3vl mmproj 合并支持，已提交到其本地 git 分支，更新时保留）** 和 ComfyUI-KJNodes。

控制面是 **attach 模式**：不管理引擎进程，只调 HTTP API（/prompt /history /object_info /view /interrupt /queue）。新环境要求：ComfyUI ≥ v0.39 且**以新代码启动**（DualCLIPLoader 的 type 枚举含 ltxv；引擎进程若在代码更新前启动需重启才能生效——踩过这个坑）。

## 4. 已实现（截至交接）

- 认证（pbkdf2 + HMAC token 24h）、多用户隔离（学生只见自己，admin/teacher 见全部）
- 单槽调度器：FIFO 单槽、运行中取消走 interrupt、断线对账（/history）、事件经 Notifier 广播 → `/ws/jobs?token=` WebSocket + 前端 3 秒重连 + 10 秒轮询兜底
- 任务产物按用户落盘 `data/users/{uid}/outputs/{job_id}/`，经引擎 /view 下载（支持 images/videos/gifs/audio 键）
- 图片上传 `POST /api/uploads`（留档 + 分发引擎 input/，20MB 限制）
- `/api/templates` 返回每模板的 defaults + **按模板隔离的必需/可选缺失清单**（可选缺失只提醒不堵门）
- 前端三页：登录、创作（模板切换自动回填默认值、按 capability 显隐图片上传）、任务队列（进度条、取消、图片缩略图 + `<video>` 预览）

**5 个模板**（`backend/app/compiler/templates.py`，模型文件候选制 + resolver 自动适配在场变体）：

| 模板 id | 能力 | 状态 |
|---|---|---|
| qwen-image-2.1-turbo | 文生图 | ✅ 真机验证过（w4a8 编码器 + int8 DiT + turbo，512² 约 15s/张，双账号并发+单槽排队+隔离全链路演示通过，出图经视觉模型确认） |
| ltx-2.3-i2v | 图生视频(音视频联合) | 休眠（模型文件被用户清理腾空间；代码就绪，unsloth GGUF 配方 + DualCLIPLoaderGGUF type=ltxv） |
| ltx-2b-i2v-distilled | 图生视频(低显存档) | 代码就绪，等 3 个文件（对标标书"6GB 跑基础视频模型"） |
| wan-2.1-1.3b-t2v | 文生视频 480p | 代码就绪，等 3 个文件 |
| fast-hunyuan-t2v-720p | 文生视频 720p 6步 | 代码就绪，等 3 个文件 |

## 5. 待下载模型清单（用户在慢慢下，放引擎 models/ 对应目录）

- **LTX2B**（~11.7G）：`Lightricks/LTX-Video` 的 `ltxv-2b-0.9.6-distilled-04-25.safetensors`(5.9G→diffusion_models) + `vae/diffusion_pytorch_model.safetensors`(1.6G，**改名 ltxv_vae.safetensors**→vae) + `city96/t5-v1_1-xxl-encoder-gguf` 的 `t5-v1_1-xxl-encoder-Q4_K_M.gguf`(4.2G→text_encoders)
- **Wan1.3B**（~9.2G）：`Comfy-Org/Wan_2.1_ComfyUI_repackaged` split_files 下的 `wan2.1_t2v_1.3B_fp16.safetensors`(2.6G) + `umt5_xxl_fp8_e4m3fn_scaled.safetensors`(6.3G) + `wan_2.1_vae.safetensors`(0.24G)
- **FastHunyuan**（~14.4G）：`city96/FastHunyuan-gguf` 的 `fast-hunyuan-video-t2v-720p-Q3_K_S.gguf`(5.7G→diffusion_models) + `Comfy-Org/HunyuanVideo_repackaged` 的 `llava_llama3_fp8_scaled.safetensors`(8.5G→text_encoders) + `hunyuan_video_vae_bf16.safetensors`(0.46G→vae)
- 镜像统一用 hf-mirror.com（直连 HF 不通）
- 每套到位后：跑真机冒烟（提交 512 级小参数任务 → DONE → 校验 mp4），FastHunyuan 先用 49 帧短时长验证

## 6. 关键教训（勿重蹈）

1. **文本编码器维度必须匹配**：qwen2.1 系列 DiT 要 4096 维（qwen3vl_8b），盘上 4b 版(2560维)会报 normalized_shape 错——所以有"模型清单启动校验"机制，别删。
2. ComfyUI 官方 workflow 模板包在 `miniconda site-packages/comfyui_workflow_templates_json/templates/`，是权威接线来源；UI 模板里的 subgraph 定义在 JSON 的 `definitions.subgraphs` 里挖。
3. ComfyUI-GGUF 的 DualCLIPLoaderGGUF 的 **type 枚举继承自核心 DualCLIPLoader**，clip_name 选项来自 text_encoders 目录（safetensors 和 gguf 都吃，按扩展名分流加载器）。
4. FastAPI 路由参数必须显式 `Request` 注入；`pkill -f` 会自杀（命令行含匹配串），用 `lsof -ti:端口 | xargs kill`。
5. 标书疑点（待商务澄清）：i7-14700K 实为 20C/28T（标书笔误 14C/20T）；Mac 型号措辞统一写 Apple Silicon。

## 7. 下一步（按优先级）

1. **真机冒烟**三套新模板（文件到位后逐个：提交→DONE→验证产物）
2. **多步 DSL 链**（M2 核心，验收演示"文字→图→视频"）：Job 表已预留 DSL 形态；调度器扩展 steps 数组顺序执行、步骤间产物传递（上一步 output 图 → LoadImage），ffmpeg 合成配音留 TTS 之后
3. **TTS 模板**：原机器 HF 缓存有 Qwen3-TTS-12Hz-1.7B / VoxCPM2 / MaskGCT 可参考选型（新服务器需自下）
4. **显存 Profile**（6GB/8GB/12GB/AUTO）+ 自研 custom node 上报实时显存（验收截图物料）
5. M3+：管理后台（配额/用户管理 UI）、离线安装包、加密狗、Mac/MPS 适配、夜班冒烟、Agent（本地 LLM 拆解→DSL）

## 8. 工作方式约定

- 开发流程用 superpowers 技能族：brainstorm → spec（docs/superpowers/specs/）→ plan（docs/superpowers/plans/）→ 执行；计划文档有 checkbox
- 提交粒度：一任务一提交，中文 message（feat:/fix:/test:/docs:/chore:），随时 push origin feature/m1-prototype
- 测试先行，假引擎（tests/fakes.py）保持与模板模型清单同步
- YAGNI 红线：不上 PG/MQ/微服务，不做 SSO，单机两进程结构即终态
