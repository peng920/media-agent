# AIGC 教学创作平台 技术设计文档

- 日期：2026-10-12
- 状态：设计已与负责人逐模块评审通过（本文为其定稿）
- 产品命名属商务决策，本文统一以「平台」指代

## 1. 背景与目标

教育采购项目（交互媒体制作，标书序号 12）。交付物 = **本地化全栈 AIGC 生成系统（软件平台）+ 双形态硬件终端**。核心约束：

- **双架构**：同一套系统须运行于 Windows/NVIDIA CUDA 与 Apple Silicon/macOS（Metal/MPS）；
- **全离线**：部署、模型加载、文生图、图生视频全流程不依赖外网；
- **教学级多用户**：学生 / 教师 / 管理员三角色，含配额、队列与管理后台；
- **验收驱动**：标书 12 条技术参数逐条对应可演示 / 可截图物料（见 §4.9）。

目标硬件：

| 设备 | 配置 |
|---|---|
| Lenovo ThinkStation P2 图形工作站 | i7-14700K（20C/28T）、RTX 3060 12GB、32GB DDR4、2TB NVMe SSD、Windows |
| MacBook（Apple M5） | 统一内存以实际到货为准（16–24GB）、macOS |

**标书疑点记录**（提交商务侧书面澄清，技术侧不受阻）：

1. 标书写「i7-14700K，14 核 20 线程」。实际 14700K 为 20 核 28 线程；14C/20T 是 i5-14600K 规格，系笔误。按 14700K 实机交付（20C/28T ≥ 14C/20T，正偏离）。
2. 标书要求双端运行截图；应答文本写「Mac M4」，实际到货为 M5。对外物料统一措辞「Apple Silicon」并与实物一致。

## 2. 核心决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | ComfyUI（pin 版本，headless）作为唯一生成执行引擎 | Wan / HunyuanVideo / LTX / Flux / Qwen-Image 一线支持；低显存能力原生；新模型跟进快；自研重写五模型族 × 双架构无收益 |
| D2 | 旁挂控制面，**不打 ComfyUI 内核补丁**；深层数据经自研 custom node 获取 | 上游可持续跟版（对应要求 11）；规避 GPL 传染面；custom node 面窄、随版本 pin |
| D3 | 控制面队列是唯一队列，ComfyUI 作**单槽执行器**（一次一图） | 绕开其单用户 / 无鉴权 / 队列不识人的设计；配额、公平调度、取消语义在控制面天然成立 |
| D4 | 所有 GPU 负载（生成 / TTS / Agent LLM）一律经执行面串行执行 | 显存治理单点成立；控制面自身保持 CPU-only |
| D5 | 引入**任务 DSL** 中间层；Agent LLM 不直接生成 ComfyUI 图 JSON | 隔离上游节点 schema 漂移；模板可独立测试 |
| D6 | 许可证策略：控制面闭源；custom node 遵循 GPL 兼容发布；不修改 ComfyUI 核心 | 本标书无知识产权硬性要求（已确认）；进程隔离 + HTTP 通信属「独立程序」形态，控制面未来如需报软著有清晰边界 |

## 3. 总体架构

```
┌────────────────────────────────────────────────────────────┐
│ 前端（Vue3 + Element Plus，全部自研）                          │
│ 学生端（表单/模板/画廊/进度） 教师端（+画布+模板管理+学情）        │
│ 管理端（用户/配额/模型/系统监控/授权/验收面板）                  │
├────────────────────────────────────────────────────────────┤
│ 控制面 fastapi（产品本体，CPU-only，长驻）                      │
│ 认证/RBAC │ 任务队列+配额调度器 │ 模型注册表 │ 资产库(按用户隔离)  │
│ Agent 编排器 │ ffmpeg 合成 worker │ 授权(加密狗) │ 能力矩阵       │
│ SQLite(WAL) │ WebSocket 进度推送                               │
├────────────────────────────────────────────────────────────┤
│ 执行面：ComfyUI 子进程（127.0.0.1:8188，仅本机，headless）      │
│   Windows/CUDA 与 macOS/MPS 同一份代码，启动参数不同             │
│   + vendored custom nodes（显存上报/TTS/LLM/视频合成辅助）       │
└────────────────────────────────────────────────────────────┘
```

进程视图（单机，两个长驻进程）：

- **控制面** `fastapi`：含前端静态资源、调度器、Agent 编排器、ffmpeg 合成 worker（CPU 异步任务）、SQLite。
- **执行面** ComfyUI 子进程：仅监听 `127.0.0.1:8188`，由控制面拉起、健康监控、按需重启（见 §4.3 显存 Profile 切换）。对外零暴露。

不变量：**GPU 只在执行面出现；一切 GPU 计算过单槽队列串行**；控制面崩溃不影响正在运行的任务完成落盘（执行面独立跑完当前图，控制面恢复后收账）。

目录布局：

```
install/
  app/        控制面 + 前端构建产物
  engine/     pin 版本 ComfyUI + vendored custom nodes + Python 运行时
  ffmpeg/
data/
  models/     ComfyUI 标准模型目录（checkpoint/diffusion_models/vae/lora/...）
  users/{uid}/outputs/{job_id}/    产物按用户隔离
  db/app.sqlite（WAL）
  logs/
```

## 4. 模块设计

### 4.1 认证与多用户（要求 8）

- 角色：`student` / `teacher` / `admin`。本地账号库（argon2 哈希），session token 认证；单机部署，不做外接 LDAP/SSO。
- 会话隔离：任务、产物、提示词历史、画廊全部按 uid 隔离；学生只见自己的内容；教师可查看全部学生的队列与产物（单机教学场景，不引入班级分组模型，批量导入 CSV 中的班级字段仅作备注）；管理员见系统全局。
- 管理后台：用户增删启停、批量导入（班级名单 CSV）、配额档位管理、队列管控（全局暂停/恢复，用于课堂控制）。
- 验收演示：两个学生账号同时提交任务 → 双方各自看到自己在队列中的位置与进度（要求 3+8 联合演示）。

### 4.2 任务生命周期与队列（要求 3）

**任务 DSL**（JSON，声明式，schema 版本化）：

```json
{
  "schema": 1,
  "name": "科普短视频-示例",
  "steps": [
    {"capability": "text-to-image", "template": "flux-standard",
     "params": {"prompt": "...", "aspect": "16:9"}},
    {"capability": "image-to-video", "template": "wan-480p",
     "params": {"motion": "...", "frames": 81, "use_prev": "image"}},
    {"capability": "tts", "template": "cosyvoice2-zh",
     "params": {"text": "...", "voice": "teacher-female"}},
    {"capability": "compose", "params": {"mux_audio": true, "subtitles": true}}
  ]
}
```

步骤间数据经 `use_prev` 显式引用，编排器负责传递中间产物。

**状态机**：`QUEUED → COMPILING → RUNNING(step k/n) → POST(ffmpeg 合成) → DONE | FAILED | CANCELLED`

- 提交：表单 / 模板 / Agent 产出 → DSL 校验（params 对模板 schema）→ 配额检查 → 入全局队列。
- 调度：**按用户轮转**（round-robin，单用户同刻至多占 1 个运行位，防止单人刷满）；教师优先级高于学生（可配置）。
- 执行：调度器将队头 DSL 编译为 ComfyUI 图（模板 = 已验证的图 + 参数注入），单槽投递 `/prompt`；`/ws` 进度映射为 step k/n 与百分比。
- 取消：排队任务 = 纯队列移除；运行中任务 = 调 `/interrupt`，产物按已完成 step 保留。
- 断点续收：控制面重启后从执行面 `/history` 对账，恢复任务账本。

**连续提交 ≥3 个任务自动排队**（要求 3 原文）由队列语义天然满足，验收以录屏呈现。

### 4.3 显存治理（要求 2、10）

- **显存 Profile**：`6GB / 8GB / 12GB / AUTO`，管理台可切。Profile 决定：模板的量化档位选择（fp8 / GGUF Q4-Q8）、block-swap 档位、VAE tiling 参数、分辨率上限。
- **切换语义**：Profile 变更 → 队列排空后**重启执行面子进程**（控制面不重启，服务在线——满足「切模型/切档不重启服务」的要求表述）；AUTO 档按当前卡实测显存自动选模板档。
- **实时上报**：自研 custom node 每 tick 上报（CUDA：`torch.cuda.mem_get_info`；MPS：`recommendedMaxWorkingSetSize`），进管理台「运行状态」页，即验收所需显存占用截图的正式来源。
- **6GB 复现机制**：6GB 档以 LTX-Video（轻量视频模型）+ GGUF 量化为演示对象，执行面以受限分配器启动真实复现「6GB 显存运行基础视频模型」，截图来自实际运行界面，非人工修饰。
- **Agent LLM 显存纪律**：Qwen3-4B 量化模型**按需加载、任务间卸载**（D4 之下由调度器统一编排）。GPU 推理走执行面单槽；**CPU 回退（低显存档位 / Mac 端）在控制面侧 worker 运行，不占 GPU 单槽**（14700K 20 线程可承载，Agent 非交互式，延迟可接受）。
- LoRA 轻量训练（要求 10 的 16GB 表述）：单机 12GB 档仅支持小分辨率 LoRA 微调演示（Flux Lite / SDXL 类 + 显存优化），文档明确标注能力边界，不做超卖承诺。

### 4.4 双平台适配（要求 4）

- **能力矩阵**（数据文件，非硬编码）：每模型 × 每平台标注 `full / degraded / unavailable`。降级模板 = 降分辨率、减步数、换量化格式。初始值为工程判断，**由夜班冒烟实测校准**（§4.9）；Mac 上标记 `unavailable` 的能力在 UI 隐藏。
- **MPS 技术路线**：fp8 计算在 MPS 不可用 → 走 GGUF 反量化 + fp16/bf16 计算；FlashAttention 类算子缺失 → PyTorch SDPA 回退；长视频 OOM → 分段解码。全部封装在模板层，用户无感。
- **性能达标证据**：标书要求「不低于 Apple 官方 PyTorch MPS 后端」——执行面底层即 PyTorch MPS，天然满足；另做**首次安装自检 benchmark**（每能力一个标准任务，双端跑，结果存档）作为书面对照证据。
- 控制面、前端、调度、DSL、模板体系两端**完全同构**，差异全部收敛在能力矩阵与执行面启动参数。

### 4.5 模型管理（要求 1、7、11）

- **模型包格式**：`zip/tar + manifest.json`（文件清单、SHA256、模型族/变体/量化档、最低显存、适用平台、版本）。离线分发，U 盘/内网拷贝导入。
- **一键导入**：解包 → 校验 → 落 `data/models/` → 注册表登记 → 能力矩阵刷新，管理台显示进度与结果。
- **注册表视图**：模型列表、版本、空间占用、适用能力、启停；LoRA 统一登记，模板参数中可勾选叠加。
- **升级机制（要求 11）**：平台版本记录页（版本号、变更日志、兼容性说明：模型包/模板/引擎三者版本兼容矩阵）；新模型 = 导入新模型包 +（必要时）随平台发布更新 vendored custom nodes（pin 版本），教学界面与流程零变化。

### 4.6 Agent 编排（要求 9）

- **本地 LLM**：Qwen3-4B-Instruct 量化（GGUF），经执行面单槽按需加载（D4）；输入 = 用户自然语言 + 能力目录（few-shot 示例含「5 秒科普短视频」），输出 = 任务 DSL 草稿。
- **可视化确认界面**：学生可查看 / 修改 Agent 拆解出的 DSL（分镜、提示词、参数）再执行——教学上呈现「拆解 → 确认 → 执行」过程，也是验收演示看点。
- **模板链**：「生成一段 5 秒科普短视频，含画面、旁白和风格描述」→ 文生图 → 图生视频 → CosyVoice2 中文旁白 → ffmpeg 合成（音视频混流 + 字幕）。模板链本身预置，Agent 只负责填参，保证演示稳定。
- TTS 选型 CosyVoice2（Apache-2.0，中文效果好，显存可控）；模型与代码均入离线包。

### 4.7 授权与加密狗（要求 12）

- 加密狗选国内主流厂商（深思数盾类，具体型号商务定），SDK 集成于控制面。
- **校验链**：启动校验（狗在位 + 设备指纹绑定 + 期限有效）→ 运行期心跳复验（分钟级）。
- **优雅降级**：校验失败 → 拒绝新任务提交、管理台横幅告警；**正在运行的任务允许完成**（教学场景不黑屏打断学生）。
- 期限管理在狗内；到期行为同上。防复制：软件与设备指纹 + 狗双因子。

### 4.8 离线交付（要求 5）

- **Windows**：便携包（自带 Python 运行时 + engine + app）+ `启动.bat`；**macOS**：脚本包/.app 壳 + 同结构 engine。
- 程序包与模型包分离分发（模型包大文件单独拷贝导入）。
- 首次启动自检：GPU/驱动/Metal、依赖完整性、模型包完整性、能力矩阵生成；全流程零外网依赖（含字体、ffmpeg、Python 环境）。
- 断网演示流程标准化：拔网线 → 重启服务 → 模型加载 → 文生图 → 图生视频，录屏为验收物料。

### 4.9 测试与验收物料（要求 2–9、11）

**夜班自动化冒烟**（两台真机各跑，cron 调管理 API）：

- 范围：每模板 × 每平台，最低步数各一任务；断言产物存在；采集墙钟时间、显存峰值入库。
- 产出：趋势图进「运行状态」页；失败即管理台告警。防上游 ComfyUI 升级悄悄破坏模板；沉淀的数据同时充当双端性能对照证据（§4.4）。

**验收物料映射**（每条要求 → 物料，管理台「运行状态 / 系统信息」页即为截图正式来源）：

| 要求 | 物料 |
|---|---|
| 1 全栈引擎 | 平台总览演示 + 各能力界面录屏 |
| 2 低显存 | 6GB/8GB 档运行截图、显存占用面板截图、任务队列面板截图、模型切换截图 |
| 3 队列/动态加载 | 连提 3 任务自动排队录屏；切模型不重启服务录屏 |
| 4 双架构 | Windows/NVIDIA 与 Mac 两端界面截图 + 自检 benchmark 对照报告 |
| 5 离线 | 断网全流程录屏 + 安装包/模型包实物 |
| 6 工作流 | 文字→图→视频自动流程录屏（模板链） |
| 7 模型生态 | 模型列表/空间占用/一键导入/版本管理界面截图 |
| 8 多用户 | 双学生账号并发提交录屏 + 管理后台截图 |
| 9 Agent | 「5 秒科普短视频」现场演示（拆解确认→执行） |
| 10 硬件约束 | 部署文档 + 单机部署确认 |
| 11 持续升级 | 平台版本记录页截图 + 兼容性说明文档 |
| 12 授权 | 加密狗演示：拔狗→拒新任务、运行中任务正常完成录屏 |

## 5. 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 控制面 | Python 3.11 + FastAPI + uvicorn | 单进程；WebSocket 进度 |
| 存储 | SQLite（WAL）+ 文件系统 | 单机规模，明确不上 PG |
| 前端 | Vue3 + Element Plus + Vite | 管理端组件生态成熟 |
| 执行面 | ComfyUI（pin 版本）+ vendored custom nodes | 核心原生支持目标模型族 |
| 量化 | ComfyUI-GGUF 等社区节点（pin） | CUDA/MPS 双端可用 |
| LLM | Qwen3-4B GGUF（llama.cpp 系节点/运行时） | Agent 拆解 |
| TTS | CosyVoice2 | Apache-2.0 |
| 音视频合成 | ffmpeg（内置分发） | 混流、字幕 |
| 授权 | 加密狗厂商 SDK | 商务定型号 |

## 6. 数据模型（核心表）

```
users(id, name, role, quota_profile_id, status, created_at)
quota_profiles(id, max_jobs_per_day, gpu_minutes_per_day, priority, ...)
jobs(id, user_id, dsl_json, status, submit_at, start_at, finish_at,
     gpu_ms, vram_peak_mb, error, outputs_json)
templates(id, capability, name, graph_ref, params_schema, min_vram_mb,
          platforms, version, origin[builtin|teacher], visibility)
model_registry(id, family, variant, quant, path, size_bytes, sha256,
               imported_at, enabled)
model_packages(id, name, version, manifest_json, imported_at, status)
smoke_runs(id, template_id, platform, ts, ok, wall_s, vram_peak_mb, note)
license_events(ts, kind, ok)
platform_caps(model_family, platform, level)      -- 能力矩阵
settings(key, value)
```

## 7. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| MPS 算子缺失/慢（视频模型尤甚） | 高 | 能力矩阵降级模板；GGUF+fp16 路线；夜班实测校准；Mac 演示脚本控制在短时长任务 |
| 上游 ComfyUI 版本漂移破坏模板 | 高 | pin + vendored；升级走分支测试 + 夜班冒烟全绿再发版 |
| Agent LLM 与生成任务抢显存 | 中 | D4 单槽串行 + 按需加载卸载；低档/CPU 回退 |
| GPL-3.0 传染 | 低（已确认无产权硬性要求） | 不改内核；custom node GPL 兼容；控制面独立进程通信；未来报软著以控制面为界 |
| 标书 CPU 参数笔误 / M4-M5 措辞 | 低 | §1 疑点记录，商务书面澄清；对外物料统一「Apple Silicon」 |
| 加密狗 SDK 平台差异（Win/macOS） | 中 | 选型时验证双端 SDK；Mac 端备选软授权（设备指纹）兜底 |

## 8. 里程碑

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| M1 产品核 | 控制面骨架 + 单槽执行 + 文生图表单 + 队列/进度 | 两端各跑通「提交→生成→落盘→画廊」 |
| M2 能力全家桶 | 图生视频/TTS/合成模板链 + 模型包管理 + 显存 Profile + 运行状态页 | 要求 1/2/3/6/7 物料可产出 |
| M3 多用户与打包 | 账号/配额/管理后台 + 双平台离线安装包 + 首装自检 benchmark | 要求 4/5/8 物料可产出；全新机器断网装机即用 |
| M4 Agent 与授权 | Agent 拆解+确认界面 + 加密狗 + 夜班冒烟全量 + 验收演练 | 要求 9/11/12 物料可产出；全 12 条物料清单齐 |

人力宽裕的用法：M2/M3 并行推进；Agent 的 LLM 集成自 M3 提前启动；夜班冒烟自 M2 起持续运行积累数据。

## 9. 明确不做（YAGNI）

- 不上 PostgreSQL / 消息队列 / 微服务拆分——单机两进程结构即终态；
- 不做实时协同编辑、云端同步、移动端；
- 不做外接 SSO/LDAP——本地账号体系即终态；
- 不自研生成 pipeline（全部经 ComfyUI）；
- 不在 12GB 单机上承诺大分辨率视频或全量 LoRA 训练（能力边界如实标注）。
