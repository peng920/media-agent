# M1 产品核打样 实施计划（FastAPI 控制面 + ComfyUI attach）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打样 M1 产品核——FastAPI 控制面（认证、单槽队列、任务 DSL 编译、进度推送、产物隔离落盘）驱动本机已在运行的 ComfyUI（attach 模式），跑通「学生登录 → 提交文生图 → 排队 → 生成 → 画廊取图」全流程。

**Architecture:** 控制面是唯一队列（D3），ComfyUI 作单槽执行器，经 HTTP/WS API 通信；所有图由模板编译器从参数生成（D5），模板代码内置经 2026-10-12 真机校准的节点图；产物经引擎 `/view` 接口下载到 `data/users/{uid}/outputs/{job_id}/`（attach 模式不碰引擎文件系统）。引擎为外部进程（pid 由用户管理），控制面不负责拉起。

**Tech Stack:** Python 3.12 / FastAPI / SQLModel(SQLite WAL) / httpx / websockets / pytest(+pytest-asyncio)；前端 React18 + Vite + TS + Ant Design（任务 8-9）。

## Global Constraints

- 引擎地址 `http://127.0.0.1:8188`，attach 模式，不启动/不重启引擎进程
- 单槽执行：同一时刻最多 1 个任务投递给引擎；其余排队（FIFO）
- 用户隔离：任务与产物按 uid 存储；学生只见自己的任务
- 模板 `qwen-image-2.1-turbo` 依赖模型文件（2026-10-12 校准结论）：
  - `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors`（在盘）
  - `text_encoders/qwen3vl_8b_int8_convrot.safetensors`（**待用户手动下载**，盘上 4B 版维度 2560 ≠ 模型要求 4096）
  - `vae/qwen_image_2.1_vae_bf16.safetensors`（在盘）
  - `loras/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors`（在盘）
- 产物下载走引擎 `/view?filename=&subfolder=&type=output`，不读引擎磁盘
- 测试默认离线（Fake 引擎）；真机测试标记 `@pytest.mark.live`，不在 CI 自动跑
- 提交信息用中文前缀 `feat:/test:/chore:`，一次任务一提交

---

### Task 1: 后端脚手架（配置/DB/数据模型/种子用户）

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/models.py`
- Create: `backend/app/db.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Test: `backend/tests/test_db_seed.py`

**Interfaces:**
- Produces: `Settings`（字段 `engine_base_url:str="http://127.0.0.1:8188"`、`data_dir:Path=Path("data")`、`db_path:Path`（默认 `data/db/app.sqlite`）、`token_secret:str="dev-secret"`、`seed_users:bool=True`）；`get_settings()`（lru_cache 单例）
- Produces: 表 `User(id:int pk, username:str unique, password_hash:str, role:str)`；表 `Job(id:int pk, user_id:int fk, template_id:str, params:json(str), status:str, prompt_id:str|None, error:str|None, progress:float(0-1), outputs:json(str|None), created_at:datetime, started_at, finished_at)`；状态常量 `JobStatus`（QUEUED/RUNNING/DONE/FAILED/CANCELLED）
- Produces: `init_db(settings)`（建表+种子用户 admin/admin123、student1/stu123、student2/stu123，argon2 太重、用 stdlib pbkdf2，见 Task 2 的 `hash_password`）；`get_session(settings)` 生成器

- [ ] **Step 1: 初始化 uv 项目**

```bash
cd /data/github/zimage21/backend && uv init --no-readme --python 3.12 . && uv add fastapi 'uvicorn[standard]' httpx websockets sqlmodel pydantic-settings && uv add --dev pytest pytest-asyncio
```

`backend/pyproject.toml` 追加（uv init 后 Edit）：

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["live: 需要本机 ComfyUI 运行中的真机测试"]
```

- [ ] **Step 2: 写 config.py**

```python
# backend/app/config.py
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZIMG_", env_file=".env", extra="ignore")
    engine_base_url: str = "http://127.0.0.1:8188"
    data_dir: Path = Path("data")
    token_secret: str = "dev-secret"
    seed_users: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "app.sqlite"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3: 写 models.py**

```python
# backend/app/models.py
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

class JobStatus:
    QUEUED = "QUEUED"; RUNNING = "RUNNING"; DONE = "DONE"
    FAILED = "FAILED"; CANCELLED = "CANCELLED"

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str  # student / teacher / admin

class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    template_id: str
    params: str            # JSON 编码的模板参数
    status: str = JobStatus.QUEUED
    prompt_id: str | None = None
    error: str | None = None
    progress: float = 0.0
    outputs: str | None = None   # JSON: [{"filename","subfolder","url"}]
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
```

- [ ] **Step 4: 写 db.py（含种子）**

```python
# backend/app/db.py
import hashlib, os, secrets
from collections.abc import Generator
from sqlmodel import SQLModel, Session, create_engine, select
from app.config import Settings
from app.models import User

def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return salt.hex() + "$" + dk.hex()

def verify_password(pw: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split("$", 1)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 200_000)
    return secrets.compare_digest(dk.hex(), dk_hex)

def init_db(settings: Settings) -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    if not settings.seed_users:
        return
    with Session(engine) as s:
        if s.exec(select(User).where(User.username == "admin")).first():
            return
        for name, pw, role in [("admin", "admin123", "admin"),
                               ("student1", "stu123", "student"),
                               ("student2", "stu123", "student")]:
            s.add(User(username=name, password_hash=hash_password(pw), role=role))
        s.commit()

def session_factory(settings: Settings):
    engine = create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})
    def _get() -> Generator[Session, None, None]:
        with Session(engine) as s:
            yield s
    return _get
```

- [ ] **Step 5: 写测试（conftest 提供 tmp 数据目录的 settings 与 session）**

```python
# backend/tests/conftest.py
import pytest
from app.config import Settings
from app import db as appdb

@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(data_dir=tmp_path, seed_users=True)
    appdb.init_db(s)
    return s

@pytest.fixture
def session(settings):
    yield from appdb.session_factory(settings)()
```

```python
# backend/tests/test_db_seed.py
from sqlmodel import select
from app.models import User

def test_seed_users(settings, session):
    users = session.exec(select(User)).all()
    assert {u.username for u in users} == {"admin", "student1", "student2"}
    assert all("$" in u.password_hash for u in users)

def test_password_roundtrip():
    from app.db import hash_password, verify_password
    h = hash_password("abc")
    assert verify_password("abc", h) and not verify_password("abd", h)
```

- [ ] **Step 6: 跑测试**

Run: `cd backend && uv run pytest tests/test_db_seed.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit** `git add backend && git commit -m "feat: 后端脚手架——配置/SQLite模型/种子用户"`

---

### Task 2: 认证（登录 / HMAC 令牌 / 依赖注入）

**Files:**
- Create: `backend/app/auth.py`
- Create: `backend/app/schemas.py`
- Test: `backend/tests/test_auth.py`

**Interfaces:**
- Consumes: `User`, `session`（Task 1）
- Produces: `issue_token(user)->str`；`parse_token(token)->tuple[int,str]|None`（uid, role，过期或坏签名返回 None，有效期 24h）；`get_current_user` FastAPI 依赖（读 `Authorization: Bearer <token>`，401 on 失败）；DTO：`LoginIn(username,password)`、`TokenOut(token,username,role)`

- [ ] **Step 1: 写 auth.py**

```python
# backend/app/auth.py
import base64, hashlib, hmac, time
from fastapi import Depends, HTTPException, Request
from sqlmodel import Session, select
from app.config import get_settings
from app.models import User

def _sign(msg: str) -> str:
    key = get_settings().token_secret.encode()
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()

def issue_token(user: User) -> str:
    exp = int(time.time()) + 86400
    body = f"{user.id}.{user.role}.{exp}"
    b = base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")
    return f"{b}.{_sign(b)}"

def parse_token(token: str) -> tuple[int, str] | None:
    try:
        b, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(_sign(b), sig):
            return None
        pad = "=" * (-len(b) % 4)
        uid, role, exp = base64.urlsafe_b64decode(b + pad).decode().split(".")
        if int(exp) < time.time():
            return None
        return int(uid), role
    except Exception:
        return None

def get_current_user(request: Request, session: Session) -> User:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    parsed = parse_token(token) if token else None
    if not parsed:
        raise HTTPException(401, "未登录或令牌失效")
    user = session.get(User, parsed[0])
    if not user:
        raise HTTPException(401, "用户不存在")
    return user
```

- [ ] **Step 2: 写 schemas.py（本任务只加认证部分）**

```python
# backend/app/schemas.py
from pydantic import BaseModel

class LoginIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    token: str
    username: str
    role: str
```

- [ ] **Step 3: 写测试**

```python
# backend/tests/test_auth.py
from app.auth import issue_token, parse_token
from app.models import User

def _user(uid=7, role="student"):
    return User(id=uid, username="x", password_hash="h$", role=role)

def test_token_roundtrip():
    t = issue_token(_user())
    assert parse_token(t) == (7, "student")

def test_bad_token():
    assert parse_token("7.student.99999.deadbeef") is None
    assert parse_token("") is None
```

- [ ] **Step 4: 跑测试** `uv run pytest tests/test_auth.py -v` → 2 passed
- [ ] **Step 5: Commit** `git commit -am "feat: HMAC令牌认证"`

---

### Task 3: 引擎客户端（ComfyUI HTTP API 封装）

**Files:**
- Create: `backend/app/engine/__init__.py`
- Create: `backend/app/engine/client.py`
- Test: `backend/tests/test_engine_client.py`（respx 或本地 fake http server；用 `httpx.MockTransport` 零依赖）

**Interfaces:**
- Produces: `EngineClient(base_url)`：异步方法 `health()->dict`（GET /system_stats）、`submit(graph:dict, client_id:str)->str`（POST /prompt，返回 prompt_id，400 时抛 `EngineError(detail)`）、`history(prompt_id)->dict`、`interrupt()->None`、`download(filename,subfolder)->bytes`（GET /view）、`model_options(node,class_field)->list[str]`（GET /object_info/{node}，取下拉项——用于模型清单校验）；`EngineError(Exception)`

- [ ] **Step 1: 写 client.py**

```python
# backend/app/engine/client.py
import httpx

class EngineError(Exception):
    pass

class EngineClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=60)

    async def _get(self, path: str) -> dict | bytes:
        r = await self._http.get(path)
        r.raise_for_status()
        return r.json() if "json" in r.headers.get("content-type", "") else r.content

    async def health(self) -> dict:
        return await self._get("/system_stats")

    async def submit(self, graph: dict, client_id: str) -> str:
        r = await self._http.post("/prompt", json={"client_id": client_id, "prompt": graph})
        if r.status_code != 200:
            raise EngineError(f"引擎拒单({r.status_code}): {r.text[:300]}")
        data = r.json()
        if "error" in data:
            raise EngineError(f"图校验失败: {data['error']} nodes={list(data.get('node_errors', {}).keys())}")
        return data["prompt_id"]

    async def history(self, prompt_id: str) -> dict:
        return (await self._get(f"/history/{prompt_id}")).get(prompt_id, {})

    async def interrupt(self) -> None:
        await self._http.post("/interrupt")

    async def download(self, filename: str, subfolder: str = "") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": "output"}
        r = await self._http.get("/view", params=params)
        r.raise_for_status()
        return r.content

    async def model_options(self, node: str, field: str) -> list[str]:
        info = (await self._get(f"/object_info/{node}"))[node]
        return info["input"]["required"][field][0]
```

- [ ] **Step 2: 写测试（httpx.MockTransport 打桩引擎）**

```python
# backend/tests/test_engine_client.py
import httpx, pytest
from app.engine.client import EngineClient, EngineError

def _client(handler) -> EngineClient:
    ec = EngineClient("http://test")
    ec._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ec

async def test_submit_ok():
    async def h(request):
        return httpx.Response(200, json={"prompt_id": "pid-1"})
    ec = _client(h)
    assert await ec.submit({"1": {}}, "c1") == "pid-1"

async def test_submit_validation_error():
    async def h(request):
        return httpx.Response(400, json={"error": "bad", "node_errors": {"7": "x"}})
    ec = _client(h)
    with pytest.raises(EngineError):
        await ec.submit({}, "c1")

async def test_history_empty_for_unknown():
    async def h(request):
        return httpx.Response(200, json={})
    ec = _client(h)
    assert await ec.history("nope") == {}
```

- [ ] **Step 3: 跑测试** `uv run pytest tests/test_engine_client.py -v` → 3 passed
- [ ] **Step 4: Commit** `git commit -am "feat: 引擎HTTP客户端"`

---

### Task 4: 模板编译器（校准图 + 模型清单校验）

**Files:**
- Create: `backend/app/compiler/__init__.py`
- Create: `backend/app/compiler/templates.py`
- Test: `backend/tests/test_compiler.py`

**Interfaces:**
- Consumes: `EngineClient.model_options`（Task 3）
- Produces: `Template` 协议：`id/name/capability/params_model(pydantic)/required_models: dict[node,dict[field,expected_file]]`、`build(params)->dict`（API 格式图）；`TEMPLATES: dict[str, Template]`；`validate_models(engine)->list[str]`（返回缺失文件清单，空=齐备）
- Produces: `QwenT2IParams(BaseModel)`：`prompt:str`、`width:int=1024`、`height:int=1024`、`steps:int=6`、`seed:int|None=None`（None→随机）、`use_lora:bool=True`

- [ ] **Step 1: 写 templates.py（图结构 = 2026-10-12 真机校准通过的那张，仅文本编码器待换 8B 文件）**

```python
# backend/app/compiler/templates.py
import random
from pydantic import BaseModel, Field

class QwenT2IParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    steps: int = Field(default=6, ge=1, le=50)
    seed: int | None = None
    use_lora: bool = True

UNET = "qwen_image_2.1_turbo_int8_convrot.safetensors"
TE   = "qwen3vl_8b_int8_convrot.safetensors"   # 待下载（4B 版维度不匹配，见计划头部）
VAE  = "qwen_image_2.1_vae_bf16.safetensors"
LORA = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"

def build_qwen_t2i(p: QwenT2IParams) -> dict:
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    model_src = ["1", 0]
    g: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": None, "positive": ["5", 0], "negative": ["5", 0],
              "latent_image": ["6", 0], "seed": seed, "steps": p.steps, "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "zimg"}},
    }
    if p.use_lora:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": 1.0}}
        model_src = ["4", 0]
    g["7"]["inputs"]["model"] = model_src
    return g

class Template:
    def __init__(self, id, name, capability, params_model, builder, required_models):
        self.id, self.name, self.capability = id, name, capability
        self.params_model, self.builder, self.required_models = params_model, builder, required_models

QWEN_T2I = Template(
    id="qwen-image-2.1-turbo", name="Qwen-Image 2.1 Turbo 文生图", capability="text-to-image",
    params_model=QwenT2IParams, builder=build_qwen_t2i,
    required_models={
        "UNETLoader": {"unet_name": UNET},
        "CLIPLoader": {"clip_name": TE},
        "VAELoader": {"vae_name": VAE},
        "LoraLoaderModelOnly": {"lora_name": LORA},
    })

TEMPLATES: dict[str, Template] = {QWEN_T2I.id: QWEN_T2I}

async def validate_models(engine) -> list[str]:
    """对照引擎各加载节点的下拉清单，返回缺失的模型文件名列表。"""
    missing = []
    for node, fields in QWEN_T2I.required_models.items():
        for field, expected in fields.items():
            try:
                opts = await engine.model_options(node, field)
            except Exception as e:
                missing.append(f"{expected}（无法查询引擎: {e}）"); continue
            if expected not in opts:
                missing.append(expected)
    return missing
```

- [ ] **Step 2: 写测试**

```python
# backend/tests/test_compiler.py
import pytest
from app.compiler.templates import TEMPLATES, QwenT2IParams, build_qwen_t2i, validate_models

def test_build_with_lora():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat"))
    assert g["7"]["inputs"]["model"] == ["4", 0]
    assert g["4"]["class_type"] == "LoraLoaderModelOnly"
    assert g["9"]["class_type"] == "SaveImage"

def test_build_without_lora():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat", use_lora=False))
    assert "4" not in g and g["7"]["inputs"]["model"] == ["1", 0]

def test_param_validation():
    with pytest.raises(Exception):
        QwenT2IParams(prompt="x", width=100)  # 非32倍数

class FakeEngine:
    def __init__(self, opts): self.opts = opts
    async def model_options(self, node, field): return self.opts[(node, field)]

async def test_validate_models_reports_missing():
    ok = FakeEngine({("UNETLoader", "unet_name"): ["qwen_image_2.1_turbo_int8_convrot.safetensors"]})
    miss = await validate_models(ok)
    assert miss and any("qwen3vl_8b" in m for m in miss)
```

- [ ] **Step 3: 跑测试** `uv run pytest tests/test_compiler.py -v` → 4 passed
- [ ] **Step 4: Commit** `git commit -am "feat: 文生图模板编译器与模型清单校验"`

---

### Task 5: 单槽调度器（生命周期 + 产物落盘 + 事件通知）

**Files:**
- Create: `backend/app/scheduler.py`
- Create: `backend/app/events.py`
- Test: `backend/tests/test_scheduler.py`

**Interfaces:**
- Consumes: `EngineClient`（Task 3）、`Template.build`（Task 4）、`session_factory`/`JobStatus`（Task 1）
- Produces: `Notifier`（`publish(event:dict)`、`subscribe()->asyncio.Queue`、`unsubscribe(q)`）；`Scheduler(engine, session_factory, settings, notifier)`：`async run_forever()`（启动为后台任务）、`request_cancel(job_id)`（QUEUED→直接 CANCELLED；RUNNING→`engine.interrupt()` 置取消标志）、`enqueue(job_id)`（唤醒）

- [ ] **Step 1: 写 events.py**

```python
# backend/app/events.py
import asyncio

class Notifier:
    def __init__(self):
        self._subs: set[asyncio.Queue] = set()

    async def publish(self, event: dict) -> None:
        for q in list(self._subs):
            q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)
```

- [ ] **Step 2: 写 scheduler.py**

```python
# backend/app/scheduler.py
import asyncio, json, uuid
from pathlib import Path
from sqlmodel import Session, select
from app.config import Settings
from app.engine.client import EngineClient
from app.compiler.templates import TEMPLATES
from app.events import Notifier
from app.models import Job, JobStatus, utcnow

class Scheduler:
    def __init__(self, engine: EngineClient, session_factory, settings: Settings, notifier: Notifier):
        self.engine, self.sf, self.settings, self.notifier = engine, session_factory, settings, notifier
        self._wakeup = asyncio.Event()
        self._cancel_requested: set[int] = set()
        self._current: int | None = None
        self._progress_cb = None  # 可选：引擎ws进度回调 fn(job_id, value, max)

    def enqueue(self, job_id: int) -> None:
        self._wakeup.set()

    def request_cancel(self, job_id: int) -> None:
        self._cancel_requested.add(job_id)
        self._wakeup.set()

    async def run_forever(self) -> None:
        while True:
            self._wakeup.clear()
            with next(self.sf()) as s:
                job = s.exec(select(Job).where(Job.status == JobStatus.QUEUED)
                             .order_by(Job.created_at, Job.id)).first()
            if not job:
                await self._wakeup.wait(); continue
            await self._execute(job.id)
            await asyncio.sleep(0.1)  # 让 DB 状态落稳

    def _patch(self, job_id: int, **kw) -> Job:
        with next(self.sf()) as s:
            job = s.get(Job, job_id)
            for k, v in kw.items(): setattr(job, k, v)
            s.add(job); s.commit(); s.refresh(job)
            return job

    async def _notify(self, job: Job) -> None:
        await self.notifier.publish({"job_id": job.id, "user_id": job.user_id,
                                     "status": job.status, "progress": job.progress,
                                     "error": job.error, "outputs": json.loads(job.outputs or "[]")})

    async def _execute(self, job_id: int) -> None:
        with next(self.sf()) as s:
            job = s.get(Job, job_id)
            tpl = TEMPLATES[job.template_id]
            params = tpl.params_model(**json.loads(job.params))
        self._current = job_id
        try:
            graph = tpl.build(params)
            prompt_id = await self.engine.submit(graph, client_id=str(uuid.uuid4()))
            job = self._patch(job_id, status=JobStatus.RUNNING, prompt_id=prompt_id, started_at=utcnow())
            await self._notify(job)
            outputs = await self._await(prompt_id, job_id)
        except Exception as e:
            job = self._patch(job_id, status=JobStatus.FAILED, error=str(e)[:500], finished_at=utcnow())
            await self._notify(job); return
        finally:
            self._current = None

        if job_id in self._cancel_requested:
            self._cancel_requested.discard(job_id)
            job = self._patch(job_id, status=JobStatus.CANCELLED, finished_at=utcnow())
            await self._notify(job); return
        saved = self._save_outputs(job_id, outputs)
        job = self._patch(job_id, status=JobStatus.DONE, progress=1.0, outputs=json.dumps(saved), finished_at=utcnow())
        await self._notify(job)

    async def _await(self, prompt_id: str, job_id: int) -> list[dict]:
        """轮询 history 直到完成；运行中若收到取消则 interrupt。"""
        import time
        t0 = time.time()
        while time.time() - t0 < 3600:
            if job_id in self._cancel_requested and prompt_id:
                await self.engine.interrupt()
            h = await self.engine.history(prompt_id)
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m[1].get("exception_message", "") for m in st.get("messages", [])
                        if m[0] == "execution_error"]
                raise RuntimeError("引擎执行失败: " + "; ".join(msgs)[:300])
            if st.get("completed"):
                outs = []
                for node_out in h.get("outputs", {}).values():
                    outs.extend(node_out.get("images", []))
                return outs
            await asyncio.sleep(1.0)
        raise TimeoutError("任务超时(1h)")

    def _save_outputs(self, job_id: int, engine_outputs: list[dict]) -> list[dict]:
        """同步下载（引擎空闲才会走到这，阻塞事件循环短时可接受；打样从简）。"""
        import httpx
        out_dir = self.settings.data_dir / "users"
        saved = []
        with next(self.sf()) as s:
            uid = s.get(Job, job_id).user_id
        for i, img in enumerate(engine_outputs):
            fname = f"{job_id}_{i}_{img['filename']}"
            (out_dir / str(uid) / "outputs" / str(job_id)).mkdir(parents=True, exist_ok=True)
            (out_dir / str(uid) / "outputs" / str(job_id) / fname).write_bytes(
                httpx.get(f"{self.engine.base_url}/view",
                          params={"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                  "type": img.get("type", "output")}, timeout=120).content)
            saved.append({"filename": fname, "url": f"/files/{uid}/outputs/{job_id}/{fname}"})
        return saved
```

- [ ] **Step 3: 写测试（FakeEngine 完整生命周期）**

```python
# backend/tests/test_scheduler.py
import asyncio, json
import httpx
import pytest
from app.events import Notifier
from app.models import Job, JobStatus
from app.scheduler import Scheduler

class FakeEngine:
    def __init__(self):
        self.submitted, self.interrupted = [], False
        self._result = {"status": {"completed": True, "status_str": "success"},
                        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}}
    async def submit(self, graph, client_id):
        self.submitted.append(graph); return "pid-1"
    async def history(self, pid): return self._result
    async def interrupt(self): self.interrupted = True

@pytest.fixture
def make_job(settings, session):
    def _make(n=1):
        jobs = [Job(user_id=2, template_id="qwen-image-2.1-turbo",
                    params=json.dumps({"prompt": "cat"})) for _ in range(n)]
        for j in jobs: session.add(j)
        session.commit()
        return jobs if n > 1 else jobs[0]
    return _make

async def test_lifecycle_done(settings, monkeypatch):
    from app import db as appdb
    fake = FakeEngine()
    n = Notifier(); q = n.subscribe()
    sched = Scheduler(fake, appdb.session_factory(settings), settings, n)
    with next(appdb.session_factory(settings)()) as s:
        job = s.exec(__import__("sqlmodel").select(Job)).first() or None
    make = None
    with next(appdb.session_factory(settings)()) as s:
        from sqlmodel import select
        if not s.exec(select(Job)).first():
            j = Job(user_id=1, template_id="qwen-image-2.1-turbo", params='{"prompt":"cat"}')
            s.add(j); s.commit(); s.refresh(j); job = j
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, content=b"PNG"))
    task = asyncio.create_task(sched.run_forever())
    ev = await asyncio.wait_for(q.get(), 5)
    assert ev["status"] == "RUNNING"
    ev = await asyncio.wait_for(q.get(), 5)
    assert ev["status"] == "DONE" and ev["outputs"][0]["url"].startswith("/files/")
    task.cancel()
```

（此测试用独立 tmp 数据库、FakeEngine 与 monkeypatch 掉产物下载；验收点：RUNNING→DONE 两条事件、outputs 落盘路径正确。）

- [ ] **Step 4: 跑测试** `uv run pytest tests/test_scheduler.py -v` → 1 passed
- [ ] **Step 5: Commit** `git commit -am "feat: 单槽调度器与事件通知"`

---

### Task 6: API 路由 + WebSocket + 应用装配

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes_auth.py`
- Create: `backend/app/api/routes_jobs.py`
- Create: `backend/app/api/routes_misc.py`
- Create: `backend/app/main.py`
- Modify: `backend/app/schemas.py`（追加 Job DTO）
- Test: `backend/tests/test_api_jobs.py`

**Interfaces:**
- Consumes: 前五个任务的全部产物
- Produces: `POST /api/auth/login`→TokenOut；`GET /api/auth/me`；`GET /api/templates`（含 `missing_models`）；`POST /api/jobs`→JobOut（校验模板与参数，入队）；`GET /api/jobs`（本人任务，admin/teacher 见全部）；`GET /api/jobs/{id}`；`POST /api/jobs/{id}/cancel`；`WS /ws/jobs?token=`（推送本人任务事件）；`GET /api/health`（db+引擎）；静态：`/files/*` 挂 data 目录、`/` 挂前端构建产物（若存在）
- Produces: `create_app(settings=None)->FastAPI`（测试用），全局 `app = create_app()`；`app.state.scheduler/notifier/engine/settings/session_factory`

- [ ] **Step 1: schemas.py 追加**

```python
class JobOut(BaseModel):
    id: int; template_id: str; status: str; progress: float; error: str | None
    outputs: list[dict] = []; created_at: str; finished_at: str | None
class JobIn(BaseModel):
    template_id: str
    params: dict
class TemplateOut(BaseModel):
    id: str; name: str; capability: str; defaults: dict; missing_models: list[str]
```

- [ ] **Step 2: routes_auth.py**

```python
# backend/app/api/routes_auth.py
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.auth import issue_token
from app.db import verify_password
from app.models import User
from app.schemas import LoginIn, TokenOut

router = APIRouter(prefix="/api/auth")

@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, session: Session = Depends(...)):
    from app.main import get_session_dep
    user = session.exec(select(User).where(User.username == body.username)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "用户名或密码错误")
    t = issue_token(user)
    return TokenOut(token=t, username=user.username, role=user.role)
```

（实际实现把 session 依赖统一为 `app.api.deps` 里的 `get_session_dep`，此处省略重复定义。）

- [ ] **Step 3: routes_jobs.py**

```python
# backend/app/api/routes_jobs.py
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select
from app.api.deps import get_current_user, get_session_dep
from app.compiler.templates import TEMPLATES, validate_models
from app.models import Job, JobStatus, User
from app.schemas import JobIn, JobOut
from app.scheduler import Scheduler

router = APIRouter(prefix="/api/jobs")

def _out(j: Job) -> JobOut:
    return JobOut(id=j.id, template_id=j.template_id, status=j.status, progress=j.progress,
                  error=j.error, outputs=json.loads(j.outputs or "[]"),
                  created_at=j.created_at.isoformat(), finished_at=j.finished_at.isoformat() if j.finished_at else None)

@router.post("", response_model=JobOut)
def submit(body: JobIn, request: Request, user: User = Depends(get_current_user),
           session: Session = Depends(get_session_dep)):
    tpl = TEMPLATES.get(body.template_id) or HTTPException(404, "模板不存在")
    if isinstance(tpl, HTTPException): raise tpl
    try:
        tpl.params_model(**body.params)
    except Exception as e:
        raise HTTPException(422, f"参数不合法: {e}")
    job = Job(user_id=user.id, template_id=body.template_id, params=json.dumps(body.params, ensure_ascii=False))
    session.add(job); session.commit(); session.refresh(job)
    request.app.state.scheduler.enqueue(job.id)
    return _out(job)

@router.get("", response_model=list[JobOut])
def list_jobs(user: User = Depends(get_current_user), session: Session = Depends(get_session_dep)):
    q = select(Job).order_by(Job.id.desc()).limit(100)
    if user.role == "student":
        q = q.where(Job.user_id == user.id)
    return [_out(j) for j in session.exec(q)]

@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel(job_id: int, request: Request, user: User = Depends(get_current_user),
           session: Session = Depends(get_session_dep)):
    job = session.get(Job, job_id)
    if not job or (user.role == "student" and job.user_id != user.id):
        raise HTTPException(404)
    if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
        return _out(job)
    if job.status == JobStatus.QUEUED:
        job.status = JobStatus.CANCELLED; session.add(job); session.commit(); session.refresh(job)
    else:
        request.app.state.scheduler.request_cancel(job_id)
    return _out(job)
```

- [ ] **Step 4: routes_misc.py（templates/health/ws）**

```python
# backend/app/api/routes_misc.py
import asyncio, json
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from app.api.deps import get_session_dep
from app.auth import parse_token
from app.compiler.templates import TEMPLATES, validate_models
from app.schemas import TemplateOut

router = APIRouter(prefix="/api")

@router.get("/templates", response_model=list[TemplateOut])
async def templates(request: Request):
    engine = request.app.state.engine
    missing = await validate_models(engine)
    return [TemplateOut(id=t.id, name=t.name, capability=t.capability,
                        defaults=t.params_model().model_dump(), missing_models=missing)
            for t in TEMPLATES.values()]

@router.get("/health")
async def health(request: Request):
    db_ok, engine_ok, engine_ver = True, False, ""
    try:
        engine_ver = (await request.app.state.engine.health()).get("system", {}).get("comfyui_version", "")
        engine_ok = True
    except Exception:
        pass
    return {"db": db_ok, "engine": engine_ok, "engine_version": engine_ver}

async def ws_jobs(websocket: WebSocket):
    await websocket.accept()
    user = None
    token = websocket.query_params.get("token", "")
    parsed = parse_token(token) if token else None
    if not parsed:
        await websocket.send_json({"type": "error", "detail": "未认证"}); await websocket.close(); return
    uid, role = parsed
    q = websocket.app.state.notifier.subscribe()
    try:
        while True:
            ev = await q.get()
            if role != "student" or ev.get("user_id") == uid:
                await websocket.send_json(ev)
    except WebSocketDisconnect:
        pass
    finally:
        websocket.app.state.notifier.unsubscribe(q)
```

- [ ] **Step 5: main.py 装配（含 /ws/jobs 注册与静态目录）**

```python
# backend/app/main.py
import asyncio, contextlib
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from app.config import Settings, get_settings
from app import db as appdb
from app.engine.client import EngineClient
from app.events import Notifier
from app.scheduler import Scheduler
from app.api.routes_auth import router as r_auth
from app.api.routes_jobs import router as r_jobs
from app.api.routes_misc import router as r_misc, ws_jobs

def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="AIGC 教学创作平台")
    appdb.init_db(s)
    app.state.settings = s
    app.state.engine = EngineClient(s.engine_base_url)
    app.state.notifier = Notifier()
    app.state.session_factory = appdb.session_factory(s)
    app.state.scheduler = Scheduler(app.state.engine, app.state.session_factory, s, app.state.notifier)

    @app.on_event("startup")
    async def _start():
        app.state.sched_task = asyncio.create_task(app.state.scheduler.run_forever())

    @app.on_event("shutdown")
    async def _stop():
        app.state.sched_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.sched_task

    app.include_router(r_auth); app.include_router(r_jobs); app.include_router(r_misc)
    app.websocket("/ws/jobs")(ws_jobs)

    files_dir = s.data_dir
    files_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(files_dir)), name="files")
    web_dist = Path(__file__).parent.parent / "frontend_dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
    return app

app = create_app()
```

依赖注入 `app/api/deps.py`：`get_session_dep`（从 `request.app.state.session_factory` 取）与重新导出的 `get_current_user`——注意 `get_current_user` 需要 session，改成 `Depends(get_session_dep)` 版本（Task 2 的签名保持不变，内部改从依赖取 session）。

- [ ] **Step 6: API 集成测试（FastAPI TestClient + FakeEngine 替换 app.state.engine）**

```python
# backend/tests/test_api_jobs.py
import json
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings
from tests.test_scheduler import FakeEngine

@pytest.fixture
def client(settings):
    app = create_app(settings)
    app.state.engine = FakeEngine()
    # 重建 scheduler 使用 fake engine
    from app.scheduler import Scheduler
    app.state.scheduler = Scheduler(app.state.engine, app.state.session_factory, settings, app.state.notifier)
    with TestClient(app) as c:
        yield c

def _login(c, u="student1", p="stu123"):
    r = c.post("/api/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}

def test_login_and_me_flow(client):
    h = _login(client)
    assert client.get("/api/jobs", headers=h).status_code == 200

def test_submit_and_isolation(client):
    h1 = _login(client, "student1")
    h2 = _login(client, "student2")
    r = client.post("/api/jobs", headers=h1,
                    json={"template_id": "qwen-image-2.1-turbo", "params": {"prompt": "cat"}})
    assert r.status_code == 200 and r.json()["status"] == "QUEUED"
    mine = client.get("/api/jobs", headers=h1).json()
    other = client.get("/api/jobs", headers=h2).json()
    assert len(mine) == 1 and len(other) == 0  # 学生隔离

def test_submit_bad_params(client):
    h = _login(client)
    r = client.post("/api/jobs", headers=h,
                    json={"template_id": "qwen-image-2.1-turbo", "params": {"prompt": "x", "width": 100}})
    assert r.status_code == 422
```

- [ ] **Step 7: 跑全部测试** `uv run pytest -v`（排除 live）→ 全绿
- [ ] **Step 8: Commit** `git commit -am "feat: API/WS 路由与应用装配"`

---

### Task 7: 真机冒烟脚本（live，待 8B 编码器到位）

**Files:**
- Create: `backend/scripts/live_smoke.py`
- Create: `backend/tests/live/test_engine_live.py`

**Interfaces:**
- Consumes: 全部前序产物；前置条件 `qwen3vl_8b_int8_convrot.safetensors` 已放入引擎 text_encoders/

- [ ] **Step 1: live_smoke.py**：启动 `uvicorn app.main:app --port 8090` → 登录 student1 → `GET /api/templates` 断言 `missing_models==[]` → `POST /api/jobs`（512x512, 4 步）→ 轮询 `GET /api/jobs/{id}` 至 DONE → 下载 outputs[0].url 校验 PNG 头 → 打印耗时。
- [ ] **Step 2: live 标记测试**：`@pytest.mark.live` 包装同样的断言。
- [ ] **Step 3: 手工执行** `uv run python scripts/live_smoke.py`，Expected: `SMOKE OK <秒数>`；图出 `data/users/1/outputs/`。
- [ ] **Step 4: Commit** `git commit -am "test: 真机冒烟脚本"`

---

### Task 8: 前端骨架（Vite+React+TS+AntD + 登录页）

**Files:**
- Create: `frontend/`（vite 脚手架）
- Create: `frontend/src/api.ts`、`frontend/src/auth.ts`、`frontend/src/pages/Login.tsx`、`frontend/src/App.tsx`（改写）

- [ ] **Step 1:** `npm create vite@latest frontend -- --template react-ts && cd frontend && npm i antd && npm i -D @types/node`；`vite.config.ts` 加 `server.proxy['/api']` 与 `['/ws']`、`['/files']` → `http://127.0.0.1:8090`。
- [ ] **Step 2:** `src/api.ts`：fetch 封装（带 Bearer token，401 跳登录）+ `login/submitJob/listJobs/cancelJob/listTemplates/health`。
- [ ] **Step 3:** `src/auth.ts`：token/username/role 存 localStorage；`Login.tsx`：antd Form（用户名/密码/登录按钮），登录成功写 localStorage 并跳 `/`。
- [ ] **Step 4:** `App.tsx`：未登录重定向 `/login`；`react-router` 两个路由 `/login` 与 `/`（主页=Task 9 的创作页）。
- [ ] **Step 5:** `npm run build` 通过 → `git add frontend && git commit -m "feat: 前端骨架与登录页"`

---

### Task 9: 创作页 + 任务队列页（打样收口）

**Files:**
- Create: `frontend/src/pages/Create.tsx`、`frontend/src/pages/Jobs.tsx`、`frontend/src/ws.ts`

- [ ] **Step 1:** `Create.tsx`：模板下拉（`GET /api/templates`，缺模型时黄条提示 missing_models）+ 参数表单（prompt/宽/高/步数/seed/LoRA 开关，antd Form）→ 提交 → 跳转任务页。
- [ ] **Step 2:** `ws.ts`：`new WebSocket(\`ws://${location.host}/ws/jobs?token=${token}\`)`，断线 3s 重连。
- [ ] **Step 3:** `Jobs.tsx`：antd Table（id/模板/状态 Tag/进度条/时间/操作-取消/产物缩略图 `<Image>`）；WS 事件合入行状态；学生只见自己（后端已保证）。
- [ ] **Step 4:** 双账号手工验证：两个浏览器分别登 student1/student2，各提 2 个任务 → 观察 4 任务排队、单槽逐个执行、各自只见自己的。
- [ ] **Step 5:** `npm run build` 通过；FastAPI 挂载 `frontend_dist`（`npm run build --outDir ../backend/frontend_dist`）后 `uvicorn` 单进程演示。Commit `feat: 创作页与任务队列页`。

---

### Task 10: README 与收口

- [ ] `README.md`：开发启动（uvicorn 8090 / vite dev）、种子账号、引擎前置条件（attach 8188 + 模型清单 + 8B 编码器下载链接）、live 冒烟跑法。
- [ ] 全量测试 + 提交。设计文档 §4.2/§4.9 与本计划对齐性检查（DSL 多步链、显存 Profile 留 M2，不提前实现——YAGNI）。
