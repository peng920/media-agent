import asyncio
import json
import time
import uuid
from collections.abc import Generator

import httpx
from sqlmodel import Session, select

from app.compiler.templates import TEMPLATES
from app.config import Settings
from app.engine.client import EngineClient
from app.events import Notifier
from app.models import Job, JobStatus, utcnow


class Scheduler:
    """控制面唯一队列：单槽驱动引擎，全生命周期事件经 Notifier 广播。"""

    def __init__(self, engine: EngineClient, session_factory, settings: Settings, notifier: Notifier):
        self.engine = engine
        self.sf = session_factory  # () -> Generator[Session]
        self.settings = settings
        self.notifier = notifier
        self._wakeup = asyncio.Event()
        self._cancel_requested: set[int] = set()

    def enqueue(self, job_id: int) -> None:
        self._wakeup.set()

    def request_cancel(self, job_id: int) -> None:
        self._cancel_requested.add(job_id)
        self._wakeup.set()

    def _session(self) -> Generator[Session, None, None]:
        return self.sf()

    def _patch(self, job_id: int, **kw) -> Job:
        with next(self._session()) as s:
            job = s.get(Job, job_id)
            for k, v in kw.items():
                setattr(job, k, v)
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    async def _notify(self, job: Job) -> None:
        await self.notifier.publish(
            {"job_id": job.id, "user_id": job.user_id, "status": job.status,
             "progress": job.progress, "error": job.error,
             "outputs": json.loads(job.outputs or "[]")}
        )

    async def run_forever(self) -> None:
        while True:
            self._wakeup.clear()
            with next(self._session()) as s:
                job = s.exec(
                    select(Job).where(Job.status == JobStatus.QUEUED)
                    .order_by(Job.created_at, Job.id)
                ).first()
            if not job:
                await self._wakeup.wait()
                continue
            await self._execute(job.id)
            await asyncio.sleep(0.1)

    async def _execute(self, job_id: int) -> None:
        with next(self._session()) as s:
            job = s.get(Job, job_id)
            tpl = TEMPLATES[job.template_id]
            params = tpl.params_model(**json.loads(job.params))
        try:
            graph = tpl.build(params)
            prompt_id = await self.engine.submit(graph, client_id=str(uuid.uuid4()))
            job = self._patch(job_id, status=JobStatus.RUNNING, prompt_id=prompt_id,
                              started_at=utcnow(), progress=0.0)
            await self._notify(job)
            outputs = await self._await(prompt_id, job_id)

            if job_id in self._cancel_requested:
                self._cancel_requested.discard(job_id)
                job = self._patch(job_id, status=JobStatus.CANCELLED, finished_at=utcnow())
                await self._notify(job)
                return

            saved = self._save_outputs(job_id, outputs)
            job = self._patch(job_id, status=JobStatus.DONE, progress=1.0,
                              outputs=json.dumps(saved), finished_at=utcnow())
            await self._notify(job)
        except Exception as e:
            job = self._patch(job_id, status=JobStatus.FAILED, error=str(e)[:500],
                              finished_at=utcnow())
            await self._notify(job)

    async def _await(self, prompt_id: str, job_id: int) -> list[dict]:
        """轮询 history 直到完成/出错；取消请求则 interrupt。"""
        t0 = time.time()
        while time.time() - t0 < 3600:
            if job_id in self._cancel_requested:
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
        """经引擎 /view 下载产物到按用户隔离目录（单槽空闲期，同步下载可接受）。"""
        with next(self._session()) as s:
            uid = s.get(Job, job_id).user_id
        out_dir = self.settings.data_dir / "users" / str(uid) / "outputs" / str(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, img in enumerate(engine_outputs):
            fname = f"{job_id}_{i}_{img['filename']}"
            content = httpx.get(
                f"{self.engine.base_url}/view",
                params={"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output")},
                timeout=120,
            ).content
            (out_dir / fname).write_bytes(content)
            saved.append({"filename": fname, "url": f"/files/users/{uid}/outputs/{job_id}/{fname}"})
        return saved
