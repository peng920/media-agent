import asyncio
import json

import httpx
import pytest
from sqlmodel import select

from app import db as appdb
from app.events import Notifier
from app.models import Job, JobStatus
from app.scheduler import Scheduler
from tests.fakes import FakeEngine


def _make_job(settings, n=1):
    sf = appdb.session_factory(settings)
    with next(sf()) as s:
        jobs = [Job(user_id=1, template_id="qwen-image-2.1-turbo",
                    params=json.dumps({"prompt": "cat"})) for _ in range(n)]
        for j in jobs:
            s.add(j)
        s.commit()
        for j in jobs:
            s.refresh(j)
        return jobs if n > 1 else jobs[0]


@pytest.fixture
def patched_download(monkeypatch, tmp_path):
    def fake_get(url, params=None, timeout=None):
        return httpx.Response(200, content=b"\x89PNG-fake")
    monkeypatch.setattr(httpx, "get", fake_get)


async def test_lifecycle_done(settings, patched_download):
    fake = FakeEngine()
    notifier = Notifier()
    q = notifier.subscribe()
    sched = Scheduler(fake, appdb.session_factory(settings), settings, notifier)
    job = _make_job(settings)

    task = asyncio.create_task(sched.run_forever())
    try:
        ev1 = await asyncio.wait_for(q.get(), 5)
        assert ev1 == {"job_id": job.id, "user_id": 1, "status": "RUNNING",
                       "progress": 0.0, "error": None, "outputs": []}
        ev2 = await asyncio.wait_for(q.get(), 5)
        assert ev2["status"] == "DONE"
        assert ev2["outputs"][0]["url"] == f"/files/users/1/outputs/{job.id}/{job.id}_0_a.png"
        assert fake.submitted and fake.submitted[0]["9"]["class_type"] == "SaveImage"
    finally:
        task.cancel()

    sf = appdb.session_factory(settings)
    with next(sf()) as s:
        final = s.exec(select(Job)).first()
        assert final.status == JobStatus.DONE
        assert (settings.data_dir / "users" / "1" / "outputs" / str(job.id)).exists()


async def test_engine_failure_marks_failed(settings):
    from tests.fakes import FailingEngine
    fake = FailingEngine()
    notifier = Notifier()
    q = notifier.subscribe()
    sched = Scheduler(fake, appdb.session_factory(settings), settings, notifier)
    job = _make_job(settings)

    task = asyncio.create_task(sched.run_forever())
    try:
        ev = await asyncio.wait_for(q.get(), 5)
        assert ev["status"] == "FAILED"
        assert "engine down" in ev["error"]
    finally:
        task.cancel()
    sf = appdb.session_factory(settings)
    with next(sf()) as s:
        assert s.get(Job, job.id).status == JobStatus.FAILED


async def test_single_slot_serializes(settings, patched_download):
    """两个任务必须串行执行（单槽核心约束）。"""
    fake = FakeEngine()
    notifier = Notifier()
    q = notifier.subscribe()
    sched = Scheduler(fake, appdb.session_factory(settings), settings, notifier)
    _make_job(settings, n=2)

    task = asyncio.create_task(sched.run_forever())
    try:
        statuses = []
        for _ in range(4):  # 2 任务 × RUNNING/DONE
            ev = await asyncio.wait_for(q.get(), 5)
            statuses.append((ev["job_id"], ev["status"]))
        # RUNNING 出现时，前一个必已 DONE
        running_seen = set()
        for job_id, st in statuses:
            if st == "RUNNING":
                assert job_id not in running_seen
                running_seen.add(job_id)
            else:
                assert st == "DONE"
        assert len(running_seen) == 2
    finally:
        task.cancel()
