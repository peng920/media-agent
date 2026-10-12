import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app import db as appdb
from app.main import create_app
from app.models import Job
from app.scheduler import Scheduler
from tests.fakes import FakeEngine

import httpx

PNG = b"\x89PNG-fake"


@pytest.fixture
def client(settings, monkeypatch):
    app = create_app(settings)
    fake = FakeEngine()
    app.state.engine = fake
    app.state.scheduler = Scheduler(fake, app.state.session_factory, settings,
                                     app.state.notifier)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, content=PNG))
    with TestClient(app) as c:
        yield c


def _login(c, u="student1", p="stu123"):
    r = c.post("/api/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["engine"] is True


def test_templates_endpoint(client):
    r = client.get("/api/templates")
    assert r.status_code == 200
    data = r.json()
    assert data[0]["id"] == "qwen-image-2.1-turbo"
    assert data[0]["defaults"]["width"] == 1024
    assert data[0]["defaults"]["prompt"] is None


def test_login_me_and_unauthorized(client):
    h = _login(client)
    assert client.get("/api/auth/me", headers=h).json()["username"] == "student1"
    assert client.get("/api/jobs").status_code == 401


def test_submit_and_isolation(client, settings):
    h1 = _login(client, "student1")
    h2 = _login(client, "student2")
    r = client.post("/api/jobs", headers=h1,
                    json={"template_id": "qwen-image-2.1-turbo",
                          "params": {"prompt": "cat"}})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "QUEUED"

    import time
    deadline = time.time() + 5
    while time.time() < deadline:
        mine = client.get("/api/jobs", headers=h1).json()
        if mine and mine[0]["status"] == "DONE":
            break
        time.sleep(0.1)
    assert mine[0]["status"] == "DONE"
    assert mine[0]["outputs"][0]["url"].startswith("/files/")

    other = client.get("/api/jobs", headers=h2).json()
    assert other == []  # 学生隔离

    files = client.get(mine[0]["outputs"][0]["url"])
    assert files.status_code == 200 and files.content == PNG


def test_admin_sees_all(client):
    ha = _login(client, "admin", "admin123")
    _login(client, "student1")
    client.post("/api/jobs", headers=ha,
                json={"template_id": "qwen-image-2.1-turbo",
                      "params": {"prompt": "x"}})
    assert len(client.get("/api/jobs", headers=ha).json()) == 1


def test_submit_bad_params(client):
    h = _login(client)
    r = client.post("/api/jobs", headers=h,
                    json={"template_id": "qwen-image-2.1-turbo",
                          "params": {"prompt": "x", "width": 100}})
    assert r.status_code == 422
    r = client.post("/api/jobs", headers=h,
                    json={"template_id": "nope", "params": {}})
    assert r.status_code == 404


def test_cancel_queued(client):
    ha = _login(client, "admin", "admin123")
    # 提交两个：第一个会被立刻执行，第二个排队
    for _ in range(2):
        r = client.post("/api/jobs", headers=ha,
                        json={"template_id": "qwen-image-2.1-turbo",
                              "params": {"prompt": "x"}})
    second = client.get("/api/jobs", headers=ha).json()[0]["id"]  # desc 序，最新的是第二个
    r = client.post(f"/api/jobs/{second}/cancel", headers=ha)
    assert r.status_code == 200
    assert r.json()["status"] == "CANCELLED"
