import time


def _login(c, u="student1", p="stu123"):
    r = c.post("/api/auth/login", json={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["engine"] is True


def test_login_me_and_unauthorized(client):
    h = _login(client)
    assert client.get("/api/auth/me", headers=h).json()["username"] == "student1"
    assert client.get("/api/jobs").status_code == 401


def test_templates_endpoint(client):
    r = client.get("/api/templates")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    by_id = {t["id"]: t for t in data}
    qwen = by_id["qwen-image-2.1-turbo"]
    assert qwen["defaults"]["width"] == 1024
    assert qwen["defaults"]["prompt"] is None
    assert qwen["missing_models"] == []
    assert len(qwen["missing_optional_models"]) == 1  # LoRA 不在场，仅提示
    ltx = by_id["ltx-2.3-i2v"]
    assert ltx["capability"] == "image-to-video"
    assert ltx["missing_models"] == []                # FakeEngine 全量提供
    assert ltx["defaults"]["length"] == 25
    ltx2b = by_id["ltx-2b-i2v-distilled"]
    assert ltx2b["missing_models"] == []
    assert ltx2b["defaults"]["steps"] == 8 and ltx2b["defaults"]["cfg"] == 1.0


def test_submit_and_isolation(client):
    h1 = _login(client, "student1")
    h2 = _login(client, "student2")
    r = client.post("/api/jobs", headers=h1,
                    json={"template_id": "qwen-image-2.1-turbo",
                          "params": {"prompt": "cat"}})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "QUEUED"

    deadline = time.time() + 5
    mine = []
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
    assert files.status_code == 200


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
    # LTX 缺 image 参数 → 422
    r = client.post("/api/jobs", headers=h,
                    json={"template_id": "ltx-2.3-i2v", "params": {"prompt": "x"}})
    assert r.status_code == 422


def test_cancel_queued(client):
    ha = _login(client, "admin", "admin123")
    # 提交两个：第一个会被立刻执行，第二个排队
    for _ in range(2):
        client.post("/api/jobs", headers=ha,
                    json={"template_id": "qwen-image-2.1-turbo",
                          "params": {"prompt": "x"}})
    second = client.get("/api/jobs", headers=ha).json()[0]["id"]  # desc 序，最新的是第二个
    r = client.post(f"/api/jobs/{second}/cancel", headers=ha)
    assert r.status_code == 200
    assert r.json()["status"] == "CANCELLED"
