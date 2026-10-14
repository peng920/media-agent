PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def test_upload_image(client, settings):
    h = {"Authorization": _token(client)}
    r = client.post("/api/uploads", headers=h,
                    files={"file": ("cat.png", PNG, "image/png")})
    assert r.status_code == 200, r.text
    data = r.json()
    name = data["image"]
    assert name.startswith("zimg_") and name.endswith(".png")
    # 控制面留档与引擎 input 目录都落盘（student1 的 uid=2）
    assert (settings.data_dir / "users" / "2" / "uploads" / name).read_bytes() == PNG
    assert (settings.engine_input_dir / name).read_bytes() == PNG
    # 留档可通过 /files 访问
    assert client.get(data["url"]).status_code == 200


def test_upload_rejects_bad_type(client):
    h = {"Authorization": _token(client)}
    r = client.post("/api/uploads", headers=h,
                    files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def test_upload_requires_auth(client):
    r = client.post("/api/uploads",
                    files={"file": ("cat.png", PNG, "image/png")})
    assert r.status_code == 401


def _token(c):
    r = c.post("/api/auth/login", json={"username": "student1", "password": "stu123"})
    return f"Bearer {r.json()['token']}"
