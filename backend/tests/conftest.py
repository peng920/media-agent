import httpx
import pytest
from fastapi.testclient import TestClient

from app import db as appdb
from app.config import Settings
from app.main import create_app
from app.scheduler import Scheduler
from tests.fakes import FakeEngine

PNG = b"\x89PNG-fake"


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(data_dir=tmp_path, engine_input_dir=tmp_path / "engine_input",
                 seed_users=True)
    appdb.init_db(s)
    return s


@pytest.fixture
def session(settings):
    yield from appdb.session_factory(settings)()


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
