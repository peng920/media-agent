import pytest

from app import db as appdb
from app.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(data_dir=tmp_path, seed_users=True)
    appdb.init_db(s)
    return s


@pytest.fixture
def session(settings):
    yield from appdb.session_factory(settings)()
