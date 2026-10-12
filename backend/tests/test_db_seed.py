from sqlmodel import select

from app.db import hash_password, verify_password
from app.models import User


def test_seed_users(settings, session):
    users = session.exec(select(User)).all()
    assert {u.username for u in users} == {"admin", "student1", "student2"}
    assert all("$" in u.password_hash for u in users)


def test_password_roundtrip():
    h = hash_password("abc")
    assert verify_password("abc", h)
    assert not verify_password("abd", h)
