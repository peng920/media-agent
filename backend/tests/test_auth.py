from app.auth import issue_token, parse_token
from app.models import User


def _user(uid=7, role="student") -> User:
    return User(id=uid, username="x", password_hash="00$", role=role)


def test_token_roundtrip():
    t = issue_token(_user())
    assert parse_token(t) == (7, "student")


def test_bad_token():
    assert parse_token("7.student.99999.deadbeef") is None
    assert parse_token("") is None
    assert parse_token("garbage") is None
