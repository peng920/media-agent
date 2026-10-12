import hashlib
import secrets
from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine, select

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


def _engine_for(settings: Settings):
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False}
    )


def init_db(settings: Settings) -> None:
    SQLModel.metadata.create_all(_engine_for(settings))
    if not settings.seed_users:
        return
    with Session(_engine_for(settings)) as s:
        if s.exec(select(User).where(User.username == "admin")).first():
            return
        for name, pw, role in [
            ("admin", "admin123", "admin"),
            ("student1", "stu123", "student"),
            ("student2", "stu123", "student"),
        ]:
            s.add(User(username=name, password_hash=hash_password(pw), role=role))
        s.commit()


def session_factory(settings: Settings):
    engine = _engine_for(settings)

    def _get() -> Generator[Session, None, None]:
        with Session(engine) as s:
            yield s

    return _get
