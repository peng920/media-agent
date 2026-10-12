from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class JobStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str  # student / teacher / admin


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    template_id: str
    params: str  # JSON 编码的模板参数
    status: str = JobStatus.QUEUED
    prompt_id: str | None = None
    error: str | None = None
    progress: float = 0.0
    outputs: str | None = None  # JSON: [{"filename","subfolder","url"}]
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
