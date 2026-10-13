from pydantic import BaseModel


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    token: str
    username: str
    role: str


class JobIn(BaseModel):
    template_id: str
    params: dict


class JobOut(BaseModel):
    id: int
    template_id: str
    status: str
    progress: float
    error: str | None
    outputs: list[dict] = []
    created_at: str
    finished_at: str | None


class TemplateOut(BaseModel):
    id: str
    name: str
    capability: str
    defaults: dict
    missing_models: list[str]          # 必需缺失 → 禁止提交
    missing_optional_models: list[str] = []  # 可选缺失 → 仅提示
