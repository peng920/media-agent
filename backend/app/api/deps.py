from collections.abc import Generator

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from app.auth import parse_token
from app.models import User


def get_session_dep(request: Request) -> Generator[Session, None, None]:
    yield from request.app.state.session_factory()


def get_current_user(request: Request,
                     session: Session = Depends(get_session_dep)) -> User:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    parsed = parse_token(token) if token else None
    if not parsed:
        raise HTTPException(401, "未登录或令牌失效")
    user = session.get(User, parsed[0])
    if not user:
        raise HTTPException(401, "用户不存在")
    return user
