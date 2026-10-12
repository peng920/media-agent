from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.deps import get_current_user, get_session_dep
from app.auth import issue_token
from app.db import verify_password
from app.models import User
from app.schemas import LoginIn, TokenOut

router = APIRouter(prefix="/api/auth")


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, session: Session = Depends(get_session_dep)) -> TokenOut:
    user = session.exec(select(User).where(User.username == body.username)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "用户名或密码错误")
    token = issue_token(user)
    return TokenOut(token=token, username=user.username, role=user.role)


@router.get("/me", response_model=TokenOut)
def me(user: User = Depends(get_current_user)) -> TokenOut:
    return TokenOut(token="", username=user.username, role=user.role)
