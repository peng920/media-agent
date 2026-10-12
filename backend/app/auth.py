import base64
import hashlib
import hmac
import time

from app.config import get_settings
from app.models import User


def _sign(msg: str) -> str:
    key = get_settings().token_secret.encode()
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()


def issue_token(user: User) -> str:
    exp = int(time.time()) + 86400
    body = f"{user.id}.{user.role}.{exp}"
    b = base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")
    return f"{b}.{_sign(b)}"


def parse_token(token: str) -> tuple[int, str] | None:
    try:
        b, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(_sign(b), sig):
            return None
        pad = "=" * (-len(b) % 4)
        uid, role, exp = base64.urlsafe_b64decode(b + pad).decode().split(".")
        if int(exp) < time.time():
            return None
        return int(uid), role
    except Exception:
        return None
