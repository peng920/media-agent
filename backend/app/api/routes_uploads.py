import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.api.deps import get_current_user
from app.models import User

router = APIRouter(prefix="/api/uploads")

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_BYTES = 20 * 1024 * 1024


@router.post("")
async def upload_image(request: Request, file: UploadFile,
                       user: User = Depends(get_current_user)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(422, f"仅支持 {'/'.join(sorted(ALLOWED_EXT))}，收到 {ext or '未知类型'}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(422, "图片超过 20MB 上限")
    if not data:
        raise HTTPException(422, "空文件")

    # 引擎侧文件名（LoadImage 引用），控制面留档一份
    name = f"zimg_{user.id}_{uuid.uuid4().hex[:8]}{ext}"
    settings = request.app.state.settings
    archive_dir = settings.data_dir / "users" / str(user.id) / "uploads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / name).write_bytes(data)

    engine_input = Path(settings.engine_input_dir)
    engine_input.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive_dir / name, engine_input / name)
    return {"image": name, "url": f"/files/users/{user.id}/uploads/{name}"}
