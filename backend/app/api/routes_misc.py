from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic_core import PydanticUndefined

from app.auth import parse_token
from app.compiler.templates import TEMPLATES, validate_models
from app.schemas import TemplateOut

router = APIRouter(prefix="/api")


def _field_defaults(params_model) -> dict:
    return {
        name: (None if f.default is PydanticUndefined else f.default)
        for name, f in params_model.model_fields.items()
    }


@router.get("/templates", response_model=list[TemplateOut])
async def templates(request: Request) -> list[TemplateOut]:
    missing = await validate_models(request.app.state.engine)
    return [
        TemplateOut(id=t.id, name=t.name, capability=t.capability,
                    defaults=_field_defaults(t.params_model), missing_models=missing)
        for t in TEMPLATES.values()
    ]


@router.get("/health")
async def health(request: Request) -> dict:
    engine_ok, engine_ver = False, ""
    try:
        stats = await request.app.state.engine.health()
        engine_ver = stats.get("system", {}).get("comfyui_version", "")
        engine_ok = True
    except Exception:
        pass
    return {"db": True, "engine": engine_ok, "engine_version": engine_ver}


async def ws_jobs(websocket: WebSocket) -> None:
    await websocket.accept()
    token = websocket.query_params.get("token", "")
    parsed = parse_token(token) if token else None
    if not parsed:
        await websocket.send_json({"type": "error", "detail": "未认证"})
        await websocket.close()
        return
    uid, role = parsed
    q = websocket.app.state.notifier.subscribe()
    try:
        while True:
            ev = await q.get()
            if role != "student" or ev.get("user_id") == uid:
                await websocket.send_json(ev)
    except WebSocketDisconnect:
        pass
    finally:
        websocket.app.state.notifier.unsubscribe(q)
