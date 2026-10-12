import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from app import db as appdb
from app.api.routes_auth import router as r_auth
from app.api.routes_jobs import router as r_jobs
from app.api.routes_misc import router as r_misc, ws_jobs
from app.config import Settings, get_settings
from app.engine.client import EngineClient
from app.events import Notifier
from app.scheduler import Scheduler


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="AIGC 教学创作平台")
    appdb.init_db(s)

    app.state.settings = s
    app.state.engine = EngineClient(s.engine_base_url)
    app.state.notifier = Notifier()
    app.state.session_factory = appdb.session_factory(s)
    app.state.scheduler = Scheduler(app.state.engine, app.state.session_factory, s,
                                    app.state.notifier)

    @app.on_event("startup")
    async def _start() -> None:
        app.state.sched_task = asyncio.create_task(app.state.scheduler.run_forever())

    @app.on_event("shutdown")
    async def _stop() -> None:
        app.state.sched_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.sched_task

    app.include_router(r_auth)
    app.include_router(r_jobs)
    app.include_router(r_misc)
    app.websocket("/ws/jobs")(ws_jobs)

    s.data_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(s.data_dir)), name="files")
    web_dist = Path(__file__).resolve().parent.parent / "frontend_dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
    return app


app = create_app()
