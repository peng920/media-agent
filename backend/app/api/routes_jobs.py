import json

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from app.api.deps import get_current_user, get_session_dep
from app.compiler.templates import TEMPLATES
from app.models import Job, JobStatus, User
from app.schemas import JobIn, JobOut

router = APIRouter(prefix="/api/jobs")


def _out(j: Job) -> JobOut:
    return JobOut(
        id=j.id, template_id=j.template_id, status=j.status, progress=j.progress,
        error=j.error, outputs=json.loads(j.outputs or "[]"),
        created_at=j.created_at.isoformat(),
        finished_at=j.finished_at.isoformat() if j.finished_at else None,
    )


@router.post("", response_model=JobOut)
async def submit(body: JobIn, request: Request,
                 user: User = Depends(get_current_user),
                 session: Session = Depends(get_session_dep)) -> JobOut:
    tpl = TEMPLATES.get(body.template_id)
    if not tpl:
        raise HTTPException(404, "模板不存在")
    try:
        tpl.params_model(**body.params)
    except Exception as e:
        raise HTTPException(422, f"参数不合法: {e}")
    job = Job(user_id=user.id, template_id=body.template_id,
              params=json.dumps(body.params, ensure_ascii=False))
    session.add(job)
    session.commit()
    session.refresh(job)
    request.app.state.scheduler.enqueue(job.id)
    return _out(job)


@router.get("", response_model=list[JobOut])
def list_jobs(user: User = Depends(get_current_user),
              session: Session = Depends(get_session_dep)) -> list[JobOut]:
    q = select(Job).order_by(Job.id.desc()).limit(100)
    if user.role == "student":
        q = q.where(Job.user_id == user.id)  # type: ignore[arg-type]
    return [_out(j) for j in session.exec(q)]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, user: User = Depends(get_current_user),
            session: Session = Depends(get_session_dep)) -> JobOut:
    job = session.get(Job, job_id)
    if not job or (user.role == "student" and job.user_id != user.id):
        raise HTTPException(404)
    return _out(job)


@router.post("/{job_id}/cancel", response_model=JobOut)
async def cancel(job_id: int, request: Request,
                 user: User = Depends(get_current_user),
                 session: Session = Depends(get_session_dep)) -> JobOut:
    job = session.get(Job, job_id)
    if not job or (user.role == "student" and job.user_id != user.id):
        raise HTTPException(404)
    if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
        return _out(job)
    if job.status == JobStatus.QUEUED:
        job.status = JobStatus.CANCELLED
        session.add(job)
        session.commit()
        session.refresh(job)
    else:
        request.app.state.scheduler.request_cancel(job_id)
    return _out(job)
