export interface TemplateItem {
  id: string
  name: string
  capability: string
  defaults: Record<string, unknown>
  missing_models: string[]
  missing_optional_models?: string[]
}

export interface JobOutput {
  filename: string
  url: string
}

export interface JobItem {
  id: number
  template_id: string
  status: string
  progress: number
  error: string | null
  outputs: JobOutput[]
  created_at: string
  finished_at: string | null
}

export class ApiError extends Error {}

async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem('token') ?? ''
  const res = await fetch(path, {
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(opts.headers ?? {}),
    },
  })
  if (res.status === 401) {
    localStorage.clear()
    window.location.href = '/login'
    throw new ApiError('未登录')
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string }
    throw new ApiError(body.detail ?? `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export const login = (username: string, password: string) =>
  api<{ token: string; username: string; role: string }>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })

export const listTemplates = () => api<TemplateItem[]>('/api/templates')

export const submitJob = (template_id: string, params: Record<string, unknown>) =>
  api<JobItem>('/api/jobs', { method: 'POST', body: JSON.stringify({ template_id, params }) })

export const listJobs = () => api<JobItem[]>('/api/jobs')

export const cancelJob = (id: number) =>
  api<JobItem>(`/api/jobs/${id}/cancel`, { method: 'POST' })

export const getHealth = () =>
  api<{ db: boolean; engine: boolean; engine_version: string }>('/api/health')
