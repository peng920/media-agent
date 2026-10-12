export interface JobEvent {
  job_id: number
  user_id: number
  status: string
  progress: number
  error: string | null
  outputs: { filename: string; url: string }[]
}

export type WsStatus = 'connecting' | 'open' | 'closed'

/** 连接 /ws/jobs，事件回调 + 3 秒断线重连；返回清理函数。 */
export function connectJobsWS(
  onEvent: (ev: JobEvent) => void,
  onStatus: (s: WsStatus) => void,
): () => void {
  let stopped = false
  let timer: number | undefined
  let ws: WebSocket | undefined

  const connect = () => {
    if (stopped) return
    onStatus('connecting')
    const token = localStorage.getItem('token') ?? ''
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/ws/jobs?token=${encodeURIComponent(token)}`)
    ws.onopen = () => onStatus('open')
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data) as JobEvent)
      } catch {
        /* 忽略非 JSON 消息 */
      }
    }
    ws.onclose = () => {
      onStatus('closed')
      if (!stopped) timer = window.setTimeout(connect, 3000)
    }
    ws.onerror = () => ws?.close()
  }
  connect()

  return () => {
    stopped = true
    if (timer) clearTimeout(timer)
    ws?.close()
  }
}
