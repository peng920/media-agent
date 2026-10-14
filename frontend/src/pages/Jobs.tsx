import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Badge, Button, Card, Image, Popconfirm, Progress, Space, Table, Tag, Tooltip,
  Typography, message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { cancelJob, listJobs, type JobItem } from '../api'
import { connectJobsWS, type JobEvent, type WsStatus } from '../ws'

const STATUS_TAG: Record<string, { color: string; text: string }> = {
  QUEUED: { color: 'default', text: '排队中' },
  RUNNING: { color: 'processing', text: '生成中' },
  DONE: { color: 'success', text: '完成' },
  FAILED: { color: 'error', text: '失败' },
  CANCELLED: { color: 'warning', text: '已取消' },
}

function upsert(jobs: JobItem[], ev: JobEvent): JobItem[] {
  const exists = jobs.some((j) => j.id === ev.job_id)
  const patched: JobItem = {
    id: ev.job_id, template_id: '', status: ev.status, progress: ev.progress,
    error: ev.error, outputs: ev.outputs, created_at: '', finished_at: null,
  }
  if (exists) {
    return jobs.map((j) =>
      j.id === ev.job_id ? { ...j, status: ev.status, progress: ev.progress, error: ev.error, outputs: ev.outputs.length ? ev.outputs : j.outputs } : j,
    )
  }
  return [patched, ...jobs]
}

export default function Jobs() {
  const [jobs, setJobs] = useState<JobItem[]>([])
  const [wsStatus, setWsStatus] = useState<WsStatus>('connecting')
  const jobsRef = useRef<JobItem[]>([])

  const refresh = useCallback(() => {
    listJobs()
      .then((js) => {
        jobsRef.current = js
        setJobs(js)
      })
      .catch((e) => message.error((e as Error).message))
  }, [])

  useEffect(() => {
    refresh()
    const stop = connectJobsWS(
      (ev) => setJobs((prev) => upsert(prev, ev)),
      setWsStatus,
    )
    const timer = window.setInterval(refresh, 10000) // WS 断线时的兜底轮询
    return () => {
      stop()
      clearInterval(timer)
    }
  }, [refresh])

  const onCancel = async (id: number) => {
    try {
      await cancelJob(id)
      message.success(`任务 #${id} 已请求取消`)
      refresh()
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  const columns: ColumnsType<JobItem> = [
    { title: 'ID', dataIndex: 'id', width: 70 },
    { title: '模板', dataIndex: 'template_id', width: 200 },
    {
      title: '状态',
      dataIndex: 'status',
      width: 110,
      render: (s: string) => {
        const t = STATUS_TAG[s] ?? { color: 'default', text: s }
        return <Tag color={t.color}>{t.text}</Tag>
      },
    },
    {
      title: '进度',
      dataIndex: 'progress',
      width: 160,
      render: (p: number, row) =>
        row.status === 'RUNNING' ? (
          <Progress percent={Math.round(p * 100)} size="small" status="active" />
        ) : (
          <Progress
            percent={row.status === 'DONE' ? 100 : 0}
            size="small"
            status={row.status === 'FAILED' ? 'exception' : 'normal'}
            showInfo={row.status === 'DONE'}
          />
        ),
    },
    {
      title: '产物',
      dataIndex: 'outputs',
      render: (outputs: JobItem['outputs']) => (
        <Space>
          {outputs.map((o) =>
            /\.(mp4|webm|mov)$/i.test(o.url) ? (
              <video key={o.url} src={o.url} controls width={200}
                style={{ borderRadius: 8, background: '#000' }} />
            ) : (
              <Image.PreviewGroup key={o.url}>
                <Image src={o.url} width={72} height={72} style={{ objectFit: 'cover' }} />
              </Image.PreviewGroup>
            ),
          )}
        </Space>
      ),
    },
    {
      title: '操作',
      width: 110,
      render: (_, row) =>
        ['QUEUED', 'RUNNING'].includes(row.status) ? (
          <Popconfirm title="确定取消该任务？" onConfirm={() => onCancel(row.id)}>
            <Button size="small" danger>
              取消
            </Button>
          </Popconfirm>
        ) : row.status === 'FAILED' && row.error ? (
          <Tooltip title={row.error}>
            <Typography.Text type="secondary" style={{ cursor: 'help' }}>
              错误详情
            </Typography.Text>
          </Tooltip>
        ) : null,
    },
  ]

  return (
    <Card
      title="任务队列"
      extra={
        <Space>
          <Badge
            status={wsStatus === 'open' ? 'success' : 'error'}
            text={wsStatus === 'open' ? '实时推送已连接' : '实时推送断开，10 秒轮询兜底'}
          />
          <Button size="small" onClick={refresh}>
            刷新
          </Button>
        </Space>
      }
    >
      <Table
        rowKey="id"
        columns={columns}
        dataSource={jobs}
        pagination={{ pageSize: 20 }}
      />
    </Card>
  )
}
