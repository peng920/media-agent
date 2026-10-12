import { useEffect, useState } from 'react'
import {
  Alert, Button, Card, Col, Form, Input, InputNumber, message, Row, Select, Switch,
} from 'antd'
import { useNavigate } from 'react-router-dom'
import { listTemplates, submitJob, type TemplateItem } from '../api'

interface FormValues {
  template_id: string
  prompt: string
  width: number
  height: number
  steps: number
  seed?: number
  use_lora: boolean
}

export default function Create() {
  const navigate = useNavigate()
  const [templates, setTemplates] = useState<TemplateItem[]>([])
  const [selected, setSelected] = useState<TemplateItem | null>(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    listTemplates()
      .then((ts) => {
        setTemplates(ts)
        setSelected(ts[0] ?? null)
      })
      .catch((e) => message.error((e as Error).message))
  }, [])

  const missing = selected?.missing_models ?? []

  const onFinish = async (v: FormValues) => {
    setSubmitting(true)
    try {
      const params: Record<string, unknown> = {
        prompt: v.prompt, width: v.width, height: v.height,
        steps: v.steps, use_lora: v.use_lora,
      }
      if (v.seed !== undefined && v.seed !== null) params.seed = v.seed
      const job = await submitJob(v.template_id, params)
      message.success(`任务 #${job.id} 已提交`)
      navigate('/jobs')
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card title="创作" style={{ maxWidth: 860 }}>
      {missing.length > 0 && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message="引擎缺少以下模型文件，任务将被拒绝"
          description={missing.join('、')}
        />
      )}
      <Form
        layout="vertical"
        onFinish={onFinish}
        initialValues={{
          template_id: selected?.id,
          width: 1024, height: 1024, steps: 6, use_lora: true,
        }}
      >
        <Form.Item name="template_id" label="模板" rules={[{ required: true }]}>
          <Select
            options={templates.map((t) => ({ value: t.id, label: t.name }))}
            value={selected?.id}
            onChange={(id) => setSelected(templates.find((t) => t.id === id) ?? null)}
          />
        </Form.Item>
        <Form.Item
          name="prompt"
          label="提示词"
          rules={[{ required: true, message: '请输入提示词' }]}
        >
          <Input.TextArea rows={4} placeholder="描述你想生成的画面…" />
        </Form.Item>
        <Row gutter={16}>
          <Col span={6}>
            <Form.Item name="width" label="宽度" rules={[{ required: true }]}>
              <InputNumber min={256} max={2048} step={32} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name="height" label="高度" rules={[{ required: true }]}>
              <InputNumber min={256} max={2048} step={32} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name="steps" label="步数" rules={[{ required: true }]}>
              <InputNumber min={1} max={50} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name="seed" label="种子（留空随机）">
              <InputNumber min={0} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Form.Item name="use_lora" label="启用 Turbo LoRA" valuePropName="checked">
          <Switch />
        </Form.Item>
        <Button
          type="primary"
          htmlType="submit"
          loading={submitting}
          disabled={missing.length > 0}
        >
          提交任务
        </Button>
      </Form>
    </Card>
  )
}
