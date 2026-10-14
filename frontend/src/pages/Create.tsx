import { useEffect, useState } from 'react'
import {
  Alert, Button, Card, Col, Form, Input, InputNumber, message, Row, Select, Switch,
  Upload,
} from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { listTemplates, submitJob, uploadImage, type TemplateItem } from '../api'

interface FormValues {
  template_id: string
  prompt: string
  image?: string
  width: number
  height: number
  length?: number
  cfg?: number
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
  const missingOptional = selected?.missing_optional_models ?? []
  const isVideo = selected?.capability === 'image-to-video'

  const onFinish = async (v: FormValues) => {
    setSubmitting(true)
    try {
      const params: Record<string, unknown> = {
        prompt: v.prompt, width: v.width, height: v.height,
        steps: v.steps,
      }
      if (v.seed !== undefined && v.seed !== null) params.seed = v.seed
      if (isVideo) {
        params.image = v.image
        params.length = v.length
        params.cfg = v.cfg
      } else {
        params.use_lora = v.use_lora
      }
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
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          message="引擎缺少必需模型文件，暂不能提交"
          description={missing.join('、')}
        />
      )}
      {missing.length === 0 && missingOptional.length > 0 && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message="可选模型缺失（关闭对应开关即可正常提交）"
          description={missingOptional.join('、')}
        />
      )}
      <Form
        layout="vertical"
        onFinish={onFinish}
        key={selected?.id ?? 'none'}
        initialValues={{
          template_id: selected?.id,
          width: isVideo ? 768 : 1024,
          height: isVideo ? 512 : 1024,
          steps: isVideo ? 20 : 6,
          length: 25,
          cfg: 3.0,
          use_lora: true,
        }}
      >
        <Form.Item name="template_id" label="模板" rules={[{ required: true }]}>
          <Select
            options={templates.map((t) => ({ value: t.id, label: t.name }))}
            value={selected?.id}
            onChange={(id) => setSelected(templates.find((t) => t.id === id) ?? null)}
          />
        </Form.Item>
        {isVideo && (
          <Form.Item
            name="image"
            label="起始图片"
            rules={[{ required: true, message: '请先上传图片' }]}
          >
            <Upload
              accept=".png,.jpg,.jpeg,.webp"
              maxCount={1}
              listType="picture-card"
              customRequest={async ({ file, onSuccess, onError }) => {
                try {
                  const r = await uploadImage(file as File)
                  onSuccess?.(r)
                } catch (e) {
                  onError?.(e as Error)
                }
              }}
            >
              <div>
                <UploadOutlined />
                <div style={{ marginTop: 8 }}>上传图片</div>
              </div>
            </Upload>
          </Form.Item>
        )}
        <Form.Item
          name="prompt"
          label="提示词"
          rules={[{ required: true, message: '请输入提示词' }]}
        >
          <Input.TextArea
            rows={4}
            placeholder={isVideo
              ? '描述画面如何随时间运动，以及需要的环境音/对白（音视频联合生成）…'
              : '描述你想生成的画面…'}
          />
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
        {isVideo ? (
          <Row gutter={16}>
            <Col span={6}>
              <Form.Item name="length" label="帧数（8n+1）" rules={[{ required: true }]}>
                <InputNumber min={9} max={121} step={8} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="cfg" label="CFG" rules={[{ required: true }]}>
                <InputNumber min={1} max={10} step={0.5} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        ) : (
          <Form.Item name="use_lora" label="启用 Turbo LoRA" valuePropName="checked">
            <Switch />
          </Form.Item>
        )}
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
