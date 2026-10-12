import { Button, Card, Form, Input, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { login } from '../api'
import { saveAuth } from '../auth'

export default function Login() {
  const navigate = useNavigate()

  const onFinish = async (values: { username: string; password: string }) => {
    try {
      const r = await login(values.username, values.password)
      saveAuth(r.token, r.username, r.role)
      message.success(`欢迎，${r.username}`)
      navigate('/')
    } catch (e) {
      message.error((e as Error).message || '登录失败')
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Card title="AIGC 教学创作平台" style={{ width: 360 }}>
        <Form layout="vertical" onFinish={onFinish}>
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input placeholder="student1 / admin" autoFocus />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password placeholder="stu123" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block>
            登录
          </Button>
        </Form>
      </Card>
    </div>
  )
}
