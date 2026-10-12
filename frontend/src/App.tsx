import type { ReactElement } from 'react'
import { Layout, Menu, Space, Tag, Typography } from 'antd'
import {
  Navigate, Route, Routes, useLocation, useNavigate,
} from 'react-router-dom'
import { getUser, logout } from './auth'
import Create from './pages/Create'
import Jobs from './pages/Jobs'
import Login from './pages/Login'

const { Header, Content } = Layout

function RequireAuth({ children }: { children: ReactElement }) {
  return getUser() ? children : <Navigate to="/login" replace />
}

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const user = getUser()

  if (location.pathname === '/login') {
    return user ? <Navigate to="/" replace /> : <Routes><Route path="/login" element={<Login />} /></Routes>
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: '#fff',
          borderBottom: '1px solid #eee',
        }}
      >
        <Space size="large">
          <Typography.Title level={4} style={{ margin: 0 }}>
            AIGC 教学创作平台
          </Typography.Title>
          <Menu
            mode="horizontal"
            selectedKeys={[location.pathname]}
            style={{ flex: 1, minWidth: 240 }}
            onClick={(e) => navigate(e.key)}
            items={[
              { key: '/', label: '创作' },
              { key: '/jobs', label: '任务队列' },
            ]}
          />
        </Space>
        <Space>
          <Tag color={user?.role === 'admin' ? 'gold' : user?.role === 'teacher' ? 'blue' : 'green'}>
            {user?.username}（{user?.role === 'admin' ? '管理员' : user?.role === 'teacher' ? '教师' : '学生'}）
          </Tag>
          <Typography.Link
            onClick={() => {
              logout()
              navigate('/login')
            }}
          >
            退出
          </Typography.Link>
        </Space>
      </Header>
      <Content style={{ padding: 24, maxWidth: 1080, width: '100%', margin: '0 auto' }}>
        <Routes>
          <Route path="/" element={<RequireAuth><Create /></RequireAuth>} />
          <Route path="/jobs" element={<RequireAuth><Jobs /></RequireAuth>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Content>
    </Layout>
  )
}
