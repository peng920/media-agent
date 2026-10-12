export interface AuthUser {
  token: string
  username: string
  role: string
}

export function saveAuth(token: string, username: string, role: string): void {
  localStorage.setItem('token', token)
  localStorage.setItem('username', username)
  localStorage.setItem('role', role)
}

export function getUser(): AuthUser | null {
  const token = localStorage.getItem('token')
  if (!token) return null
  return {
    token,
    username: localStorage.getItem('username') ?? '',
    role: localStorage.getItem('role') ?? '',
  }
}

export function logout(): void {
  localStorage.removeItem('token')
  localStorage.removeItem('username')
  localStorage.removeItem('role')
}
