import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = 'http://127.0.0.1:8090'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/files': { target: backend, changeOrigin: true },
      '/ws': { target: backend.replace('http', 'ws'), ws: true },
    },
  },
  build: { outDir: '../backend/frontend_dist', emptyOutDir: true },
})
