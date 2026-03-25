// frontend/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],

  server: {
    host: '0.0.0.0', // required inside Docker
    port: 5173,
    proxy: {
      // DEV proxy: strips /api prefix before hitting FastAPI
      // PROD equivalent is handled by Nginx (docker/nginx/nginx.conf)
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            console.error('[vite proxy] Backend unreachable:', err.message)
            if (!res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'application/json' })
            }
            res.end(JSON.stringify({
              error: 'Backend service unavailable. Please try again later.',
            }))
          })
        },
      },
    },
  },
})
