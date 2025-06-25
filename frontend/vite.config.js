import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())
  return {
    plugins: [react()],
    define: {
      'import.meta.env.VITE_API_URL': JSON.stringify(env.VITE_API_URL),
    },
    preview: {
      port: parseInt(env.PORT) || 4173,
      host: true,
      allowedHosts: ['netxplore-frontend.onrender.com'],
    },
  }
})
