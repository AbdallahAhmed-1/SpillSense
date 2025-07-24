import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy GET /files → http://localhost:8000/files
      '/files': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy POST /upload → http://localhost:8000/upload
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy GET /download/... → http://localhost:8000/download/...
      '/download': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Proxy POST /chat → http://localhost:8000/chat
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/artifacts': 'http://localhost:8000',
    }
  }
});
