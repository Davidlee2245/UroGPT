/**
 * UroGPT UI - Main Entry Point
 * 
 * Quickstart:
 * 1. npm create vite@latest urogpt-ui -- --template react-ts
 * 2. cd urogpt-ui
 * 3. npm i
 * 4. npm i -D tailwindcss postcss autoprefixer
 * 5. npx tailwindcss init -p
 * 6. Configure tailwind.config.ts (see tailwind.config.ts)
 * 7. Add Tailwind directives to src/index.css
 * 8. npm i lucide-react
 * 9. npm run dev
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

