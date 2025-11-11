import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'urogpt': {
          50: '#e6f7f3',
          100: '#b3e8dd',
          200: '#80d9c7',
          300: '#4dcab1',
          400: '#1abb9b',
          500: '#10a37f',
          600: '#0d8c6f',
          700: '#0a7a5f',
          800: '#07684f',
          900: '#04563f',
        }
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
      },
    },
  },
  plugins: [],
} satisfies Config

