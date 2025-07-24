/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html","./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      // 1. Custom font
      fontFamily: {
        sans: ['Roboto', 'sans-serif'],
      },
      // 2. Your brand colors (sample: adjust hex to match your palette)
      colors: {
        brand: {
          50:  '#e9f2ff',
          100: '#c8ddff',
          200: '#a3c7ff',
          300: '#7eb2ff',
          400: '#5aa0ff',
          500: '#388eff',    // primary
          600: '#226ed6',
          700: '#184fa4',
          800: '#113472',
          900: '#0b1f41',
        },
        neutral: {
          50:  '#fafafa',
          100: '#f4f4f4',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
        },
      },
    },
  },
  plugins: [],
};
