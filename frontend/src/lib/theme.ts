/**
 * Phase 9 / Session 9.1 — minimal MUI theme.
 *
 * The pre-existing ``RootLayout`` imports ``{ theme } from '@/lib/theme'``
 * but the file was missing on main (typecheck failure surfaced by the
 * Session 9.1 frontend rewrite). We ship a sensible default — primary +
 * secondary palette tuned for the gradient-heavy login page; later
 * sessions can replace with a richer brand palette without touching
 * this import.
 */

import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1e3a8a',  // ORION deep blue
    },
    secondary: {
      main: '#0d9488',  // ORION teal accent
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
    button: {
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 8,
  },
})
