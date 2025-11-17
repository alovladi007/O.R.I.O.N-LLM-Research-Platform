'use client'

import { AppBar as MuiAppBar, Toolbar, Typography, Box } from '@mui/material'

export function AppBar() {
  return (
    <MuiAppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          🔬 ORION Platform
        </Typography>
      </Toolbar>
    </MuiAppBar>
  )
}
