'use client'

import Link from 'next/link'
import { AppBar as MuiAppBar, Toolbar, Typography, Box, Button } from '@mui/material'
import { Science, Search, TableChart, Home } from '@mui/icons-material'

export function AppBar() {
  return (
    <MuiAppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ mr: 4 }}>
          🔬 ORION Platform
        </Typography>
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
          <Button
            component={Link}
            href="/"
            color="inherit"
            startIcon={<Home />}
          >
            Home
          </Button>
          <Button
            component={Link}
            href="/structures"
            color="inherit"
            startIcon={<TableChart />}
          >
            Structures
          </Button>
          <Button
            component={Link}
            href="/design"
            color="inherit"
            startIcon={<Search />}
          >
            Design Search
          </Button>
        </Box>
      </Toolbar>
    </MuiAppBar>
  )
}
