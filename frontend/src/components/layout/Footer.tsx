'use client'

import { Box, Container, Typography } from '@mui/material'

export function Footer() {
  return (
    <Box component="footer" sx={{ py: 3, px: 2, mt: 'auto', backgroundColor: '#f5f5f5' }}>
      <Container maxWidth="lg">
        <Typography variant="body2" color="text.secondary" align="center">
          © {new Date().getFullYear()} ORION Platform - Materials Science AI
        </Typography>
      </Container>
    </Box>
  )
}
