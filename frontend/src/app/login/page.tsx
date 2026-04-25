'use client'

import { useState, FormEvent, useMemo } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import {
  Container,
  Box,
  TextField,
  Button,
  Typography,
  Link,
  Alert,
  CircularProgress,
  Paper,
  Grid,
  useTheme,
  useMediaQuery,
} from '@mui/material'
import { motion } from 'framer-motion'
import { Lock, Mail } from '@mui/icons-material'
import toast from 'react-hot-toast'

import { useAuth } from '@/lib/auth-context'
import { formatErrorMessage } from '@/lib/api'

interface LoginFormData {
  username: string  // backend accepts username OR email here
  password: string
}

export default function LoginPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { login } = useAuth()
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'))

  const next = useMemo(
    () => searchParams.get('next') ?? '/dashboard',
    [searchParams],
  )

  const [formData, setFormData] = useState<LoginFormData>({ username: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

  const validateForm = (): boolean => {
    const errs: Record<string, string> = {}
    if (!formData.username.trim()) errs.username = 'Username or email is required'
    if (!formData.password) errs.password = 'Password is required'
    setValidationErrors(errs)
    return Object.keys(errs).length === 0
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
    if (validationErrors[name]) {
      setValidationErrors((prev) => ({ ...prev, [name]: '' }))
    }
  }

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!validateForm()) {
      setError('Please fix the errors below')
      return
    }
    setLoading(true)
    setError('')
    try {
      await login(formData)
      toast.success('Login successful!')
      router.push(next)
    } catch (err) {
      const msg = formatErrorMessage(err)
      setError(msg)
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
        py: 4,
      }}
    >
      <Container maxWidth="sm">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Paper
            elevation={8}
            sx={{ p: { xs: 3, md: 4 }, borderRadius: 2, backgroundColor: 'background.paper' }}
          >
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                <Box
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: '50%',
                    background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Lock sx={{ color: 'white', fontSize: 32 }} />
                </Box>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                Welcome Back
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Sign in to access the ORION Platform
              </Typography>
            </Box>

            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
              >
                <Alert
                  severity="error"
                  sx={{ mb: 3, borderRadius: 1 }}
                  onClose={() => setError('')}
                  data-testid="login-error"
                >
                  {error}
                </Alert>
              </motion.div>
            )}

            <Box component="form" onSubmit={handleSubmit} noValidate>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Username or Email"
                    name="username"
                    type="text"
                    value={formData.username}
                    onChange={handleInputChange}
                    error={!!validationErrors.username}
                    helperText={validationErrors.username}
                    placeholder="scientist@orion.dev"
                    disabled={loading}
                    inputProps={{ 'data-testid': 'login-username' }}
                    InputProps={{
                      startAdornment: <Mail sx={{ mr: 1.5, color: 'text.secondary' }} />,
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Password"
                    name="password"
                    type="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    error={!!validationErrors.password}
                    helperText={validationErrors.password}
                    placeholder="Enter your password"
                    disabled={loading}
                    inputProps={{ 'data-testid': 'login-password' }}
                    InputProps={{
                      startAdornment: <Lock sx={{ mr: 1.5, color: 'text.secondary' }} />,
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Button
                    fullWidth
                    variant="contained"
                    size="large"
                    type="submit"
                    disabled={loading}
                    data-testid="login-submit"
                    sx={{
                      py: 1.5,
                      fontSize: '1rem',
                      fontWeight: 600,
                      textTransform: 'none',
                      background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                    }}
                  >
                    {loading ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CircularProgress size={20} sx={{ color: 'white' }} />
                        Signing in…
                      </Box>
                    ) : (
                      'Sign In'
                    )}
                  </Button>
                </Grid>

                <Grid item xs={12}>
                  <Typography align="center" variant="body2">
                    Don&apos;t have an account?{' '}
                    <Link href="/register" underline="hover" sx={{ color: 'primary.main', fontWeight: 600 }}>
                      Create an account
                    </Link>
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </motion.div>

        {!isMobile && (
          <Box sx={{ mt: 4, textAlign: 'center', color: 'rgba(255,255,255,0.7)' }}>
            <Typography variant="body2">ORION AI-Driven Materials Science Platform</Typography>
          </Box>
        )}
      </Container>
    </Box>
  )
}
