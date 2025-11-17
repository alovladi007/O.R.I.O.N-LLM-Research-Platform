'use client'

import { useState, FormEvent } from 'react'
import { useRouter } from 'next/navigation'
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

interface LoginFormData {
  email: string
  password: string
}

interface LoginResponse {
  access_token: string
  token_type: string
  user: {
    id: string
    email: string
    username: string
    full_name?: string
  }
}

export default function LoginPage() {
  const router = useRouter()
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'))

  // Form state
  const [formData, setFormData] = useState<LoginFormData>({
    email: '',
    password: '',
  })

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

  /**
   * Validate form data
   */
  const validateForm = (): boolean => {
    const errors: Record<string, string> = {}

    if (!formData.email.trim()) {
      errors.email = 'Email is required'
    } else if (!isValidEmail(formData.email)) {
      errors.email = 'Please enter a valid email address'
    }

    if (!formData.password) {
      errors.password = 'Password is required'
    } else if (formData.password.length < 6) {
      errors.password = 'Password must be at least 6 characters'
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }

  /**
   * Validate email format
   */
  const isValidEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  /**
   * Handle input change
   */
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }))
    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: '',
      }))
    }
  }

  /**
   * Handle form submission
   */
  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    // Validate form
    if (!validateForm()) {
      setError('Please fix the errors below')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/auth/login`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: formData.email,
            password: formData.password,
          }),
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Login failed')
      }

      const data: LoginResponse = await response.json()

      // Store token in localStorage
      localStorage.setItem('auth_token', data.access_token)

      // Show success message
      toast.success('Login successful!')

      // Redirect to home page
      router.push('/')
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred during login'
      setError(errorMessage)
      toast.error(errorMessage)
      console.error('Login error:', err)
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
            sx={{
              p: { xs: 3, md: 4 },
              borderRadius: 2,
              backgroundColor: 'background.paper',
            }}
          >
            {/* Header */}
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  mb: 2,
                }}
              >
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

            {/* Error Alert */}
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <Alert
                  severity="error"
                  sx={{ mb: 3, borderRadius: 1 }}
                  onClose={() => setError('')}
                >
                  {error}
                </Alert>
              </motion.div>
            )}

            {/* Login Form */}
            <Box component="form" onSubmit={handleSubmit} noValidate>
              <Grid container spacing={2}>
                {/* Email Field */}
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Email Address"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    error={!!validationErrors.email}
                    helperText={validationErrors.email}
                    placeholder="you@example.com"
                    disabled={loading}
                    InputProps={{
                      startAdornment: (
                        <Mail sx={{ mr: 1.5, color: 'text.secondary' }} />
                      ),
                    }}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        '&:hover fieldset': {
                          borderColor: 'primary.main',
                        },
                      },
                    }}
                  />
                </Grid>

                {/* Password Field */}
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
                    InputProps={{
                      startAdornment: (
                        <Lock sx={{ mr: 1.5, color: 'text.secondary' }} />
                      ),
                    }}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        '&:hover fieldset': {
                          borderColor: 'primary.main',
                        },
                      },
                    }}
                  />
                </Grid>

                {/* Remember Me & Forgot Password */}
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      Keep me signed in
                    </Typography>
                    <Link
                      href="/forgot-password"
                      underline="hover"
                      sx={{
                        fontSize: '0.875rem',
                        cursor: 'pointer',
                        color: 'primary.main',
                        '&:hover': {
                          textDecoration: 'underline',
                        },
                      }}
                    >
                      Forgot Password?
                    </Link>
                  </Box>
                </Grid>

                {/* Submit Button */}
                <Grid item xs={12}>
                  <Button
                    fullWidth
                    variant="contained"
                    size="large"
                    type="submit"
                    disabled={loading}
                    sx={{
                      py: 1.5,
                      fontSize: '1rem',
                      fontWeight: 600,
                      textTransform: 'none',
                      background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                      '&:hover': {
                        opacity: 0.9,
                      },
                      '&:disabled': {
                        opacity: 0.6,
                      },
                    }}
                  >
                    {loading ? (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CircularProgress size={20} sx={{ color: 'white' }} />
                        Signing in...
                      </Box>
                    ) : (
                      'Sign In'
                    )}
                  </Button>
                </Grid>

                {/* Divider */}
                <Grid item xs={12}>
                  <Box sx={{ position: 'relative', my: 2 }}>
                    <Box
                      sx={{
                        position: 'absolute',
                        left: 0,
                        right: 0,
                        top: '50%',
                        transform: 'translateY(-50%)',
                        borderTop: `1px solid ${theme.palette.divider}`,
                      }}
                    />
                    <Typography
                      variant="caption"
                      sx={{
                        position: 'relative',
                        px: 2,
                        display: 'inline-block',
                        left: '50%',
                        transform: 'translateX(-50%)',
                        backgroundColor: 'background.paper',
                        color: 'text.secondary',
                      }}
                    >
                      Don't have an account?
                    </Typography>
                  </Box>
                </Grid>

                {/* Register Link */}
                <Grid item xs={12}>
                  <Typography align="center" variant="body2">
                    <Link
                      href="/register"
                      underline="none"
                      sx={{
                        color: 'primary.main',
                        fontWeight: 600,
                        cursor: 'pointer',
                        '&:hover': {
                          textDecoration: 'underline',
                        },
                      }}
                    >
                      Create an account
                    </Link>
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Footer Text */}
            <Typography
              variant="caption"
              align="center"
              sx={{
                display: 'block',
                mt: 3,
                color: 'text.secondary',
              }}
            >
              By signing in, you agree to our{' '}
              <Link href="/terms" underline="hover">
                Terms of Service
              </Link>
              {' '}and{' '}
              <Link href="/privacy" underline="hover">
                Privacy Policy
              </Link>
            </Typography>
          </Paper>
        </motion.div>

        {/* Responsive spacing */}
        {!isMobile && (
          <Box
            sx={{
              mt: 4,
              textAlign: 'center',
              color: 'rgba(255,255,255,0.7)',
            }}
          >
            <Typography variant="body2">
              ORION AI-Driven Materials Science Platform
            </Typography>
          </Box>
        )}
      </Container>
    </Box>
  )
}
