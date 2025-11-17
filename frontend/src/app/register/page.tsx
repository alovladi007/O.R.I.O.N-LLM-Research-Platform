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
  FormControlLabel,
  Checkbox,
} from '@mui/material'
import { motion } from 'framer-motion'
import { PersonAdd, Mail, Lock, Person } from '@mui/icons-material'
import toast from 'react-hot-toast'

interface RegisterFormData {
  email: string
  username: string
  fullName: string
  password: string
  confirmPassword: string
  agreeToTerms: boolean
}

interface RegisterResponse {
  id: string
  email: string
  username: string
  full_name?: string
  message: string
}

interface ValidationErrors {
  email?: string
  username?: string
  fullName?: string
  password?: string
  confirmPassword?: string
  agreeToTerms?: string
}

export default function RegisterPage() {
  const router = useRouter()
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'))

  // Form state
  const [formData, setFormData] = useState<RegisterFormData>({
    email: '',
    username: '',
    fullName: '',
    password: '',
    confirmPassword: '',
    agreeToTerms: false,
  })

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [success, setSuccess] = useState<string>('')
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({})

  /**
   * Validate email format
   */
  const isValidEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  /**
   * Validate password strength
   */
  const validatePasswordStrength = (password: string): { isValid: boolean; message: string } => {
    if (password.length < 8) {
      return { isValid: false, message: 'Password must be at least 8 characters' }
    }

    const hasUpperCase = /[A-Z]/.test(password)
    const hasLowerCase = /[a-z]/.test(password)
    const hasNumbers = /\d/.test(password)
    const hasSpecialChar = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)

    if (!hasUpperCase || !hasLowerCase || !hasNumbers) {
      return {
        isValid: false,
        message: 'Password must contain uppercase, lowercase, and numbers',
      }
    }

    return { isValid: true, message: '' }
  }

  /**
   * Validate username format
   */
  const validateUsername = (username: string): string => {
    if (username.length < 3) {
      return 'Username must be at least 3 characters'
    }
    if (username.length > 30) {
      return 'Username must be less than 30 characters'
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
      return 'Username can only contain letters, numbers, underscores, and hyphens'
    }
    return ''
  }

  /**
   * Validate entire form
   */
  const validateForm = (): boolean => {
    const errors: ValidationErrors = {}

    // Email validation
    if (!formData.email.trim()) {
      errors.email = 'Email is required'
    } else if (!isValidEmail(formData.email)) {
      errors.email = 'Please enter a valid email address'
    }

    // Username validation
    if (!formData.username.trim()) {
      errors.username = 'Username is required'
    } else {
      const usernameError = validateUsername(formData.username)
      if (usernameError) {
        errors.username = usernameError
      }
    }

    // Full name validation
    if (!formData.fullName.trim()) {
      errors.fullName = 'Full name is required'
    } else if (formData.fullName.length < 2) {
      errors.fullName = 'Full name must be at least 2 characters'
    }

    // Password validation
    if (!formData.password) {
      errors.password = 'Password is required'
    } else {
      const passwordValidation = validatePasswordStrength(formData.password)
      if (!passwordValidation.isValid) {
        errors.password = passwordValidation.message
      }
    }

    // Confirm password validation
    if (!formData.confirmPassword) {
      errors.confirmPassword = 'Please confirm your password'
    } else if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match'
    }

    // Terms agreement validation
    if (!formData.agreeToTerms) {
      errors.agreeToTerms = 'You must agree to the terms and conditions'
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }

  /**
   * Handle input change
   */
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, checked, type } = e.target

    if (type === 'checkbox') {
      setFormData(prev => ({
        ...prev,
        [name]: checked,
      }))
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value,
      }))
    }

    // Clear validation error for this field
    if (validationErrors[name as keyof ValidationErrors]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: undefined,
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
    setSuccess('')

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/auth/register`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: formData.email,
            username: formData.username,
            full_name: formData.fullName,
            password: formData.password,
          }),
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Registration failed')
      }

      const data: RegisterResponse = await response.json()

      // Show success message
      setSuccess(
        'Registration successful! Redirecting to login page in 3 seconds...'
      )
      toast.success('Account created successfully!')

      // Redirect to login after 3 seconds
      setTimeout(() => {
        router.push('/login')
      }, 3000)
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'An error occurred during registration'
      setError(errorMessage)
      toast.error(errorMessage)
      console.error('Registration error:', err)
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
                  <PersonAdd sx={{ color: 'white', fontSize: 32 }} />
                </Box>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                Create Account
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Join the ORION Platform community
              </Typography>
            </Box>

            {/* Success Alert */}
            {success && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <Alert severity="success" sx={{ mb: 3, borderRadius: 1 }}>
                  {success}
                </Alert>
              </motion.div>
            )}

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

            {/* Registration Form */}
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
                  />
                </Grid>

                {/* Username Field */}
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Username"
                    name="username"
                    type="text"
                    value={formData.username}
                    onChange={handleInputChange}
                    error={!!validationErrors.username}
                    helperText={validationErrors.username}
                    placeholder="username"
                    disabled={loading}
                    InputProps={{
                      startAdornment: (
                        <Person sx={{ mr: 1.5, color: 'text.secondary' }} />
                      ),
                    }}
                  />
                </Grid>

                {/* Full Name Field */}
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Full Name"
                    name="fullName"
                    type="text"
                    value={formData.fullName}
                    onChange={handleInputChange}
                    error={!!validationErrors.fullName}
                    helperText={validationErrors.fullName}
                    placeholder="John Doe"
                    disabled={loading}
                    InputProps={{
                      startAdornment: (
                        <Person sx={{ mr: 1.5, color: 'text.secondary' }} />
                      ),
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
                    helperText={
                      validationErrors.password ||
                      'Min 8 chars, uppercase, lowercase, numbers'
                    }
                    placeholder="Enter a strong password"
                    disabled={loading}
                    InputProps={{
                      startAdornment: (
                        <Lock sx={{ mr: 1.5, color: 'text.secondary' }} />
                      ),
                    }}
                  />
                </Grid>

                {/* Confirm Password Field */}
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Confirm Password"
                    name="confirmPassword"
                    type="password"
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    error={!!validationErrors.confirmPassword}
                    helperText={validationErrors.confirmPassword}
                    placeholder="Confirm your password"
                    disabled={loading}
                    InputProps={{
                      startAdornment: (
                        <Lock sx={{ mr: 1.5, color: 'text.secondary' }} />
                      ),
                    }}
                  />
                </Grid>

                {/* Terms Agreement */}
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        name="agreeToTerms"
                        checked={formData.agreeToTerms}
                        onChange={handleInputChange}
                        disabled={loading}
                        color="primary"
                      />
                    }
                    label={
                      <Typography variant="body2">
                        I agree to the{' '}
                        <Link href="/terms" underline="hover">
                          Terms of Service
                        </Link>
                        {' '}and{' '}
                        <Link href="/privacy" underline="hover">
                          Privacy Policy
                        </Link>
                      </Typography>
                    }
                  />
                  {validationErrors.agreeToTerms && (
                    <Typography variant="caption" color="error">
                      {validationErrors.agreeToTerms}
                    </Typography>
                  )}
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
                        Creating Account...
                      </Box>
                    ) : (
                      'Create Account'
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
                      Already have an account?
                    </Typography>
                  </Box>
                </Grid>

                {/* Login Link */}
                <Grid item xs={12}>
                  <Typography align="center" variant="body2">
                    <Link
                      href="/login"
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
                      Sign in to your account
                    </Link>
                  </Typography>
                </Grid>
              </Grid>
            </Box>
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
