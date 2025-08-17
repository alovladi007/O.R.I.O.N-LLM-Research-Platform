'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import {
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  Avatar,
  IconButton,
  useTheme,
  useMediaQuery,
} from '@mui/material'
import {
  Science,
  Psychology,
  Memory,
  Speed,
  Security,
  CloudUpload,
  ArrowForward,
  GitHub,
  LinkedIn,
  Twitter,
  PlayArrow,
} from '@mui/icons-material'
import { HeroSection } from '@/components/landing/HeroSection'
import { FeaturesSection } from '@/components/landing/FeaturesSection'
import { DemoSection } from '@/components/landing/DemoSection'
import { TestimonialsSection } from '@/components/landing/TestimonialsSection'
import { PricingSection } from '@/components/landing/PricingSection'
import { CTASection } from '@/components/landing/CTASection'

const features = [
  {
    icon: <Psychology sx={{ fontSize: 40 }} />,
    title: 'AI-Powered Discovery',
    description: 'Leverage cutting-edge LLMs to generate novel material candidates based on desired properties.',
    color: '#3f51b5',
  },
  {
    icon: <Science sx={{ fontSize: 40 }} />,
    title: 'Simulation Integration',
    description: 'Seamlessly run DFT, molecular dynamics, and FEA simulations with automated workflows.',
    color: '#009688',
  },
  {
    icon: <Memory sx={{ fontSize: 40 }} />,
    title: 'Knowledge Graph',
    description: 'Navigate a comprehensive materials ontology with intelligent relationship mapping.',
    color: '#ff5722',
  },
  {
    icon: <Speed sx={{ fontSize: 40 }} />,
    title: 'Real-time Collaboration',
    description: 'Work together with your team in real-time with live updates and shared workspaces.',
    color: '#4caf50',
  },
  {
    icon: <Security sx={{ fontSize: 40 }} />,
    title: 'Enterprise Security',
    description: 'Bank-level encryption, OAuth2 authentication, and fine-grained access control.',
    color: '#673ab7',
  },
  {
    icon: <CloudUpload sx={{ fontSize: 40 }} />,
    title: 'Cloud-Native',
    description: 'Scalable infrastructure that grows with your research needs.',
    color: '#2196f3',
  },
]

const stats = [
  { value: '10M+', label: 'Materials in Database' },
  { value: '50K+', label: 'Simulations Run' },
  { value: '1000+', label: 'Research Teams' },
  { value: '99.9%', label: 'Uptime SLA' },
]

export default function HomePage() {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'))
  const router = useRouter()
  const [videoOpen, setVideoOpen] = useState(false)

  return (
    <>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
          color: 'white',
          pt: { xs: 8, md: 12 },
          pb: { xs: 6, md: 10 },
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Typography
                  variant="h1"
                  sx={{
                    fontSize: { xs: '2.5rem', md: '3.5rem' },
                    fontWeight: 700,
                    mb: 2,
                  }}
                >
                  Accelerate Materials Discovery with AI
                </Typography>
                <Typography
                  variant="h5"
                  sx={{
                    mb: 4,
                    opacity: 0.9,
                    fontWeight: 400,
                  }}
                >
                  From theoretical concepts to laboratory protocols in minutes, not months.
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Button
                    variant="contained"
                    size="large"
                    sx={{
                      bgcolor: 'white',
                      color: 'primary.main',
                      '&:hover': {
                        bgcolor: 'grey.100',
                      },
                    }}
                    endIcon={<ArrowForward />}
                    onClick={() => router.push('/signup')}
                  >
                    Start Free Trial
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    sx={{
                      borderColor: 'white',
                      color: 'white',
                      '&:hover': {
                        borderColor: 'white',
                        bgcolor: 'rgba(255,255,255,0.1)',
                      },
                    }}
                    startIcon={<PlayArrow />}
                    onClick={() => setVideoOpen(true)}
                  >
                    Watch Demo
                  </Button>
                </Box>
              </motion.div>
            </Grid>
            <Grid item xs={12} md={6}>
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Box
                  component="img"
                  src="/images/hero-dashboard.png"
                  alt="ORION Dashboard"
                  sx={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: 2,
                    boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
                  }}
                />
              </motion.div>
            </Grid>
          </Grid>
        </Container>
        {/* Decorative elements */}
        <Box
          sx={{
            position: 'absolute',
            top: -100,
            right: -100,
            width: 400,
            height: 400,
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.05)',
          }}
        />
      </Box>

      {/* Stats Section */}
      <Box sx={{ py: 8, bgcolor: 'background.paper' }}>
        <Container maxWidth="lg">
          <Grid container spacing={4}>
            {stats.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Box textAlign="center">
                    <Typography
                      variant="h3"
                      sx={{
                        fontWeight: 700,
                        color: 'primary.main',
                        mb: 1,
                      }}
                    >
                      {stat.value}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Box sx={{ py: 10, bgcolor: 'grey.50' }}>
        <Container maxWidth="lg">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
          >
            <Typography
              variant="h2"
              align="center"
              sx={{
                mb: 2,
                fontWeight: 700,
              }}
            >
              Everything You Need for Materials Research
            </Typography>
            <Typography
              variant="h6"
              align="center"
              color="text.secondary"
              sx={{ mb: 8, maxWidth: 800, mx: 'auto' }}
            >
              A comprehensive platform that combines AI, simulation, and collaboration tools
              to accelerate your materials discovery workflow.
            </Typography>
          </motion.div>

          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card
                    sx={{
                      height: '100%',
                      transition: 'transform 0.3s',
                      '&:hover': {
                        transform: 'translateY(-8px)',
                      },
                    }}
                  >
                    <CardContent sx={{ p: 4 }}>
                      <Avatar
                        sx={{
                          bgcolor: feature.color,
                          width: 80,
                          height: 80,
                          mb: 3,
                        }}
                      >
                        {feature.icon}
                      </Avatar>
                      <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                        {feature.title}
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        {feature.description}
                      </Typography>
                    </CardContent>
                  </Card>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Demo Section */}
      <DemoSection />

      {/* Testimonials */}
      <TestimonialsSection />

      {/* Pricing */}
      <PricingSection />

      {/* CTA Section */}
      <CTASection />
    </>
  )
}