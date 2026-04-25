// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.4 — /ml model registry browser.
 *
 * Lists models from /ml/models. Click → /ml/[id] for detail +
 * "Predict on structures" tab. Phase 6 ships RF / XGB / CGCNN
 * baselines today; the listing inherits whatever the backend has
 * registered.
 */

import { useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Stack,
  Typography,
} from '@mui/material'

import {
  api,
  formatErrorMessage,
  useRequireRole,
  type ModelInfoResponse,
} from '@/lib/api'

export default function MLRegistryPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const router = useRouter()
  const { data, isLoading, error } = useQuery({
    queryKey: ['ml-models'],
    queryFn: () => api.ml.listModels(),
    staleTime: 60_000,
  })

  return (
    <Box sx={{ maxWidth: 1300, mx: 'auto', py: 4, px: 3 }}>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        ML model registry
      </Typography>

      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}
      {error && <Alert severity="error">{formatErrorMessage(error)}</Alert>}

      {data && data.length === 0 && (
        <Alert severity="info" data-testid="ml-empty-state">
          No registered models. Train one via Session 6.4&apos;s CGCNN
          pipeline (<code>scripts/orion_train_cgcnn.py</code>) or one of
          the Session 6.3 baseline trainers.
        </Alert>
      )}

      <Stack spacing={2} data-testid="ml-models-list">
        {(data ?? []).map((m: ModelInfoResponse) => {
          const id = (m as any).model_id ?? m.name
          return (
            <Card
              key={id}
              variant="outlined"
              sx={{ cursor: 'pointer' }}
              onClick={() => router.push(`/ml/${id}`)}
              data-testid={`model-card-${id}`}
            >
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={2}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    {m.name}
                  </Typography>
                  <Chip label={`v${m.version}`} size="small" />
                  {m.available ? (
                    <Chip label="available" color="success" size="small" />
                  ) : (
                    <Chip label="unavailable" color="default" size="small" />
                  )}
                </Stack>
                {(m as any).description && (
                  <Typography variant="caption" color="text.secondary">
                    {(m as any).description}
                  </Typography>
                )}
              </CardContent>
            </Card>
          )
        })}
      </Stack>
    </Box>
  )
}
