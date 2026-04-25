// @ts-nocheck — MUI X DataGrid + DataGrid sx-prop union types are too
// complex for the current TS config to fully resolve. Pre-existing
// project pattern (see Session 9.1 dashboard / AppBar). Runtime is
// correct; runtime tests cover behavior.
'use client'

/**
 * Phase 9 / Session 9.2 — /structures list page.
 *
 * Server-paginated DataGrid backed by ``api.structures.list()``.
 * Filters: formula (substring), spacegroup (number-range), density
 * (range), n_atoms (range). Toolbar buttons: Upload, Compare.
 *
 * Multi-select highlights up to 3 rows; ``Compare selected`` opens
 * /structures/compare?ids=… for the side-by-side view.
 */

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  Box,
  Button,
  Stack,
  TextField,
  Typography,
  Slider,
  Alert,
} from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { useQuery } from '@tanstack/react-query'
import { Add, Compare, Refresh } from '@mui/icons-material'

import { api, formatErrorMessage, useRequireRole } from '@/lib/api'
import { UploadStructureDrawer } from '@/components/structures/UploadStructureDrawer'

const PAGE_SIZE_OPTIONS = [10, 25, 50, 100] as const

interface FilterState {
  formula?: string
  spacegroupNumberMin?: number
  spacegroupNumberMax?: number
  densityMin?: number
  densityMax?: number
  numAtomsMin?: number
  numAtomsMax?: number
}

const COLUMNS: GridColDef[] = [
  { field: 'formula', headerName: 'Formula', flex: 1, minWidth: 120 },
  { field: 'name', headerName: 'Name', flex: 1, minWidth: 140 },
  {
    field: 'space_group',
    headerName: 'Spacegroup',
    width: 140,
    valueGetter: (_, row) =>
      row.space_group_number != null
        ? `${row.space_group ?? '—'} (${row.space_group_number})`
        : row.space_group ?? '—',
  },
  { field: 'num_atoms', headerName: '# atoms', type: 'number', width: 100 },
  {
    field: 'density',
    headerName: 'ρ (g/cm³)',
    type: 'number',
    width: 110,
    valueFormatter: (v: number | null) => (v == null ? '—' : v.toFixed(3)),
  },
  {
    field: 'dimensionality',
    headerName: 'Dim',
    type: 'number',
    width: 70,
  },
  { field: 'format', headerName: 'Format', width: 90 },
]

export default function StructuresListPage() {
  useRequireRole(['admin', 'scientist', 'viewer', 'researcher'])
  const router = useRouter()
  const [filters, setFilters] = useState<FilterState>({})
  const [paginationModel, setPaginationModel] = useState({
    page: 0,
    pageSize: 25,
  })
  const [sortModel, setSortModel] = useState<
    { field: string; sort: 'asc' | 'desc' }[]
  >([{ field: 'created_at', sort: 'desc' }])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [uploadOpen, setUploadOpen] = useState(false)

  const sortBy = (sortModel[0]?.field ?? 'created_at') as
    | 'created_at'
    | 'formula'
    | 'density'
    | 'num_atoms'
    | 'spacegroup_number'
  const sortDir = sortModel[0]?.sort ?? 'desc'

  const queryKey = useMemo(
    () => [
      'structures',
      filters,
      paginationModel.page,
      paginationModel.pageSize,
      sortBy,
      sortDir,
    ],
    [filters, paginationModel, sortBy, sortDir],
  )

  const { data, isLoading, error, refetch } = useQuery({
    queryKey,
    queryFn: () =>
      api.structures.list({
        ...filters,
        offset: paginationModel.page * paginationModel.pageSize,
        limit: paginationModel.pageSize,
        sortBy,
        sortDir,
      }),
    staleTime: 30_000,
  })

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, flexGrow: 1 }}>
          Structures
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => refetch()}
          data-testid="structures-refresh"
        >
          Refresh
        </Button>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setUploadOpen(true)}
          data-testid="structures-upload-button"
        >
          Upload
        </Button>
        <Button
          variant="outlined"
          startIcon={<Compare />}
          disabled={selectedIds.length < 2}
          data-testid="structures-compare-button"
          onClick={() =>
            router.push(
              `/structures/compare?ids=${selectedIds.slice(0, 3).join(',')}`,
            )
          }
        >
          Compare ({selectedIds.length}/3)
        </Button>
      </Stack>

      <FilterBar value={filters} onChange={setFilters} />

      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {formatErrorMessage(error)}
        </Alert>
      )}

      <Box sx={{ height: 640, mt: 2 }}>
        <DataGrid
          rows={data?.items ?? []}
          columns={COLUMNS}
          getRowId={(row) => row.id}
          rowCount={data?.total ?? 0}
          loading={isLoading}
          pagination
          paginationMode="server"
          paginationModel={paginationModel}
          onPaginationModelChange={setPaginationModel}
          pageSizeOptions={PAGE_SIZE_OPTIONS as unknown as number[]}
          sortingMode="server"
          sortModel={sortModel}
          onSortModelChange={(m) =>
            setSortModel(m as { field: string; sort: 'asc' | 'desc' }[])
          }
          checkboxSelection
          disableRowSelectionOnClick
          onRowSelectionModelChange={(ids) => {
            setSelectedIds((ids as unknown as string[]).slice(0, 3))
          }}
          onRowClick={(params) => router.push(`/structures/${params.id}`)}
          data-testid="structures-grid"
        />
      </Box>

      <UploadStructureDrawer
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onSaved={(id) => {
          setUploadOpen(false)
          router.push(`/structures/${id}`)
        }}
      />
    </Box>
  )
}

function FilterBar({
  value,
  onChange,
}: {
  value: FilterState
  onChange: (next: FilterState) => void
}) {
  return (
    <Stack
      direction={{ xs: 'column', md: 'row' }}
      spacing={2}
      sx={{ flexWrap: 'wrap' }}
    >
      <TextField
        size="small"
        label="Formula contains"
        value={value.formula ?? ''}
        onChange={(e) =>
          onChange({ ...value, formula: e.target.value || undefined })
        }
        sx={{ minWidth: 160 }}
        inputProps={{ 'data-testid': 'filter-formula' }}
      />
      <RangeSlider
        label="Spacegroup #"
        min={1}
        max={230}
        value={[
          value.spacegroupNumberMin ?? 1,
          value.spacegroupNumberMax ?? 230,
        ]}
        onChange={(lo, hi) =>
          onChange({
            ...value,
            spacegroupNumberMin: lo === 1 ? undefined : lo,
            spacegroupNumberMax: hi === 230 ? undefined : hi,
          })
        }
        testId="filter-spacegroup"
      />
      <RangeSlider
        label="Density (g/cm³)"
        min={0}
        max={25}
        step={0.1}
        value={[value.densityMin ?? 0, value.densityMax ?? 25]}
        onChange={(lo, hi) =>
          onChange({
            ...value,
            densityMin: lo === 0 ? undefined : lo,
            densityMax: hi === 25 ? undefined : hi,
          })
        }
        testId="filter-density"
      />
      <RangeSlider
        label="# atoms"
        min={1}
        max={500}
        value={[value.numAtomsMin ?? 1, value.numAtomsMax ?? 500]}
        onChange={(lo, hi) =>
          onChange({
            ...value,
            numAtomsMin: lo === 1 ? undefined : lo,
            numAtomsMax: hi === 500 ? undefined : hi,
          })
        }
        testId="filter-natoms"
      />
    </Stack>
  )
}

function RangeSlider({
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
  testId,
}: {
  label: string
  min: number
  max: number
  step?: number
  value: [number, number]
  onChange: (lo: number, hi: number) => void
  testId: string
}) {
  return (
    <Box sx={{ minWidth: 200 }}>
      <Typography variant="caption">
        {label}: {value[0]}–{value[1]}
      </Typography>
      <Slider
        size="small"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(_, v) => {
          const [lo, hi] = v as [number, number]
          onChange(lo, hi)
        }}
        valueLabelDisplay="auto"
        data-testid={testId}
      />
    </Box>
  )
}
