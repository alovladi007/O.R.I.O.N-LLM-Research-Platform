'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardActionArea,
  TextField,
  InputAdornment,
  Stack,
  Chip,
  Pagination,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  Button,
  Autocomplete,
  Slider,
} from '@mui/material';
import {
  Search,
  FilterList,
  Add,
  ViewModule,
  ViewList,
  Sort,
} from '@mui/icons-material';
import { listStructures, formatErrorMessage } from '@/lib/api';
import { Structure, StructureListParams } from '@/types/structures';
import { getUniqueElements } from '@/utils/elementColors';

const ITEMS_PER_PAGE = 12;

export default function StructuresListPage() {
  const router = useRouter();
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filters, setFilters] = useState<StructureListParams>({
    skip: 0,
    limit: ITEMS_PER_PAGE,
    sort_by: 'formula',
    order: 'asc',
  });
  const [showFilters, setShowFilters] = useState(false);

  // Fetch structures
  const {
    data,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['structures', filters, searchQuery],
    queryFn: () => {
      const params = {
        ...filters,
        skip: (page - 1) * ITEMS_PER_PAGE,
        limit: ITEMS_PER_PAGE,
        ...(searchQuery && { formula: searchQuery }),
      };
      return listStructures(params);
    },
  });

  const totalPages = data ? Math.ceil(data.total / ITEMS_PER_PAGE) : 0;

  const handlePageChange = (_: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleStructureClick = (id: string) => {
    router.push(`/structures/${id}`);
  };

  const handleFilterChange = (key: keyof StructureListParams, value: any) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setPage(1);
  };

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setPage(1);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h4" component="h1" fontWeight="bold" gutterBottom>
              Crystal Structures
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Browse and analyze atomic structures from our database
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => router.push('/structures/upload')}
          >
            Upload Structure
          </Button>
        </Stack>
      </Box>

      {/* Search and Filters */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Stack spacing={2}>
          <Stack direction="row" spacing={2}>
            {/* Search */}
            <TextField
              fullWidth
              placeholder="Search by formula (e.g., Si, Fe2O3, GaN)"
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />

            {/* Sort */}
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={filters.sort_by || 'formula'}
                label="Sort By"
                onChange={(e) => handleFilterChange('sort_by', e.target.value)}
              >
                <MenuItem value="formula">Formula</MenuItem>
                <MenuItem value="num_atoms">Number of Atoms</MenuItem>
                <MenuItem value="band_gap">Band Gap</MenuItem>
                <MenuItem value="formation_energy">Formation Energy</MenuItem>
                <MenuItem value="volume">Volume</MenuItem>
              </Select>
            </FormControl>

            {/* Order */}
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Order</InputLabel>
              <Select
                value={filters.order || 'asc'}
                label="Order"
                onChange={(e) => handleFilterChange('order', e.target.value as 'asc' | 'desc')}
              >
                <MenuItem value="asc">Ascending</MenuItem>
                <MenuItem value="desc">Descending</MenuItem>
              </Select>
            </FormControl>

            {/* Toggle Filters */}
            <Button
              variant="outlined"
              startIcon={<FilterList />}
              onClick={() => setShowFilters(!showFilters)}
            >
              Filters
            </Button>
          </Stack>

          {/* Advanced Filters */}
          {showFilters && (
            <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Dimensionality</InputLabel>
                    <Select
                      value={filters.dimensionality || ''}
                      label="Dimensionality"
                      onChange={(e) => handleFilterChange('dimensionality', e.target.value || undefined)}
                    >
                      <MenuItem value="">All</MenuItem>
                      <MenuItem value={3}>3D</MenuItem>
                      <MenuItem value={2}>2D</MenuItem>
                      <MenuItem value={1}>1D</MenuItem>
                      <MenuItem value={0}>0D</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Crystal System</InputLabel>
                    <Select
                      value={filters.crystal_system || ''}
                      label="Crystal System"
                      onChange={(e) => handleFilterChange('crystal_system', e.target.value || undefined)}
                    >
                      <MenuItem value="">All</MenuItem>
                      <MenuItem value="cubic">Cubic</MenuItem>
                      <MenuItem value="tetragonal">Tetragonal</MenuItem>
                      <MenuItem value="orthorhombic">Orthorhombic</MenuItem>
                      <MenuItem value="hexagonal">Hexagonal</MenuItem>
                      <MenuItem value="trigonal">Trigonal</MenuItem>
                      <MenuItem value="monoclinic">Monoclinic</MenuItem>
                      <MenuItem value="triclinic">Triclinic</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Stability</InputLabel>
                    <Select
                      value={filters.is_stable !== undefined ? String(filters.is_stable) : ''}
                      label="Stability"
                      onChange={(e) => {
                        const value = e.target.value;
                        handleFilterChange('is_stable', value === '' ? undefined : value === 'true');
                      }}
                    >
                      <MenuItem value="">All</MenuItem>
                      <MenuItem value="true">Stable</MenuItem>
                      <MenuItem value="false">Unstable</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography gutterBottom>
                    Band Gap (eV): {filters.min_band_gap || 0} - {filters.max_band_gap || 10}
                  </Typography>
                  <Slider
                    value={[filters.min_band_gap || 0, filters.max_band_gap || 10]}
                    onChange={(_, value) => {
                      const [min, max] = value as number[];
                      handleFilterChange('min_band_gap', min);
                      handleFilterChange('max_band_gap', max);
                    }}
                    valueLabelDisplay="auto"
                    min={0}
                    max={10}
                    step={0.1}
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </Stack>
      </Paper>

      {/* Results */}
      {isLoading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh">
          <CircularProgress size={60} />
        </Box>
      ) : error ? (
        <Alert severity="error">
          Failed to load structures: {formatErrorMessage(error)}
        </Alert>
      ) : !data || data.items.length === 0 ? (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No structures found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your search criteria or filters
          </Typography>
        </Paper>
      ) : (
        <>
          {/* Results Count */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Showing {(page - 1) * ITEMS_PER_PAGE + 1} -{' '}
              {Math.min(page * ITEMS_PER_PAGE, data.total)} of {data.total} structures
            </Typography>
          </Box>

          {/* Grid View */}
          <Grid container spacing={3}>
            {data.items.map((structure) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={structure.id}>
                <StructureCard structure={structure} onClick={() => handleStructureClick(structure.id)} />
              </Grid>
            ))}
          </Grid>

          {/* Pagination */}
          {totalPages > 1 && (
            <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={handlePageChange}
                color="primary"
                size="large"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </>
      )}
    </Container>
  );
}

// Structure Card Component
const StructureCard: React.FC<{
  structure: Structure;
  onClick: () => void;
}> = ({ structure, onClick }) => {
  const elements = getUniqueElements(structure.atomic_species);

  return (
    <Card
      sx={{
        height: '100%',
        transition: 'all 0.3s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 4,
        },
      }}
    >
      <CardActionArea onClick={onClick} sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" component="div" gutterBottom fontWeight="bold">
            {structure.formula}
          </Typography>

          {structure.material_name && (
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {structure.material_name}
            </Typography>
          )}

          <Stack spacing={1.5} sx={{ mt: 2 }}>
            {/* Elements */}
            {elements.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">
                  Elements
                </Typography>
                <Stack direction="row" spacing={0.5} flexWrap="wrap" sx={{ mt: 0.5 }}>
                  {elements.map((el) => (
                    <Chip key={el} label={el} size="small" />
                  ))}
                </Stack>
              </Box>
            )}

            {/* Properties */}
            <Box>
              <Stack spacing={0.5}>
                {structure.num_atoms !== undefined && (
                  <Typography variant="caption">
                    <strong>Atoms:</strong> {structure.num_atoms}
                  </Typography>
                )}

                {structure.dimensionality !== undefined && (
                  <Typography variant="caption">
                    <strong>Dimensionality:</strong> {structure.dimensionality}D
                  </Typography>
                )}

                {structure.band_gap !== undefined && (
                  <Typography variant="caption">
                    <strong>Band Gap:</strong> {structure.band_gap.toFixed(2)} eV
                  </Typography>
                )}

                {structure.space_group && (
                  <Typography variant="caption">
                    <strong>Space Group:</strong> {structure.space_group}
                  </Typography>
                )}

                {structure.crystal_system && (
                  <Typography variant="caption">
                    <strong>System:</strong> {structure.crystal_system}
                  </Typography>
                )}
              </Stack>
            </Box>

            {/* Status Chips */}
            <Stack direction="row" spacing={0.5} flexWrap="wrap">
              {structure.is_stable !== undefined && (
                <Chip
                  label={structure.is_stable ? 'Stable' : 'Unstable'}
                  size="small"
                  color={structure.is_stable ? 'success' : 'warning'}
                  variant="outlined"
                />
              )}
              {structure.band_gap !== undefined && (
                <Chip
                  label={structure.band_gap > 0 ? 'Semiconductor' : 'Metal'}
                  size="small"
                  color={structure.band_gap > 0 ? 'info' : 'default'}
                  variant="outlined"
                />
              )}
            </Stack>
          </Stack>
        </CardContent>
      </CardActionArea>
    </Card>
  );
};
