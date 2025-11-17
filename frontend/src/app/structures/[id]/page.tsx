'use client';

import React, { useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  Container,
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  Stack,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Download,
  PlayArrow,
  Psychology,
  ArrowBack,
  Share,
  Favorite,
  FavoriteBorder,
  MoreVert,
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import { StructureViewer3D } from '@/components/structures/StructureViewer3D';
import {
  getStructure,
  downloadStructure,
  runSimulation,
  predictProperties,
  downloadBlob,
  formatErrorMessage,
} from '@/lib/api';
import { Structure } from '@/types/structures';
import { CRYSTAL_SYSTEMS } from '@/types/structures';

export default function StructureDetailPage() {
  const params = useParams();
  const router = useRouter();
  const structureId = params.id as string;

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [simulationDialogOpen, setSimulationDialogOpen] = useState(false);
  const [predictionDialogOpen, setPredictionDialogOpen] = useState(false);
  const [isFavorite, setIsFavorite] = useState(false);

  // Fetch structure data
  const {
    data: structure,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['structure', structureId],
    queryFn: () => getStructure(structureId),
    enabled: !!structureId,
  });

  // Download mutation
  const downloadMutation = useMutation({
    mutationFn: ({ format }: { format: 'cif' | 'poscar' | 'xyz' | 'json' | 'xsf' }) =>
      downloadStructure(structureId, format),
    onSuccess: (blob, variables) => {
      const filename = `${structure?.formula || 'structure'}.${variables.format}`;
      downloadBlob(blob, filename);
      toast.success(`Downloaded ${filename}`);
    },
    onError: (error) => {
      toast.error(`Download failed: ${formatErrorMessage(error)}`);
    },
  });

  // Simulation mutation
  const simulationMutation = useMutation({
    mutationFn: ({ type, params }: { type: string; params?: any }) =>
      runSimulation(structureId, type, params),
    onSuccess: (data) => {
      toast.success(`Simulation job ${data.job_id} started`);
      setSimulationDialogOpen(false);
      router.push(`/simulations/${data.job_id}`);
    },
    onError: (error) => {
      toast.error(`Simulation failed: ${formatErrorMessage(error)}`);
    },
  });

  // Prediction mutation
  const predictionMutation = useMutation({
    mutationFn: ({ properties }: { properties: string[] }) =>
      predictProperties(structureId, properties),
    onSuccess: () => {
      toast.success('Property prediction started');
      setPredictionDialogOpen(false);
      refetch();
    },
    onError: (error) => {
      toast.error(`Prediction failed: ${formatErrorMessage(error)}`);
    },
  });

  const handleDownload = (format: 'cif' | 'poscar' | 'xyz' | 'json' | 'xsf') => {
    downloadMutation.mutate({ format });
    setAnchorEl(null);
  };

  const handleRunSimulation = (type: string) => {
    simulationMutation.mutate({ type });
  };

  const handlePredictProperties = (properties: string[]) => {
    predictionMutation.mutate({ properties });
  };

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  if (error || !structure) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          Failed to load structure: {formatErrorMessage(error)}
        </Alert>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => router.push('/structures')}
          sx={{ mt: 2 }}
        >
          Back to Structures
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
          <IconButton onClick={() => router.push('/structures')} size="small">
            <ArrowBack />
          </IconButton>
          <Typography variant="h4" component="h1" fontWeight="bold">
            {structure.formula}
          </Typography>
          <IconButton onClick={() => setIsFavorite(!isFavorite)}>
            {isFavorite ? <Favorite color="error" /> : <FavoriteBorder />}
          </IconButton>
          <IconButton>
            <Share />
          </IconButton>
          <Box sx={{ flexGrow: 1 }} />
          <Button
            variant="outlined"
            startIcon={<Download />}
            onClick={(e) => setAnchorEl(e.currentTarget)}
          >
            Download
          </Button>
          <Button
            variant="outlined"
            startIcon={<PlayArrow />}
            onClick={() => setSimulationDialogOpen(true)}
          >
            Run Simulation
          </Button>
          <Button
            variant="contained"
            startIcon={<Psychology />}
            onClick={() => setPredictionDialogOpen(true)}
          >
            Predict Properties
          </Button>
        </Stack>

        {structure.material_name && (
          <Typography variant="h6" color="text.secondary">
            {structure.material_name}
          </Typography>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* 3D Viewer */}
        <Grid item xs={12} lg={8}>
          <StructureViewer3D structure={structure} height={600} />
        </Grid>

        {/* Properties Panel */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={2}>
            {/* Basic Information */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Basic Information
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Stack spacing={1.5}>
                  <PropertyRow label="Formula" value={structure.formula} />
                  {structure.space_group && (
                    <PropertyRow
                      label="Space Group"
                      value={`${structure.space_group} (${structure.space_group_number})`}
                    />
                  )}
                  {structure.crystal_system && (
                    <PropertyRow
                      label="Crystal System"
                      value={CRYSTAL_SYSTEMS[structure.crystal_system]?.name || structure.crystal_system}
                    />
                  )}
                  {structure.dimensionality !== undefined && (
                    <PropertyRow
                      label="Dimensionality"
                      value={`${structure.dimensionality}D`}
                    />
                  )}
                  <PropertyRow
                    label="Number of Atoms"
                    value={structure.num_atoms || structure.atomic_positions?.length || 0}
                  />
                  {structure.volume && (
                    <PropertyRow
                      label="Volume"
                      value={`${structure.volume.toFixed(2)} Ų`}
                    />
                  )}
                  {structure.density && (
                    <PropertyRow
                      label="Density"
                      value={`${structure.density.toFixed(2)} g/cm³`}
                    />
                  )}
                </Stack>
              </CardContent>
            </Card>

            {/* Lattice Parameters */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Lattice Parameters
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Stack spacing={1.5}>
                  {structure.a && <PropertyRow label="a" value={`${structure.a.toFixed(4)} Å`} />}
                  {structure.b && <PropertyRow label="b" value={`${structure.b.toFixed(4)} Å`} />}
                  {structure.c && <PropertyRow label="c" value={`${structure.c.toFixed(4)} Å`} />}
                  {structure.alpha && <PropertyRow label="α" value={`${structure.alpha.toFixed(2)}°`} />}
                  {structure.beta && <PropertyRow label="β" value={`${structure.beta.toFixed(2)}°`} />}
                  {structure.gamma && <PropertyRow label="γ" value={`${structure.gamma.toFixed(2)}°`} />}
                </Stack>
              </CardContent>
            </Card>

            {/* Electronic Properties */}
            {(structure.band_gap !== undefined || structure.formation_energy !== undefined) && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Electronic Properties
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Stack spacing={1.5}>
                    {structure.band_gap !== undefined && (
                      <PropertyRow
                        label="Band Gap"
                        value={
                          <Stack direction="row" spacing={1} alignItems="center">
                            <span>{structure.band_gap.toFixed(3)} eV</span>
                            {structure.is_gap_direct !== undefined && (
                              <Chip
                                label={structure.is_gap_direct ? 'Direct' : 'Indirect'}
                                size="small"
                                color={structure.is_gap_direct ? 'success' : 'default'}
                              />
                            )}
                          </Stack>
                        }
                      />
                    )}
                    {structure.formation_energy !== undefined && (
                      <PropertyRow
                        label="Formation Energy"
                        value={`${structure.formation_energy.toFixed(3)} eV/atom`}
                      />
                    )}
                    {structure.energy_above_hull !== undefined && (
                      <PropertyRow
                        label="Energy Above Hull"
                        value={`${structure.energy_above_hull.toFixed(3)} eV/atom`}
                      />
                    )}
                    {structure.is_stable !== undefined && (
                      <PropertyRow
                        label="Stability"
                        value={
                          <Chip
                            label={structure.is_stable ? 'Stable' : 'Unstable'}
                            size="small"
                            color={structure.is_stable ? 'success' : 'warning'}
                          />
                        }
                      />
                    )}
                  </Stack>
                </CardContent>
              </Card>
            )}

            {/* Magnetic Properties */}
            {(structure.magnetic_ordering || structure.total_magnetization !== undefined) && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Magnetic Properties
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Stack spacing={1.5}>
                    {structure.magnetic_ordering && (
                      <PropertyRow label="Magnetic Ordering" value={structure.magnetic_ordering} />
                    )}
                    {structure.total_magnetization !== undefined && (
                      <PropertyRow
                        label="Total Magnetization"
                        value={`${structure.total_magnetization.toFixed(3)} μB`}
                      />
                    )}
                  </Stack>
                </CardContent>
              </Card>
            )}
          </Stack>
        </Grid>
      </Grid>

      {/* Download Menu */}
      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => setAnchorEl(null)}>
        <MenuItem onClick={() => handleDownload('cif')}>CIF Format</MenuItem>
        <MenuItem onClick={() => handleDownload('poscar')}>POSCAR (VASP)</MenuItem>
        <MenuItem onClick={() => handleDownload('xyz')}>XYZ Format</MenuItem>
        <MenuItem onClick={() => handleDownload('json')}>JSON Format</MenuItem>
        <MenuItem onClick={() => handleDownload('xsf')}>XSF Format</MenuItem>
      </Menu>

      {/* Simulation Dialog */}
      <SimulationDialog
        open={simulationDialogOpen}
        onClose={() => setSimulationDialogOpen(false)}
        onSubmit={handleRunSimulation}
        isLoading={simulationMutation.isPending}
      />

      {/* Prediction Dialog */}
      <PredictionDialog
        open={predictionDialogOpen}
        onClose={() => setPredictionDialogOpen(false)}
        onSubmit={handlePredictProperties}
        isLoading={predictionMutation.isPending}
      />
    </Container>
  );
}

// Helper component for property rows
const PropertyRow: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
  <Box>
    <Typography variant="caption" color="text.secondary" display="block">
      {label}
    </Typography>
    <Typography variant="body2" fontWeight="medium">
      {value}
    </Typography>
  </Box>
);

// Simulation Dialog Component
const SimulationDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  onSubmit: (type: string) => void;
  isLoading: boolean;
}> = ({ open, onClose, onSubmit, isLoading }) => {
  const [simulationType, setSimulationType] = useState('dft');

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Run Simulation</DialogTitle>
      <DialogContent>
        <FormControl fullWidth sx={{ mt: 2 }}>
          <InputLabel>Simulation Type</InputLabel>
          <Select
            value={simulationType}
            label="Simulation Type"
            onChange={(e) => setSimulationType(e.target.value)}
          >
            <MenuItem value="dft">DFT (Density Functional Theory)</MenuItem>
            <MenuItem value="md">Molecular Dynamics</MenuItem>
            <MenuItem value="phonon">Phonon Calculations</MenuItem>
            <MenuItem value="elastic">Elastic Constants</MenuItem>
          </Select>
        </FormControl>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isLoading}>
          Cancel
        </Button>
        <Button
          onClick={() => onSubmit(simulationType)}
          variant="contained"
          disabled={isLoading}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Start Simulation'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// Prediction Dialog Component
const PredictionDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  onSubmit: (properties: string[]) => void;
  isLoading: boolean;
}> = ({ open, onClose, onSubmit, isLoading }) => {
  const [selectedProperties, setSelectedProperties] = useState<string[]>(['band_gap']);

  const properties = [
    { value: 'band_gap', label: 'Band Gap' },
    { value: 'formation_energy', label: 'Formation Energy' },
    { value: 'bulk_modulus', label: 'Bulk Modulus' },
    { value: 'shear_modulus', label: 'Shear Modulus' },
  ];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Predict Properties</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Select properties to predict using ML models
        </Typography>
        <Stack spacing={1}>
          {properties.map((prop) => (
            <Box key={prop.value}>
              <input
                type="checkbox"
                id={prop.value}
                checked={selectedProperties.includes(prop.value)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedProperties([...selectedProperties, prop.value]);
                  } else {
                    setSelectedProperties(selectedProperties.filter((p) => p !== prop.value));
                  }
                }}
              />
              <label htmlFor={prop.value} style={{ marginLeft: 8 }}>
                {prop.label}
              </label>
            </Box>
          ))}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isLoading}>
          Cancel
        </Button>
        <Button
          onClick={() => onSubmit(selectedProperties)}
          variant="contained"
          disabled={isLoading || selectedProperties.length === 0}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Predict'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
