/**
 * Orchestrator Dashboard Page
 *
 * Session 30: Control Plane for Nanomaterials AGI
 */

"use client";

import React, { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  CircularProgress,
  LinearProgress,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import RefreshIcon from "@mui/icons-material/Refresh";
import SettingsIcon from "@mui/icons-material/Settings";

interface OrchestratorState {
  id: string;
  name: string;
  mode: string;
  is_active: boolean;
  run_count: number;
  total_simulations_launched: number;
  total_experiments_launched: number;
  total_trainings_launched: number;
  last_run_at: string | null;
  stats: {
    active_campaigns: number;
    pending_jobs: number;
    running_jobs: number;
    completed_jobs_last_24h: number;
    pending_experiments: number;
    total_structures: number;
    total_labeled_samples: number;
    models_ready_for_retrain: string[];
    campaigns_needing_attention: string[];
  } | null;
}

interface OrchestratorRun {
  id: string;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  success: boolean;
  triggered_by: string;
  actions: {
    campaigns_advanced: string[];
    simulations_launched: number;
    experiments_launched: number;
    models_retrained: string[];
  };
}

export default function OrchestratorPage() {
  const [state, setState] = useState<OrchestratorState | null>(null);
  const [runs, setRuns] = useState<OrchestratorRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchState = async () => {
    try {
      const response = await fetch("/api/orchestrator/state?name=default");
      if (!response.ok) throw new Error("Failed to fetch orchestrator state");
      const data = await response.json();
      setState(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const fetchRuns = async () => {
    try {
      const response = await fetch("/api/orchestrator/runs?name=default&limit=10");
      if (!response.ok) throw new Error("Failed to fetch orchestrator runs");
      const data = await response.json();
      setRuns(data);
    } catch (err) {
      console.error("Failed to fetch runs:", err);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    await Promise.all([fetchState(), fetchRuns()]);
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const handleRunOnce = async () => {
    setRunning(true);
    try {
      const response = await fetch("/api/orchestrator/run_once?name=default", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) throw new Error("Failed to run orchestrator");
      await fetchData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setRunning(false);
    }
  };

  const handleToggleActive = async () => {
    if (!state) return;
    const endpoint = state.is_active ? "deactivate" : "activate";
    try {
      const response = await fetch(`/api/orchestrator/${endpoint}?name=default`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`Failed to ${endpoint} orchestrator`);
      await fetchState();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  if (loading && !state) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Orchestrator Control Plane
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Central control for NANO-OS AGI - manages campaigns, schedules simulations, and triggers experiments
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Status Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Status
              </Typography>
              <Box display="flex" alignItems="center" gap={1}>
                <Chip
                  label={state?.is_active ? "ACTIVE" : "INACTIVE"}
                  color={state?.is_active ? "success" : "default"}
                  size="small"
                />
                <Chip label={state?.mode || "MANUAL"} size="small" variant="outlined" />
              </Box>
              <Typography variant="h6" sx={{ mt: 1 }}>
                {state?.run_count || 0} runs
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Active Campaigns
              </Typography>
              <Typography variant="h4">
                {state?.stats?.active_campaigns || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {state?.stats?.campaigns_needing_attention?.length || 0} need attention
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Jobs
              </Typography>
              <Typography variant="h4">
                {(state?.stats?.running_jobs || 0) + (state?.stats?.pending_jobs || 0)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {state?.stats?.completed_jobs_last_24h || 0} completed (24h)
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Launched
              </Typography>
              <Typography variant="body2">
                Simulations: {state?.total_simulations_launched || 0}
              </Typography>
              <Typography variant="body2">
                Experiments: {state?.total_experiments_launched || 0}
              </Typography>
              <Typography variant="body2">
                Trainings: {state?.total_trainings_launched || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Actions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Actions
          </Typography>
          <Box display="flex" gap={2} flexWrap="wrap">
            <Button
              variant="contained"
              startIcon={<PlayArrowIcon />}
              onClick={handleRunOnce}
              disabled={running || !state?.is_active}
            >
              {running ? "Running..." : "Run Orchestrator Step"}
            </Button>
            <Button
              variant="outlined"
              startIcon={state?.is_active ? <PauseIcon /> : <PlayArrowIcon />}
              onClick={handleToggleActive}
            >
              {state?.is_active ? "Deactivate" : "Activate"}
            </Button>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={fetchData}
            >
              Refresh
            </Button>
            <Button
              variant="outlined"
              startIcon={<SettingsIcon />}
              disabled
            >
              Configure
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* System Statistics */}
      {state?.stats && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={4}>
                <Typography variant="body2" color="text.secondary">
                  Total Structures
                </Typography>
                <Typography variant="h6">
                  {state.stats.total_structures}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <Typography variant="body2" color="text.secondary">
                  Labeled Samples
                </Typography>
                <Typography variant="h6">
                  {state.stats.total_labeled_samples}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <Typography variant="body2" color="text.secondary">
                  Pending Experiments
                </Typography>
                <Typography variant="h6">
                  {state.stats.pending_experiments}
                </Typography>
              </Grid>
            </Grid>

            {state.stats.models_ready_for_retrain.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Models Ready for Retraining
                </Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  {state.stats.models_ready_for_retrain.map((model) => (
                    <Chip key={model} label={model} color="warning" size="small" />
                  ))}
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      )}

      {/* Recent Runs */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Orchestrator Runs
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Started</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Triggered By</TableCell>
                  <TableCell>Campaigns</TableCell>
                  <TableCell>Simulations</TableCell>
                  <TableCell>Experiments</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {runs.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No runs yet
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  runs.map((run) => (
                    <TableRow key={run.id}>
                      <TableCell>
                        {new Date(run.started_at).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        {run.duration_seconds
                          ? `${run.duration_seconds.toFixed(1)}s`
                          : "-"}
                      </TableCell>
                      <TableCell>{run.triggered_by}</TableCell>
                      <TableCell>
                        {run.actions.campaigns_advanced.length}
                      </TableCell>
                      <TableCell>
                        {run.actions.simulations_launched}
                      </TableCell>
                      <TableCell>
                        {run.actions.experiments_launched}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={run.success ? "SUCCESS" : "FAILED"}
                          color={run.success ? "success" : "error"}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Container>
  );
}
