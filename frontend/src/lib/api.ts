import axios, { AxiosError, AxiosInstance } from 'axios';
import { Structure, StructureListParams } from '@/types/structures';
import { DesignSearchParams, DesignSearchResult, DesignStats } from '@/types/design';

// API base URL from environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor for auth tokens
apiClient.interceptors.request.use(
  (config) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - clear token and redirect to login
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Error message formatter
export function formatErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message || 'An error occurred';
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unknown error occurred';
}

// ===========================
// STRUCTURE APIs
// ===========================

export interface PaginatedStructures {
  items: Structure[];
  total: number;
  skip: number;
  limit: number;
}

export async function listStructures(params?: StructureListParams): Promise<PaginatedStructures> {
  const response = await apiClient.get('/structures', { params });
  return response.data;
}

export async function getStructure(id: string): Promise<Structure> {
  const response = await apiClient.get(`/structures/${id}`);
  return response.data;
}

export interface CreateStructureInput {
  formula: string;
  structure_data: string;
  format?: 'cif' | 'poscar' | 'xyz';
  tags?: string[];
  metadata?: Record<string, any>;
}

export async function createStructure(input: CreateStructureInput): Promise<Structure> {
  const response = await apiClient.post('/structures', input);
  return response.data;
}

export async function updateStructure(id: string, updates: Partial<Structure>): Promise<Structure> {
  const response = await apiClient.put(`/structures/${id}`, updates);
  return response.data;
}

export async function deleteStructure(id: string): Promise<void> {
  await apiClient.delete(`/structures/${id}`);
}

export interface ParseStructureInput {
  structure_data: string;
  format: 'cif' | 'poscar' | 'xyz';
}

export async function parseStructure(input: ParseStructureInput): Promise<Structure> {
  const response = await apiClient.post('/structures/parse', input);
  return response.data;
}

export async function exportStructure(id: string, format: 'cif' | 'poscar' | 'xyz'): Promise<string> {
  const response = await apiClient.get(`/structures/${id}/export`, {
    params: { format },
  });
  return response.data.structure_data;
}

// ===========================
// SIMULATION APIs
// ===========================

export interface SubmitSimulationInput {
  structure_id: string;
  simulation_type: 'dft' | 'md' | 'fea';
  parameters?: Record<string, any>;
}

export interface SimulationJob {
  id: string;
  structure_id: string;
  simulation_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  results?: Record<string, any>;
}

export async function submitSimulationJob(input: SubmitSimulationInput): Promise<SimulationJob> {
  const response = await apiClient.post('/jobs', input);
  return response.data;
}

export async function getSimulationJob(id: string): Promise<SimulationJob> {
  const response = await apiClient.get(`/jobs/${id}`);
  return response.data;
}

export async function listSimulationJobs(structureId?: string): Promise<SimulationJob[]> {
  const response = await apiClient.get('/jobs', {
    params: structureId ? { structure_id: structureId } : undefined,
  });
  return response.data.items;
}

// ===========================
// ML PREDICTION APIs
// ===========================

export interface PredictPropertiesInput {
  structure_id: string;
  properties: string[];
}

export interface PredictedProperty {
  property_name: string;
  predicted_value: number;
  uncertainty?: number;
  unit?: string;
}

export interface PredictionResult {
  structure_id: string;
  properties: PredictedProperty[];
  model_version: string;
  timestamp: string;
}

export async function predictProperties(input: PredictPropertiesInput): Promise<PredictionResult> {
  const response = await apiClient.post('/ml/predict', input);
  return response.data;
}

// ===========================
// DESIGN SEARCH APIs
// ===========================

export async function searchDesigns(params: DesignSearchParams): Promise<DesignSearchResult> {
  const response = await apiClient.post('/design/search', params);
  return response.data;
}

export async function getDesignStats(): Promise<DesignStats> {
  const response = await apiClient.get('/design/stats');
  return response.data;
}

export interface OptimizeDesignInput {
  target_properties: Record<string, { min?: number; max?: number; target?: number }>;
  constraints?: Record<string, any>;
  population_size?: number;
  generations?: number;
}

export async function optimizeDesign(input: OptimizeDesignInput): Promise<DesignSearchResult> {
  const response = await apiClient.post('/design/optimize', input);
  return response.data;
}

// ===========================
// PROVENANCE APIs
// ===========================

export interface ProvenanceRecord {
  id: string;
  entity_type: string;
  entity_id: string;
  action: string;
  actor: string;
  timestamp: string;
  metadata?: Record<string, any>;
  parent_id?: string;
}

export async function getProvenance(entityType: string, entityId: string): Promise<ProvenanceRecord[]> {
  const response = await apiClient.get(`/provenance/${entityType}/${entityId}`);
  return response.data;
}

export async function createProvenanceRecord(record: Omit<ProvenanceRecord, 'id' | 'timestamp'>): Promise<ProvenanceRecord> {
  const response = await apiClient.post('/provenance', record);
  return response.data;
}

// ===========================
// MATERIALS APIs
// ===========================

export interface Material {
  id: string;
  formula: string;
  name?: string;
  structure_id?: string;
  properties?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export async function listMaterials(params?: { formula?: string; limit?: number; skip?: number }): Promise<{ items: Material[]; total: number }> {
  const response = await apiClient.get('/materials', { params });
  return response.data;
}

export async function getMaterial(id: string): Promise<Material> {
  const response = await apiClient.get(`/materials/${id}`);
  return response.data;
}

// ===========================
// WORKFLOW APIs
// ===========================

export interface WorkflowTemplate {
  id: string;
  name: string;
  description?: string;
  steps: any[];
  created_at: string;
}

export async function listWorkflowTemplates(): Promise<WorkflowTemplate[]> {
  const response = await apiClient.get('/workflows/templates');
  return response.data.items;
}

export async function executeWorkflow(templateId: string, inputs: Record<string, any>): Promise<{ workflow_id: string }> {
  const response = await apiClient.post(`/workflows/templates/${templateId}/execute`, inputs);
  return response.data;
}

// ===========================
// AUTH APIs (placeholder)
// ===========================

export interface LoginInput {
  username: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    username: string;
    email: string;
  };
}

export async function login(input: LoginInput): Promise<AuthResponse> {
  const response = await apiClient.post('/auth/login', input);
  if (typeof window !== 'undefined') {
    localStorage.setItem('auth_token', response.data.access_token);
  }
  return response.data;
}

export async function logout(): Promise<void> {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('auth_token');
  }
  await apiClient.post('/auth/logout');
}

export async function getCurrentUser() {
  const response = await apiClient.get('/auth/me');
  return response.data;
}

export default apiClient;
