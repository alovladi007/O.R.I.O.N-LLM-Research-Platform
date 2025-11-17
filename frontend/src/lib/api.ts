/**
 * API client for ORION platform
 * Provides typed functions for backend API calls
 */

import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';
import {
  Structure,
  StructureListParams,
  StructureListResponse,
  Material,
} from '@/types/structures';
import {
  DesignSearchRequest,
  DesignSearchResponse,
  DesignStats,
} from '@/types/design';
import {
  ProvenanceChain,
  ProvenanceTimeline,
  ProvenanceSummary,
} from '@/types/provenance';

// ==================== Authentication Types ====================

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  full_name: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user?: {
    id: string;
    email: string;
    username: string;
    full_name?: string;
  };
}

export interface User {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  created_at?: string;
  updated_at?: string;
}

// API base URL from environment variables
// Use empty string to make requests through Next.js proxy (avoiding CORS)
// The proxy is configured in next.config.js to forward /api/v1/* to the backend
const API_BASE_URL = typeof window !== 'undefined' && window.location.hostname === 'localhost'
  ? '' // Use Next.js proxy for local development
  : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

/**
 * Create configured axios instance
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor for adding auth token
 */
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor for error handling
 */
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data as any;

      if (status === 401) {
        // Unauthorized - redirect to login
        if (typeof window !== 'undefined') {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
      } else if (status === 403) {
        console.error('Forbidden:', data.detail || 'Access denied');
      } else if (status === 404) {
        console.error('Not found:', data.detail || 'Resource not found');
      } else if (status >= 500) {
        console.error('Server error:', data.detail || 'Internal server error');
      }
    } else if (error.request) {
      // Request made but no response
      console.error('Network error: No response from server');
    } else {
      // Error in request setup
      console.error('Request error:', error.message);
    }

    return Promise.reject(error);
  }
);

/**
 * Generic API error class
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Handle API errors consistently
 */
function handleApiError(error: any): never {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    const data = axiosError.response?.data as any;
    throw new ApiError(
      data?.detail || error.message,
      axiosError.response?.status,
      data
    );
  }
  throw error;
}

// ==================== Structure API ====================

/**
 * Get a single structure by ID
 */
export async function getStructure(id: string): Promise<Structure> {
  try {
    const response = await apiClient.get<Structure>(`/api/v1/structures/${id}`);
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * List structures with filtering and pagination
 */
export async function listStructures(
  params?: StructureListParams
): Promise<StructureListResponse> {
  try {
    const response = await apiClient.get<StructureListResponse>(
      '/api/v1/structures',
      { params }
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Search structures by formula or criteria
 */
export async function searchStructures(
  query: string,
  params?: Partial<StructureListParams>
): Promise<StructureListResponse> {
  try {
    const response = await apiClient.get<StructureListResponse>(
      '/api/v1/structures/search',
      {
        params: {
          q: query,
          ...params,
        },
      }
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Export structure in specified format
 */
export async function downloadStructure(
  id: string,
  format: 'cif' | 'poscar' | 'xyz' | 'json' | 'xsf' = 'cif'
): Promise<Blob> {
  try {
    const response = await apiClient.get(`/api/v1/structures/${id}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Create a new structure
 */
export async function createStructure(data: Partial<Structure>): Promise<Structure> {
  try {
    const response = await apiClient.post<Structure>('/api/v1/structures', data);
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Update an existing structure
 */
export async function updateStructure(
  id: string,
  data: Partial<Structure>
): Promise<Structure> {
  try {
    const response = await apiClient.put<Structure>(
      `/api/v1/structures/${id}`,
      data
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Delete a structure
 */
export async function deleteStructure(id: string): Promise<void> {
  try {
    await apiClient.delete(`/api/v1/structures/${id}`);
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Material API ====================

/**
 * Get a single material by ID
 */
export async function getMaterial(id: string): Promise<Material> {
  try {
    const response = await apiClient.get<Material>(`/api/v1/materials/${id}`);
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * List materials with filtering
 */
export async function listMaterials(params?: {
  skip?: number;
  limit?: number;
  formula?: string;
  elements?: string[];
}): Promise<{ items: Material[]; total: number }> {
  try {
    const response = await apiClient.get('/api/v1/materials', { params });
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Simulation API ====================

/**
 * Run simulation on a structure
 */
export async function runSimulation(
  structureId: string,
  simulationType: string,
  parameters?: Record<string, any>
): Promise<{ job_id: string }> {
  try {
    const response = await apiClient.post('/api/v1/simulations', {
      structure_id: structureId,
      simulation_type: simulationType,
      parameters,
    });
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Get simulation job status
 */
export async function getSimulationStatus(jobId: string): Promise<any> {
  try {
    const response = await apiClient.get(`/api/v1/simulations/${jobId}`);
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Prediction API ====================

/**
 * Predict properties for a structure
 */
export async function predictProperties(
  structureId: string,
  properties: string[]
): Promise<any> {
  try {
    const response = await apiClient.post('/api/v1/predictions', {
      structure_id: structureId,
      properties,
    });
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Upload API ====================

/**
 * Upload structure file (CIF, POSCAR, etc.)
 */
export async function uploadStructureFile(
  file: File,
  metadata?: Record<string, any>
): Promise<Structure> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    const response = await apiClient.post<Structure>(
      '/api/v1/structures/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Utility Functions ====================

/**
 * Download file from blob
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

/**
 * Format error message for display
 */
export function formatErrorMessage(error: any): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as any;
    return data?.detail || error.message || 'An error occurred';
  }
  return error.message || 'An unknown error occurred';
}

// ==================== Design API ====================

/**
 * Search for materials matching design criteria
 */
export async function searchDesigns(
  request: DesignSearchRequest
): Promise<DesignSearchResponse> {
  try {
    const response = await apiClient.post<DesignSearchResponse>(
      '/api/v1/design/search',
      request
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Get design search statistics
 */
export async function getDesignStats(): Promise<DesignStats> {
  try {
    const response = await apiClient.get<DesignStats>('/api/v1/design/stats');
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Provenance API ====================

/**
 * Get complete provenance chain for an entity
 */
export async function getProvenance(
  entityType: string,
  entityId: string
): Promise<ProvenanceChain> {
  try {
    const response = await apiClient.get<ProvenanceChain>(
      `/api/v1/provenance/${entityType}/${entityId}`
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Get provenance timeline for UI visualization
 */
export async function getProvenanceTimeline(
  entityType: string,
  entityId: string
): Promise<ProvenanceTimeline> {
  try {
    const response = await apiClient.get<ProvenanceTimeline>(
      `/api/v1/provenance/${entityType}/${entityId}/timeline`
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Get provenance summary statistics
 */
export async function getProvenanceSummary(
  entityType: string,
  entityId: string
): Promise<ProvenanceSummary> {
  try {
    const response = await apiClient.get<ProvenanceSummary>(
      `/api/v1/provenance/${entityType}/${entityId}/summary`
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Get provenance for a simulation job (convenience method)
 */
export async function getJobProvenance(jobId: string): Promise<ProvenanceChain> {
  try {
    const response = await apiClient.get<ProvenanceChain>(
      `/api/v1/provenance/job/${jobId}`
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Get provenance for a prediction (convenience method)
 */
export async function getPredictionProvenance(predictionId: string): Promise<ProvenanceChain> {
  try {
    const response = await apiClient.get<ProvenanceChain>(
      `/api/v1/provenance/prediction/${predictionId}`
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

// ==================== Authentication API ====================

/**
 * Login with email and password
 */
export async function login(request: LoginRequest): Promise<AuthResponse> {
  try {
    const response = await apiClient.post<AuthResponse>(
      '/api/v1/auth/login',
      request
    );
    // Store token in localStorage
    if (response.data.access_token && typeof window !== 'undefined') {
      localStorage.setItem('auth_token', response.data.access_token);
    }
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Register new user
 */
export async function register(request: RegisterRequest): Promise<AuthResponse> {
  try {
    const response = await apiClient.post<AuthResponse>(
      '/api/v1/auth/register',
      request
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Logout user (clear token)
 */
export async function logout(): Promise<void> {
  try {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
    // Optionally call backend logout endpoint
    await apiClient.post('/api/v1/auth/logout');
  } catch (error) {
    // Ignore logout errors, token is already cleared
    console.warn('Logout error:', error);
  }
}

/**
 * Get current user information
 */
export async function getCurrentUser(): Promise<User> {
  try {
    const response = await apiClient.get<User>('/api/v1/auth/me');
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Refresh authentication token
 */
export async function refreshToken(): Promise<AuthResponse> {
  try {
    const response = await apiClient.post<AuthResponse>('/api/v1/auth/refresh');
    // Store new token
    if (response.data.access_token && typeof window !== 'undefined') {
      localStorage.setItem('auth_token', response.data.access_token);
    }
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Request password reset
 */
export async function requestPasswordReset(email: string): Promise<{ message: string }> {
  try {
    const response = await apiClient.post<{ message: string }>(
      '/api/v1/auth/password-reset/request',
      { email }
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}

/**
 * Reset password with token
 */
export async function resetPassword(
  token: string,
  newPassword: string
): Promise<{ message: string }> {
  try {
    const response = await apiClient.post<{ message: string }>(
      '/api/v1/auth/password-reset/confirm',
      { token, new_password: newPassword }
    );
    return response.data;
  } catch (error) {
    return handleApiError(error);
  }
}
