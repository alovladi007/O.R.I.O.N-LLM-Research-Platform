/**
 * Phase 9 / Session 9.1 — legacy hand-written API helpers.
 *
 * Pre-Phase-9 the frontend's `lib/api.ts` was a 600-line file of
 * hand-written wrappers and named exports. Session 9.1 replaced the
 * core (axios instance + auth + types pulled from the backend's
 * OpenAPI schema) but kept the legacy helper functions live in this
 * module so the existing structures / design / provenance pages
 * keep compiling.
 *
 * Sessions 9.2 (structures), 9.3 (jobs / workflows), and 9.4
 * (campaigns / ML) replace the relevant legacy wrappers as part of
 * each page rewrite. When this file is empty, delete it.
 */

import {
  apiClient,
  ApiError,
  formatErrorMessage as fmt,
} from './api';

import type {
  Structure,
  StructureListParams,
  StructureListResponse,
  Material,
} from '@/types/structures';
import type {
  DesignSearchRequest,
  DesignSearchResponse,
  DesignStats,
} from '@/types/design';
import type {
  ProvenanceChain,
  ProvenanceTimeline,
  ProvenanceSummary,
} from '@/types/provenance';

// Re-export the formatter so existing imports keep working.
export const formatErrorMessage = fmt;
export { ApiError };

// --------------------------------------------------------------------
// Structures (legacy — replaced by `api.structures.*` in Session 9.2)
// --------------------------------------------------------------------

export async function getStructure(id: string): Promise<Structure> {
  try {
    const r = await apiClient.get<Structure>(`/structures/${id}`);
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function listStructures(
  params?: StructureListParams,
): Promise<StructureListResponse> {
  try {
    const r = await apiClient.get<StructureListResponse>('/structures', { params });
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function downloadStructure(
  id: string,
  format: 'cif' | 'poscar' | 'xyz' | 'json' | 'xsf' = 'cif',
): Promise<Blob> {
  try {
    const r = await apiClient.get(`/structures/${id}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function uploadStructureFile(
  file: File,
  metadata?: Record<string, unknown>,
): Promise<Structure> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }
    const r = await apiClient.post<Structure>('/structures/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

// --------------------------------------------------------------------
// Materials (legacy)
// --------------------------------------------------------------------

export async function getMaterial(id: string): Promise<Material> {
  try {
    const r = await apiClient.get<Material>(`/materials/${id}`);
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

// --------------------------------------------------------------------
// Simulation / Prediction (legacy — still used by structures detail page)
// --------------------------------------------------------------------

export async function runSimulation(
  structureId: string,
  simulationType: string,
  parameters?: Record<string, unknown>,
): Promise<{ job_id: string }> {
  try {
    const r = await apiClient.post('/simulations', {
      structure_id: structureId,
      simulation_type: simulationType,
      parameters,
    });
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function predictProperties(
  structureId: string,
  properties: string[],
): Promise<unknown> {
  try {
    const r = await apiClient.post('/predictions', {
      structure_id: structureId,
      properties,
    });
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

/** Trigger a save-as on a Blob via a synthetic anchor click. */
export function downloadBlob(blob: Blob, filename: string): void {
  if (typeof window === 'undefined') return;
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

// --------------------------------------------------------------------
// Design (legacy — replaced by `api.campaigns.*` in Session 9.4)
// --------------------------------------------------------------------

export async function searchDesigns(
  request: DesignSearchRequest,
): Promise<DesignSearchResponse> {
  try {
    const r = await apiClient.post<DesignSearchResponse>('/design/search', request);
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function getDesignStats(): Promise<DesignStats> {
  try {
    const r = await apiClient.get<DesignStats>('/design/stats');
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

// --------------------------------------------------------------------
// Provenance (legacy)
// --------------------------------------------------------------------

export async function getProvenance(
  entityType: string,
  entityId: string,
): Promise<ProvenanceChain> {
  try {
    const r = await apiClient.get<ProvenanceChain>(
      `/provenance/${entityType}/${entityId}`,
    );
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function getProvenanceTimeline(
  entityType: string,
  entityId: string,
): Promise<ProvenanceTimeline> {
  try {
    const r = await apiClient.get<ProvenanceTimeline>(
      `/provenance/${entityType}/${entityId}/timeline`,
    );
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}

export async function getProvenanceSummary(
  entityType: string,
  entityId: string,
): Promise<ProvenanceSummary> {
  try {
    const r = await apiClient.get<ProvenanceSummary>(
      `/provenance/${entityType}/${entityId}/summary`,
    );
    return r.data;
  } catch (e) {
    throw new ApiError(formatErrorMessage(e));
  }
}
