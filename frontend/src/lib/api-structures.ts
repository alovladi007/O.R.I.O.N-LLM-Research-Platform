/**
 * Phase 9 / Session 9.2 — typed wrappers for /api/v1/structures.
 *
 * One module per backend resource. Pages call ``api.structures.list({...})``
 * via `lib/api.ts` (which re-exports the namespace) and get back the
 * generated openapi types. No hand-written DTOs.
 *
 * Endpoints
 * ---------
 *   GET  /structures               → list({filters, sort, paging})
 *   GET  /structures/:id           → get(id)
 *   POST /structures               → create(body)
 *   PUT  /structures/:id           → update(id, body)
 *   DELETE /structures/:id         → remove(id)
 *   POST /structures/parse         → parse({text, format})
 *   GET  /structures/:id/export    → exportFile(id, format) → Blob
 */

import type { AxiosResponse } from 'axios'

import { apiClient, type Schemas } from './api'

// ---------- types pulled from the generated schema ------------------

export type StructureResponse = Schemas['StructureResponse']
export type StructureCreate = Schemas['StructureCreate']
export type StructureUpdate = Schemas['StructureUpdate']
export type StructureParseRequest = Schemas['StructureParseRequest']
export type StructureParseResponse = Schemas['StructureParseResponse']

// ---------- list parameters -----------------------------------------

export interface StructureListParams {
  formula?: string
  spacegroupNumber?: number
  spacegroupNumberMin?: number
  spacegroupNumberMax?: number
  densityMin?: number
  densityMax?: number
  numAtomsMin?: number
  numAtomsMax?: number
  dimensionality?: number
  format?: string
  materialId?: string
  sortBy?:
    | 'created_at'
    | 'formula'
    | 'density'
    | 'num_atoms'
    | 'spacegroup_number'
  sortDir?: 'asc' | 'desc'
  offset?: number
  limit?: number
}

export interface StructureListResult {
  items: StructureResponse[]
  total: number
}

// ---------- API surface ---------------------------------------------

function paramsForList(p: StructureListParams = {}): Record<string, unknown> {
  // Map camelCase JS keys to the snake_case query params the backend
  // declares. Skip undefined so the URL stays clean.
  const o: Record<string, unknown> = {}
  if (p.formula !== undefined) o.formula = p.formula
  if (p.spacegroupNumber !== undefined) o.spacegroup_number = p.spacegroupNumber
  if (p.spacegroupNumberMin !== undefined)
    o.spacegroup_number_min = p.spacegroupNumberMin
  if (p.spacegroupNumberMax !== undefined)
    o.spacegroup_number_max = p.spacegroupNumberMax
  if (p.densityMin !== undefined) o.density_min = p.densityMin
  if (p.densityMax !== undefined) o.density_max = p.densityMax
  if (p.numAtomsMin !== undefined) o.num_atoms_min = p.numAtomsMin
  if (p.numAtomsMax !== undefined) o.num_atoms_max = p.numAtomsMax
  if (p.dimensionality !== undefined) o.dimensionality = p.dimensionality
  if (p.format !== undefined) o.format = p.format
  if (p.materialId !== undefined) o.material_id = p.materialId
  if (p.sortBy !== undefined) o.sort_by = p.sortBy
  if (p.sortDir !== undefined) o.sort_dir = p.sortDir
  if (p.offset !== undefined) o.offset = p.offset
  if (p.limit !== undefined) o.limit = p.limit
  return o
}

export const structures = {
  async list(params: StructureListParams = {}): Promise<StructureListResult> {
    const r = await apiClient.get<StructureResponse[]>('/structures', {
      params: paramsForList(params),
    })
    const total = parseInt(
      r.headers['x-total-count'] ?? `${r.data.length}`,
      10,
    )
    return { items: r.data, total }
  },

  async get(id: string): Promise<StructureResponse> {
    const r = await apiClient.get<StructureResponse>(`/structures/${id}`)
    return r.data
  },

  async create(body: StructureCreate): Promise<StructureResponse> {
    const r = await apiClient.post<StructureResponse>('/structures', body)
    return r.data
  },

  async update(
    id: string,
    body: StructureUpdate,
  ): Promise<StructureResponse> {
    const r = await apiClient.put<StructureResponse>(
      `/structures/${id}`,
      body,
    )
    return r.data
  },

  async remove(id: string): Promise<void> {
    await apiClient.delete(`/structures/${id}`)
  },

  async parse(body: StructureParseRequest): Promise<StructureParseResponse> {
    const r = await apiClient.post<StructureParseResponse>(
      '/structures/parse',
      body,
    )
    return r.data
  },

  async exportFile(
    id: string,
    format: 'cif' | 'poscar' | 'xyz' | 'json' = 'cif',
  ): Promise<Blob> {
    const r: AxiosResponse<Blob> = await apiClient.get(
      `/structures/${id}/export`,
      {
        params: { format },
        responseType: 'blob',
      },
    )
    return r.data
  },
}
