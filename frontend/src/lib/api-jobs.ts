/**
 * Phase 9 / Session 9.3 — typed wrappers for /api/v1/jobs and
 * /api/v1/workflow-runs.
 */

import { apiClient, type Schemas } from './api'

// ---------- jobs ----------------------------------------------------

export type SimulationJobResponse = Schemas['SimulationJobResponse']

export interface JobListParams {
  status?: string
  engine?: string
  kind?: string
  ownerId?: string
  offset?: number
  limit?: number
}

export interface JobListResult {
  items: SimulationJobResponse[]
  total: number
}

export const jobs = {
  async list(params: JobListParams = {}): Promise<JobListResult> {
    const o: Record<string, unknown> = {}
    if (params.status) o.status = params.status
    if (params.engine) o.engine = params.engine
    if (params.kind) o.kind = params.kind
    if (params.ownerId) o.owner_id = params.ownerId
    if (params.offset !== undefined) o.offset = params.offset
    if (params.limit !== undefined) o.limit = params.limit
    const r = await apiClient.get<SimulationJobResponse[]>('/jobs', { params: o })
    const total = parseInt(r.headers['x-total-count'] ?? `${r.data.length}`, 10)
    return { items: r.data, total: isNaN(total) ? r.data.length : total }
  },

  async get(id: string): Promise<SimulationJobResponse> {
    const r = await apiClient.get<SimulationJobResponse>(`/jobs/${id}`)
    return r.data
  },

  /**
   * POST /jobs/{id}/cancel. Returns the updated job. The backend
   * returns 409 if the job is already in a terminal state — let
   * the caller's catch handle that.
   */
  async cancel(id: string): Promise<SimulationJobResponse> {
    const r = await apiClient.post<SimulationJobResponse>(`/jobs/${id}/cancel`)
    return r.data
  },

  /**
   * POST /jobs (re-run): submit a brand-new job whose inputs come
   * from the supplied job dict. Caller is expected to strip out
   * server-assigned fields (id, status, timestamps) before passing.
   */
  async create(body: unknown): Promise<SimulationJobResponse> {
    const r = await apiClient.post<SimulationJobResponse>('/jobs', body)
    return r.data
  },

  /** Tail static log (one-shot text body). */
  async logsText(id: string, tail = 200): Promise<string> {
    const r = await apiClient.get<string>(`/jobs/${id}/logs`, {
      params: { tail },
      responseType: 'text',
      transformResponse: (d) => d,
    })
    return r.data
  },
}

// ---------- workflow runs ------------------------------------------

export type WorkflowRunResponse = Schemas['WorkflowRunResponse']

export const workflowRuns = {
  async get(id: string): Promise<WorkflowRunResponse> {
    const r = await apiClient.get<WorkflowRunResponse>(`/workflow-runs/${id}`)
    return r.data
  },

  async cancel(id: string): Promise<WorkflowRunResponse> {
    const r = await apiClient.post<WorkflowRunResponse>(
      `/workflow-runs/${id}/cancel`,
    )
    return r.data
  },

  async manifest(id: string): Promise<unknown> {
    const r = await apiClient.get(`/workflow-runs/${id}/manifest`)
    return r.data
  },
}
