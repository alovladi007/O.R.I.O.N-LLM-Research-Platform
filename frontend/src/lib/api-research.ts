/**
 * Phase 9 / Session 9.4 — typed wrappers for /al, /bo, /ml.
 */

import { apiClient, type Schemas } from './api'

// ---------- AL --------------------------------------------------

export type ALCampaignCreate = Schemas['ALCampaignCreate']
export type ALCampaignResponse = Schemas['ALCampaignResponse']
export type ALCycleResponse = Schemas['ALCycleResponse']

export const al = {
  async list(): Promise<ALCampaignResponse[]> {
    const r = await apiClient.get<ALCampaignResponse[]>('/al/campaigns')
    return r.data
  },

  async create(body: ALCampaignCreate): Promise<ALCampaignResponse> {
    const r = await apiClient.post<ALCampaignResponse>('/al/campaigns', body)
    return r.data
  },

  async get(id: string): Promise<ALCampaignResponse> {
    const r = await apiClient.get<ALCampaignResponse>(`/al/campaigns/${id}`)
    return r.data
  },
}

// ---------- BO --------------------------------------------------

export type BOSuggestRequest = Schemas['BOSuggestRequest']
export type BOSuggestResponse = Schemas['BOSuggestResponse']
export type ParetoFrontRequest = Schemas['ParetoFrontRequest']
export type ParetoFrontResponse = Schemas['ParetoFrontResponse']

export const bo = {
  async suggest(body: BOSuggestRequest): Promise<BOSuggestResponse> {
    const r = await apiClient.post<BOSuggestResponse>('/bo/suggest', body)
    return r.data
  },

  async paretoFront(body: ParetoFrontRequest): Promise<ParetoFrontResponse> {
    const r = await apiClient.post<ParetoFrontResponse>(
      '/bo/pareto-front',
      body,
    )
    return r.data
  },
}

// ---------- ML --------------------------------------------------

export type ModelInfoResponse = Schemas['ModelInfoResponse']
export type PropertyPredictionRequest = Schemas['PropertyPredictionRequest']
export type PropertyPredictionResponse = Schemas['PropertyPredictionResponse']

export const ml = {
  async listModels(): Promise<ModelInfoResponse[]> {
    const r = await apiClient.get<ModelInfoResponse[]>('/ml/models')
    return r.data
  },

  async getModel(modelId: string): Promise<ModelInfoResponse> {
    const r = await apiClient.get<ModelInfoResponse>(`/ml/models/${modelId}`)
    return r.data
  },

  async predict(
    body: PropertyPredictionRequest,
  ): Promise<PropertyPredictionResponse> {
    const r = await apiClient.post<PropertyPredictionResponse>(
      '/ml/properties',
      body,
    )
    return r.data
  },
}
