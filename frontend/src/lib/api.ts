/**
 * Phase 9 / Session 9.1 — typed ORION API client.
 *
 * One axios instance, cookie-based auth (no localStorage), refresh-
 * on-401 interceptor with single-flight retry. All request / response
 * shapes are pulled from `src/types/api.generated.ts` which is
 * generated from the backend's live OpenAPI schema (see
 * `scripts/gen-api-types.sh`). Hand-written DTOs are deprecated.
 *
 * Auth model
 * ----------
 *
 * The frontend always uses the cookie-mode path:
 *   - POST /auth/login?mode=cookie         → server sets httpOnly cookies
 *   - all subsequent requests              → cookies travel automatically
 *     via `withCredentials: true`
 *   - POST /auth/refresh?mode=cookie       → server reads cookie, sets
 *     fresh ones
 *   - POST /auth/logout                    → server clears cookies
 *
 * No JWT ever lives in JavaScript memory longer than the response
 * promise — that's the whole point of cookie mode (Phase 9 / Session
 * 9.1 in the roadmap; httpOnly cookies are immune to XSS-driven
 * token theft, the primary risk of the old `localStorage` flow).
 *
 * Backward compatibility
 * ----------------------
 *
 * The bearer flow is still supported by the backend (curl / Postman
 * / Python SDK use it). This client doesn't expose bearer tokens —
 * if a non-browser caller needs them, it should hit the API
 * directly without going through this module.
 */

import axios, {
  type AxiosError,
  type AxiosInstance,
  type AxiosRequestConfig,
  type InternalAxiosRequestConfig,
} from 'axios';
import type { components, paths } from '@/types/api.generated';

// --------------------------------------------------------------------
// Generated-type re-exports — every page imports from `lib/api`, not
// directly from the generated file, so future schema renames are
// limited to this one module.
// --------------------------------------------------------------------

export type Schemas = components['schemas'];
export type Paths = paths;

export type LoginRequest = Schemas['UserLogin'];
export type Token = Schemas['Token'];
export type UserResponse = Schemas['UserResponse'];

// --------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------

const DEFAULT_BASE_URL = 'http://localhost:8002/api/v1';

/**
 * Resolved base URL.
 *
 * Order:
 *   1. `NEXT_PUBLIC_ORION_API_URL` env (set in `.env.local` for dev,
 *      set in CI / staging / prod via the Next.js build env).
 *   2. Hard-coded default `http://localhost:8002/api/v1`.
 */
function resolveBaseUrl(): string {
  if (typeof process !== 'undefined') {
    const fromEnv = process.env.NEXT_PUBLIC_ORION_API_URL;
    if (fromEnv && fromEnv.length > 0) return fromEnv;
  }
  return DEFAULT_BASE_URL;
}

// --------------------------------------------------------------------
// Axios instance
// --------------------------------------------------------------------

export const apiClient: AxiosInstance = axios.create({
  baseURL: resolveBaseUrl(),
  timeout: 30_000,
  // Required for the cookie-mode auth: instructs the browser to send
  // cookies cross-origin (Next.js dev → Uvicorn) and to accept the
  // Set-Cookie response header from the backend.
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --------------------------------------------------------------------
// Refresh-on-401 interceptor
//
// Single-flight: while a refresh is in progress, all other 401
// responses queue up on the same promise. After a successful
// refresh, queued requests retry with the fresh cookie. If the
// refresh itself fails or returns 401, every queued request rejects
// and we redirect to /login.
// --------------------------------------------------------------------

let refreshInFlight: Promise<void> | null = null;

async function refreshAccessToken(): Promise<void> {
  // The backend reads the orion_refresh_token cookie when ?mode=cookie
  // is set. No body required.
  await apiClient.post('/auth/refresh?mode=cookie');
}

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as
      | (InternalAxiosRequestConfig & { __retried?: boolean })
      | undefined;
    if (!error.response || !original) return Promise.reject(error);

    const status = error.response.status;
    const isAuthCall =
      original.url?.includes('/auth/login') ||
      original.url?.includes('/auth/refresh');

    // Only attempt refresh on 401 from a non-auth route, and only once.
    if (status !== 401 || isAuthCall || original.__retried) {
      if (status === 401 && typeof window !== 'undefined') {
        // Final 401 — bounce to login with the original path so we
        // can come back after re-auth.
        const next = encodeURIComponent(window.location.pathname + window.location.search);
        window.location.href = `/login?next=${next}`;
      }
      return Promise.reject(error);
    }

    original.__retried = true;
    try {
      if (!refreshInFlight) {
        refreshInFlight = refreshAccessToken().finally(() => {
          refreshInFlight = null;
        });
      }
      await refreshInFlight;
    } catch {
      // Refresh failed — propagate the original error so the caller's
      // catch block fires; the next 401 will redirect.
      return Promise.reject(error);
    }
    // Retry with the same config; the new cookies were set on the
    // refresh response and will travel automatically.
    return apiClient.request(original);
  },
);

// --------------------------------------------------------------------
// Error helpers
// --------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function formatErrorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { detail?: string } | undefined;
    return data?.detail ?? error.message ?? 'An error occurred';
  }
  if (error instanceof Error) return error.message;
  return 'An unknown error occurred';
}

function rethrow(err: unknown): never {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as { detail?: string } | undefined;
    throw new ApiError(
      data?.detail ?? err.message,
      err.response?.status,
      data,
    );
  }
  throw err;
}

// --------------------------------------------------------------------
// Typed wrappers
// --------------------------------------------------------------------

export const auth = {
  /**
   * Cookie-mode login. Sets httpOnly cookies on success and returns
   * the user payload (no body token usable by JS — that's by design).
   */
  async login(body: LoginRequest): Promise<Token> {
    try {
      const r = await apiClient.post<Token>(
        '/auth/login?mode=cookie',
        body,
      );
      return r.data;
    } catch (e) {
      rethrow(e);
    }
  },

  /**
   * Force a token refresh. The interceptor calls this automatically
   * on 401 — most callers don't need to.
   */
  async refresh(): Promise<void> {
    try {
      await apiClient.post('/auth/refresh?mode=cookie');
    } catch (e) {
      rethrow(e);
    }
  },

  async logout(): Promise<void> {
    try {
      await apiClient.post('/auth/logout');
    } catch {
      /* swallow — logout is best-effort */
    }
  },

  async me(): Promise<UserResponse> {
    try {
      const r = await apiClient.get<UserResponse>('/auth/me');
      return r.data;
    } catch (e) {
      rethrow(e);
    }
  },
};

// --------------------------------------------------------------------
// Generic typed-route helpers — for endpoints that don't have a
// dedicated wrapper yet. Pages should prefer the dedicated wrappers.
// --------------------------------------------------------------------

export async function getJson<T>(
  url: string,
  config?: AxiosRequestConfig,
): Promise<T> {
  try {
    const r = await apiClient.get<T>(url, config);
    return r.data;
  } catch (e) {
    rethrow(e);
  }
}

export async function postJson<T, B = unknown>(
  url: string,
  body: B,
  config?: AxiosRequestConfig,
): Promise<T> {
  try {
    const r = await apiClient.post<T>(url, body, config);
    return r.data;
  } catch (e) {
    rethrow(e);
  }
}

// --------------------------------------------------------------------
// Re-export the shape of every endpoint as a structured catalog.
// Pages can import `api.structures.list({page, ...})` etc. once
// the wrapper exists; the underlying types are pulled from `paths`.
// --------------------------------------------------------------------

import { structures } from './api-structures';

// Re-export auth-context hooks so pages have one import surface for
// "platform-wide concerns" (api wrappers + auth + role gate).
export { useAuth, useRequireRole } from './auth-context';
export { structures } from './api-structures';
export type {
  StructureListParams,
  StructureListResult,
  StructureResponse as StructureRow,
  StructureCreate as StructureCreateBody,
  StructureUpdate as StructureUpdateBody,
  StructureParseRequest,
  StructureParseResponse,
} from './api-structures';

export const api = {
  auth,
  structures,
  getJson,
  postJson,
  // jobs, ml, al, bo, agent — wrappers added by Sessions 9.3 / 9.4.
};

// --------------------------------------------------------------------
// Legacy named-exports (Session 9.1 compat layer)
//
// Re-exported from ./api-legacy so the existing structures / design /
// provenance pages keep compiling until 9.2 / 9.3 / 9.4 rewrite them.
// New code should NOT import these — use `api.*` instead.
// --------------------------------------------------------------------

export {
  getStructure,
  listStructures,
  downloadStructure,
  uploadStructureFile,
  getMaterial,
  runSimulation,
  predictProperties,
  downloadBlob,
  searchDesigns,
  getDesignStats,
  getProvenance,
  getProvenanceTimeline,
  getProvenanceSummary,
} from './api-legacy';

export default api;
