// =============================================================================
// Axios Base Client — Dexmiq API with Authenticated Session & Auto-Refresh
// =============================================================================

import axios, { AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { tokenService } from './tokenService.ts';
import { refreshTokens } from './authService.ts';

/**
 * Pre-configured Axios instance pointing at the Dexmiq backend.
 *
 * Base URL resolution order:
 *   1. VITE_API_URL environment variable (Render production standard)
 *   2. VITE_API_BASE_URL environment variable (legacy project convention)
 *   3. Falls back to '/api' which Vite proxies to http://127.0.0.1:8000
 */
export function getApiBaseUrl(): string {
  const envBase =
    typeof import.meta !== 'undefined' && import.meta.env
      ? import.meta.env.VITE_API_BASE_URL
      : undefined;
  const envUrl =
    typeof import.meta !== 'undefined' && import.meta.env
      ? import.meta.env.VITE_API_URL
      : undefined;
  const rawUrl = envBase || envUrl || '/api';
  return rawUrl.replace(/\/+$/, '');
}

export const API_BASE_URL = getApiBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30_000, // 30 seconds
});

// ---------------------------------------------------------------------------
// Request Interceptor — attach JWT Bearer token
// ---------------------------------------------------------------------------
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = tokenService.getAccessToken();
    if (token && token !== 'undefined' && token !== 'null' && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ---------------------------------------------------------------------------
// Error Normalization Helper
// ---------------------------------------------------------------------------
function extractErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosErr = error as AxiosError<{ detail?: unknown }>;
    const detail = axiosErr.response?.data?.detail;
    if (detail) {
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((item) => (typeof item === 'object' && item && 'msg' in item ? (item as any).msg : JSON.stringify(item)))
          .join('; ');
      }
      return JSON.stringify(detail);
    }
    if (axiosErr.response?.status === 429) {
      const retryAfter = axiosErr.response.headers?.['retry-after'];
      return retryAfter
        ? `Rate limit exceeded. Please retry after ${retryAfter} seconds.`
        : 'Rate limit exceeded. Please try again shortly.';
    }
    if (axiosErr.message) {
      return axiosErr.message;
    }
  }
  return error instanceof Error ? error.message : String(error);
}

// ---------------------------------------------------------------------------
// Response Interceptor — automatic token refresh & error normalization
// ---------------------------------------------------------------------------
interface CustomRequestConfig extends InternalAxiosRequestConfig {
  _retry?: boolean;
}

apiClient.interceptors.response.use(
  (response) => response,
  async (error: unknown) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      const originalRequest = error.config as CustomRequestConfig;

      // Do NOT attempt refresh for /auth/ endpoints (login, signup, refresh itself)
      const url = originalRequest?.url || '';
      const isAuthEndpoint =
        url.includes('/auth/login') ||
        url.includes('/auth/signup') ||
        url.includes('/auth/refresh');

      if (!isAuthEndpoint && originalRequest && !originalRequest._retry) {
        originalRequest._retry = true;

        try {
          // Attempt token refresh via rotated refresh token
          const tokens = await refreshTokens();
          // Update original request with new access token and retry
          originalRequest.headers = originalRequest.headers || {};
          if (tokens?.access_token) {
            originalRequest.headers.Authorization = `Bearer ${tokens.access_token}`;
          }
          return apiClient(originalRequest);
        } catch (refreshErr) {
          return Promise.reject(new Error(extractErrorMessage(refreshErr)));
        }
      }
    }

    return Promise.reject(new Error(extractErrorMessage(error)));
  }
);

// ---------------------------------------------------------------------------
// Health Check (Phase 1 Connectivity Test)
// ---------------------------------------------------------------------------
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await apiClient.get('/health');
    return response.data?.status === 'ok';
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
}

export default apiClient;
