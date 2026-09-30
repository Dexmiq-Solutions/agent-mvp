// =============================================================================
// Axios Base Client — Dexmiq API
// =============================================================================

import axios from 'axios';

/**
 * Pre-configured Axios instance pointing at the Dexmiq backend.
 *
 * Base URL resolution order:
 *   1. VITE_API_BASE_URL environment variable (set in .env or .env.local)
 *   2. Falls back to localhost:8000 for local development
 */
const apiClient = axios.create({
  // In dev, requests go to /api/* which Vite proxies to http://localhost:8000
  // In production, set VITE_API_BASE_URL to the real backend URL
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30_000, // 30 seconds — generous for synchronous LLM calls
});

// ---------------------------------------------------------------------------
// Response Interceptor — normalise errors into a consistent shape
// ---------------------------------------------------------------------------
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Surface the backend's `detail` field when available
    if (axios.isAxiosError(error) && error.response?.data?.detail) {
      const detail = error.response.data.detail;
      const message = typeof detail === 'string'
        ? detail
        : JSON.stringify(detail);
      return Promise.reject(new Error(message));
    }
    return Promise.reject(error);
  },
);

export default apiClient;
