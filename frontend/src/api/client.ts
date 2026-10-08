import { tokenService } from '../services/tokenService.ts';
import { refreshTokens } from '../services/authService.ts';

const envBase =
  typeof import.meta !== 'undefined' && import.meta.env
    ? import.meta.env.VITE_API_BASE_URL
    : undefined;
const envUrl =
  typeof import.meta !== 'undefined' && import.meta.env
    ? import.meta.env.VITE_API_URL
    : undefined;
const API_BASE = (envBase || envUrl || '').replace(/\/+$/, '');

async function parseErrorDetail(response: Response): Promise<string> {
  let errorDetail = `Request failed with status ${response.status}`;
  try {
    const errorJson = await response.json();
    if (errorJson.detail) {
      errorDetail = typeof errorJson.detail === 'string'
        ? errorJson.detail
        : Array.isArray(errorJson.detail)
          ? errorJson.detail.map((d: any) => (d && typeof d === 'object' && 'msg' in d ? d.msg : JSON.stringify(d))).join('; ')
          : JSON.stringify(errorJson.detail);
    }
  } catch {
    const text = await response.text().catch(() => '');
    if (text) errorDetail = text;
  }
  return errorDetail;
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
  
  const headers = new Headers(options.headers || {});
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const token = tokenService.getAccessToken();
  if (token && token !== 'undefined' && token !== 'null' && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  let response = await fetch(url, {
    ...options,
    headers,
  });

  // Handle 401 Unauthorized with token refresh and retry
  const isAuthEndpoint =
    endpoint.includes('/auth/login') ||
    endpoint.includes('/auth/signup') ||
    endpoint.includes('/auth/refresh');

  if (response.status === 401 && !isAuthEndpoint) {
    try {
      const tokens = await refreshTokens();
      if (tokens?.access_token) {
        headers.set('Authorization', `Bearer ${tokens.access_token}`);
      }
      response = await fetch(url, {
        ...options,
        headers,
      });
    } catch {
      // Refresh failed; throw the original or expiration error
      throw new Error('Session expired. Please log in again.');
    }
  }

  if (!response.ok) {
    const errorDetail = await parseErrorDetail(response);
    throw new Error(errorDetail);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export { API_BASE };
