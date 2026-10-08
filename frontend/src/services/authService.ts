// =============================================================================
// Auth Service — Authentication, Registration, Session, and Token Refresh
// =============================================================================

import axios from 'axios';
import apiClient, { getApiBaseUrl } from './api.ts';
import { tokenService } from './tokenService.ts';
import type {
  UserLoginRequest,
  UserSignupRequest,
  TokenResponse,
  UserResponse,
} from '../types/index.ts';

type AuthExpiredCallback = () => void;
const authExpiredListeners = new Set<AuthExpiredCallback>();

export function onAuthExpired(callback: AuthExpiredCallback): () => void {
  authExpiredListeners.add(callback);
  return () => {
    authExpiredListeners.delete(callback);
  };
}

export function notifyAuthExpired(): void {
  authExpiredListeners.forEach((callback) => {
    try {
      callback();
    } catch (e) {
      console.error('Error in auth expired listener:', e);
    }
  });
}

// In-flight refresh mutex to prevent duplicate refresh calls when concurrent requests return 401
let refreshPromise: Promise<TokenResponse> | null = null;

export async function signup(payload: UserSignupRequest): Promise<UserResponse> {
  const { data } = await apiClient.post<UserResponse>('/auth/signup', {
    email: payload.email.trim().toLowerCase(),
    password: payload.password,
  });
  return data;
}

export async function login(payload: UserLoginRequest): Promise<TokenResponse> {
  const { data } = await apiClient.post<TokenResponse>('/auth/login', {
    email: payload.email.trim().toLowerCase(),
    password: payload.password,
  });

  tokenService.setTokens(data.access_token, data.refresh_token);
  return data;
}

export async function refreshTokens(): Promise<TokenResponse> {
  // If a refresh is already in-flight, reuse it
  if (refreshPromise) {
    return refreshPromise;
  }

  const currentRefreshToken = tokenService.getRefreshToken();
  if (!currentRefreshToken) {
    tokenService.clearTokens();
    notifyAuthExpired();
    return Promise.reject(new Error('No refresh token available.'));
  }

  const cleanBase = getApiBaseUrl();

  refreshPromise = (async () => {
    try {
      const response = await axios.post<TokenResponse>(
        `${cleanBase}/auth/refresh`,
        { refresh_token: currentRefreshToken },
        { headers: { 'Content-Type': 'application/json' } }
      );

      const tokens = response.data;
      tokenService.setTokens(tokens.access_token, tokens.refresh_token);
      return tokens;
    } catch (error) {
      tokenService.clearTokens();
      notifyAuthExpired();
      throw error;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

export async function logout(): Promise<void> {
  const currentRefreshToken = tokenService.getRefreshToken();
  if (currentRefreshToken) {
    try {
      await apiClient.post('/auth/logout', {
        refresh_token: currentRefreshToken,
      });
    } catch (e) {
      // Don't let backend logout error block client-side token wipe
      console.warn('Backend logout encountered error (cleaning client session anyway):', e);
    }
  }

  tokenService.clearTokens();
  notifyAuthExpired();
}

export async function getCurrentUser(): Promise<UserResponse> {
  const { data } = await apiClient.get<UserResponse>('/auth/me');
  return data;
}
