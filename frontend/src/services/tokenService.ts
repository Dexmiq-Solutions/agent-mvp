// =============================================================================
// Token Storage Service — Centralized Access & Refresh Token Management
// =============================================================================

const ACCESS_TOKEN_KEY = 'agent_mvp_access_token';
const REFRESH_TOKEN_KEY = 'agent_mvp_refresh_token';

/**
 * Safely access and persist authentication tokens in browser storage.
 * Does not expose or log tokens.
 */
export const tokenService = {
  getAccessToken(): string | null {
    try {
      return localStorage.getItem(ACCESS_TOKEN_KEY);
    } catch {
      return null;
    }
  },

  getRefreshToken(): string | null {
    try {
      return localStorage.getItem(REFRESH_TOKEN_KEY);
    } catch {
      return null;
    }
  },

  setTokens(accessToken: string, refreshToken: string): void {
    try {
      localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
      localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
    } catch (e) {
      console.error('Failed to persist auth tokens:', e);
    }
  },

  clearTokens(): void {
    try {
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
    } catch (e) {
      console.error('Failed to clear auth tokens:', e);
    }
  },

  hasTokens(): boolean {
    return Boolean(this.getAccessToken() || this.getRefreshToken());
  },
};
