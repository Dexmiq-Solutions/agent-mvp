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
      const token = localStorage.getItem(ACCESS_TOKEN_KEY);
      if (!token || token === 'undefined' || token === 'null' || token.trim() === '') {
        return null;
      }
      return token;
    } catch {
      return null;
    }
  },

  getRefreshToken(): string | null {
    try {
      const token = localStorage.getItem(REFRESH_TOKEN_KEY);
      if (!token || token === 'undefined' || token === 'null' || token.trim() === '') {
        return null;
      }
      return token;
    } catch {
      return null;
    }
  },

  setTokens(accessToken?: string | null, refreshToken?: string | null): void {
    try {
      if (accessToken && accessToken !== 'undefined' && accessToken !== 'null' && accessToken.trim() !== '') {
        localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
      } else {
        localStorage.removeItem(ACCESS_TOKEN_KEY);
      }
      if (refreshToken && refreshToken !== 'undefined' && refreshToken !== 'null' && refreshToken.trim() !== '') {
        localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
      } else {
        localStorage.removeItem(REFRESH_TOKEN_KEY);
      }
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
