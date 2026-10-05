// =============================================================================
// Frontend Authentication & Token Integration Tests
// =============================================================================

import test from 'node:test';
import assert from 'node:assert/strict';

// Helper to simulate global fetch
function setupMockFetch(responder) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    return responder(url, options);
  };
  return () => {
    globalThis.fetch = originalFetch;
  };
}

// Mock localStorage for Node.js test environment
function setupMockLocalStorage() {
  const store = new Map();
  const originalLocalStorage = globalThis.localStorage;

  globalThis.localStorage = {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
    removeItem(key) {
      store.delete(key);
    },
    clear() {
      store.clear();
    },
  };

  return () => {
    globalThis.localStorage = originalLocalStorage;
  };
}

test('tokenService - safely stores, retrieves, and clears tokens', async () => {
  const restoreStorage = setupMockLocalStorage();

  try {
    const { tokenService } = await import('../src/services/tokenService.ts');
    
    // Initially empty
    tokenService.clearTokens();
    assert.equal(tokenService.getAccessToken(), null);
    assert.equal(tokenService.getRefreshToken(), null);
    assert.equal(tokenService.hasTokens(), false);

    // Set tokens
    tokenService.setTokens('access-token-123', 'refresh-token-456');
    assert.equal(tokenService.getAccessToken(), 'access-token-123');
    assert.equal(tokenService.getRefreshToken(), 'refresh-token-456');
    assert.equal(tokenService.hasTokens(), true);

    // Clear tokens
    tokenService.clearTokens();
    assert.equal(tokenService.getAccessToken(), null);
    assert.equal(tokenService.getRefreshToken(), null);
    assert.equal(tokenService.hasTokens(), false);
  } finally {
    restoreStorage();
  }
});

test('apiRequest - attaches Bearer token from tokenService automatically', async () => {
  const restoreStorage = setupMockLocalStorage();
  let capturedAuthHeader = null;

  const restoreFetch = setupMockFetch(async (_url, options) => {
    const headers = new Headers(options?.headers || {});
    capturedAuthHeader = headers.get('Authorization');
    return {
      ok: true,
      status: 200,
      json: async () => ({ status: 'ok' }),
    };
  });

  try {
    const { tokenService } = await import('../src/services/tokenService.ts');
    tokenService.setTokens('jwt-test-token', 'refresh-test-token');

    const { apiRequest } = await import('../src/api/client.ts');
    await apiRequest('/test-endpoint');

    assert.equal(capturedAuthHeader, 'Bearer jwt-test-token');
  } finally {
    restoreFetch();
    restoreStorage();
  }
});

test('authService - login stores tokens and getCurrentUser sends Bearer token', async () => {
  const restoreStorage = setupMockLocalStorage();
  const { default: apiClient } = await import('../src/services/api.ts');
  const originalAdapter = apiClient.defaults.adapter;

  apiClient.defaults.adapter = async (config) => {
    const url = String(config.url || '');
    if (url.includes('/auth/login')) {
      return {
        data: {
          access_token: 'new-jwt-token',
          refresh_token: 'new-refresh-token',
          token_type: 'bearer',
          expires_in: 900,
        },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      };
    }

    if (url.includes('/auth/me')) {
      const authHeader = config.headers?.Authorization;
      assert.equal(authHeader, 'Bearer new-jwt-token');
      return {
        data: {
          id: 'user-uuid-1',
          email: 'test@example.com',
          is_active: true,
        },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      };
    }

    return { data: {}, status: 404, statusText: 'Not Found', headers: {}, config };
  };

  try {
    const { tokenService } = await import('../src/services/tokenService.ts');
    const { login, getCurrentUser } = await import('../src/services/authService.ts');

    const tokenResponse = await login({ email: 'test@example.com', password: 'password123' });
    assert.equal(tokenResponse.access_token, 'new-jwt-token');
    assert.equal(tokenService.getAccessToken(), 'new-jwt-token');
    assert.equal(tokenService.getRefreshToken(), 'new-refresh-token');

    const user = await getCurrentUser();
    assert.equal(user.email, 'test@example.com');
    assert.equal(user.id, 'user-uuid-1');
  } finally {
    apiClient.defaults.adapter = originalAdapter;
    restoreStorage();
  }
});

test('authService - logout revokes refresh token and clears client tokens', async () => {
  const restoreStorage = setupMockLocalStorage();
  const { default: apiClient } = await import('../src/services/api.ts');
  const originalAdapter = apiClient.defaults.adapter;
  let logoutCalledWithToken = null;

  apiClient.defaults.adapter = async (config) => {
    const url = String(config.url || '');
    if (url.includes('/auth/logout')) {
      const data = typeof config.data === 'string' ? JSON.parse(config.data) : config.data;
      logoutCalledWithToken = data?.refresh_token;
      return { data: {}, status: 204, statusText: 'No Content', headers: {}, config };
    }
    return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
  };

  try {
    const { tokenService } = await import('../src/services/tokenService.ts');
    tokenService.setTokens('access-to-clear', 'refresh-to-revoke');

    const { logout } = await import('../src/services/authService.ts');
    await logout();

    assert.equal(logoutCalledWithToken, 'refresh-to-revoke');
    assert.equal(tokenService.getAccessToken(), null);
    assert.equal(tokenService.getRefreshToken(), null);
  } finally {
    apiClient.defaults.adapter = originalAdapter;
    restoreStorage();
  }
});
