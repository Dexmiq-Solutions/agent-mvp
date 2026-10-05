// =============================================================================
// AuthContext — Global Authenticated State Provider
// =============================================================================

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from 'react';
import {
  getCurrentUser,
  login as apiLogin,
  signup as apiSignup,
  logout as apiLogout,
  refreshTokens,
  onAuthExpired,
} from '../services/authService';
import { tokenService } from '../services/tokenService';
import type {
  UserResponse,
  UserLoginRequest,
  UserSignupRequest,
} from '../types';

interface AuthContextType {
  user: UserResponse | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: UserLoginRequest) => Promise<void>;
  signup: (credentials: UserSignupRequest) => Promise<UserResponse>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchUserProfile = useCallback(async () => {
    try {
      const userData = await getCurrentUser();
      setUser(userData);
      setIsAuthenticated(true);
    } catch {
      // Access token may have failed; attempt token refresh
      try {
        await refreshTokens();
        const freshUser = await getCurrentUser();
        setUser(freshUser);
        setIsAuthenticated(true);
      } catch {
        tokenService.clearTokens();
        setUser(null);
        setIsAuthenticated(false);
      }
    }
  }, []);

  // Initialize session on mount
  useEffect(() => {
    let isMounted = true;

    async function initializeAuth() {
      if (tokenService.hasTokens()) {
        try {
          await fetchUserProfile();
        } catch {
          if (isMounted) {
            setUser(null);
            setIsAuthenticated(false);
          }
        }
      } else {
        setUser(null);
        setIsAuthenticated(false);
      }

      if (isMounted) {
        setIsLoading(false);
      }
    }

    initializeAuth();

    // Listen for global auth session expiry from interceptors
    const unsubscribe = onAuthExpired(() => {
      if (isMounted) {
        setUser(null);
        setIsAuthenticated(false);
      }
    });

    return () => {
      isMounted = false;
      unsubscribe();
    };
  }, [fetchUserProfile]);

  const login = useCallback(async (credentials: UserLoginRequest) => {
    await apiLogin(credentials);
    const userData = await getCurrentUser();
    setUser(userData);
    setIsAuthenticated(true);
  }, []);

  const signup = useCallback(async (credentials: UserSignupRequest) => {
    return await apiSignup(credentials);
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } finally {
      setUser(null);
      setIsAuthenticated(false);
    }
  }, []);

  const refreshUser = useCallback(async () => {
    await fetchUserProfile();
  }, [fetchUserProfile]);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated,
      isLoading,
      login,
      signup,
      logout,
      refreshUser,
    }),
    [user, isAuthenticated, isLoading, login, signup, logout, refreshUser]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
