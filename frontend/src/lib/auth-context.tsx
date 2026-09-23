'use client';

/**
 * Authentication Context & Provider
 * 
 * JWT Access/Refresh token yönetimi, kullanıcı durumu, otomatik
 * header enjeksiyonu ve oturum kontrolü.
 */

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import axios from 'axios';
import { useQueryClient } from '@tanstack/react-query';
import { User, LoginResponse, RegisterRequest } from '@/types/auth';
import { API_BASE_URL } from '@/lib/api';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (usernameOrEmail: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (data: RegisterRequest) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const ACCESS_TOKEN_KEY = 'ailab_access_token';
export const REFRESH_TOKEN_KEY = 'ailab_refresh_token';
export const USER_KEY = 'ailab_user';

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    axios.get(`${API_BASE_URL}/api/v1/auth/me`, { withCredentials: true })
      .then(res => {
        if (isMounted) setUser(res.data.user);
      })
      .catch(async () => {
        try {
          await axios.post(`${API_BASE_URL}/api/v1/auth/refresh`, {}, { withCredentials: true });
          const res = await axios.get(`${API_BASE_URL}/api/v1/auth/me`, { withCredentials: true });
          if (isMounted) setUser(res.data.user);
        } catch {
          if (isMounted) setUser(prev => (prev ? prev : null));
        }
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Oturum açma
  const login = useCallback(async (usernameOrEmail: string, password: string) => {
    try {
      const response = await axios.post<LoginResponse>(`${API_BASE_URL}/api/v1/auth/login`, {
        username_or_email: usernameOrEmail,
        password: password,
      });

      const data = response.data;
      queryClient.clear();
      setToken(data.access_token);
      setUser(data.user);



      return { success: true };
    } catch (error: any) {
      const msg = error.response?.data?.detail || 'Giriş yapılamadı. Lütfen bilgilerinizi kontrol edin.';
      return { success: false, error: msg };
    }
  }, []);

  // Yeni kullanıcı kaydı
  const register = useCallback(async (data: RegisterRequest) => {
    try {
      await axios.post(`${API_BASE_URL}/api/v1/auth/register`, data);
      // Kayıt başarılı olduğunda otomatik giriş yap
      return await login(data.username, data.password);
    } catch (error: any) {
      const msg = error.response?.data?.detail || 'Kayıt işlemi başarısız oldu.';
      return { success: false, error: msg };
    }
  }, [login]);

  // Oturumu kapatma
  const logout = useCallback(() => {
    axios.post(`${API_BASE_URL}/api/v1/auth/logout`, {}, { withCredentials: true }).catch(() => {});
    queryClient.clear();
    setUser(null);
    setToken(null);
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }, []);

  // Profil yenileme
  const refreshUser = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/v1/auth/me`, {
        withCredentials: true,
      });
      if (res.data?.user) {
        setUser(res.data.user);

      }
    } catch (err) {
      console.warn('Profil yenilenemedi:', err);
    }
  }, [token]);

  const value = {
    user,
    token,
    isAuthenticated: !!user,
    isLoading,
    login,
    register,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

const defaultAuthContext: AuthContextType = {
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: false,
  login: async () => ({ success: false, error: 'AuthProvider bulunamadı' }),
  register: async () => ({ success: false, error: 'AuthProvider bulunamadı' }),
  logout: () => {},
  refreshUser: async () => {},
};

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  return context || defaultAuthContext;
}
