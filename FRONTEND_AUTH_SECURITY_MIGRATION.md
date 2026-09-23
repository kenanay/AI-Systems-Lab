# Frontend Auth Security Migration Guide

## Mevcut Durum (localStorage)

Şu anda `frontend/src/contexts/auth-context.tsx` access token'ı localStorage'da saklıyor:

```tsx
// Mevcut (GÜVENLİ DEĞİL)
localStorage.setItem('access_token', response.access_token);
const token = localStorage.getItem('access_token');
```

**Güvenlik Sorunu:**
- ❌ XSS (Cross-Site Scripting) saldırılarına açık
- ❌ JavaScript'ten erişilebilir
- ❌ Token çalınabilir

---

## Hedef: HttpOnly Cookie

```tsx
// Hedef (GÜVENLİ)
// Backend cookie set eder, frontend okuyamaz
```

**Güvenlik Avantajları:**
- ✅ HttpOnly flag - JavaScript erişemez
- ✅ Secure flag - Sadece HTTPS
- ✅ SameSite=Strict - CSRF koruması
- ✅ XSS saldırılarına karşı koruma

---

## Migration Adımları

### 1. Backend Değişiklikleri

#### 1.1. Login Endpoint Güncelleme

```python
# backend/routers/auth.py

from fastapi import Response
from datetime import timedelta

@router.post("/login")
async def login(
    response: Response,  # Response inject edilecek
    form_data: OAuth2PasswordRequestForm = Depends()
):
    # ... authentication logic ...
    
    # Access token oluştur
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=30)
    )
    
    # Refresh token oluştur
    refresh_token = create_refresh_token(
        data={"sub": user.username},
        expires_delta=timedelta(days=7)
    )
    
    # HttpOnly cookie'leri set et
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,          # JavaScript erişemez
        secure=True,            # Sadece HTTPS (production)
        samesite="strict",      # CSRF koruması
        max_age=1800,           # 30 dakika (saniye cinsinden)
        path="/"
    )
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=604800,         # 7 gün
        path="/api/v1/auth"     # Sadece refresh endpoint'i için
    )
    
    # Response body (UI için bilgi, token içermeyen)
    return {
        "message": "Login successful",
        "user": {
            "username": user.username,
            "role": user.role,
            "email": user.email
        },
        "expires_in": 1800  # seconds
    }
```

#### 1.2. Logout Endpoint

```python
@router.post("/logout")
async def logout(response: Response):
    # Cookie'leri sil
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/api/v1/auth")
    
    return {"message": "Logged out successfully"}
```

#### 1.3. Token Refresh Endpoint

```python
@router.post("/refresh")
async def refresh_token(
    response: Response,
    request: Request
):
    # Refresh token'ı cookie'den al
    refresh_token = request.cookies.get("refresh_token")
    
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")
    
    # Token doğrula ve yeni access token oluştur
    try:
        payload = decode_token(refresh_token, "refresh")
        username = payload.get("sub")
        
        # Yeni access token
        new_access_token = create_access_token(
            data={"sub": username, "role": user.role},
            expires_delta=timedelta(minutes=30)
        )
        
        # Yeni cookie set et
        response.set_cookie(
            key="access_token",
            value=new_access_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=1800,
            path="/"
        )
        
        return {
            "message": "Token refreshed",
            "expires_in": 1800
        }
    
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
```

#### 1.4. Dependency Güncellemesi

```python
# backend/security/dependencies.py

async def get_current_user(request: Request) -> UserInDB:
    # Cookie'den token al
    token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = decode_token(token, "access")
        username = payload.get("sub")
        # ... user fetch ...
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

### 2. Frontend Değişiklikleri

#### 2.1. Auth Context Güncellemesi

```tsx
// frontend/src/contexts/auth-context.tsx

'use client';

import { createContext, useContext, useState, useEffect } from 'react';
import { api } from '@/lib/api';

interface User {
  username: string;
  role: string;
  email?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Session check on mount
  useEffect(() => {
    checkSession();
  }, []);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (user) {
      // Her 25 dakikada bir refresh (30 dk expiry'den önce)
      const interval = setInterval(refreshAuth, 25 * 60 * 1000);
      return () => clearInterval(interval);
    }
  }, [user]);

  const checkSession = async () => {
    try {
      // Backend'e istek at - cookie varsa user bilgisi dönecek
      const response = await fetch('/api/v1/auth/me', {
        credentials: 'include'  // Cookie'leri gönder
      });
      
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        setUser(null);
      }
    } catch (error) {
      console.error('Session check failed:', error);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch('/api/v1/auth/login', {
      method: 'POST',
      body: formData,
      credentials: 'include'  // Cookie'leri al
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data = await response.json();
    setUser(data.user);
  };

  const logout = async () => {
    try {
      await fetch('/api/v1/auth/logout', {
        method: 'POST',
        credentials: 'include'
      });
    } finally {
      setUser(null);
    }
  };

  const refreshAuth = async () => {
    try {
      const response = await fetch('/api/v1/auth/refresh', {
        method: 'POST',
        credentials: 'include'
      });

      if (!response.ok) {
        // Refresh failed - logout
        setUser(null);
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout,
        refreshAuth
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```

#### 2.2. API Client Güncellemesi

```tsx
// frontend/src/lib/api.ts

// Tüm API isteklerinde credentials: 'include' kullan
export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`/api/v1${endpoint}`, {
    ...options,
    credentials: 'include',  // Cookie'leri otomatik gönder
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (response.status === 401) {
    // Unauthorized - redirect to login
    window.location.href = '/login';
    throw new Error('Unauthorized');
  }

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Request failed');
  }

  return response.json();
}
```

#### 2.3. Login Page Güncellemesi

```tsx
// frontend/src/app/login/page.tsx

'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      await login(username, password);
      router.push('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Login form UI */}
    </form>
  );
}
```

---

### 3. CORS Konfigürasyonu

```python
# backend/main.py

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development
        "https://your-domain.com"  # Production
    ],
    allow_credentials=True,  # Cookie'ler için gerekli
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 4. Environment Variables

```bash
# .env

# Development
SECURE_COOKIES=false

# Production
SECURE_COOKIES=true
```

```python
# backend/config.py

import os

SECURE_COOKIES = os.getenv("SECURE_COOKIES", "false").lower() == "true"
```

---

## Migration Checklist

### Backend
- [ ] Login endpoint cookie set eder
- [ ] Logout endpoint cookie siler
- [ ] Refresh endpoint ekle
- [ ] `/auth/me` endpoint ekle (session check için)
- [ ] `get_current_user` dependency cookie okur
- [ ] CORS `allow_credentials=True` set edildi

### Frontend
- [ ] AuthContext localStorage kodlarını kaldır
- [ ] `credentials: 'include'` tüm fetch'lere eklendi
- [ ] Auto-refresh mekanizması eklendi
- [ ] Session check on mount
- [ ] Login/logout UI güncellemesi

### Testing
- [ ] Login flow test edildi
- [ ] Logout flow test edildi
- [ ] Token refresh test edildi
- [ ] Session persistence test edildi
- [ ] XSS koruması doğrulandı (JavaScript token okuyamıyor)

---

## Rollback Plan

Eğer migration sorunlu olursa:

1. Backend: Eski `/login` endpoint'i geri koy (token response body'de)
2. Frontend: AuthContext eski haline dön
3. localStorage kullanımını geri aç

---

## Güvenlik Notları

✅ **HttpOnly:** JavaScript cookie'ye erişemez  
✅ **Secure:** Sadece HTTPS (production)  
✅ **SameSite=Strict:** CSRF koruması  
✅ **Max-Age:** Token expiry kontrolü  
✅ **Path:** Cookie scope sınırlaması  

**Not:** Development'ta `secure=False` kullanabilirsiniz (HTTP için).
Production'da mutlaka `secure=True` kullanın.

---

## Estimasyon

**Backend değişiklikler:** ~2 saat  
**Frontend değişiklikler:** ~3 saat  
**Testing:** ~2 saat  
**Toplam:** ~1 gün

---

**Status:** 📋 Hazır (Implementation için)  
**Öncelik:** P1 (Önemli güvenlik iyileştirmesi)
