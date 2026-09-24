'use client';

/**
 * Giriş (Login) Sayfası
 * 
 * JWT tabanlı kimlik doğrulama, hızlı demo giriş butonları ve şık dark/glassmorphic arayüz.
 */

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';

export default function LoginPage() {
  const router = useRouter();
  const { login, isAuthenticated, user, logout } = useAuth();

  const [usernameOrEmail, setUsernameOrEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const getRedirectTarget = () => {
    if (typeof window === 'undefined') return '/profile';
    const next = new URLSearchParams(window.location.search).get('next');
    return next && next.startsWith('/') && !next.startsWith('//') ? next : '/profile';
  };

  React.useEffect(() => {
    const reason = new URLSearchParams(window.location.search).get('reason');
    if (reason === 'session-expired' || reason === 'unauthorized') {
      setErrorMsg('Bu sayfayı kullanmak için giriş yapmanız gerekiyor.');
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameOrEmail.trim() || !password) {
      setErrorMsg('Lütfen kullanıcı adı ve parolanızı girin.');
      return;
    }

    setErrorMsg(null);
    setLoading(true);

    const result = await login(usernameOrEmail.trim(), password);
    setLoading(false);

    if (result.success) {
      router.push(getRedirectTarget());
    } else {
      setErrorMsg(result.error || 'Giriş yapılamadı.');
    }
  };

  const handleQuickLogin = async (role: 'admin' | 'researcher') => {
    setErrorMsg(null);
    setLoading(true);

    const creds = role === 'admin' 
      ? { u: 'admin', p: 'admin' }
      : { u: 'researcher', p: 'researcher123' };

    setUsernameOrEmail(creds.u);
    setPassword(creds.p);

    const result = await login(creds.u, creds.p);
    setLoading(false);

    if (result.success) {
      router.push(getRedirectTarget());
    } else {
      setErrorMsg(result.error || 'Demo girişi başarısız oldu.');
    }
  };

  if (isAuthenticated && user) {
    return (
      <div className="min-h-[80vh] flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-gray-200/80 p-8 text-center space-y-4">
          <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto text-2xl font-bold">
            {user.username.charAt(0).toUpperCase()}
          </div>
          <h2 className="text-xl font-bold text-gray-900">Zaten Giriş Yapılmış</h2>
          <p className="text-sm text-gray-600">
            <span className="font-semibold text-gray-900">@{user.username}</span> olarak oturumunuz açık.
            (Rol: <span className="uppercase font-mono text-xs px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">{user.role}</span>)
          </p>
          <div className="flex flex-col gap-2 pt-2">
            <Link
              href="/profile"
              className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl text-sm transition-all shadow-md hover:shadow-lg text-center"
            >
              Profil & API Anahtarlarına Git
            </Link>
            <button
              onClick={logout}
              className="w-full py-2.5 px-4 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-xl text-sm transition-all"
            >
              Farklı Hesapla Giriş Yap (Çıkış)
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[85vh] flex items-center justify-center p-4 bg-gradient-to-b from-gray-50 via-gray-100/50 to-gray-50">
      <div className="max-w-md w-full space-y-6">
        {/* Brand Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-lg shadow-blue-500/30 text-3xl">
            🤖
          </div>
          <h1 className="text-2xl font-black tracking-tight text-gray-900">
            Local AI Research Lab
          </h1>
          <p className="text-sm text-gray-500">
            Kurumsal Model Eğitimi, Güvenli API & Benchmark Platformu
          </p>
        </div>

        {/* Card */}
        <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-xl shadow-gray-200/50 border border-gray-200/80 p-8 space-y-6">
          <div className="flex items-center justify-between border-b border-gray-100 pb-4">
            <h2 className="text-lg font-bold text-gray-900">Hesaba Giriş Yap</h2>
            <span className="text-xs font-mono text-gray-600 bg-gray-100 px-2 py-0.5 rounded">
              JWT HS256
            </span>
          </div>

          {errorMsg && (
            <div className="p-3.5 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl flex items-start gap-2.5 animate-in fade-in">
              <span className="text-base leading-none">⚠️</span>
              <div className="flex-1 text-xs leading-relaxed">{errorMsg}</div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="login-username" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">
                Kullanıcı Adı veya E-posta
              </label>
              <input
                id="login-username"
                type="text"
                required
                value={usernameOrEmail}
                onChange={(e) => setUsernameOrEmail(e.target.value)}
                placeholder="örn: admin veya researcher@ailab.local"
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 text-sm transition-all"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="login-password" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider">
                  Parola
                </label>
              </div>
              <input
                id="login-password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 text-sm transition-all"
              />
            </div>

            <button
              id="login-submit-button"
              type="submit"
              disabled={loading}
              className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium rounded-xl text-sm transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Doğrulanıyor...</span>
                </>
              ) : (
                <span>Oturum Aç</span>
              )}
            </button>
          </form>

          {/* Quick Demo Login Buttons */}
          <div className="pt-2 border-t border-gray-100">
            <span className="block text-[11px] font-semibold text-gray-600 uppercase tracking-wider text-center mb-3">
              Geliştirici Hızlı Demo Girişi (1-Tık)
            </span>
            <div className="grid grid-cols-2 gap-2.5">
              <button
                type="button"
                onClick={() => handleQuickLogin('admin')}
                disabled={loading}
                className="flex items-center justify-center gap-1.5 py-2 px-3 bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200/80 rounded-xl text-xs font-semibold transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <span>👑</span>
                <span>Demo Admin</span>
              </button>
              <button
                type="button"
                onClick={() => handleQuickLogin('researcher')}
                disabled={loading}
                className="flex items-center justify-center gap-1.5 py-2 px-3 bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200/80 rounded-xl text-xs font-semibold transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <span>🔬</span>
                <span>Demo Researcher</span>
              </button>
            </div>
          </div>

          <div className="text-center pt-2">
            <span className="text-xs text-gray-500">
              Hesabınız yok mu?{' '}
              <Link href="/register" className="text-blue-600 font-semibold hover:underline">
                Kayıt Olun
              </Link>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
