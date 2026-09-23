'use client';

/**
 * Kayıt Ol (Register) Sayfası
 */

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth-context';

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuth();

  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('researcher');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !email.trim() || !password) {
      setErrorMsg('Lütfen tüm zorunlu alanları doldurun.');
      return;
    }

    if (password.length < 6) {
      setErrorMsg('Parola en az 6 karakter olmalıdır.');
      return;
    }

    setErrorMsg(null);
    setLoading(true);

    const result = await register({
      username: username.trim(),
      email: email.trim(),
      password: password,
      full_name: fullName.trim() || undefined,
      role: role,
    });

    setLoading(false);

    if (result.success) {
      router.push('/profile');
    } else {
      setErrorMsg(result.error || 'Kayıt başarısız oldu.');
    }
  };

  return (
    <div className="min-h-[85vh] flex items-center justify-center p-4 bg-gradient-to-b from-gray-50 via-gray-100/50 to-gray-50">
      <div className="max-w-md w-full space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-700 text-white shadow-lg shadow-indigo-500/30 text-3xl">
            ✨
          </div>
          <h1 className="text-2xl font-black tracking-tight text-gray-900">
            Yeni Araştırmacı Hesabı
          </h1>
          <p className="text-sm text-gray-500">
            AI Systems Research & Learning Platformuna Katılın
          </p>
        </div>

        <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-xl shadow-gray-200/50 border border-gray-200/80 p-8 space-y-5">
          {errorMsg && (
            <div className="p-3.5 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl flex items-start gap-2.5">
              <span className="text-base leading-none">⚠️</span>
              <div className="flex-1 text-xs leading-relaxed">{errorMsg}</div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="register-username" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                Kullanıcı Adı *
              </label>
              <input
                id="register-username"
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="örn: kenan_ay"
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 text-sm transition-all"
              />
            </div>

            <div>
              <label htmlFor="register-email" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                E-posta Adresi *
              </label>
              <input
                id="register-email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="örn: arastirmaci@ailab.local"
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 text-sm transition-all"
              />
            </div>

            <div>
              <label htmlFor="register-fullname" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                Ad Soyad (Opsiyonel)
              </label>
              <input
                id="register-fullname"
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="örn: Kenan AY"
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 text-sm transition-all"
              />
            </div>

            <div>
              <label htmlFor="register-password" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                Parola (En az 6 karakter) *
              </label>
              <input
                id="register-password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 text-sm transition-all"
              />
            </div>

            <div>
              <label htmlFor="register-role" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">
                Rol Yetkisi
              </label>
              <select
                id="register-role"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-xl border border-gray-300 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 text-sm transition-all"
              >
                <option value="researcher">🔬 Researcher (Model Eğitimi, Dataset, Benchmarks)</option>
                <option value="viewer">👁️ Viewer (Salt-Okunur Gözlemci)</option>
              </select>
            </div>

            <button
              id="register-submit-button"
              type="submit"
              disabled={loading}
              className="w-full py-3 px-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-medium rounded-xl text-sm transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 pt-3"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Hesap Oluşturuluyor...</span>
                </>
              ) : (
                <span>Hesabı Oluştur & Giriş Yap</span>
              )}
            </button>
          </form>

          <div className="text-center pt-2 border-t border-gray-100">
            <span className="text-xs text-gray-500">
              Zaten hesabınız var mı?{' '}
              <Link href="/login" className="text-indigo-600 font-semibold hover:underline">
                Giriş Yapın
              </Link>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
