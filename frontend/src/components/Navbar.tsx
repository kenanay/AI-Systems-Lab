'use client';
import { API_BASE_URL } from '@/lib/api';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';
import { useAuth } from '@/lib/auth-context';

const NAV_ITEMS = [
  { href: '/', label: 'Ana Sayfa', icon: '🏠' },
  { href: '/journey', label: 'Öğrenme Yolu', icon: '🗺️' },
  { href: '/dataset-explorer', label: 'Explorer', icon: '📚' },
  { href: '/upload', label: 'Upload', icon: '⬆️' },
  { href: '/tokenizer', label: 'Tokenizer', icon: '🔤' },
  { href: '/dataset-compiler', label: 'Compiler', icon: '⚙️' },
  { href: '/training', label: 'Training', icon: '🏋️' },
  { href: '/models', label: 'Modeller', icon: '📦' },
  { href: '/playground', label: 'Playground', icon: '💬' },
  { href: '/attention-lab', label: 'Attention Lab', icon: '👁️' },
  { href: '/embedding-lab', label: 'Embedding Lab', icon: '🎯' },
  { href: '/rag-lab', label: 'RAG Lab', icon: '🔍' },
  { href: '/math-lab', label: 'Math Lab', icon: '📐' },
  { href: '/tensor-lab', label: 'Tensor Lab', icon: '🧮' },
  { href: '/nn-lab', label: 'NN Lab', icon: '🧠' },
  { href: '/transformer-lab', label: 'Transformer Lab', icon: '🏛️' },
  { href: '/evaluation', label: 'Evaluation Lab', icon: '📊' },
  { href: '/systems-lab', label: 'Systems Lab', icon: '⚡' },
  { href: '/synthetic-lab', label: 'Synthetic Lab', icon: '🧪' },
];

export function Navbar() {
  const pathname = usePathname();
  const { user, isAuthenticated } = useAuth();
  const [engineStatus, setEngineStatus] = useState<'online' | 'checking' | 'offline'>('checking');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/inference/status`, { credentials: 'include', signal: AbortSignal.timeout(3000) });
        if (res.ok && isMounted) {
          const state = await res.json();
          setEngineStatus(state.ready && state.model_loaded ? 'online' : 'offline');
        } else if (isMounted) {
          setEngineStatus('offline');
        }
      } catch {
        if (isMounted) setEngineStatus('offline');
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-white/85 border-b border-gray-200/80 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-2.5 group">
              <span className="text-2xl transform group-hover:scale-110 transition-transform">🤖</span>
              <div className="flex flex-col">
                <span className="font-bold text-gray-900 text-lg leading-tight tracking-tight">
                  Local AI Lab
                </span>
                <span className="text-[10px] text-gray-500 font-mono tracking-wider">
                  RESEARCH INFRASTRUCTURE
                </span>
              </div>
            </Link>
            <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200/60">
              v0.1.0 Local
            </span>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {NAV_ITEMS.map((item) => {
              const isActive = item.href === '/' ? pathname === '/' : pathname?.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center space-x-1.5 px-3 py-2 rounded-md text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 font-semibold shadow-sm'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100/80'
                  }`}
                >
                  <span className="text-base">{item.icon}</span>
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right Status Indicator & Auth */}
          <div className="flex items-center space-x-3">
            <div className="hidden sm:flex items-center space-x-2 px-3 py-1 rounded-full text-xs bg-gray-100/90 border border-gray-200">
              <span
                className={`w-2 h-2 rounded-full ${
                  engineStatus === 'online'
                    ? 'bg-green-500 animate-pulse'
                    : engineStatus === 'checking'
                    ? 'bg-amber-400'
                    : 'bg-red-400'
                }`}
              />
              <span className="text-gray-600 font-medium">
                {engineStatus === 'online' ? 'Engine Ready' : engineStatus === 'checking' ? 'Connecting...' : 'Engine Standby'}
              </span>
            </div>

            {/* User Profile or Login Button */}
            {isAuthenticated && user ? (
              <Link
                href="/profile"
                className="flex items-center space-x-2 px-2.5 py-1.5 rounded-xl bg-gray-100 hover:bg-gray-200 border border-gray-200 transition-all text-xs"
              >
                <div className="w-5 h-5 rounded-full bg-blue-600 text-white font-bold flex items-center justify-center text-[10px]">
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <span className="font-semibold text-gray-800 hidden lg:inline">@{user.username}</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold uppercase bg-blue-50 text-blue-700 border border-blue-200">
                  {user.role}
                </span>
              </Link>
            ) : (
              <Link
                href="/login"
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium text-xs shadow-sm transition-all"
              >
                <span>🔑</span>
                <span>Giriş Yap</span>
              </Link>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              type="button"
              className="md:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
              aria-label="Toggle navigation menu"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-gray-200 bg-white/95 px-4 pt-2 pb-3 space-y-1 shadow-lg">
          {NAV_ITEMS.map((item) => {
            const isActive = item.href === '/' ? pathname === '/' : pathname?.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center space-x-3 px-3 py-2.5 rounded-md text-base font-medium ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <span className="text-xl">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}

          {/* Mobile Auth Item */}
          <div className="pt-2 border-t border-gray-100">
            {isAuthenticated && user ? (
              <Link
                href="/profile"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center space-x-3 px-3 py-2.5 rounded-md text-base font-semibold bg-blue-50 text-blue-700"
              >
                <span>👤</span>
                <span>@{user.username} ({user.role}) - Profil & API</span>
              </Link>
            ) : (
              <Link
                href="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center space-x-3 px-3 py-2.5 rounded-md text-base font-semibold bg-blue-600 text-white"
              >
                <span>🔑</span>
                <span>Giriş Yap / Kayıt Ol</span>
              </Link>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
