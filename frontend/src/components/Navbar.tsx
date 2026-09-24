'use client';
import { API_BASE_URL } from '@/lib/api';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/lib/auth-context';

export const NAV_GROUPS = [
  {
    title: 'Veri Hattı',
    icon: '📁',
    items: [
      { href: '/dataset-explorer', label: 'Explorer', desc: 'Dataset keşfi ve yönetimi', icon: '📚' },
      { href: '/upload', label: 'Upload', desc: 'TXT, MD, PDF dosya yükleme', icon: '⬆️' },
      { href: '/tokenizer', label: 'Tokenizer', desc: 'BPE & tokenizasyon eğitimi', icon: '🔤' },
      { href: '/dataset-compiler', label: 'Compiler', desc: 'Canonical Parquet derleme', icon: '⚙️' },
    ],
  },
  {
    title: 'Model & Eğitim',
    icon: '🚀',
    items: [
      { href: '/training', label: 'Training', desc: 'Pretrain, SFT ve LoRA stüdyosu', icon: '🏋️' },
      { href: '/models', label: 'Modeller', desc: 'Kayıtlı modeller ve sürümler', icon: '📦' },
      { href: '/playground', label: 'Playground', desc: 'İnteraktif çıkarım ve sohbet', icon: '💬' },
      { href: '/evaluation', label: 'Evaluation Lab', desc: 'Model metrikleri ve benchmark', icon: '📊' },
    ],
  },
  {
    title: 'Laboratuvarlar',
    icon: '🔬',
    items: [
      { href: '/attention-lab', label: 'Attention Lab', desc: 'Self-Attention ve görselleştirme', icon: '👁️' },
      { href: '/embedding-lab', label: 'Embedding Lab', desc: 'Vektör uzayı ve projeksiyon', icon: '🎯' },
      { href: '/rag-lab', label: 'RAG Lab', desc: 'Retrieval Augmented Generation', icon: '🔍' },
      { href: '/transformer-lab', label: 'Transformer Lab', desc: 'Katman mimarisi ve akış', icon: '🏛️' },
      { href: '/nn-lab', label: 'NN Lab', desc: 'Aktivasyon ve sinir ağları', icon: '🧠' },
      { href: '/tensor-lab', label: 'Tensor Lab', desc: 'Tensör işlemleri ve boyutlar', icon: '🧮' },
      { href: '/math-lab', label: 'Math Lab', desc: 'Lineer cebir ve kalkülüs', icon: '📐' },
      { href: '/systems-lab', label: 'Systems Lab', desc: 'GPU, bellek ve throughput', icon: '⚡' },
      { href: '/synthetic-lab', label: 'Synthetic Lab', desc: 'Sentetik veri üretimi', icon: '🧪' },
    ],
  },
];

export function Navbar() {
  const pathname = usePathname();
  const { user, isAuthenticated } = useAuth();
  const [engineStatus, setEngineStatus] = useState<'online' | 'checking' | 'offline'>('checking');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const navRef = useRef<HTMLDivElement>(null);

  // Close open dropdowns when navigating or clicking outside
  useEffect(() => {
    setOpenDropdown(null);
  }, [pathname]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
    <header className="sticky top-0 z-50 backdrop-blur-md bg-white/90 border-b border-slate-200/80 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3 shrink-0">
            <Link href="/" className="flex items-center space-x-2.5 group">
              <span className="text-2xl transform group-hover:scale-110 transition-transform">🤖</span>
              <div className="flex flex-col">
                <span className="font-bold text-slate-900 text-lg leading-tight tracking-tight">
                  Local AI Lab
                </span>
                <span className="text-[10px] text-slate-500 font-mono tracking-wider">
                  RESEARCH INFRASTRUCTURE
                </span>
              </div>
            </Link>
            <span className="hidden xl:inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200/60">
              v0.1.0 Local
            </span>
          </div>

          {/* Desktop Navigation Links */}
          <nav ref={navRef} className="hidden md:flex items-center space-x-1 lg:space-x-1.5">
            {/* Primary Direct Links */}
            <Link
              href="/"
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                pathname === '/'
                  ? 'bg-blue-50 text-blue-700 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
              }`}
            >
              <span>🏠</span>
              <span>Ana Sayfa</span>
            </Link>

            <Link
              href="/journey"
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                pathname?.startsWith('/journey')
                  ? 'bg-blue-50 text-blue-700 font-semibold shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
              }`}
            >
              <span>🗺️</span>
              <span>Öğrenme Yolu</span>
            </Link>

            {/* Categorized Dropdown Groups */}
            {NAV_GROUPS.map((group) => {
              const isGroupActive = group.items.some((item) =>
                item.href === '/' ? pathname === '/' : pathname?.startsWith(item.href)
              );
              const isOpen = openDropdown === group.title;

              return (
                <div key={group.title} className="relative group">
                  <button
                    type="button"
                    onClick={() => setOpenDropdown((prev) => (prev === group.title ? null : group.title))}
                    aria-expanded={isOpen}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      isGroupActive || isOpen
                        ? 'bg-blue-50 text-blue-700 font-semibold'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
                    }`}
                  >
                    <span>{group.icon}</span>
                    <span>{group.title}</span>
                    <svg
                      className={`w-3.5 h-3.5 ml-0.5 text-slate-400 group-hover:text-slate-600 transition-transform ${
                        isOpen ? 'rotate-180 text-blue-600' : 'group-hover:rotate-180'
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {/* Dropdown Menu Wrapper with Hover Tunnel Bridge & Grace Period Delay */}
                  <div
                    className={`absolute left-0 top-full pt-1.5 z-50 transition-all duration-150 ${
                      isOpen
                        ? 'opacity-100 visible translate-y-0 pointer-events-auto'
                        : 'opacity-0 invisible translate-y-1 pointer-events-none group-hover:opacity-100 group-hover:visible group-hover:translate-y-0 group-hover:pointer-events-auto delay-150 group-hover:delay-0'
                    }`}
                  >
                    {/* Invisible Safe Hover Bridge: prevents cursor drop when moving across boundary */}
                    <div className="absolute -top-3 left-0 w-full h-3 bg-transparent pointer-events-auto" />

                    {/* Inner Menu Card */}
                    <div
                      className={`${
                        group.items.length > 5 ? 'w-80 sm:w-96' : 'w-64'
                      } rounded-xl bg-white shadow-xl border border-slate-200/90 py-2 relative`}
                    >
                      <div className="px-3 py-1.5 border-b border-slate-100 mb-1 flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                          {group.title}
                        </span>
                        <span className="text-[11px] text-slate-400 font-mono">
                          {group.items.length} modül
                        </span>
                      </div>

                      <div className={group.items.length > 5 ? 'grid grid-cols-2 gap-1 px-1.5' : 'space-y-0.5 px-1.5'}>
                        {group.items.map((item) => {
                          const isActive = pathname?.startsWith(item.href);
                          return (
                            <Link
                              key={item.href}
                              href={item.href}
                              onClick={() => setOpenDropdown(null)}
                              className={`flex items-start space-x-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors ${
                                isActive
                                  ? 'bg-blue-50 text-blue-700 font-medium'
                                  : 'text-slate-700 hover:bg-slate-100/80 hover:text-slate-900'
                              }`}
                            >
                              <span className="text-base shrink-0 mt-0.5">{item.icon}</span>
                              <div className="min-w-0">
                                <div className="font-medium text-xs sm:text-sm leading-tight truncate">
                                  {item.label}
                                </div>
                                {group.items.length <= 5 && (
                                  <div className="text-[11px] text-slate-500 truncate leading-tight mt-0.5">
                                    {item.desc}
                                  </div>
                                )}
                              </div>
                            </Link>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </nav>

          {/* Right Status Indicator & Auth */}
          <div className="flex items-center space-x-3 shrink-0">
            <div className="hidden sm:flex items-center space-x-2 px-3 py-1 rounded-full text-xs bg-slate-100/90 border border-slate-200">
              <span
                className={`w-2 h-2 rounded-full ${
                  engineStatus === 'online'
                    ? 'bg-emerald-500 animate-pulse'
                    : engineStatus === 'checking'
                    ? 'bg-amber-400'
                    : 'bg-rose-400'
                }`}
              />
              <span className="text-slate-600 font-medium">
                {engineStatus === 'online' ? 'Engine Ready' : engineStatus === 'checking' ? 'Connecting...' : 'Engine Standby'}
              </span>
            </div>

            {/* User Profile or Login Button */}
            {isAuthenticated && user ? (
              <div className="flex items-center space-x-2">
                <Link
                  href="/profile"
                  className="flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 hover:text-slate-900 hover:bg-slate-100 border border-slate-200 transition-colors"
                >
                  <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-[10px]">
                    {user.username.charAt(0).toUpperCase()}
                  </span>
                  <span className="font-semibold max-w-[100px] truncate">{user.username}</span>
                  <span className="text-[10px] px-1.5 py-0.2 bg-slate-200 text-slate-600 rounded">
                    {user.role}
                  </span>
                </Link>
              </div>
            ) : (
              <Link
                href="/login"
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 transition shadow-xs"
              >
                <span>🔑</span>
                <span>Giriş Yap</span>
              </Link>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              type="button"
              className="md:hidden p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100"
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
        <div className="md:hidden border-b border-slate-200 bg-white/98 px-4 pt-3 pb-6 space-y-4 shadow-xl max-h-[calc(100vh-4rem)] overflow-y-auto">
          {/* Quick Direct Links */}
          <div className="space-y-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 px-3">
              Temel
            </span>
            <div className="grid grid-cols-2 gap-1 pt-1">
              <Link
                href="/"
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center space-x-2.5 px-3 py-2 rounded-lg text-sm font-medium ${
                  pathname === '/' ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                <span>🏠</span>
                <span>Ana Sayfa</span>
              </Link>
              <Link
                href="/journey"
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center space-x-2.5 px-3 py-2 rounded-lg text-sm font-medium ${
                  pathname?.startsWith('/journey') ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                <span>🗺️</span>
                <span>Öğrenme Yolu</span>
              </Link>
            </div>
          </div>

          {/* Grouped Sections */}
          {NAV_GROUPS.map((group) => (
            <div key={group.title} className="space-y-1 pt-2 border-t border-slate-100">
              <div className="flex items-center justify-between px-3">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  {group.title}
                </span>
                <span className="text-[11px] text-slate-400 font-mono">
                  {group.icon}
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 pt-1">
                {group.items.map((item) => {
                  const isActive = pathname?.startsWith(item.href);
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium ${
                        isActive
                          ? 'bg-blue-50 text-blue-700 font-semibold'
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <span className="text-lg">{item.icon}</span>
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Mobile Auth Item */}
          <div className="pt-3 border-t border-slate-200">
            {isAuthenticated && user ? (
              <Link
                href="/profile"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-semibold bg-blue-50 text-blue-700"
              >
                <span>👤</span>
                <span>@{user.username} ({user.role}) - Profil & API</span>
              </Link>
            ) : (
              <Link
                href="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center justify-center space-x-2 px-3 py-2.5 rounded-lg text-sm font-semibold bg-blue-600 text-white"
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
