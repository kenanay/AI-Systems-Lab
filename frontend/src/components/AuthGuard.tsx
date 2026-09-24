'use client';

import { ReactNode, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';

const PUBLIC_PATHS = new Set(['/', '/login', '/register']);

function isPublicPath(pathname: string | null) {
  return pathname ? PUBLIC_PATHS.has(pathname) : true;
}

export function AuthGuard({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();
  const isPublic = isPublicPath(pathname);

  useEffect(() => {
    if (isPublic || isLoading || isAuthenticated || typeof window === 'undefined') {
      return;
    }

    const current = `${window.location.pathname}${window.location.search}`;
    const next = encodeURIComponent(current);
    router.replace(`/login?next=${next}&reason=session-expired`);
  }, [isAuthenticated, isLoading, isPublic, pathname, router]);

  if (isPublic) {
    return <>{children}</>;
  }

  if (isLoading || !isAuthenticated) {
    return (
      <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center bg-slate-50 px-4">
        <div className="rounded-xl border border-slate-200 bg-white px-5 py-4 text-center shadow-sm">
          <div className="mx-auto mb-3 h-6 w-6 animate-spin rounded-full border-2 border-slate-200 border-t-indigo-600" />
          <p className="text-sm text-slate-600">
            {isLoading ? 'Oturum kontrol ediliyor…' : 'Login sayfasına yönlendiriliyorsunuz…'}
          </p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
