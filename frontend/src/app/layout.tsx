/**
 * Root Layout
 */

import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import '@/styles/globals.css';
import { Providers } from './providers';
import { Navbar } from '@/components/Navbar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Local AI Research Lab',
  description: 'Local-First AI Systems Research & Learning Platform - Developed by Kenan AY, Kütahya, TÜRKİYE',
  authors: [{ name: 'Kenan AY' }],
  keywords: ['AI', 'Machine Learning', 'Deep Learning', 'Research', 'Education', 'PyTorch', 'Transformer'],
  creator: 'Kenan AY',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr">
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen flex flex-col bg-gray-50 text-gray-900">
            <Navbar />
            <main className="flex-1">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
