/** @type {import('next').NextConfig} */
const apiUrl = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Docker deployment için standalone output
  output: 'standalone',
  
  // API rewrite (backend proxy)
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
  
  // Environment variables
  env: {
    // Keep the browser client, server rewrite, and launcher-selected port aligned.
    API_URL: apiUrl,
    NEXT_PUBLIC_API_URL: apiUrl,
  },
};

module.exports = nextConfig;
