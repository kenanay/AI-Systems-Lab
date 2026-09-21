/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Docker deployment için standalone output
  output: 'standalone',
  
  // API rewrite (backend proxy)
  async rewrites() {
    const apiUrl = process.env.API_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
  
  // Environment variables
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig;
