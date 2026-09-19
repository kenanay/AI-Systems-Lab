/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // API rewrite (backend proxy)
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  
  // Environment variables
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig;
