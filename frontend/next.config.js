/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep `next dev` artifacts separate from production builds. Sharing `.next`
  // can leave a running dev server serving HTML for assets that no longer exist.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
};

module.exports = nextConfig;
