import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static site: pages fetch their data in the browser, from the live API in development
  // (NEXT_PUBLIC_API_URL) or from the public/data snapshot written by `uv run export-static`.
  output: "export",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
