import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/inspect": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
    },
    watch: {
      // The backend rewrites logs/ and data/ on every request; watching them
      // causes a full page reload per API call.
      ignored: ["**/logs/**", "**/data/**", "**/models/**", "**/dist/**", "**/vendor/**"],
    },
  },
  optimizeDeps: { exclude: [] },
  build: {
    target: "es2020",
  },
});
