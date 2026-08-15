import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: { strictPort: true, port: 5173 },
  preview: { strictPort: true, port: 4173 },
  test: {
    exclude: ["tests/e2e/**", "node_modules/**", "dist/**"]
  }
});
