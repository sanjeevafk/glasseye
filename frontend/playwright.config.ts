import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const python = process.env.GLASSEYE_PYTHON ?? path.join(root, ".venv", "bin", "python");

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 180_000,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure"
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "PYTHONPATH=backend " + python + " -m uvicorn app.main:app --host 127.0.0.1 --port 8000",
      cwd: root,
      url: "http://127.0.0.1:8000/health",
      timeout: 180_000,
      reuseExistingServer: true
    },
    {
      command: "npm run dev",
      cwd: here,
      url: "http://127.0.0.1:5173",
      timeout: 60_000,
      reuseExistingServer: true
    }
  ]
});
