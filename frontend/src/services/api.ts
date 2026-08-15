import type { MissionResult } from "../types";

export const backendUrl = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

async function jsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed with " + response.status);
  }
  return response.json() as Promise<T>;
}

export async function loadLatestDemo(): Promise<MissionResult | null> {
  const response = await fetch(backendUrl + "/api/demo/latest");
  if (response.status === 404) {
    return null;
  }
  return jsonResponse<MissionResult>(response);
}

export async function runDemo(): Promise<MissionResult> {
  return jsonResponse<MissionResult>(
    await fetch(backendUrl + "/api/demo/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    })
  );
}

export function artifactUrl(reference: string): string {
  return backendUrl + "/" + reference.replace(/^\/+/, "");
}
