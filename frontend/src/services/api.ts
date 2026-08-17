import type { ImageInspectionResult, MissionResult, SampleImage } from "../types";

export const backendUrl = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

async function jsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    if (response.status === 502 || response.status === 503 || response.status === 504) {
      throw new Error("Render server was waking up from idle state. Please click again.");
    }
    const text = await response.text();
    if (text.includes("<!DOCTYPE") || text.includes("<html")) {
      throw new Error(`Server returned HTTP ${response.status}. Please refresh and retry.`);
    }
    try {
      const parsed = JSON.parse(text);
      throw new Error(parsed.detail || parsed.message || `Request failed with ${response.status}`);
    } catch {
      throw new Error(text || `Request failed with ${response.status}`);
    }
  }
  return response.json() as Promise<T>;
}

export async function loadLatestDemo(): Promise<MissionResult | null> {
  try {
    const response = await fetch(backendUrl + "/api/demo/latest");
    if (response.status === 404) {
      return null;
    }
    return await jsonResponse<MissionResult>(response);
  } catch {
    return null;
  }
}

export async function runDemo(): Promise<MissionResult> {
  const response = await fetch(backendUrl + "/api/demo/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  });
  return jsonResponse<MissionResult>(response);
}

export async function inspectImage(
  file: File | Blob,
  options: { confidence?: number; modelChoice?: string; runVlm?: boolean } = {}
): Promise<ImageInspectionResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (options.confidence !== undefined) {
    formData.append("confidence", options.confidence.toString());
  }
  if (options.modelChoice) {
    formData.append("model_choice", options.modelChoice);
  }
  if (options.runVlm !== undefined) {
    formData.append("run_vlm", options.runVlm.toString());
  }

  const response = await fetch(backendUrl + "/api/inspect/image", {
    method: "POST",
    body: formData,
  });
  return jsonResponse<ImageInspectionResult>(response);
}

export async function loadSampleImages(): Promise<SampleImage[]> {
  const response = await fetch(backendUrl + "/api/inspect/samples");
  if (!response.ok) return [];
  return jsonResponse<SampleImage[]>(response);
}

export function artifactUrl(reference: string): string {
  return backendUrl + "/" + reference.replace(/^\/+/, "");
}
