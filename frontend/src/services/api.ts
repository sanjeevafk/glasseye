import type { ImageInspectionResult, MissionResult, SampleImage } from "../types";

export const backendUrl = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

interface FetchRetryOptions extends RequestInit {
  maxRetries?: number;
  baseDelayMs?: number;
}

/** Fetch wrapper with automatic retry and exponential backoff for Render cold starts (502/503/504). */
export async function fetchWithRetry(url: string, options: FetchRetryOptions = {}): Promise<Response> {
  const { maxRetries = 3, baseDelayMs = 1200, ...fetchOptions } = options;
  let attempt = 0;

  while (attempt <= maxRetries) {
    try {
      const res = await fetch(url, fetchOptions);
      const isColdStart = res.status === 502 || res.status === 503 || res.status === 504;
      if (isColdStart && attempt < maxRetries) {
        attempt += 1;
        const delay = baseDelayMs * Math.pow(1.8, attempt - 1);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }
      return res;
    } catch (err) {
      if (attempt < maxRetries) {
        attempt += 1;
        const delay = baseDelayMs * Math.pow(1.8, attempt - 1);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }
      throw err;
    }
  }
  throw new Error("Render server is starting up. Please click again in a few seconds.");
}

export async function checkServerHealth(): Promise<boolean> {
  try {
    const res = await fetch(backendUrl + "/health");
    return res.ok;
  } catch {
    return false;
  }
}

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
    const response = await fetchWithRetry(backendUrl + "/api/demo/latest", { maxRetries: 2 });
    if (response.status === 404) {
      return null;
    }
    return await jsonResponse<MissionResult>(response);
  } catch {
    return null;
  }
}

export async function runDemo(): Promise<MissionResult> {
  const response = await fetchWithRetry(backendUrl + "/api/demo/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    maxRetries: 3,
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

  const response = await fetchWithRetry(backendUrl + "/api/inspect/image", {
    method: "POST",
    body: formData,
    maxRetries: 2,
  });
  return jsonResponse<ImageInspectionResult>(response);
}

export async function loadSampleImages(): Promise<SampleImage[]> {
  try {
    const response = await fetchWithRetry(backendUrl + "/api/inspect/samples", { maxRetries: 2 });
    if (!response.ok) return [];
    return jsonResponse<SampleImage[]>(response);
  } catch {
    return [];
  }
}

export async function inspectVideo(
  fileOrSample: File | Blob | { sampleFilename: string },
  options: { confidence?: number; sampleFps?: number; modelChoice?: string } = {}
): Promise<import("../types").VideoInspectionResult> {
  const formData = new FormData();
  if (fileOrSample instanceof Blob) {
    formData.append("file", fileOrSample);
  } else if (fileOrSample && typeof fileOrSample === "object" && "sampleFilename" in fileOrSample) {
    formData.append("sample_filename", fileOrSample.sampleFilename);
  }
  if (options.confidence !== undefined) {
    formData.append("confidence", options.confidence.toString());
  }
  if (options.sampleFps !== undefined) {
    formData.append("sample_fps", options.sampleFps.toString());
  }
  if (options.modelChoice) {
    formData.append("model_choice", options.modelChoice);
  }

  const response = await fetchWithRetry(backendUrl + "/api/inspect/video", {
    method: "POST",
    body: formData,
    maxRetries: 2,
  });
  return jsonResponse<import("../types").VideoInspectionResult>(response);
}

export async function loadSampleVideos(): Promise<import("../types").SampleVideo[]> {
  try {
    const response = await fetchWithRetry(backendUrl + "/api/inspect/video/samples", { maxRetries: 2 });
    if (!response.ok) return [];
    return jsonResponse<import("../types").SampleVideo[]>(response);
  } catch {
    return [];
  }
}

export function artifactUrl(reference: string): string {
  return backendUrl + "/" + reference.replace(/^\/+/, "");
}
