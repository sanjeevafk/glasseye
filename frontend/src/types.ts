export type DefectClass = "cleanable_surface_issue" | "structural_issue";
export type IssueStatus =
  | "IDLE"
  | "INSPECTING"
  | "DETECTED"
  | "EVIDENCE_READY"
  | "DECIDED"
  | "CLEANING"
  | "REINSPECTING"
  | "RESOLVED"
  | "UNRESOLVED"
  | "ESCALATED"
  | "REVIEW";

export interface Evidence {
  evidence_id: string;
  frame_id: string;
  timestamp: number;
  artifact_ref: string;
  bbox_xyxy: number[];
}

export interface FacadeIssue {
  issue_id: string;
  track_id: number;
  class_name: DefectClass;
  confidence: number;
  bbox_xyxy: number[];
  location: { panel_id: string; normalized_centroid: number[] };
  evidence: Evidence[];
  decision: { outcome: "CLEAN" | "ESCALATE" | "REVIEW"; reason_code: string; confidence: number };
  status: IssueStatus;
  action_taken?: string | null;
  verification_reason?: string | null;
}

export interface MissionEvent {
  event_id: string;
  mission_id: string;
  sequence: number;
  timestamp: number;
  event_type: string;
  source: string;
  issue_id?: string | null;
  track_id?: number | null;
  reason_code: string;
  evidence_refs: string[];
  payload: Record<string, unknown>;
}

export interface VideoInference {
  video_id: string;
  fps: number;
  frame_count: number;
  sampled_frames: number;
  model_version: string;
  elapsed_seconds: number;
  frames: Array<{
    frame_id: string;
    timestamp: number;
    image_id: string;
    model_version: string;
    detections: Array<{
      class_name: DefectClass;
      class_id: number;
      confidence: number;
      bbox_xyxy: number[];
      mask: null | number[][];
      track_id: number | null;
    }>;
  }>;
}

export interface MissionResult {
  mission_id: string;
  scenario_seed: number;
  model_version: string;
  model_path: string;
  state: string;
  issues: FacadeIssue[];
  events: MissionEvent[];
  preinspection: VideoInference;
  reinspection: VideoInference;
  event_log_ref: string;
  replay_digest: string;
  inference_benchmark: Record<string, number>;
}
