import { useEffect, useMemo, useRef, useState } from "react";
import { backendUrl, inspectVideo, loadSampleVideos } from "../services/api";
import type { SampleVideo, VideoFrameDetection, VideoInspectionResult } from "../types";

export function VideoInspector() {
  const [samples, setSamples] = useState<SampleVideo[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleVideo | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0.20);
  const [sampleFps, setSampleFps] = useState<number>(1.0);
  const [modelChoice, setModelChoice] = useState<string>("glasseye-yolo-bfdd-cubit-v1");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VideoInspectionResult | null>(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [selectedPanelFilter, setSelectedPanelFilter] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    async function fetchSamples() {
      const data = await loadSampleVideos();
      setSamples(data);
      if (data.length > 0) {
        setSelectedSample(data[0]);
        setVideoPreviewUrl(backendUrl + data[0].url);
      }
    }
    void fetchSamples();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      setVideoFile(file);
      setSelectedSample(null);
      setResult(null);
      setError(null);
      if (videoPreviewUrl && !videoPreviewUrl.startsWith("http")) {
        URL.revokeObjectURL(videoPreviewUrl);
      }
      setVideoPreviewUrl(URL.createObjectURL(file));
    }
  };

  const selectPreset = (sample: SampleVideo) => {
    setSelectedSample(sample);
    setVideoFile(null);
    setResult(null);
    setError(null);
    if (videoPreviewUrl && !videoPreviewUrl.startsWith("http")) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    setVideoPreviewUrl(backendUrl + sample.url);
  };

  const handleScan = async () => {
    setLoading(true);
    setError(null);
    try {
      let target: File | { sampleFilename: string };

      if (videoFile) {
        target = videoFile;
      } else if (selectedSample) {
        target = { sampleFilename: selectedSample.filename };
      } else {
        throw new Error("Please select a sample video or upload a flight recording.");
      }

      const resData = await inspectVideo(target, {
        confidence,
        sampleFps,
        modelChoice,
      });
      setResult(resData);
      setCurrentFrameIdx(0);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Video flight inspection failed.");
    } finally {
      setLoading(false);
    }
  };

  // Sync video playback time to active sampled frame index
  const handleVideoTimeUpdate = () => {
    if (!videoRef.current || !result || result.frames.length === 0) return;
    const curTime = videoRef.current.currentTime;
    // Find closest frame to current playback time
    let bestIdx = 0;
    let minDiff = 999999;
    result.frames.forEach((f, idx) => {
      const diff = Math.abs(f.timestamp_seconds - curTime);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = idx;
      }
    });
    setCurrentFrameIdx(bestIdx);
  };

  const activeFrame: VideoFrameDetection | null = useMemo(() => {
    if (!result || result.frames.length === 0) return null;
    return result.frames[Math.min(currentFrameIdx, result.frames.length - 1)];
  }, [result, currentFrameIdx]);

  const jumpToFrame = (idx: number) => {
    setCurrentFrameIdx(idx);
    if (videoRef.current && result && result.frames[idx]) {
      videoRef.current.currentTime = result.frames[idx].timestamp_seconds;
    }
  };

  return (
    <section className="video-inspector-section" id="drone-video-scanner">
      <div className="section-header">
        <div>
          <p className="section-kicker">AERIAL DRONE FLIGHT SCANNER</p>
          <h2>End-to-End Drone Video Inspection &amp; 3D Heatmap</h2>
          <p className="muted">
            Upload aerial drone inspection video recordings or choose pre-bundled tower scan presets.
            GlassEye slices frames, executes 640px YOLOv8 defect detection, and maps structural anomalies to facade panel coordinates.
          </p>
        </div>
      </div>

      <div className="inspector-controls">
        <div className="preset-bar">
          <span className="control-label">DRONE FLIGHT PRESETS:</span>
          <div className="preset-buttons">
            {samples.map((s) => (
              <button
                key={s.filename}
                type="button"
                className={`preset-btn ${selectedSample?.filename === s.filename ? "active" : ""}`}
                onClick={() => selectPreset(s)}
                data-testid={`preset-video-${s.filename.split(".")[0]}`}
              >
                {s.title}
              </button>
            ))}
          </div>
        </div>

        <div className="upload-dropzone" onClick={() => fileInputRef.current?.click()} data-testid="video-dropzone">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime"
            style={{ display: "none" }}
            onChange={handleFileChange}
            data-testid="video-file-input"
          />
          <div className="dropzone-inner">
            <span className="upload-icon">🎥</span>
            <p>
              <strong>{videoFile ? videoFile.name : selectedSample ? `Loaded preset: ${selectedSample.title}` : "Drop drone MP4 / WebM video here or click to browse"}</strong>
            </p>
            <span className="subtle-text">Supports MP4, WebM, and MOV drone survey clips up to 50 MB</span>
          </div>
        </div>

        <div className="parameters-strip">
          <div className="param-item">
            <label htmlFor="video-conf-slider">
              CONFIDENCE THRESHOLD: <strong>{confidence.toFixed(2)}</strong>
            </label>
            <input
              id="video-conf-slider"
              type="range"
              min="0.05"
              max="0.60"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
            />
          </div>

          <div className="param-item">
            <label htmlFor="video-fps-select">SAMPLING FREQUENCY:</label>
            <select
              id="video-fps-select"
              value={sampleFps}
              onChange={(e) => setSampleFps(parseFloat(e.target.value))}
            >
              <option value="0.5">0.5 FPS (Rapid Survey)</option>
              <option value="1.0">1.0 FPS (Standard Inspection)</option>
              <option value="2.0">2.0 FPS (Dense Detail Scan)</option>
            </select>
          </div>

          <div className="param-item">
            <label htmlFor="video-model-select">DETECTION ENGINE:</label>
            <select
              id="video-model-select"
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
            >
              <option value="glasseye-yolo-bfdd-cubit-v1">3-Way Unified 640px (Current Best Model)</option>
              <option value="glasseye-yolo-v1">Synthetic Two-Class Model</option>
            </select>
          </div>

          <button
            type="button"
            className="action-btn scan-btn"
            onClick={handleScan}
            disabled={loading}
            data-testid="analyze-video-btn"
          >
            {loading ? "PROCESSING FLIGHT FRAMES…" : "ANALYZE DRONE FLIGHT"}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-banner" role="alert" data-testid="video-error-banner">
          {error}
        </div>
      )}

      {loading && (
        <div className="loading-banner" data-testid="video-loading-banner">
          <span className="pulse-dot" />
          <span>Decoding video stream, sampling keyframes, and running YOLOv8 inference…</span>
        </div>
      )}

      {/* Video Viewport & Real-Time Telemetry HUD */}
      {videoPreviewUrl && (
        <div className="flight-inspection-dashboard">
          <div className="video-player-container">
            <div className="player-header">
              <span className="player-title">FLIGHT RECORDING: {videoFile?.name || selectedSample?.title || "Drone Flight"}</span>
              {result && (
                <span className={`status-pill ${result.health_status.toLowerCase()}`}>
                  {result.health_status.replace(/_/g, " ")}
                </span>
              )}
            </div>

            <div className="video-viewport-wrapper">
              <video
                ref={videoRef}
                src={videoPreviewUrl}
                controls
                playsInline
                className="main-video-element"
                onTimeUpdate={handleVideoTimeUpdate}
                data-testid="main-video-player"
              />

              {/* Real-time HUD Bounding Box Overlay for Active Frame */}
              {activeFrame && activeFrame.detections.length > 0 && (
                <div className="hud-overlay-layer">
                  {activeFrame.detections.map((d) => (
                    <div
                      key={d.detection_id}
                      className={`hud-box ${d.classification_type}`}
                      style={{
                        left: `${d.normalized_bbox[0] * 100}%`,
                        top: `${d.normalized_bbox[1] * 100}%`,
                        width: `${(d.normalized_bbox[2] - d.normalized_bbox[0]) * 100}%`,
                        height: `${(d.normalized_bbox[3] - d.normalized_bbox[1]) * 100}%`,
                      }}
                    >
                      <span className="hud-tag">
                        {d.panel_id} · {d.display_name} ({(d.confidence * 100).toFixed(0)}%)
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Timeline Keyframe Scrubber */}
            {result && result.frames.length > 0 && (
              <div className="timeline-scrubber" data-testid="video-timeline-scrubber">
                <div className="timeline-label">
                  <span>FLIGHT TIMELINE ({result.duration_seconds}s, {result.sampled_frames_count} sampled frames)</span>
                  <span>CURRENT: {activeFrame?.timestamp_seconds.toFixed(1)}s (Frame #{activeFrame?.frame_index})</span>
                </div>
                <div className="scrubber-track">
                  {result.frames.map((f, idx) => (
                    <button
                      key={f.frame_index}
                      type="button"
                      className={`scrubber-node ${currentFrameIdx === idx ? "active" : ""} ${f.has_critical_defect ? "critical" : f.detections_count > 0 ? "warning" : "nominal"}`}
                      onClick={() => jumpToFrame(idx)}
                      title={`Time: ${f.timestamp_seconds}s | Defects: ${f.detections_count}`}
                    >
                      {f.detections_count > 0 && <span className="defect-count-dot">{f.detections_count}</span>}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Telemetry & 4x3 Panel Heatmap Sidebar */}
          {result && (
            <div className="flight-telemetry-sidebar">
              <div className="telemetry-card">
                <h3>FACADE HEALTH SUMMARY</h3>
                <div className="stat-grid">
                  <div className="stat-box">
                    <span className="stat-num">{result.health_score}</span>
                    <span className="stat-lbl">HEALTH SCORE</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-num critical">{result.structural_defect_count}</span>
                    <span className="stat-lbl">STRUCTURAL CRACKS</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-num warning">{result.surface_defect_count}</span>
                    <span className="stat-lbl">SURFACE BLEMISHES</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-num">{result.total_detections_count}</span>
                    <span className="stat-lbl">TOTAL ANOMALIES</span>
                  </div>
                </div>

                <div className="recommendation-box">
                  <span className={`rec-badge ${result.primary_recommendation.urgency.toLowerCase()}`}>
                    {result.primary_recommendation.badge}
                  </span>
                  <p className="rec-summary">{result.primary_recommendation.summary}</p>
                </div>
              </div>

              {/* 4x3 Facade Panel Heatmap */}
              <div className="telemetry-card">
                <h3>4×3 FACADE DAMAGE HEATMAP</h3>
                <p className="subtle-text">Cumulative damage observed during drone trajectory across building panels:</p>
                <div className="panel-heatmap-grid" data-testid="panel-damage-heatmap">
                  {[0, 1, 2, 3].map((r) => (
                    <div key={`row-${r}`} className="heatmap-row">
                      {[0, 1, 2].map((c) => {
                        const pid = `P-${r}-${c}`;
                        const pdata = result.panel_damage_map[pid];
                        const isFiltered = selectedPanelFilter === pid;
                        return (
                          <div
                            key={pid}
                            className={`heatmap-cell ${pdata?.status.toLowerCase() ?? "nominal"} ${isFiltered ? "selected" : ""}`}
                            onClick={() => setSelectedPanelFilter(isFiltered ? null : pid)}
                            title={`Panel ${pid}: ${pdata?.defect_count ?? 0} defects, Max Severity: ${pdata?.max_severity ?? 0}`}
                          >
                            <span className="cell-id">{pid}</span>
                            <span className="cell-count">
                              {pdata && pdata.defect_count > 0 ? `${pdata.defect_count} hits` : "Clean"}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Highlight Frames Reel */}
      {result && result.highlight_frames.length > 0 && (
        <div className="highlight-frames-section" data-testid="video-highlights-reel">
          <h3>CRITICAL FLIGHT DETECTIONS ({result.highlight_frames.length} KEY FRAMES)</h3>
          <div className="highlights-grid">
            {result.highlight_frames.map((hf) => (
              <div
                key={`hl-${hf.frame_index}`}
                className="highlight-card"
                onClick={() => {
                  const idx = result.frames.findIndex((f) => f.frame_index === hf.frame_index);
                  if (idx !== -1) jumpToFrame(idx);
                }}
              >
                {hf.thumbnail_data_uri && (
                  <img src={hf.thumbnail_data_uri} alt={`Frame at ${hf.timestamp_seconds}s`} className="highlight-thumb" />
                )}
                <div className="highlight-meta">
                  <span><strong>{hf.timestamp_seconds}s</strong> (Frame #{hf.frame_index})</span>
                  <span className={`tag ${hf.has_critical_defect ? "structural" : "surface"}`}>
                    {hf.detections_count} defect(s)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
