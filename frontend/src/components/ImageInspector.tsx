import { useEffect, useRef, useState } from "react";
import { inspectImage, loadSampleImages, backendUrl } from "../services/api";
import type { ImageInspectionResult, SampleImage } from "../types";

export function ImageInspector() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [modelChoice, setModelChoice] = useState<string>("glasseye-yolo-bfdd-cubit-v1");
  const [confidence, setConfidence] = useState<number>(0.15);
  const [runVlm, setRunVlm] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImageInspectionResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void loadSampleImages().then(setSamples).catch(() => {});
  }, []);

  function handleFileSelect(selectedFile: File) {
    setFile(selectedFile);
    setError(null);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setResult(null);
  }

  async function handleSampleSelect(sample: SampleImage) {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch(backendUrl + sample.url);
      if (!response.ok) throw new Error("Failed to fetch sample image");
      const blob = await response.blob();
      const sampleFile = new File([blob], sample.filename, { type: "image/jpeg" });
      setFile(sampleFile);
      setPreviewUrl(URL.createObjectURL(blob));

      // Automatically run analysis on sample select
      const res = await inspectImage(sampleFile, {
        confidence,
        modelChoice,
        runVlm,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sample analysis failed");
    } finally {
      setLoading(false);
    }
  }

  async function runAnalysis() {
    if (!file) {
      setError("Please select or drop an image first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await inspectImage(file, {
        confidence,
        modelChoice,
        runVlm,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inspection failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="inspector-container" id="custom-inspection" data-testid="image-inspector">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">INTERACTIVE FAÇADE SCANNER</p>
          <h2>Upload custom façade imagery for instant YOLO defect scoring & advisory policy</h2>
        </div>
        <span className="subtle-badge">YOLOv8 + VLM</span>
      </div>

      <div className="inspector-controls-card">
        {/* Sample presets */}
        {samples.length > 0 && (
          <div className="sample-presets">
            <span className="presets-label">1-CLICK TEST PRESETS:</span>
            <div className="preset-buttons">
              {samples.map((s) => (
                <button
                  key={s.filename}
                  type="button"
                  className="preset-btn"
                  onClick={() => void handleSampleSelect(s)}
                  disabled={loading}
                >
                  <span className={`preset-dot ${s.expected_type}`} />
                  <strong>{s.title}</strong>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Drag and Drop Zone */}
        <div
          className={`dropzone ${file ? "has-file" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.dataTransfer.files?.[0]) {
              handleFileSelect(e.dataTransfer.files[0]);
            }
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files?.[0]) {
                handleFileSelect(e.target.files[0]);
              }
            }}
          />
          {previewUrl ? (
            <div className="dropzone-preview">
              <img src={previewUrl} alt="Selected preview" />
              <div className="preview-overlay">
                <span>{file?.name} ({Math.round((file?.size ?? 0) / 1024)} KB)</span>
                <small>Click or drag to change image</small>
              </div>
            </div>
          ) : (
            <div className="dropzone-prompt">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <strong>Drag & drop façade photo, or click to browse</strong>
              <p>Supports high-res JPEG, PNG, WebP (up to 15 MB)</p>
            </div>
          )}
        </div>

        {/* Options Row */}
        <div className="inspector-options">
          <div className="option-group">
            <label htmlFor="model-select">Detection Model:</label>
            <select
              id="model-select"
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              disabled={loading}
            >
              <option value="glasseye-yolo-bfdd-cubit-v1">BFDD + CUBIT Model (Recommended)</option>
              <option value="glasseye-yolo-v1">Synthetic 2-Class Model (glasseye-yolo-v1)</option>
            </select>
          </div>

          <div className="option-group">
            <label htmlFor="conf-slider">Confidence Threshold: {Math.round(confidence * 100)}%</label>
            <input
              id="conf-slider"
              type="range"
              min="0.05"
              max="0.60"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              disabled={loading}
            />
          </div>

          <div className="option-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={runVlm}
                onChange={(e) => setRunVlm(e.target.checked)}
                disabled={loading}
              />
              Enable Advisory VLM Review
            </label>
          </div>

          <button
            type="button"
            className="run-button analyze-btn"
            onClick={() => void runAnalysis()}
            disabled={loading || !file}
          >
            {loading ? "EVALUATING FAÇADE…" : "ANALYZE IMAGE WITH YOLO"}
          </button>
        </div>

        {error && <p className="error" role="alert">{error}</p>}
      </div>

      {/* Results View */}
      {result && (
        <div className="inspection-results" data-testid="inspection-result">
          {/* Integrity Score Header */}
          <div className="integrity-banner">
            <div className="score-block">
              <span className="score-label">FAÇADE INTEGRITY INDEX</span>
              <div className="score-number-row">
                <span className={`score-value ${result.health_status.toLowerCase()}`}>{result.health_score}</span>
                <span className="score-denom">/100</span>
              </div>
            </div>

            <div className="status-badge-block">
              <span className={`badge-pill ${result.health_status.toLowerCase()}`}>
                {result.primary_recommendation.badge}
              </span>
              <p className="status-summary">{result.primary_recommendation.summary}</p>
            </div>

            <div className="meta-stats">
              <div><span>ANOMALIES FOUND</span><strong>{result.detections_count}</strong></div>
              <div><span>MODEL</span><strong>{result.model_version}</strong></div>
              <div><span>DIMENSIONS</span><strong>{result.dimensions.width}×{result.dimensions.height}px</strong></div>
            </div>
          </div>

          {/* Side by side image and breakdown */}
          <div className="results-grid">
            {/* Annotated Image */}
            <div className="annotated-card">
              <div className="card-title">
                <span>YOLO BOUNDING BOX ANNOTATION</span>
                <small>4×3 Façade Panel Grid Overlay</small>
              </div>
              <div className="image-wrapper">
                <img
                  src={result.annotated_image}
                  alt={`Annotated inspection result for ${result.filename}`}
                  className="annotated-preview"
                />
              </div>
            </div>

            {/* Recommendations & Details */}
            <div className="details-col">
              {/* Action Plan */}
              <div className="detail-card policy-card">
                <div className="card-title">
                  <span>DISPATCH RECOMMENDATION & ACTION PLAN</span>
                  <span className={`urgency-tag ${result.primary_recommendation.urgency.toLowerCase()}`}>
                    URGENCY: {result.primary_recommendation.urgency}
                  </span>
                </div>
                <div className="action-readout">
                  <span className="policy-badge">{result.primary_recommendation.outcome}</span>
                  <ul className="action-steps">
                    {result.primary_recommendation.action_steps.map((step, idx) => (
                      <li key={idx}>{step}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Advisory VLM */}
              {result.vlm_review && (
                <div className="detail-card vlm-card">
                  <div className="card-title">
                    <span>ADVISORY VLM SECOND OPINION</span>
                    <span className="vlm-meta">{result.vlm_review.provider} · {result.vlm_review.latency_ms}ms</span>
                  </div>
                  <div className="vlm-verdict-row">
                    <strong className={`verdict-tag ${result.vlm_review.verdict}`}>
                      {result.vlm_review.verdict.toUpperCase()}
                    </strong>
                    <p className="vlm-rationale">{result.vlm_review.rationale}</p>
                  </div>
                </div>
              )}

              {/* Detected Anomalies Table */}
              <div className="detail-card detections-card">
                <div className="card-title">
                  <span>DETECTED ANOMALIES ({result.detections_count})</span>
                </div>
                {result.detections.length === 0 ? (
                  <p className="empty-note">No defects identified above confidence threshold.</p>
                ) : (
                  <div className="detections-list">
                    {result.detections.map((det) => (
                      <div key={det.detection_id} className={`detection-row ${det.classification_type}`}>
                        <div className="det-header">
                          <span className="det-panel">PANEL {det.panel_id}</span>
                          <strong>{det.display_name}</strong>
                          <span className="det-conf">{Math.round(det.confidence * 100)}% conf</span>
                        </div>
                        <div className="det-meta">
                          <span>Severity: {det.severity_score}/100</span>
                          <span>Area: {(det.area_fraction * 100).toFixed(2)}% of frame</span>
                          <span>BBox: [{det.bbox_xyxy.map(Math.round).join(", ")}]</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
