import { useEffect, useMemo, useState } from "react";
import { FacadeScene } from "./components/FacadeScene";
import { artifactUrl, loadLatestDemo, runDemo } from "./services/api";
import { displayClassName, statusClass } from "./status";
import type { MissionResult } from "./types";

function visibleEvents(result: MissionResult | null, cursor: number) {
  return result ? result.events.slice(0, cursor) : [];
}

function simulatedActuatorEvent(result: MissionResult | null) {
  return result?.events.find((event) => event.event_type === "SIMULATED_CLEANING_COMPLETED") ?? null;
}

export default function App() {
  const [result, setResult] = useState<MissionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showActuatorSignal, setShowActuatorSignal] = useState(false);

  useEffect(() => {
    void loadLatestDemo()
      .then((latest) => {
        setResult(latest);
        setCursor(latest?.events.length ?? 0);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Unable to load demo"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!playing || !result || cursor >= result.events.length) return;
    const handle = window.setTimeout(() => setCursor((value) => value + 1), 850);
    return () => window.clearTimeout(handle);
  }, [cursor, playing, result]);

  const events = useMemo(() => visibleEvents(result, cursor), [result, cursor]);
  const actuatorEvent = simulatedActuatorEvent(result);

  async function executeDemo() {
    setRunning(true);
    setPlaying(false);
    setError(null);
    try {
      const next = await runDemo();
      setResult(next);
      setCursor(next.events.length);
      setShowActuatorSignal(Boolean(simulatedActuatorEvent(next)));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Demo execution failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">GLASSEYE / FACADE MISSION CONTROL</p>
          <h1>See the defect. Prove the outcome.</h1>
        </div>
        <div className="simulation-pill">SIMULATION ONLY · NO PHYSICAL ACTUATORS</div>
      </header>

      <section className="hero">
        <div>
          <p className="section-kicker">DETERMINISTIC CLOSED LOOP</p>
          <h2>YOLO inspection → evidence → policy → simulation → reinspection</h2>
          <p className="muted">
            A fixed facade scenario runs through actual trained-model video inference. The panel map,
            evidence, outcome, and replay are all projections of the same append-only event log.
          </p>
        </div>
        <button
          className="run-button"
          type="button"
          onClick={() => void executeDemo()}
          disabled={running}
          data-testid="run-demo"
        >
          {running ? "RUNNING YOLO INSPECTION…" : "RUN DETERMINISTIC DEMO"}
        </button>
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      {showActuatorSignal && actuatorEvent && (
        <div className="actuator-backdrop" role="presentation">
          <section
            aria-describedby="actuator-command-detail"
            aria-labelledby="actuator-command-title"
            aria-modal="true"
            className="actuator-modal"
            data-testid="actuator-command-modal"
            role="dialog"
          >
            <p className="section-kicker">DRONE BRAIN / COMMAND SIGNAL</p>
            <h2 id="actuator-command-title">CLEANING COMMAND DISPATCHED</h2>
            <p id="actuator-command-detail">
              Target: facade panel {result?.issues.find((issue) => issue.issue_id === actuatorEvent.issue_id)?.location.panel_id ?? "—"}.
              The mission policy approved a surface-cleaning action and recorded the command in the event log.
            </p>
            <div className="command-readout">
              <span>COMMAND ID</span>
              <strong>{String(actuatorEvent.payload.action_id ?? actuatorEvent.event_id)}</strong>
              <span>STATUS</span>
              <strong>{String(actuatorEvent.payload.status ?? "SIMULATED_COMPLETE")}</strong>
            </div>
            <p className="simulation-note">Software command signal only — no physical actuator was controlled.</p>
            <button type="button" onClick={() => setShowActuatorSignal(false)}>ACKNOWLEDGE</button>
          </section>
        </div>
      )}

      <section className="metrics" aria-label="Mission metrics">
        <div><span>MISSION</span><strong>{result?.state ?? (loading ? "LOADING" : "STANDBY")}</strong></div>
        <div><span>MODEL</span><strong>{result?.model_version ?? "—"}</strong></div>
        <div><span>EVENTS</span><strong>{result?.events.length ?? 0}</strong></div>
        <div><span>SEEDED SCENARIO</span><strong>{result?.scenario_seed ?? "—"}</strong></div>
      </section>

      <section className="dashboard-grid">
        <article className="panel facade-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">KNOWN FACADE GEOMETRY</p>
              <h2>Panel state</h2>
            </div>
            <span className="subtle-badge">THREE.JS</span>
          </div>
          <FacadeScene issues={result?.issues ?? []} />
          <div className="legend">
            <span><i className="dot green" /> resolved</span>
            <span><i className="dot red" /> escalated</span>
            <span><i className="dot blue" /> active / review</span>
          </div>
        </article>

        <article className="panel timeline-panel">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">APPEND-ONLY EVENT LOG</p>
              <h2>Replay timeline</h2>
            </div>
            <span className="subtle-badge">{cursor}/{result?.events.length ?? 0}</span>
          </div>
          <div className="replay-controls">
            <button
              type="button"
              onClick={() => setPlaying((value) => !value)}
              disabled={!result || cursor >= result.events.length}
            >
              {playing ? "PAUSE" : "PLAY"}
            </button>
            <button type="button" onClick={() => { setPlaying(false); setCursor(0); }} disabled={!result}>RESET</button>
            <button
              type="button"
              onClick={() => setCursor((value) => Math.min(value + 1, result?.events.length ?? 0))}
              disabled={!result || cursor >= result.events.length}
            >
              STEP
            </button>
          </div>
          <ol className="timeline" data-testid="timeline">
            {events.map((event) => (
              <li key={event.event_id}>
                <time>{event.timestamp.toFixed(1)}s</time>
                <div>
                  <strong>{event.event_type.replaceAll("_", " ")}</strong>
                  <span>{event.reason_code.replaceAll("_", " ")}</span>
                </div>
              </li>
            ))}
            {!events.length && <li className="empty">Run the seeded mission to populate its evidence trail.</li>}
          </ol>
        </article>
      </section>

      <section className="issue-section">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">INSPECTION OUTCOMES</p>
            <h2>Tracked facade issues</h2>
          </div>
          {result && <span className="subtle-badge">REPLAY {result.replay_digest.slice(0, 10)}</span>}
        </div>
        <div className="issue-grid">
          {(result?.issues ?? []).map((issue) => (
            <article className="issue-card" key={issue.issue_id} data-testid={"issue-" + issue.class_name}>
              <div className="issue-title">
                <span className={"status " + statusClass(issue.status)}>{issue.status}</span>
                <span>Panel {issue.location.panel_id}</span>
              </div>
              <h3>{displayClassName(issue.class_name)}</h3>
              <p className="confidence">{Math.round(issue.confidence * 100)}% detector confidence · Track {issue.track_id}</p>
              {issue.evidence[0] && (
                <img
                  className="evidence"
                  src={artifactUrl(issue.evidence[0].artifact_ref)}
                  alt={"Evidence crop for " + displayClassName(issue.class_name)}
                />
              )}
              <dl>
                <div><dt>Policy</dt><dd>{issue.decision.outcome}</dd></div>
                <div><dt>Action</dt><dd>{issue.action_taken ?? "Awaiting action"}</dd></div>
                <div><dt>Verification</dt><dd>{issue.verification_reason ?? "Not yet reinspected"}</dd></div>
              </dl>
            </article>
          ))}
          {!result && <p className="empty">No issues have been inspected yet.</p>}
        </div>
      </section>
    </main>
  );
}
