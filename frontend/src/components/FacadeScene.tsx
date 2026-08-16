import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { FacadeIssue } from "../types";

interface Props {
  issues: FacadeIssue[];
}

function markerColor(issue: FacadeIssue): THREE.ColorRepresentation {
  if (issue.status === "RESOLVED") return "#5fc98a";
  if (issue.status === "ESCALATED" || issue.status === "UNRESOLVED") return "#ff647c";
  if (issue.class_name === "structural_issue") return "#eb6c36";
  return "#e8a33d";
}

function fallbackClass(status: string): string {
  if (status === "RESOLVED") return "fallback-resolved";
  if (status === "ESCALATED" || status === "UNRESOLVED") return "fallback-escalated";
  return "fallback-active";
}

function FallbackPanelMap({ issues }: Props) {
  return (
    <div className="facade-canvas facade-fallback" data-testid="facade-canvas">
      <div className="fallback-note">
        3D PANEL MAP UNAVAILABLE — WebGL is disabled in this browser.
      </div>
      <ol className="fallback-panels">
        {issues.map((issue) => (
          <li key={issue.issue_id} className={fallbackClass(issue.status)}>
            <span>Panel {issue.location.panel_id}</span>
            <strong>{issue.status}</strong>
          </li>
        ))}
        {!issues.length && (
          <li className="fallback-empty">Run the seeded mission to populate the panel map.</li>
        )}
      </ol>
    </div>
  );
}

export function FacadeScene({ issues }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [webglAvailable, setWebglAvailable] = useState(true);

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const container: HTMLDivElement = element;

    let renderer: THREE.WebGLRenderer | null = null;
    let observer: ResizeObserver | null = null;
    try {
      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#19140e");
      const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
      camera.position.set(0, 0.2, 8.2);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.replaceChildren(renderer.domElement);

      const ambient = new THREE.AmbientLight("#d8cdb6", 1.1);
      const key = new THREE.DirectionalLight("#f2e9d8", 1.3);
      key.position.set(1.5, 2.5, 5);
      scene.add(ambient, key);

      const facade = new THREE.Group();
      const panelGeometry = new THREE.BoxGeometry(1.55, 1.17, 0.14);
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 4; column += 1) {
          const material = new THREE.MeshStandardMaterial({
            color: (row + column) % 2 === 0 ? "#8a7f6c" : "#958a74",
            roughness: 0.82,
            metalness: 0.08
          });
          const panel = new THREE.Mesh(panelGeometry, material);
          panel.position.set(-2.325 + column * 1.55, 1.17 - row * 1.17, 0);
          facade.add(panel);
        }
      }
      scene.add(facade);

      const markerGeometry = new THREE.SphereGeometry(0.14, 28, 20);
      for (const issue of issues) {
        const [normalizedX, normalizedY] = issue.location.normalized_centroid;
        const marker = new THREE.Mesh(
          markerGeometry,
          new THREE.MeshStandardMaterial({
            color: markerColor(issue),
            emissive: markerColor(issue),
            emissiveIntensity: 0.55,
            metalness: 0.15,
            roughness: 0.35
          })
        );
        marker.position.set(-3.1 + normalizedX * 6.2, 1.75 - normalizedY * 3.5, 0.28);
        scene.add(marker);
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.22, 0.025, 10, 30),
          new THREE.MeshBasicMaterial({ color: markerColor(issue) })
        );
        ring.position.copy(marker.position);
        ring.position.z -= 0.03;
        scene.add(ring);
      }

      function render() {
        if (!renderer) return;
        const width = Math.max(1, container.clientWidth);
        const height = Math.max(260, container.clientHeight);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.render(scene, camera);
      }
      observer = new ResizeObserver(render);
      observer.observe(container);
      render();
    } catch (error) {
      // WebGL unavailable (blocked GPU, VM, remote desktop): degrade to a
      // static panel list instead of crashing the whole dashboard.
      console.warn("FacadeScene: WebGL unavailable, showing fallback panel map.", error);
      setWebglAvailable(false);
      container.replaceChildren();
    }
    return () => {
      if (observer) observer.disconnect();
      if (renderer) renderer.dispose();
      container.replaceChildren();
    };
  }, [issues]);

  return webglAvailable ? (
    <div
      ref={host}
      className="facade-canvas"
      data-testid="facade-canvas"
      aria-label="Three dimensional facade panel map"
    />
  ) : (
    <FallbackPanelMap issues={issues} />
  );
}
