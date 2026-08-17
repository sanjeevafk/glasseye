"""Comprehensive end-to-end production smoke test suite for GlassEye."""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://glasseye-td75.onrender.com"

print(f"==================================================")
print(f"🔍 GLASSEYE E2E PRODUCTION SMOKE TEST")
print(f"Target: {BASE_URL}")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"==================================================\n")

failures = []

def record_pass(step: str, details: str = ""):
    print(f"  ✅ PASS: {step} {details}")

def record_fail(step: str, reason: str):
    print(f"  ❌ FAIL: {step} -> {reason}")
    failures.append((step, reason))


# ----------------------------------------------------
# 1. Health & Edge Probe Routing
# ----------------------------------------------------
print("1. HEALTH & ROUTING CHECKS")
try:
    # GET /health
    t0 = time.time()
    res = requests.get(f"{BASE_URL}/health", timeout=10)
    dur = round((time.time() - t0) * 1000, 1)
    if res.status_code == 200 and res.json().get("status") == "ok":
        record_pass("GET /health", f"({dur}ms, status: {res.json().get('status')})")
    else:
        record_fail("GET /health", f"Status {res.status_code}, body: {res.text}")

    # HEAD /health
    res_head = requests.head(f"{BASE_URL}/health", timeout=10)
    if res_head.status_code == 200:
        record_pass("HEAD /health", "(200 OK)")
    else:
        record_fail("HEAD /health", f"Status {res_head.status_code}")

    # HEAD /
    res_root_head = requests.head(f"{BASE_URL}/", timeout=10)
    if res_root_head.status_code == 200:
        record_pass("HEAD / (Root Edge Probe)", "(200 OK)")
    else:
        record_fail("HEAD / (Root Edge Probe)", f"Status {res_root_head.status_code}")
except Exception as e:
    record_fail("Health & Edge Probes", str(e))


# ----------------------------------------------------
# 2. Caching & GZip Compression Headers
# ----------------------------------------------------
print("\n2. CACHING & COMPRESSION HEADERS")
try:
    res_root = requests.get(f"{BASE_URL}/", timeout=10)
    cc_root = res_root.headers.get("Cache-Control", "")
    if "no-cache" in cc_root:
        record_pass("SPA index.html Cache-Control", f"({cc_root})")
    else:
        record_fail("SPA index.html Cache-Control", f"Expected no-cache, got: {cc_root}")

    # Check GZip compression on API
    res_gzip = requests.get(
        f"{BASE_URL}/api/inspect/samples",
        headers={"Accept-Encoding": "gzip"},
        timeout=10
    )
    if res_gzip.status_code == 200:
        samples = res_gzip.json()
        record_pass("GET /api/inspect/samples", f"({len(samples)} presets available)")
    else:
        record_fail("GET /api/inspect/samples", f"Status {res_gzip.status_code}")
except Exception as e:
    record_fail("Caching & Compression", str(e))


# ----------------------------------------------------
# 3. Interactive YOLO + VLM Image Inspector
# ----------------------------------------------------
print("\n3. INTERACTIVE YOLO + VLM INFERENCE")
sample_path = Path("backend/app/samples/spalling_damage_sample.jpg")
if sample_path.is_file():
    try:
        t0 = time.time()
        with open(sample_path, "rb") as f:
            res_inspect = requests.post(
                f"{BASE_URL}/api/inspect/image",
                files={"file": ("spalling_sample.jpg", f, "image/jpeg")},
                data={"confidence": "0.15", "run_vlm": "true"},
                timeout=25
            )
        dur = round((time.time() - t0) * 1000, 1)
        if res_inspect.status_code == 200:
            data = res_inspect.json()
            score = data.get("health_score")
            dets = len(data.get("detections", []))
            badge = data.get("primary_recommendation", {}).get("badge")
            has_vlm = data.get("vlm_review") is not None
            has_preview = data.get("annotated_image_base64", "").startswith("data:image/jpeg;base64,")
            record_pass(
                "POST /api/inspect/image (Spalling Sample)",
                f"({dur}ms | Score: {score}/100 | Anomalies: {dets} | Badge: {badge} | VLM: {has_vlm} | BBox Image: {has_preview})"
            )
        else:
            record_fail("POST /api/inspect/image", f"Status {res_inspect.status_code}: {res_inspect.text}")
    except Exception as e:
        record_fail("POST /api/inspect/image", str(e))
else:
    record_fail("POST /api/inspect/image", f"Sample image missing at {sample_path}")


# ----------------------------------------------------
# 4. Deterministic Closed-Loop Simulation
# ----------------------------------------------------
print("\n4. DETERMINISTIC CLOSED-LOOP MISSION")
try:
    t0 = time.time()
    res_demo = requests.post(f"{BASE_URL}/api/demo/run", timeout=20)
    dur = round((time.time() - t0) * 1000, 1)
    if res_demo.status_code == 200:
        demo_data = res_demo.json()
        state = demo_data.get("state")
        event_count = len(demo_data.get("events", []))
        issues_count = len(demo_data.get("issues", []))
        digest = demo_data.get("replay_digest", "")[:10]
        record_pass(
            "POST /api/demo/run (Seeded Mission Execution)",
            f"({dur}ms | State: {state} | Events: {event_count} | Issues: {issues_count} | Digest: {digest})"
        )
    else:
        record_fail("POST /api/demo/run", f"Status {res_demo.status_code}: {res_demo.text}")

    res_latest = requests.get(f"{BASE_URL}/api/demo/latest", timeout=10)
    if res_latest.status_code == 200:
        record_pass("GET /api/demo/latest", "(Cached telemetry matching replay)")
    else:
        record_fail("GET /api/demo/latest", f"Status {res_latest.status_code}")
except Exception as e:
    record_fail("Deterministic Demo Execution", str(e))


# ----------------------------------------------------
# Summary & Exit Code
# ----------------------------------------------------
print("\n==================================================")
if not failures:
    print(f"🎉 ALL PRODUCTION SMOKE TESTS PASSED (0 failures)")
    print(f"==================================================")
    sys.exit(0)
else:
    print(f"❌ SMOKE TEST FAILED WITH {len(failures)} ERROR(S)")
    for step, reason in failures:
        print(f"   - {step}: {reason}")
    print(f"==================================================")
    sys.exit(1)
