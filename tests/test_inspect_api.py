import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SCRIPT = PROJECT_ROOT / "backend" / "serve.py"
TEST_IMAGES = PROJECT_ROOT / "data" / "test_images"
PORT = 8010
BASE_URL = f"http://127.0.0.1:{PORT}"


def _ensure_backend_running():
    env = dict(os.environ, PORT=str(PORT))
    proc = subprocess.Popen(
        [sys.executable, str(BACKEND_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=1) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("Backend did not start in time")


def _post_image(image_path: Path, threshold=0.5):
    boundary = "boundary-do-not-use-this"
    with image_path.open("rb") as f:
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8")
        body += f.read()
        body += (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="confidence_threshold"\r\n\r\n'
            f"{threshold}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/api/inspect",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _assert_valid_payload(payload):
    assert isinstance(payload, dict), "Inspection response should be JSON object"
    assert payload.get("predicted_class") in {
        "no_defect",
        "crack",
        "scratch",
        "discoloration",
    }, "Unexpected predicted class"
    confidence = float(payload.get("confidence", -1))
    assert 0.0 <= confidence <= 1.0, "Confidence must be in 0..1"
    assert payload.get("risk_level") in {"Critical", "High", "Medium", "Low"}
    assert isinstance(payload.get("rpn"), (int, float))
    probs = payload.get("probabilities") or {}
    assert isinstance(probs, dict)
    assert abs(sum(probs.values()) - 1.0) < 1e-6, "Probabilities should sum to 1"


def test_detects_each_defect_class_correctly():
    proc = _ensure_backend_running()
    try:
        expected = {
            "crack_sample.png": "crack",
            "scratch_sample.png": "scratch",
            "discoloration_sample.png": "discoloration",
            "clean_sample.png": "no_defect",
        }
        for name, exp in expected.items():
            img = TEST_IMAGES / name
            assert img.exists(), f"missing test image {name}; run backend/make_test_images.py"
            payload = _post_image(img)
            _assert_valid_payload(payload)
            assert payload["predicted_class"] == exp, (
                f"{name}: expected {exp}, got {payload['predicted_class']} "
                f"(probs={payload['probabilities']})"
            )
            assert payload["threshold_met"] == (payload["confidence"] >= 0.5)
    finally:
        _shutdown(proc)


def test_real_carbon_crack_detected():
    proc = _ensure_backend_running()
    try:
        img = TEST_IMAGES / "real_carbon_crack.png"
        assert img.exists(), "missing real sample crop"
        payload = _post_image(img)
        _assert_valid_payload(payload)
        assert payload["predicted_class"] == "crack", (
            f"real carbon sample should be crack, got {payload['predicted_class']}"
        )
        assert payload["confidence"] >= 0.4
        assert payload["risk_level"] in {"Critical", "High"}
    finally:
        _shutdown(proc)


def test_history_is_recorded_and_listed():
    proc = _ensure_backend_running()
    try:
        img = TEST_IMAGES / "crack_sample.png"
        first = _post_image(img)
        _assert_valid_payload(first)
        with urllib.request.urlopen(f"{BASE_URL}/api/inspections", timeout=5) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        assert isinstance(rows, list) and len(rows) >= 1
        assert any(r.get("id") == first["id"] for r in rows), "new inspection should appear in history"
    finally:
        _shutdown(proc)


def test_non_image_upload_rejected():
    proc = _ensure_backend_running()
    try:
        bad = PROJECT_ROOT / "data" / "README.md"
        try:
            payload = _post_image(bad)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        raise AssertionError(f"expected 400 for non-image upload, got {payload}")
    finally:
        _shutdown(proc)


def _shutdown(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
