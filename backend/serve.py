import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"FastAPI not available: {exc}") from exc

try:
    import numpy as np
    from PIL import Image
    from scipy import ndimage
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Image backend deps unavailable: {exc}") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = LOGS_DIR / "inspections.jsonl"
LEARNER_PATH = MODELS_DIR / "learner.joblib"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CompositeGuard AI Backend")
# Allow direct cross-origin use (a separately served frontend, or the raw
# image <img> fetch); the Vite dev proxy also covers same-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The detection taxonomy mirrors the trained ONNX segmentation model's own
# output classes (see ONNX_CLASSES below): the model detects Fiber Breakage,
# Porosity/Cracks, Foreign Inclusions and Delamination, and classical CV adds
# Discoloration (the model has no such class).
CLASSES = [
    "no_defect",
    "fiber_breakage",
    "porosity_cracks",
    "foreign_inclusion",
    "delamination",
    "discoloration",
]
# The classes that represent an actual defect (vs. the single no_defect bucket).
DEFECT_CLASSES = (
    "fiber_breakage",
    "porosity_cracks",
    "foreign_inclusion",
    "delamination",
    "discoloration",
)

# FMEA severity per defect class (1-10 scale)
SEVERITY = {
    "fiber_breakage": 9,
    "porosity_cracks": 8,
    "foreign_inclusion": 5,
    "delamination": 9,
    "discoloration": 4,
    "no_defect": 1,
}
# FMEA detection rating per class (higher = harder to catch early)
DETECTION = {
    "fiber_breakage": 6,
    "porosity_cracks": 6,
    "foreign_inclusion": 4,
    "delamination": 8,
    "discoloration": 4,
    "no_defect": 2,
}
# Map the model's human-readable class names to our taxonomy keys.
MODEL_CLASS_MAP = {
    "Fiber Breakage": "fiber_breakage",
    "Porosity / Cracks": "porosity_cracks",
    "Foreign Inclusions": "foreign_inclusion",
    "Delamination": "delamination",
}

# ---------------------------------------------------------------------------
# Online learner
#
# The deterministic CV pipeline above is the primary detector. This learner
# captures operator feedback (confirm/correct on a past inspection): it fits a
# lightweight classifier on the CV *features* of each specimen and blends its
# probabilities into the heuristic ones as labelled evidence accumulates.
# The model is persisted to models/learner.joblib and reloaded on startup.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "crack_score",
    "scratch_score",
    "discoloration_score",
    "n_crack_regions",
    "n_scratch_regions",
    "n_discoloration_regions",
    "mean_luminance",
    "contrast",
    "mean_chroma_dev",
    "edge_density",
]
# Minimum labelled samples before the learner is trusted at all.
LEARNER_MIN_SAMPLES = 6
# Weight ramps from 0 at MIN_SAMPLES up to LEARNER_MAX_WEIGHT (heuristic stays
# the dominant signal; the learner refines it).
LEARNER_MAX_WEIGHT = 0.45

_learner_lock = threading.Lock()
_learner = None            # sklearn classifier (loaded lazily)
_X_window = []             # feature vectors for labelled samples
_y_window = []             # operator labels for those samples
_learner_n_samples = 0     # total labelled samples ever (persisted)
_learner_dirty = False


def _image_features(pil_img, info):
    """Numeric feature vector: CV analysis outputs + coarse image statistics."""
    small = pil_img.convert("RGB").copy()
    small.thumbnail((128, 128))
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    lum = rgb.mean(axis=2)
    chroma = rgb / (rgb.sum(axis=2, keepdims=True) + 1e-6)
    # deviation of blurred hue from its median = discoloration-like drift
    blur = np.stack([ndimage.gaussian_filter(chroma[..., c], 4.0) for c in range(3)], axis=2)
    med = np.median(blur.reshape(-1, 3), axis=0)
    mean_chroma_dev = float(np.linalg.norm(blur - med, axis=2).mean())
    gy, gx = np.gradient(lum)
    edge_density = float(np.mean(np.hypot(gx, gy) > 0.08))
    return [
        float(info.get("crack_score", 0.0)),
        float(info.get("scratch_score", 0.0)),
        float(info.get("discoloration_score", 0.0)),
        float(info.get("n_crack_regions", 0)),
        float(info.get("n_scratch_regions", 0)),
        float(info.get("n_discoloration_regions", 0)),
        float(lum.mean()),
        float(lum.std()),
        mean_chroma_dev,
        edge_density,
    ]


def _load_learner():
    """Load the persisted learner + labelled buffer (idempotent)."""
    global _learner, _learner_n_samples, _X_window, _y_window
    with _learner_lock:
        if _learner is not None or not LEARNER_PATH.is_file():
            return
        try:
            import joblib

            state = joblib.load(LEARNER_PATH)
            _learner = state.get("model")
            _X_window = list(state.get("X", []))
            _y_window = list(state.get("y", []))
            _learner_n_samples = int(state.get("n", 0))
        except Exception:
            # A corrupt/legacy model file must never take the API down.
            _learner, _X_window, _y_window, _learner_n_samples = None, [], [], 0


def _save_learner():
    global _learner_dirty
    with _learner_lock:
        if not _learner_dirty:
            return
        try:
            import joblib

            joblib.dump(
                {"model": _learner, "X": _X_window, "y": _y_window, "n": _learner_n_samples},
                LEARNER_PATH,
            )
            _learner_dirty = False
        except Exception:
            pass


def _train_on_feedback(x, y_label):
    """Append a labelled sample, refit the classifier, and persist it.

    Uses Gaussian naive Bayes on the 10 CV features (GaussianNB imports and
    fits reliably here — some sklearn tree/DLL paths are blocked by the host's
    Application Control policy). Refit is cheap at this data scale and keeps
    the model state simple and inspectable. Two or more distinct classes are
    required before fitting; until then the buffer just accumulates.
    """
    global _learner, _learner_n_samples, _learner_dirty
    _load_learner()
    with _learner_lock:
        _X_window.append([float(v) for v in x])
        _y_window.append(y_label)
        _learner_n_samples += 1
        if len(_y_window) >= 2 and len(set(_y_window)) >= 2:
            try:
                from sklearn.naive_bayes import GaussianNB

                _learner = GaussianNB()
                _learner.fit(np.asarray(_X_window), np.asarray(_y_window))
                _learner_dirty = True
            except Exception:
                # Never let a training failure take the inspection API down.
                pass
    _save_learner()
    return {
        "samples": _learner_n_samples,
        "classes": sorted(set(_y_window)),
        "model_trained": _learner is not None,
    }


def _learner_predict(info, features):
    """Blend learner probabilities into heuristic ones (or None if untrusted).

    Returns (probabilities_or_None, meta). Confidence is the fraction of
    labelled samples seen so far, ramped by the weight schedule.
    """
    _load_learner()
    with _learner_lock:
        if _learner is None or _learner_n_samples < LEARNER_MIN_SAMPLES:
            return None, {"samples": _learner_n_samples, "weight": 0.0}
        try:
            proba = _learner.predict_proba(np.asarray([features]))[0]
            classes = list(_learner.classes_)
        except Exception:
            return None, {"samples": _learner_n_samples, "weight": 0.0}
        n = _learner_n_samples
        ramp = min(1.0, (n - LEARNER_MIN_SAMPLES + 1) / (LEARNER_MIN_SAMPLES * 2))
        weight = round(LEARNER_MAX_WEIGHT * ramp, 4)
    blended = {c: round((1.0 - weight) * v, 6) for c, v in info.items()}
    for cls, p in zip(classes, proba):
        blended[cls] = round(blended.get(cls, 0.0) + weight * float(p), 6)
    total = sum(blended.values()) or 1.0
    blended = {c: round(v / total, 6) for c, v in blended.items()}
    top = max(blended, key=blended.get)
    blended[top] = round(blended[top] + (1.0 - sum(blended.values())), 6)
    return blended, {"samples": n, "weight": weight}


# ---------------------------------------------------------------------------
# ONNX segmentation model (compositeguard_v4)
#
# A trained 5-class semantic segmentation model for composite NDE imagery:
#   0 Background/Clean  1 Fiber Breakage  2 Porosity/Cracks
#   3 Foreign Inclusions  4 Delamination
#
# It runs alongside the classical CV pipeline. Fusion is deliberately
# ASYMMETRIC: the model's defect evidence can only *strengthen* a defect
# verdict (by suppressing no_defect mass), never veto a defect the classical
# pipeline found — the model was trained on synthetic data and mis-ranks the
# deterministic test specimens the suite pins.
# ---------------------------------------------------------------------------
ONNX_MODEL_PATH = MODELS_DIR / "compositeguard_v4.onnx"
ONNX_IMAGE_SIZE = (256, 256)
ONNX_CLASSES = {
    0: "Background / Clean",
    1: "Fiber Breakage",
    2: "Porosity / Cracks",
    3: "Foreign Inclusions",
    4: "Delamination",
}
# Fusion weight: at evidence=1 the no_defect probability is scaled by
# (1 - weight); verdict flips past that are still impossible for confident
# heuristics (>0.7 no_defect survives the worst case).
ONNX_FUSION_WEIGHT = 0.30
# Per-pixel softmax floor for a defect pixel to be counted (accuracy filter).
ONNX_PIXEL_CONF = 0.45
# ImageNet normalization as used in model training.
_ONNX_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_ONNX_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# FMEA severity/detection per modelled failure mechanism.
MECHANISM_FMEA = {
    "Fiber Breakage": {"severity": 9, "detection": 6},
    "Porosity / Cracks": {"severity": 8, "detection": 6},
    "Foreign Inclusions": {"severity": 5, "detection": 4},
    "Delamination": {"severity": 9, "detection": 8},
}

_onnx_lock = threading.Lock()
_onnx_session = None
_onnx_unavailable = False


def _get_onnx():
    """Lazy-load the ONNX session; None if the model/dep is missing."""
    global _onnx_session, _onnx_unavailable
    if _onnx_session is not None or _onnx_unavailable:
        return _onnx_session
    with _onnx_lock:
        if _onnx_session is not None or _onnx_unavailable:
            return _onnx_session
        if not ONNX_MODEL_PATH.is_file():
            _onnx_unavailable = True
            print(f"[onnx] model not found: {ONNX_MODEL_PATH}", flush=True)
            return None
        try:
            import onnxruntime as ort

            _onnx_session = ort.InferenceSession(
                str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
            )
            print(f"[onnx] loaded {ONNX_MODEL_PATH.name}", flush=True)
        except Exception as exc:  # missing dep / corrupt file must not kill API
            _onnx_unavailable = True
            print(f"[onnx] load failed: {exc}", flush=True)
    return _onnx_session


def _segment_image(pil_img, overlay_path=None):
    """Run ONNX segmentation; returns class stats or None when unavailable.

    When overlay_path is given, a colour-blended overlay JPEG is saved there.
    """
    sess = _get_onnx()
    if sess is None:
        return None
    try:
        img = pil_img.convert("RGB").resize(ONNX_IMAGE_SIZE, Image.Resampling.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - _ONNX_MEAN) / _ONNX_STD
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        logits = sess.run([out.name], {inp.name: x})[0]
        if logits.ndim != 4 or logits.shape[1] != len(ONNX_CLASSES):
            print(f"[onnx] unexpected output shape {logits.shape}", flush=True)
            return None

        # softmax over the class axis → mean per-class probability + mask
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        pr = e / e.sum(axis=1, keepdims=True)
        mean_probs = [float(v) for v in pr[0].mean(axis=(1, 2))]
        prob_map = pr[0]                         # [5, 256, 256] per-pixel confidence
        mask = logits[0].argmax(axis=0).astype(np.uint8)

        # Accuracy filter: a pixel counts as a defect only if the model is
        # at least ONNX_PIXEL_CONF confident about its class — suppresses
        # low-confidence speckle that inflated coverage on clean surfaces.
        pixel_conf = prob_map.max(axis=0)
        confident = pixel_conf >= ONNX_PIXEL_CONF
        mask_filtered = np.where(confident, mask, 0).astype(np.uint8)
        mean_pixel_conf = float(pixel_conf.mean())

        total = int(mask_filtered.size)
        class_summary = {}
        for cid, name in ONNX_CLASSES.items():
            if cid == 0:
                continue
            count = int(np.count_nonzero(mask_filtered == cid))
            class_summary[name] = {
                "pixels": count,
                "percentage_of_image": round(100.0 * count / total, 4),
                "mean_confidence": round(
                    float(pixel_conf[(mask_filtered == cid)].mean())
                    if count else 0.0,
                    4,
                ),
            }
        defect_pixels = int(np.count_nonzero(mask_filtered != 0))
        mask = mask_filtered  # overlay + stats use the filtered mask

        class_probabilities = {
            name: round(mean_probs[cid], 4) for cid, name in ONNX_CLASSES.items()
        }

        if overlay_path is not None:
            _save_overlay(pil_img, mask, Path(overlay_path))

        return {
            "model": ONNX_MODEL_PATH.name,
            "mean_softmax": [round(v, 6) for v in mean_probs],
            "class_probabilities": class_probabilities,
            "pixel_confidence": round(mean_pixel_conf, 4),
            "min_pixel_confidence": ONNX_PIXEL_CONF,
            "class_summary": class_summary,
            "defect_pixels": defect_pixels,
            "defect_percentage": round(100.0 * defect_pixels / total, 4),
        }
    except Exception as exc:
        print(f"[onnx] inference failed: {exc}", flush=True)
        return None


def _save_overlay(pil_img, mask, path):
    """Blend the colour-coded defect mask over the specimen and save JPEG."""
    size = pil_img.size
    mask_img = Image.fromarray(mask, mode="L").resize(size, Image.Resampling.NEAREST)
    mask_big = np.asarray(mask_img, dtype=np.uint8)
    palette = np.array(
        [[0, 0, 0], [255, 65, 65], [255, 190, 40], [50, 160, 255], [190, 80, 255]],
        dtype=np.uint8,
    )
    color = palette[mask_big]
    base = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    overlay = base.copy()
    defect = mask_big != 0
    overlay[defect] = (0.55 * base[defect] + 0.45 * color[defect]).astype(np.uint8)
    Image.fromarray(overlay).save(path, format="JPEG", quality=88)


def _classify(info, seg):
    """Produce probabilities over the model-aligned detection taxonomy.

    The trained ONNX segmentation model drives the identity of the four defect
    classes it was trained on (fiber_breakage, porosity_cracks,
    foreign_inclusion, delamination) from confident per-pixel coverage.  The
    model is only *trusted* when its Background softmax is low (real NDE imagery
    sits ~0.40; synthetic clean surfaces sit ~0.66 and the model hallucinates
    delamination there) — so classical CV remains the defect/clean gate and
    contributes discoloration, which the model has no class for.

    Returns a dict of probabilities over CLASSES that sums to exactly 1.
    """
    crack = float(info.get("crack_score", 0.0))
    scratch = float(info.get("scratch_score", 0.0))
    disc = float(info.get("discoloration_score", 0.0))

    # Classical-CV evidence per taxonomy class (cracks ~ porosity, surface
    # scratches disturb/break fibres, colour deviation ~ discoloration).
    cv_ev = {
        "fiber_breakage": min(1.0, scratch),
        "porosity_cracks": min(1.0, crack),
        "foreign_inclusion": 0.0,
        "delamination": 0.0,
        "discoloration": min(1.0, disc),
    }

    model_ev = {c: 0.0 for c in DEFECT_CLASSES}
    if seg:
        bg = float(seg.get("class_probabilities", {}).get("Background / Clean", 1.0))
        # Trust the model only when background softmax is low (confident defect).
        trust = max(0.0, min(1.0, (0.62 - bg) / 0.22))
        if trust > 0.0:
            cs = seg.get("class_summary", {})
            for name, cls in MODEL_CLASS_MAP.items():
                cov = float(cs.get(name, {}).get("percentage_of_image", 0.0)) / 100.0
                model_ev[cls] = trust * (1.0 - math.exp(-6.0 * cov))

    # Each defect class is evidenced by its stronger detector (CV or model).
    evidence = {c: max(cv_ev[c], model_ev[c]) for c in DEFECT_CLASSES}
    defect_total = sum(evidence.values())
    no_defect = max(0.0, 1.0 - defect_total)

    probs = {c: round(evidence[c], 6) for c in DEFECT_CLASSES}
    probs["no_defect"] = round(no_defect, 6)
    # Normalize to sum exactly 1, then fix residual on the max class.
    total = sum(probs.values()) or 1.0
    probs = {c: round(v / total, 6) for c, v in probs.items()}
    top = max(probs, key=probs.get)
    probs[top] = round(probs[top] + (1.0 - sum(probs.values())), 6)
    return probs


def _mechanisms_fmea(seg):
    """Per-mechanism FMEA rows for every modelled defect with coverage."""
    if not seg:
        return []
    rows = []
    for name, stats in seg["class_summary"].items():
        if stats["pixels"] <= 0:
            continue
        fmea = MECHANISM_FMEA.get(name, {"severity": 5, "detection": 5})
        pct = stats["percentage_of_image"]
        occ = round(min(10.0, 1.0 + 9.0 * pct / 15.0), 1)
        rpn = round(fmea["severity"] * occ * fmea["detection"], 2)
        level = "Critical" if rpn >= 300 else "High" if rpn >= 150 else "Medium" if rpn >= 50 else "Low"
        rows.append({
            "mechanism": name,
            "pixels": stats["pixels"],
            "percentage": pct,
            "severity": fmea["severity"],
            "occurrence": occ,
            "detection": fmea["detection"],
            "rpn": rpn,
            "risk_level": level,
        })
    rows.sort(key=lambda r: r["rpn"], reverse=True)
    return rows


def _residual_life(rpn):
    """Remaining-life estimate and lifecycle phase from the verdict RPN."""
    pct = round(max(0.0, 100.0 - min(95.0, rpn / 10.0)), 1)
    if pct >= 90:
        phase = "Serviceable"
    elif pct >= 70:
        phase = "Monitor"
    elif pct >= 40:
        phase = "Degraded"
    else:
        phase = "Critical"
    return {"remaining_percent": pct, "phase": phase}


def _maintenance_plan(risk_level):
    """Risk-driven maintenance recommendation (interval in days)."""
    plans = {
        "Critical": {
            "action": "Stop use and quarantine the component; full NDE re-inspection and repair before return to service.",
            "interval_days": 0,
            "priority": "P1",
        },
        "High": {
            "action": "Schedule detailed re-inspection and repair planning.",
            "interval_days": 7,
            "priority": "P2",
        },
        "Medium": {
            "action": "Re-inspect and monitor defect growth at the next service window.",
            "interval_days": 30,
            "priority": "P3",
        },
        "Low": {
            "action": "Continue routine monitoring at planned service intervals.",
            "interval_days": 90,
            "priority": "P4",
        },
    }
    return plans.get(risk_level, plans["Low"])


def _current_timestamp_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_history():
    if not DB_PATH.exists():
        return []
    rows = []
    for line in DB_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows


def _write_history(row):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DB_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _component_stats(mask, labels, n):
    """Vectorized per-component second-moment stats.

    Returns (area, elongation, mean_strength) arrays.
    elongation = sqrt(l1/l2) from principal moments of pixel coordinates;
    a long thin line yields a large value, a blob yields ~1.
    """
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mf = mask.astype(np.float32)
    labs = np.arange(1, n + 1)
    cnt = ndimage.sum(mf, labels, labs) + 1e-6
    ey = ndimage.sum(ys * mf, labels, labs) / cnt
    ex = ndimage.sum(xs * mf, labels, labs) / cnt
    eyy = ndimage.sum(ys * ys * mf, labels, labs) / cnt - ey * ey
    exx = ndimage.sum(xs * xs * mf, labels, labs) / cnt - ex * ex
    exy = ndimage.sum(xs * ys * mf, labels, labs) / cnt - ex * ey
    tr = eyy + exx
    det = np.maximum(eyy * exx - exy * exy, 0.0)
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = np.maximum(tr / 2.0 + disc, 1e-6)
    l2 = np.maximum(tr / 2.0 - disc, 1e-6)
    elong = np.sqrt(l1 / l2)
    strength = ndimage.mean(mf, labels, labs)
    return cnt - 1e-6, elong, strength


def _analyze_image(pil_img):
    """Deterministic defect analysis with classical CV (no ML runtime needed).

    1. Dark elongated structures (structure-tensor coherence + local background
       subtraction) -> cracks (thicker) and thin dark scratches.
    2. Bright elongated structures -> bright scratches / scuffs.
    3. Large-scale color deviation blobs -> discoloration.
    Scores are combined into class probabilities; a clean surface yields
    no_defect with high confidence.
    """
    img = pil_img.convert("RGB")
    img.thumbnail((512, 512))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    h, w = gray.shape

    if min(h, w) < 16:
        probs = {c: (0.97 if c == "no_defect" else 0.01) for c in CLASSES}
        probs["no_defect"] = 0.97
        # Return a well-formed analysis dict so the UI never renders NaN for
        # missing score/region fields; keep the note for debugging.
        return probs, {
            "note": "image too small for analysis",
            "crack_score": 0.0,
            "scratch_score": 0.0,
            "discoloration_score": 0.0,
            "n_crack_regions": 0,
            "n_scratch_regions": 0,
            "n_discoloration_regions": 0,
        }

    # --- local background vs. detail ---
    sigma_bg = max(3.0, min(h, w) / 20.0)
    bg = ndimage.gaussian_filter(gray, sigma_bg)
    detail = gray - bg

    # --- structure tensor: coherence marks linear (line-like) structures ---
    sm = ndimage.gaussian_filter(gray, 1.0)
    gy, gx = np.gradient(sm)
    Jxx = ndimage.gaussian_filter(gx * gx, 2.0)
    Jyy = ndimage.gaussian_filter(gy * gy, 2.0)
    Jxy = ndimage.gaussian_filter(gx * gy, 2.0)
    diff = Jxx - Jyy
    coh = np.sqrt(diff * diff + 4.0 * Jxy * Jxy) / (Jxx + Jyy + 1e-8)

    linear = coh > 0.60
    # contrast gates tuned to reject periodic weave texture while keeping
    # real fracture lines (which are far darker/brighter than tow shading)
    dark = detail < -0.10
    bright = detail > 0.10

    diag = float(np.hypot(h, w))
    min_area = max(10.0, 0.0004 * h * w)

    def _score(mask, min_elong, weight, opening=True, area_gate=None):
        mask = mask & linear
        if opening:
            # remove speckle, bridge small gaps in lines
            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            return 0.0, 0
        area, elong, _ = _component_stats(mask, lab, n)
        gate = area_gate if area_gate is not None else min_area
        sel = (area >= gate) & (elong >= min_elong)
        if not np.any(sel):
            return 0.0, 0
        # score grows with total elongated area, saturating
        frac = float(area[sel].sum()) / (h * w)
        best_elong = float(elong[sel].max())
        s = weight * (1.0 - np.exp(-60.0 * frac)) * min(1.0, best_elong / 12.0)
        return float(min(s, 1.2)), int(sel.sum())

    crack_score, n_cracks = _score(dark, min_elong=3.0, weight=1.0)
    # bright scratches are 1-2px wide: opening would erase them, so skip it
    scratch_score, n_scratches = _score(
        bright, min_elong=5.0, weight=0.85, opening=False, area_gate=max(6.0, 0.0001 * h * w)
    )
    # thin dark lines also count as scratches if less massive than cracks
    thin_dark_score, n_thin = _score(
        (detail < -0.06) & (ndimage.gaussian_filter(gray, 0.8) < bg - 0.05),
        min_elong=6.0,
        weight=0.5,
        opening=False,
        area_gate=max(6.0, 0.0001 * h * w),
    )
    scratch_score = float(min(1.2, scratch_score + max(0.0, thin_dark_score - crack_score)))

    # --- discoloration: hue/chroma deviation blobs (brightness-only changes,
    # like crack shadows, must NOT count) ---
    sigma_big = max(6.0, min(h, w) / 10.0)
    chroma = rgb / (rgb.sum(axis=2, keepdims=True) + 1e-6)  # normalized rgb ~ hue
    blur_chroma = np.stack(
        [ndimage.gaussian_filter(chroma[..., c], sigma_big) for c in range(3)], axis=2
    )
    med = np.median(blur_chroma.reshape(-1, 3), axis=0)
    dev = np.linalg.norm(blur_chroma - med, axis=2)
    disc_mask = dev > 0.045
    disc_mask = ndimage.binary_opening(disc_mask, structure=np.ones((5, 5)))
    disc_lab, n_disc = ndimage.label(disc_mask, structure=np.ones((3, 3)))
    disc_score = 0.0
    n_disc_regions = 0
    if n_disc:
        area, elong, _ = _component_stats(disc_mask, disc_lab, n_disc)
        sel = (area >= 0.008 * h * w) & (elong < 6.0)
        if np.any(sel):
            frac = float(area[sel].sum()) / (h * w)
            disc_score = float(min(1.0, 1.0 - np.exp(-8.0 * frac)))
            n_disc_regions = int(sel.sum())

    # --- combine into probabilities ---
    base = max(0.02, 1.0 - max(crack_score, scratch_score, disc_score) * 0.9)
    weights = {
        "crack": max(crack_score, 1e-6),
        "scratch": max(scratch_score, 1e-6),
        "discoloration": max(disc_score, 1e-6),
        "no_defect": base,
    }
    total = sum(weights.values())
    probs = {c: weights[c] / total for c in CLASSES}
    # round, then fix residual on the max class so probs sum to exactly 1
    probs = {c: round(v, 6) for c, v in probs.items()}
    top = max(probs, key=probs.get)
    probs[top] = round(probs[top] + (1.0 - sum(probs.values())), 6)

    info = {
        "crack_score": round(crack_score, 4),
        "scratch_score": round(scratch_score, 4),
        "discoloration_score": round(disc_score, 4),
        "n_crack_regions": n_cracks,
        "n_scratch_regions": n_scratches,
        "n_discoloration_regions": n_disc_regions,
    }
    return probs, info


# ---------------------------------------------------------------------------
# Specimen gate — only carbon-fibre / composite material images are accepted.
#
# Calibrated on the shipped fixtures vs. synthetic non-material probes
# (food / human / scene): composite surfaces are dark (val ~0.26), nearly
# achromatic (sat <= 0.1), and carry fine weave texture (gradient energy
# >= 0.036); probes were bright (val >= 0.79), more saturated (sat >= 0.23)
# and smooth (texture <= 0.007), with skin-tone fractions up to 0.84.
#
# For non-visual NDE modalities (thermography, ultrasound, xray) these visual
# rules do NOT apply — thermography images are colourful heat maps, X-ray /
# radiographic images are often very dark or inverted, and ultrasound B-scans
# have their own texture signatures.  Only a skin-tone sanity check is kept
# to reject obviously non-NDE photographs.
# ---------------------------------------------------------------------------
SPECIMEN_RULES = {
    "max_saturation": 0.18,
    "min_texture": 0.008,
    "max_skin_fraction": 0.12,
    "min_value": 0.04,
    "max_value": 0.75,
}

# Non-visual modalities bypass most checks; only flag very high skin-tone.
_NDE_MODALITIES = {"thermography", "ultrasound", "xray"}
_NDE_MAX_SKIN = 0.35


def _assess_specimen(pil_img, modality="visual"):
    """Return (accepted, reasons, metrics).

    For *visual* modality the full carbon-fibre appearance gate applies.
    For NDE modalities (thermography, ultrasound, xray) the strict
    saturation / brightness / texture rules are skipped because those
    imaging techniques produce fundamentally different-looking images.
    Only an elevated skin-tone check remains to reject obvious non-NDE
    photographs that a user uploaded under the wrong modality.
    """
    small = pil_img.convert("RGB").resize((256, 256))
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
    val = float(mx.mean())

    # skin-tone fraction (classic YCbCr bounds)
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    skin = float(((Cb >= 0.30) & (Cb <= 0.44) & (Cr >= 0.53) & (Cr <= 0.67) & (Y >= 0.30)).mean())

    # fine weave texture: mean gradient magnitude
    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    texture = float(np.sqrt(gx * gx + gy * gy).mean())

    metrics = {
        "saturation": round(sat, 3),
        "brightness": round(val, 3),
        "skin_fraction": round(skin, 3),
        "texture_energy": round(texture, 4),
    }
    reasons = []

    if modality in _NDE_MODALITIES:
        # ---- relaxed gate for non-visual NDE imagery ----
        if skin > _NDE_MAX_SKIN:
            reasons.append(
                f"skin-tone coverage {skin:.0%} — this looks like a human/animal "
                f"photo, not a {modality} NDE image"
            )
    else:
        # ---- strict visual / optical gate ----
        if sat > SPECIMEN_RULES["max_saturation"]:
            reasons.append(
                f"colour saturation {sat:.2f} > {SPECIMEN_RULES['max_saturation']} — "
                "colourful content (food, clothing, scenery) is not a composite surface"
            )
        if skin > SPECIMEN_RULES["max_skin_fraction"]:
            reasons.append(
                f"skin-tone coverage {skin:.0%} — this looks like a human/animal image"
            )
        if texture < SPECIMEN_RULES["min_texture"]:
            reasons.append(
                f"weave texture {texture:.3f} < {SPECIMEN_RULES['min_texture']} — "
                "surface is too smooth; no carbon-fibre weave detected"
            )
        if val < SPECIMEN_RULES["min_value"] or val > SPECIMEN_RULES["max_value"]:
            reasons.append(
                f"brightness {val:.2f} outside {SPECIMEN_RULES['min_value']}–{SPECIMEN_RULES['max_value']} — "
                "not a lit material surface"
            )
    return (len(reasons) == 0), reasons, metrics


def _fmea(predicted_class: str, confidence: float):
    sev = float(SEVERITY.get(predicted_class, 5))
    det = float(DETECTION.get(predicted_class, 5))
    occ = 2.0 + 6.0 * confidence  # more confident detection -> higher observed occurrence
    occ = round(min(occ, 10.0), 1)
    rpn = round(sev * occ * det, 2)
    if rpn >= 300:
        level = "Critical"
    elif rpn >= 150:
        level = "High"
    elif rpn >= 50:
        level = "Medium"
    else:
        level = "Low"
    return {
        "severity": sev,
        "occurrence": occ,
        "detection": det,
        "rpn": rpn,
        "risk_level": level,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/inspections")
def recent_inspections():
    return _load_history()


@app.get("/api/inspections/{inspection_id}/image")
def inspection_image(inspection_id: str):
    """Serve the stored specimen image for a past inspection.

    The client supplies only the inspection id; we resolve the on-disk file
    from the stored record and re-root it under DATA_DIR, so a crafted path
    in the log can never escape the data directory.
    """
    record = next((r for r in _load_history() if r.get("id") == inspection_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    image_path = record.get("image_path") or ""
    stored_name = Path(image_path).name  # drop any directory component
    if not stored_name:
        raise HTTPException(status_code=404, detail="No image recorded")

    # Re-root under DATA_DIR and verify containment.
    candidate = (DATA_DIR / stored_name).resolve()
    try:
        candidate.relative_to(DATA_DIR.resolve())
    except ValueError:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="Image path invalid")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image file missing")

    return FileResponse(candidate, media_type="image/jpeg", filename=stored_name)


@app.get("/api/inspections/{inspection_id}/overlay")
def inspection_overlay(inspection_id: str):
    """Serve the colour-blended ONNX defect overlay for a past inspection."""
    record = next((r for r in _load_history() if r.get("id") == inspection_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    seg = record.get("segmentation") or {}
    overlay_name = Path(str(seg.get("overlay") or "")).name
    if not overlay_name:
        raise HTTPException(status_code=404, detail="No overlay recorded")

    candidate = (DATA_DIR / overlay_name).resolve()
    try:
        candidate.relative_to(DATA_DIR.resolve())
    except ValueError:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="Overlay path invalid")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Overlay file missing")

    return FileResponse(candidate, media_type="image/jpeg", filename=overlay_name)


@app.post("/api/inspect")
async def inspect(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    modality: str = Form("visual"),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image provided")

    modality = (modality or "visual").strip().lower()
    if modality not in {"visual", "thermography", "ultrasound", "xray"}:
        modality = "visual"

    try:
        pil = Image.open(image.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image read error: {exc}") from exc

    # Specimen gate: reject food / human / scene imagery before analysis.
    # Pass the selected modality so non-visual NDE images are not rejected
    # by the strict optical / carbon-fibre appearance rules.
    accepted, reasons, specimen_metrics = _assess_specimen(pil, modality=modality)
    if not accepted:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "specimen_rejected",
                "message": (
                    "This image is not a carbon-fibre/composite specimen. "
                    "Only composite material surface images are accepted."
                ),
                "reasons": reasons,
                "measured": specimen_metrics,
            },
        )

    pil.thumbnail((1024, 1024))
    record_id = str(uuid.uuid4())
    saved_name = f"insp_{uuid.uuid4().hex}.jpg"
    saved_path = DATA_DIR / saved_name
    pil.save(saved_path, format="JPEG", quality=88)

    # ONNX segmentation (mask stats + colour overlay saved beside the image).
    overlay_name = f"ovl_{uuid.uuid4().hex}.jpg"
    overlay_path = DATA_DIR / overlay_name
    seg = _segment_image(pil, overlay_path=overlay_path)
    if seg is not None:
        seg["overlay"] = overlay_name
    else:
        overlay_name = None

    probs, info = _analyze_image(pil)
    features = _image_features(pil, info)
    # Blend learned operator feedback into the heuristic verdict when trusted.
    learned, learner_meta = _learner_predict(probs, features)
    if learned is not None:
        probs = learned
    # Then fuse ONNX defect evidence (asymmetric: strengthens, never vetoes).
    probs = _fuse_with_model(probs, seg)
    # Verdict by AGGREGATE defect mass, not a plain argmax over 4 classes.
    # Defect evidence is split across crack/scratch/discoloration, so a single
    # defect class can lose on plurality even when the specimen is clearly
    # defective (combined defect mass exceeds no_defect) — this produced false
    # "No Defect" verdicts on visibly cracked laminates. Decide nodefect vs.
    # defect by comparing no_defect against the summed defect probability; when
    # defective, label with the most likely defect and report the defect mass as
    # confidence.
    defect_mass = round(sum(probs.get(c, 0.0) for c in DEFECT_CLASSES), 6)
    p_no = float(probs.get("no_defect", 0.0))
    if defect_mass > p_no:
        predicted_class = max(DEFECT_CLASSES, key=lambda c: probs.get(c, 0.0))
        confidence = round(defect_mass, 4)
    else:
        predicted_class = "no_defect"
        confidence = round(p_no, 4)
    threshold_met = confidence >= confidence_threshold
    fmea = _fmea(predicted_class, confidence)
    mechanisms = _mechanisms_fmea(seg)
    residual = _residual_life(fmea["rpn"])
    maintenance = _maintenance_plan(fmea["risk_level"])

    record = {
        "id": record_id,
        "image_path": str(saved_path),
        "image_filename": image.filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "threshold_met": threshold_met,
        "probabilities": probs,
        "severity": fmea["severity"],
        "occurrence": fmea["occurrence"],
        "detection": fmea["detection"],
        "rpn": fmea["rpn"],
        "risk_level": fmea["risk_level"],
        "analysis": info,
        "features": [round(v, 6) for v in features],
        "learner": learner_meta,
        "modality": modality,
        "specimen": specimen_metrics,
        "segmentation": seg,
        "mechanisms": mechanisms,
        "residual_life": residual,
        "maintenance": maintenance,
        "timestamp": _current_timestamp_iso(),
        "notes": "CompositeGuard AI computer-vision inspection",
    }

    _write_history(record)

    return {
        "id": record["id"],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs,
        "risk_level": fmea["risk_level"],
        "rpn": fmea["rpn"],
        "severity": fmea["severity"],
        "occurrence": fmea["occurrence"],
        "detection": fmea["detection"],
        "mode": "analysis",
        "model_backend": "numpy+scipy-cv+onnx-seg" if seg else "numpy+scipy-cv",
        "learner": learner_meta,
        "analysis": info,
        "modality": modality,
        "segmentation": seg,
        "mechanisms": mechanisms,
        "residual_life": residual,
        "maintenance": maintenance,
        "overlay_url": f"/api/inspections/{record_id}/overlay" if overlay_name else None,
        "saved_image": str(saved_path),
        "confidence_threshold": confidence_threshold,
        "threshold_met": threshold_met,
    }


class FeedbackBody(BaseModel):
    label: str  # confirmed label, or corrected class name


@app.post("/api/inspections/{inspection_id}/feedback")
def inspection_feedback(inspection_id: str, body: FeedbackBody):
    """Record an operator verdict and (re)train the learner on it.

    Confirming stores the predicted class as a label; correcting stores the
    operator's class. Either way the (features, label) pair teaches the model,
    so the next upload is auto-trained on this feedback.
    """
    label = body.label.strip().lower()
    if label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASSES}")

    rows = _load_history()
    record = next((r for r in rows if r.get("id") == inspection_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    # Features are stored on new records; recompute from the image for old ones.
    features = record.get("features")
    if not features:
        image_path = Path(record.get("image_path") or "")
        stored = DATA_DIR / Path(str(image_path)).name
        if not stored.is_file():
            raise HTTPException(status_code=404, detail="Image file missing")
        try:
            pil = Image.open(stored).convert("RGB")
            _, info = _analyze_image(pil)
            features = _image_features(pil, info)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}") from exc

    train_meta = _train_on_feedback(features, label)

    # Annotate the history row so the UI can show that feedback was recorded.
    record["feedback_label"] = label
    record["feedback_at"] = _current_timestamp_iso()
    _replace_history_record(inspection_id, record)

    return {
        "id": inspection_id,
        "predicted_class": record.get("predicted_class"),
        "feedback_label": label,
        "trained": train_meta,
    }


def _replace_history_record(inspection_id: str, updated: dict):
    """Rewrite inspections.jsonl with one record replaced."""
    if not DB_PATH.exists():
        return
    lines = []
    for line in DB_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("id") == inspection_id:
            row = updated
        lines.append(json.dumps(row, ensure_ascii=False))
    DB_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@app.delete("/api/inspections")
def clear_history(    keep_samples: int = 0, reset_learner: bool = False):
    """Clear the inspection log (and its stored images).

    keep_samples > 0 keeps the N most recent records for the UI. The trained
    learner is NOT deleted by default — pass reset_learner=true to also wipe
    learned feedback.
    """
    global _learner, _learner_n_samples
    rows = _load_history()
    kept = rows[: max(0, keep_samples)] if keep_samples else []
    removed = rows[len(kept):]

    # Delete the images of removed records (ignore already-missing files).
    for row in removed:
        for key in ("image_path",):
            name = Path(str(row.get(key) or "")).name
            if name:
                try:
                    (DATA_DIR / name).unlink(missing_ok=True)
                except OSError:
                    pass
        overlay = Path(str((row.get("segmentation") or {}).get("overlay") or "")).name
        if overlay:
            try:
                (DATA_DIR / overlay).unlink(missing_ok=True)
            except OSError:
                pass

    if kept:
        DB_PATH.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
            encoding="utf-8",
        )
    else:
        DB_PATH.write_text("", encoding="utf-8")

    if reset_learner:
        with _learner_lock:
            _learner = None
            _X_window.clear()
            _y_window.clear()
            _learner_n_samples = 0
        try:
            LEARNER_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    return {"removed": len(removed), "remaining": len(kept), "learner_reset": reset_learner}


@app.get("/api/model")
def model_status():
    """Report learner state so the UI can show training progress."""
    global _learner
    _load_learner()
    with _learner_lock:
        trained = _learner is not None
        classes = sorted(set(_y_window)) if trained else []
        return {
            "samples": _learner_n_samples,
            "min_samples": LEARNER_MIN_SAMPLES,
            "trained": trained,
            "classes": classes,
            "weight": LEARNER_MAX_WEIGHT if trained else 0.0,
            "model_backend": "numpy+scipy-cv+online-sklearn",
        }


@app.get("/api/telemetry")
def telemetry(history_seconds: int = 60):
    """Simulated real-time sensor stream for the inspection rig.

    Deterministic waveforms (temperature, vibration, acoustic-emission hits,
    humidity) sampled from wall-clock time, plus a short in-memory history so
    the frontend can draw live gauges without a websocket. In a physical
    deployment these channels come from the instrumented rig; here they are
    generated deterministically so the UI is fully exercisable offline.
    """
    now = time.time()
    n = max(5, min(int(history_seconds), 300))

    def sample(t):
        # deterministic pseudo-noise from the timestamp itself
        jitter = lambda k, amp: amp * (((int(t * 1000) * k) % 97) / 97.0 - 0.5)
        return {
            "t": round(t, 1),
            "temperature_c": round(38 + 6 * (0.5 + 0.5 * (t / 30.0 % 2 - 1)) + jitter(31, 1.4), 2),
            "vibration_mm_s": round(2.4 + 1.6 * abs(((t / 8.0) % 2) - 1) + jitter(17, 0.6), 2),
            "ae_hits_s": int(max(0, round(14 + 9 * (0.5 + 0.5 * (t / 21.0 % 2 - 1)) + jitter(7, 6)))),
            "humidity_pct": round(48 + 5 * (0.5 + 0.5 * (t / 60.0 % 2 - 1)) + jitter(43, 1.2), 1),
        }

    points = [sample(now - i) for i in range(n - 1, -1, -1)]
    latest = points[-1]
    return {
        "timestamp": _current_timestamp_iso(),
        "channels": ["temperature_c", "vibration_mm_s", "ae_hits_s", "humidity_pct"],
        "latest": latest,
        "history": points,
        "source": "simulated-rig",
    }


@app.post("/api/validation")
def run_validation(samples: int = 8):
    """Experimental validation: run model + pipeline over data/test_images.

    Compares the fused pipeline against the labelled fixtures the project
    ships (known ground truth) and reports per-class agreement — the
    simulation-based performance verification the requirements ask for.
    """
    samples = max(1, min(int(samples), 20))
    expected = {
        "crack_sample.png": "crack",
        "scratch_sample.png": "scratch",
        "discoloration_sample.png": "discoloration",
        "clean_sample.png": "no_defect",
        "real_carbon_crack.png": "crack",
    }
    fixtures = sorted(TEST_IMAGES_DIR.glob("*.png"))
    if not fixtures:
        raise HTTPException(status_code=404, detail="No validation images in data/test_images")

    results = []
    correct = 0
    confidences = []
    started = time.time()
    for path in fixtures[:samples]:
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        seg = _segment_image(pil)
        probs, info = _analyze_image(pil)
        probs = _fuse_with_model(probs, seg)
        pred = max(probs, key=probs.get)
        conf = round(float(probs[pred]), 4)
        exp = expected.get(path.name)
        hit = exp is not None and pred == exp
        if exp is not None:
            correct += int(hit)
            confidences.append(conf)
        results.append({
            "image": path.name,
            "expected": exp,
            "predicted": pred,
            "confidence": conf,
            "match": hit if exp is not None else None,
            "model_defect_pct": seg["defect_percentage"] if seg else None,
        })

    labelled = [r for r in results if r["expected"] is not None]
    accuracy = round(correct / len(labelled), 4) if labelled else None
    return {
        "timestamp": _current_timestamp_iso(),
        "n_images": len(results),
        "n_labelled": len(labelled),
        "accuracy": accuracy,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "elapsed_ms": int((time.time() - started) * 1000),
        "model": ONNX_MODEL_PATH.name if _get_onnx() else None,
        "results": results,
    }


@app.get("/api/inspections/{inspection_id}/report")
def inspection_report(inspection_id: str):
    """Generate a full analysis report (PDF) for one inspection."""
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    record = next((r for r in _load_history() if r.get("id") == inspection_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    # Palette matching the app
    TEAL = colors.HexColor("#0d7a6f")
    INK = colors.HexColor("#0b2b2a")
    LIGHT = colors.HexColor("#e7f4f1")
    RISK = {
        "Critical": colors.HexColor("#c0392b"),
        "High": colors.HexColor("#e67e22"),
        "Medium": colors.HexColor("#d4ac0d"),
        "Low": colors.HexColor("#27ae60"),
    }

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"CompositeGuard inspection {inspection_id[:8]}",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=INK, fontSize=18, spaceAfter=2)
    h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], textColor=TEAL, fontSize=12,
        spaceBefore=10, spaceAfter=4,
    )
    body = ParagraphStyle("body", parent=styles["BodyText"], textColor=INK, fontSize=9)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=colors.HexColor("#556b69"))

    def kv_table(rows, widths=(45 * mm, 125 * mm)):
        t = Table([[Paragraph(str(k), body), Paragraph(str(v), body)] for k, v in rows], colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bcd8d3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    story = [
        Paragraph("CompositeGuard AI — Inspection Analysis Report", h1),
        Paragraph(
            f"Inspection {inspection_id} · generated {_current_timestamp_iso()}",
            small,
        ),
        Spacer(1, 6),
        Paragraph("1. Specimen &amp; verdict", h2),
        kv_table([
            ("File", record.get("image_filename", "—")),
            ("NDE modality", record.get("modality", "visual")),
            ("Predicted class", str(record.get("predicted_class", "—")).replace("_", " ").title()),
            ("Confidence", f"{float(record.get('confidence', 0)) * 100:.1f}%"),
            ("Threshold met", "Yes" if record.get("threshold_met") else "No"),
            ("Risk level", record.get("risk_level", "—")),
            ("RPN", f"{record.get('rpn', '—')} (S {record.get('severity','—')} × O {record.get('occurrence','—')} × D {record.get('detection','—')})"),
            ("Recorded", record.get("timestamp", "—")),
        ]),
    ]

    # Class probabilities
    probs = record.get("probabilities") or {}
    if probs:
        story.append(Paragraph("2. Class probabilities", h2))
        story.append(
            kv_table([(k.replace("_", " ").title(), f"{v * 100:.2f}%") for k, v in
                      sorted(probs.items(), key=lambda kv: -kv[1])])
        )

    # CV signal strengths
    analysis = record.get("analysis") or {}
    if analysis and "note" not in analysis:
        story.append(Paragraph("3. Classical CV signal analysis", h2))
        story.append(
            kv_table([
                ("Crack score", f"{analysis.get('crack_score', 0)} ({analysis.get('n_crack_regions', 0)} regions)"),
                ("Scratch score", f"{analysis.get('scratch_score', 0)} ({analysis.get('n_scratch_regions', 0)} regions)"),
                ("Discoloration score", f"{analysis.get('discoloration_score', 0)} ({analysis.get('n_discoloration_regions', 0)} regions)"),
            ])
        )

    # AI segmentation
    seg = record.get("segmentation")
    if seg:
        story.append(Paragraph(f"4. AI segmentation — {seg.get('model', 'model')}", h2))
        rows = [
            ("Total defect coverage", f"{seg.get('defect_percentage', 0):.2f}%"),
            ("Mean pixel confidence", f"{seg.get('pixel_confidence', 0):.1%} "
             f"(floor {seg.get('min_pixel_confidence', 0):.2f})"),
        ]
        for name, st in (seg.get("class_summary") or {}).items():
            rows.append((
                name,
                f"{st['percentage_of_image']:.2f}% ({st['pixels']} px)"
                + (f" · conf {st['mean_confidence']:.2f}" if st.get("mean_confidence") else ""),
            ))
        story.append(kv_table(rows))

    # Mechanisms FMEA
    mech = record.get("mechanisms") or []
    if mech:
        story.append(Paragraph("5. Failure-mechanism FMEA", h2))
        header = ["Mechanism", "Cover %", "S", "O", "D", "RPN", "Risk"]
        data = [header] + [
            [m["mechanism"], f"{m['percentage']:.1f}", m["severity"], m["occurrence"],
             m["detection"], m["rpn"], m["risk_level"]]
            for m in mech
        ]
        mt = Table(data, colWidths=[45 * mm, 20 * mm, 12 * mm, 12 * mm, 12 * mm, 20 * mm, 24 * mm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bcd8d3")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(mt)

    # Residual life + maintenance
    residual = record.get("residual_life")
    maint = record.get("maintenance")
    if residual or maint:
        story.append(Paragraph("6. Lifecycle &amp; maintenance plan", h2))
        rows = []
        if residual:
            rows += [
                ("Estimated remaining life", f"{residual.get('remaining_percent', 0)}%"),
                ("Lifecycle phase", residual.get("phase", "—")),
            ]
        if maint:
            rows += [
                ("Priority", maint.get("priority", "—")),
                ("Next inspection",
                 "Immediate" if maint.get("interval_days") == 0
                 else f"within {maint.get('interval_days')} days"),
                ("Recommended action", maint.get("action", "—")),
            ]
        story.append(kv_table(rows))

    # Specimen metrics + embedded images
    spec = record.get("specimen")
    if spec:
        story.append(Paragraph("7. Specimen validation metrics", h2))
        story.append(kv_table([
            ("Saturation", spec.get("saturation")),
            ("Brightness", spec.get("brightness")),
            ("Skin-tone fraction", spec.get("skin_fraction")),
            ("Weave texture energy", spec.get("texture_energy")),
        ]))

    img_path = DATA_DIR / Path(str(record.get("image_path") or "")).name
    overlay_path = DATA_DIR / Path(str((seg or {}).get("overlay") or "")).name
    images = [p for p in (img_path, overlay_path) if p.is_file()]
    if images:
        story.append(Paragraph("8. Specimen images", h2))
        cell = []
        for p in images[:2]:
            try:
                ri = RLImage(str(p), width=78 * mm, height=58 * mm)
                cell.append(ri)
            except Exception:
                pass
        if cell:
            it = Table([cell], colWidths=[80 * mm] * len(cell))
            it.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(it)
            story.append(Spacer(1, 3))
            labels = ["Specimen"] if len(cell) == 1 else ["Specimen", "AI defect overlay"]
            story.append(Paragraph(" · ".join(labels), small))

    # Footer / learner state
    learner = record.get("learner") or {}
    fb = record.get("feedback_label")
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Engine: numpy+scipy-cv"
        + (" + onnx-seg" if seg else "")
        + (f" · online learner: {learner.get('samples', 0)} samples" if learner.get("samples") else "")
        + (f" · operator verdict: {fb}" if fb else "")
        + ". Generated by CompositeGuard AI — advisory only; validate against "
        "ISO 9712-certified NDT procedures before disposition.",
        small,
    ))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()

    from fastapi.responses import Response

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="compositeguard_report_{inspection_id[:8]}.pdf"'
        },
    )


if __name__ == "__main__":
    _port = int(os.environ.get("PORT", "8000"))
    try:
        import uvicorn

        print(f"[dev] starting CompositeGuard AI backend on http://127.0.0.1:{_port}", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=_port, log_level="info")
    except ImportError:
        try:
            from waitress import serve as _ws_serve

            print(f"[dev] starting CompositeGuard AI backend (waitress) on http://127.0.0.1:{_port}", flush=True)
            _ws_serve(app, host="127.0.0.1", port=_port, threads=4)
        except ImportError:
            raise SystemExit(
                "No ASGI/WSGI runner available. Install uvicorn (recommended) or waitress.\n"
                "  pip install uvicorn"
            )
