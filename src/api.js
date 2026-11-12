const API_BASE = "/api";

async function postImageForInspection(imageFile, options = {}) {
  const form = new FormData();
  form.append("image", imageFile);

  if (options.confidenceThreshold !== undefined) {
    form.append("confidence_threshold", String(options.confidenceThreshold));
  }

  if (options.modality) {
    form.append("modality", options.modality);
  }

  const res = await fetch(`${API_BASE}/inspect`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    // Structured rejection (e.g. non-carbon-fibre specimen) carries a
    // detail object with human-readable reasons — surface it on the error.
    let payload = null;
    try {
      payload = await res.json();
    } catch {
      /* not JSON — fall through to status text */
    }
    const detail = payload?.detail;
    if (detail && typeof detail === "object") {
      const err = new Error(detail.message ?? "Inspection rejected");
      err.specimenRejection = detail;
      throw err;
    }
    const text = typeof detail === "string" ? detail : (payload?.detail ?? res.statusText);
    throw new Error(`Inspection request failed: ${res.status} ${text}`);
  }

  return res.json();
}

async function listRecentInspections() {
  const res = await fetch(`${API_BASE}/inspections?t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`History request failed: ${res.status} ${text}`);
  }
  return res.json();
}

// Operator feedback: confirms or corrects a past verdict; the backend
// (re)trains its online learner on every call.
async function sendInspectionFeedback(inspectionId, label) {
  const res = await fetch(`${API_BASE}/inspections/${inspectionId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Feedback failed: ${res.status} ${text}`);
  }
  return res.json();
}

// Clears the history log (and stored images) server-side.
async function clearInspectionHistory({ resetLearner = false } = {}) {
  const params = new URLSearchParams();
  if (resetLearner) params.set("reset_learner", "true");
  const qs = params.toString() ? `?${params}` : "";
  const res = await fetch(`${API_BASE}/inspections${qs}`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Clear failed: ${res.status} ${text}`);
  }
  return res.json();
}

// Online-learner state (samples collected / model trained).
async function fetchModelStatus() {
  const res = await fetch(`${API_BASE}/model`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json();
}

// Live sensor telemetry (simulated rig channels) for the monitoring panel.
async function fetchTelemetry(historySeconds = 60) {
  const res = await fetch(`${API_BASE}/telemetry?history_seconds=${historySeconds}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Telemetry request failed: ${res.status}`);
  return res.json();
}

// Experimental validation run over the labelled fixtures.
async function runValidation(samples = 8) {
  const res = await fetch(`${API_BASE}/validation?samples=${samples}`, {
    method: "POST",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Validation failed: ${res.status} ${text}`);
  }
  return res.json();
}

export {
  postImageForInspection,
  listRecentInspections,
  sendInspectionFeedback,
  clearInspectionHistory,
  fetchModelStatus,
  fetchTelemetry,
  runValidation,
  API_BASE,
};
