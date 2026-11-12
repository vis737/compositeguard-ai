import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import {
  postImageForInspection,
  listRecentInspections,
  sendInspectionFeedback,
  clearInspectionHistory,
  fetchModelStatus,
  fetchTelemetry,
  runValidation,
  API_BASE,
} from "./api";
import { DEMO_DEFECTS } from "./demoDefects";

/* Session persistence: Vite dev restarts/HMR reload the page mid-use; without
   this the uploaded preview and the last result silently vanish. */
const UPLOAD_KEY = "cg_upload_v1";
const RESULT_KEY = "cg_result_v1";
const THRESHOLD_KEY = "cg_threshold_v1";
const MAX_PERSIST_BYTES = 2_500_000; // base64 of the file must fit in sessionStorage

function loadJSON(key) {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveJSON(key, value) {
  try {
    if (value === null || value === undefined) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota exceeded — preview then survives only this session */
  }
}

function dataUrlToFile(dataUrl, name, type) {
  const [head, b64] = String(dataUrl).split(",");
  const mime = type || (head.match(/data:(.*?);/) || [])[1] || "image/jpeg";
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new File([bytes], name || "specimen", { type: mime });
}

const RISK_COLORS = {
  Critical: "#ff7a7a",
  High: "#ff9f6b",
  Medium: "#f4c96b",
  Low: "#9fd6a0",
};

const riskColor = (level) => RISK_COLORS[level] || "#5ac8b4";

const classLabel = (cls) =>
  String(cls ?? "unknown")
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

// Verdict classes an operator can confirm/correct (feeds the online learner).
const CLASS_LABELS = {
  no_defect: "No Defect",
  crack: "Crack",
  scratch: "Scratch",
  discoloration: "Discoloration",
};

const confidenceColor = (c) =>
  c >= 0.8 ? "#9fd6a0" : c >= 0.5 ? "#f4c96b" : "#ff9f6b";

const formatTime = (ts) => {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts ?? "";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const scrollToId = (id) => {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
};

/* The backend stores each specimen on disk and exposes it by inspection id. */
function inspectionImageUrl(insp) {
  return insp?.id ? `${API_BASE}/inspections/${insp.id}/image` : null;
}

function Spinner({ label = "Analyzing" }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-[#cfe8db]">
      <span
        aria-hidden
        className="h-4 w-4 animate-spin rounded-full border-2 border-[#5ac8b4]/25 border-t-[#5ac8b4]"
      />
      {label}
    </span>
  );
}

function RiskPill({ riskLevel, large = false }) {
  const color = riskColor(riskLevel);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${
        large ? "px-4 py-1.5 text-sm" : "px-3 py-1 text-xs"
      }`}
      style={{
        background: `${color}1f`,
        color,
        border: `1px solid ${color}55`,
        boxShadow: `0 0 20px -8px ${color}`,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
      {riskLevel || "Unknown"}
    </span>
  );
}

function SectionHeading({ icon, eyebrow, title, action }) {
  return (
    <div className="flex flex-col gap-4 border-b border-white/10 pb-6 md:flex-row md:items-end md:justify-between">
      <div>
        <div className="mb-3 flex items-center gap-2.5 text-[#5ac8b4]">
          <span className="flex h-7 w-7 items-center justify-center rounded-md border border-[#5ac8b4]/30 bg-[#5ac8b4]/10">
            <span className="material-symbols-outlined text-[15px]">{icon}</span>
          </span>
          <span className="font-label-telemetry text-[10px] uppercase tracking-[0.2em]">
            {eyebrow}
          </span>
        </div>
        <h2 className="font-display-hero text-3xl font-light text-white sm:text-4xl">
          {title}
        </h2>
      </div>
      {action}
    </div>
  );
}

function Navbar({ active, apiOnline, onRun }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navItems = [
    { id: "analysis", label: "Inspection" },
    { id: "results", label: "Results" },
    { id: "sensors", label: "Sensors" },
    { id: "history", label: "History" },
  ];

  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b border-white/[0.06] bg-[#00080d]/85 shadow-[0_4px_30px_rgba(90,200,180,0.07)] backdrop-blur-xl">
      <div className="flex h-20 w-full items-center justify-between px-4 sm:px-8 lg:px-16">
        <a href="#top" className="group flex cursor-pointer items-center gap-3">
          <span className="guard-glow-teal flex h-10 w-10 items-center justify-center rounded-xl border border-[#5ac8b4]/40 bg-gradient-to-br from-[#5ac8b4]/25 to-[#5ac8b4]/5 transition-transform duration-300 group-hover:scale-105">
            <span className="material-symbols-outlined text-[20px] text-[#5ac8b4]">
              radar
            </span>
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-[15px] font-extrabold tracking-[0.16em] text-[#e9f7f2]">
              OVERWORLD
            </span>
            <span className="font-label-caps mt-1.5 hidden text-[9px] uppercase text-[#7ab7af] sm:block">
              Composite Guard
            </span>
          </span>
        </a>

        <nav className="hidden items-center gap-6 md:flex">
          {navItems.map((item) => {
            const isActive = active === item.id;
            return (
              <button
                key={item.id}
                onClick={() => scrollToId(item.id)}
                className={`cursor-pointer font-label-caps text-[11px] font-semibold uppercase transition-all duration-200 ${
                  isActive
                    ? "rounded bg-[#5ac8b4] px-3 py-1 text-[#00211d] shadow-[0_0_20px_-6px_#5ac8b4]"
                    : "text-[#9cc7c0] hover:-translate-y-px hover:text-[#e9f7f2]"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-[#04141a] px-3.5 py-1.5 sm:flex">
            <span
              className={`h-2 w-2 animate-pulse rounded-full ${
                apiOnline ? "bg-[#9fd6a0] shadow-[0_0_10px_#9fd6a0]" : "bg-[#ff7a7a]"
              }`}
            />
            <span className="font-label-caps text-[10px] uppercase text-[#9cc7c0]">
              {apiOnline ? "API online" : "API offline"}
            </span>
          </div>

          <button
            onClick={onRun}
            className="hidden cursor-pointer items-center justify-center rounded-lg bg-[#5ac8b4] px-4 py-2 font-label-caps text-[11px] font-bold uppercase text-[#00211d] shadow-[0_10px_30px_-12px_rgba(90,200,180,0.9)] transition-all duration-300 hover:bg-[#78dcc8] hover:shadow-[0_10px_34px_-10px_rgba(120,220,200,0.9)] md:inline-flex"
          >
            Run Analysis
          </button>

          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="cursor-pointer rounded p-2 text-[#9cc7c0] hover:bg-[#061d23] hover:text-white md:hidden"
            aria-label="Toggle navigation menu"
          >
            <span className="material-symbols-outlined text-[24px]">
              {isMobileMenuOpen ? "close" : "menu"}
            </span>
          </button>
        </div>
      </div>

      {isMobileMenuOpen && (
        <div className="flex flex-col gap-2 border-b border-white/10 bg-[#00080d]/95 px-6 py-4 md:hidden">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                scrollToId(item.id);
                setIsMobileMenuOpen(false);
              }}
              className={`w-full cursor-pointer rounded px-3 py-2.5 text-left font-label-caps text-xs font-semibold uppercase tracking-widest ${
                active === item.id
                  ? "bg-[#5ac8b4] text-[#00211d]"
                  : "text-[#9cc7c0] hover:bg-[#061d23] hover:text-white"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </header>
  );
}

function Footer({ engine }) {
  return (
    <footer className="w-full border-t border-white/[0.08] bg-[#02090c] pt-16 pb-12 text-sm text-[#3f6a65]">
      <div aria-hidden className="h-px w-full bg-gradient-to-r from-transparent via-[#5ac8b4]/40 to-transparent" />
      <div className="mx-auto max-w-7xl space-y-12 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-5">
          <div className="space-y-4 lg:col-span-2">
            <div className="group inline-flex cursor-pointer items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-[#5ac8b4]/40 bg-[#5ac8b4]/10 transition-transform group-hover:scale-105">
                <span className="material-symbols-outlined text-[18px] text-[#5ac8b4]">
                  radar
                </span>
              </span>
              <span className="font-label-caps text-[11px] font-bold uppercase tracking-[0.25em] text-[#e9f7f2]">
                Composite Guard
              </span>
            </div>
            <p className="max-w-sm text-xs leading-relaxed text-[#9cc7c0] sm:text-sm">
              AI-assisted surface inspection for composite materials: deterministic
              defect classification, calibrated confidence, and FMEA risk priority
              scoring — computed locally by the CompositeGuard backend.
            </p>
            <div className="flex flex-wrap items-center gap-4 text-xs text-[#5ac8b4]">
              <span className="flex items-center gap-1.5">
                <span className="material-symbols-outlined text-[16px]">verified</span>
                FMEA S·O·D Risk Model
              </span>
              <span>•</span>
              <span className="flex items-center gap-1.5 text-[#78dcc8]">
                <span className="material-symbols-outlined text-[16px]">memory</span>
                {engine} engine
              </span>
            </div>
          </div>

          <div>
            <h4 className="font-label-caps mb-4 text-xs font-semibold uppercase tracking-wider text-[#cfe8db]">
              Inspection
            </h4>
            <ul className="space-y-2.5 text-xs text-[#9cc7c0]">
              <li>
                <button
                  onClick={() => scrollToId("analysis")}
                  className="cursor-pointer text-left transition-colors hover:text-[#5ac8b4]"
                >
                  Upload &amp; Analyze Specimen
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollToId("results")}
                  className="cursor-pointer text-left transition-colors hover:text-[#5ac8b4]"
                >
                  Prediction &amp; FMEA Results
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollToId("history")}
                  className="cursor-pointer text-left transition-colors hover:text-[#5ac8b4]"
                >
                  Inspection History
                </button>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-label-caps mb-4 text-xs font-semibold uppercase tracking-wider text-[#cfe8db]">
              System
            </h4>
            <ul className="space-y-2.5 font-mono text-xs text-[#9cc7c0]">
              <li>Endpoint: {API_BASE}</li>
              <li>Classes: no_defect / crack / scratch / discoloration</li>
              <li>Scoring: RPN = Severity × Occurrence × Detection</li>
            </ul>
          </div>

          <div>
            <h4 className="font-label-caps mb-4 text-xs font-semibold uppercase tracking-wider text-[#cfe8db]">
              Operating Note
            </h4>
            <div className="flex items-start gap-2 rounded-lg border border-[#f4c96b]/30 bg-[#f4c96b]/10 p-3 text-xs text-[#f4c96b]">
              <span className="material-symbols-outlined text-[16px]">warning</span>
              <span>
                Demonstration mode — results are advisory and not for safety-critical
                decisions.
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center justify-between gap-4 border-t border-white/[0.06] pt-8 text-xs text-[#3f6a65] sm:flex-row">
          <div>
            © {new Date().getFullYear()} Overworld CompositeGuard AI · Prototype build
          </div>
          <div className="flex flex-wrap items-center gap-6">
            <button
              onClick={() => scrollToId("analysis")}
              className="cursor-pointer transition-colors hover:text-white"
            >
              Inspection Console
            </button>
            <button
              onClick={() => scrollToId("history")}
              className="cursor-pointer transition-colors hover:text-white"
            >
              History
            </button>
            <span className="text-[#5ac8b4]">Not for safety-critical decisions</span>
          </div>
        </div>
      </div>
    </footer>
  );
}

/* Full-screen detail view for one history record: shows the original uploaded
   image alongside its classification, FMEA ratings and probabilities. */
function InspectionDetail({ record, onClose }) {
  const color = riskColor(record.risk_level);
  const probs = record.probabilities
    ? Object.entries(record.probabilities).sort(([, a], [, b]) => b - a)
    : [];
  const imageUrl = inspectionImageUrl(record);
  const [imgFailed, setImgFailed] = useState(false);
  const [imgLoading, setImgLoading] = useState(true);
  const [fbState, setFbState] = useState({
    submitting: false,
    done: record.feedback_label ?? null,
    error: null,
    trained: null,
  });

  const submitFeedback = async (label) => {
    setFbState((s) => ({ ...s, submitting: true, error: null }));
    try {
      const out = await sendInspectionFeedback(record.id, label);
      setFbState({
        submitting: false,
        done: label,
        error: null,
        trained: out.trained ?? null,
      });
      record.feedback_label = label; // keep the card in sync if reused
    } catch (err) {
      setFbState((s) => ({ ...s, submitting: false, error: err.message }));
    }
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  const s = record.severity ?? "—";
  const o = record.occurrence ?? "—";
  const d = record.detection ?? "—";

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-md sm:p-6"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Inspection detail"
    >
      <div
        className="animate-fade-up relative flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-b from-[#0a2a31] to-[#02090c] shadow-[0_50px_120px_-40px_rgba(0,0,0,0.9)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-start justify-between gap-4 border-b border-white/10 px-5 py-4 sm:px-7 sm:py-5">
          <div className="min-w-0">
            <div className="flex items-center gap-2.5 text-[#5ac8b4]">
              <span className="flex h-6 w-6 items-center justify-center rounded-md border border-[#5ac8b4]/30 bg-[#5ac8b4]/10">
                <span className="material-symbols-outlined text-[14px]">receipt_long</span>
              </span>
              <span className="font-label-telemetry text-[10px] uppercase tracking-[0.2em]">
                Inspection record
              </span>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <h3 className="font-display-hero truncate text-2xl font-light text-white sm:text-3xl">
                {classLabel(record.predicted_class)}
              </h3>
              <RiskPill riskLevel={record.risk_level} />
              {record.threshold_met !== undefined && (
                <span
                  className="rounded-full px-2.5 py-1 font-mono text-[10px] uppercase"
                  style={{
                    background: record.threshold_met ? "rgba(159,214,160,0.12)" : "rgba(244,201,107,0.12)",
                    color: record.threshold_met ? "#9fd6a0" : "#f4c96b",
                    border: `1px solid ${record.threshold_met ? "#9fd6a055" : "#f4c96b55"}`,
                  }}
                >
                  {record.threshold_met ? "threshold met" : "below threshold"}
                </span>
              )}
            </div>
            <p className="mt-1.5 truncate font-mono text-[11px] text-[#7ab7af]">
              {record.image_filename ?? "specimen"} · {formatTime(record.timestamp)}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <a
              href={`${API_BASE}/inspections/${record.id}/report`}
              title="Download full analysis report (PDF)"
              className="rounded-lg border border-[#78dcc8]/30 bg-[#78dcc8]/[0.08] p-2 text-[#78dcc8] transition-all duration-200 hover:border-[#78dcc8]/60 hover:bg-[#78dcc8]/15 hover:text-white"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
                picture_as_pdf
              </span>
            </a>
            <button
              onClick={onClose}
              aria-label="Close detail"
              className="cursor-pointer rounded-lg border border-white/10 bg-white/[0.04] p-2 text-[#9cc7c0] transition-all duration-200 hover:border-[#5ac8b4]/50 hover:text-white"
            >
              <span className="material-symbols-outlined text-[20px]">close</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 overflow-y-auto px-5 py-5 sm:px-7 lg:grid-cols-2">
          {/* left: the uploaded specimen image */}
          <div>
            <div className="font-label-caps mb-2 text-[10px] uppercase text-[#7ab7af]">
              Uploaded specimen
            </div>
            <div
              className="group relative flex min-h-[240px] items-center justify-center overflow-hidden rounded-2xl border border-white/10 bg-[#02090c]"
              style={{ boxShadow: `inset 0 0 60px -20px ${color}55` }}
            >
              {imageUrl && !imgFailed ? (
                <>
                  {imgLoading && (
                    <span className="absolute inset-0 flex items-center justify-center">
                      <Spinner label="Loading image…" />
                    </span>
                  )}
                  <img
                    src={imageUrl}
                    alt={`Specimen for ${record.image_filename ?? "inspection"}`}
                    onLoad={() => setImgLoading(false)}
                    onError={() => {
                      setImgFailed(true);
                      setImgLoading(false);
                    }}
                    className={`max-h-[440px] w-auto object-contain transition-opacity duration-500 ${
                      imgLoading ? "opacity-0" : "opacity-100"
                    }`}
                  />
                </>
              ) : (
                <div className="faint-grid flex flex-col items-center gap-3 px-6 py-14 text-center">
                  <span className="flex h-14 w-14 items-center justify-center rounded-2xl border border-[#3f6a65]/40 bg-white/[0.03] text-[#3f6a65]">
                    <span className="material-symbols-outlined text-[26px]">image_not_supported</span>
                  </span>
                  <p className="text-xs text-[#7ab7af]">
                    Image unavailable — the file may have been cleaned from disk.
                  </p>
                </div>
              )}

              {/* corner tag with RPN, echoes the console */}
              <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1 font-mono text-[10px] text-[#cfe8db] backdrop-blur">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
                RPN {record.rpn ?? 0}
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between rounded-xl border border-white/[0.06] bg-[#02090c]/70 px-4 py-3">
              <div className="min-w-0">
                <div className="font-label-caps text-[9px] uppercase text-[#7ab7af]">Specimen file</div>
                <div className="truncate font-mono text-xs text-white">{record.image_filename ?? "specimen"}</div>
              </div>
              <div className="shrink-0 text-right">
                <div className="font-label-caps text-[9px] uppercase text-[#7ab7af]">Recorded</div>
                <div className="font-mono text-xs text-white">{formatTime(record.timestamp)}</div>
              </div>
            </div>
          </div>

          {/* right: the analysis */}
          <div className="space-y-5">
            {/* confidence */}
            <div className="rounded-2xl border border-white/[0.07] bg-[#061d23] p-4">
              <div className="flex items-end justify-between">
                <span className="font-label-caps text-[10px] uppercase text-[#7ab7af]">Confidence</span>
                <span className="font-technical-metric text-2xl font-bold text-white">
                  {((record.confidence ?? 0) * 100).toFixed(1)}
                  <span className="text-sm text-[#7ab7af]">%</span>
                </span>
              </div>
              <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-[#02090c]">
                <div
                  className="bar-shine relative h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${Math.min(100, (record.confidence ?? 0) * 100)}%`,
                    background: `linear-gradient(90deg, ${confidenceColor(record.confidence ?? 0)}88, ${confidenceColor(
                      record.confidence ?? 0
                    )})`,
                  }}
                />
              </div>
            </div>

            {/* FMEA S/O/D */}
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Severity", value: s, icon: "warning" },
                { label: "Occurrence", value: o, icon: "repeat" },
                { label: "Detection", value: d, icon: "visibility" },
              ].map((m) => (
                <div key={m.label} className="rounded-xl border border-white/[0.07] bg-[#061d23] p-3 text-center">
                  <span className="material-symbols-outlined text-[15px] text-[#5ac8b4]">{m.icon}</span>
                  <div className="font-technical-metric mt-1.5 text-xl font-bold text-white">{m.value}</div>
                  <div className="font-label-caps text-[9px] uppercase text-[#7ab7af]">{m.label}</div>
                </div>
              ))}
            </div>

            {/* RPN banner */}
            <div
              className="flex items-center justify-between rounded-2xl border p-4"
              style={{ borderColor: `${color}55`, background: `linear-gradient(160deg, ${color}1a, #04141a 70%)` }}
            >
              <div>
                <div className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#7ab7af]">
                  Risk Priority Number
                </div>
                <div className="font-technical-metric text-3xl font-bold text-white">{record.rpn ?? 0}</div>
              </div>
              <div className="text-right font-mono text-[11px] text-[#9cc7c0]">
                S {s} × O {o} × D {d}
              </div>
            </div>

            {/* probabilities */}
            {probs.length > 0 && (
              <div className="rounded-2xl border border-white/[0.07] bg-[#061d23] p-4">
                <div className="flex items-center justify-between">
                  <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                    Class probabilities
                  </span>
                  <span className="material-symbols-outlined text-[18px] text-[#5ac8b4]">bar_chart</span>
                </div>
                <div className="mt-3.5 space-y-3">
                  {probs.map(([cls, prob], i) => (
                    <div key={cls} className="flex items-center gap-3">
                      <span
                        className={`w-28 shrink-0 font-mono text-[11px] uppercase tracking-wider ${
                          i === 0 ? "text-[#5ac8b4]" : "text-[#9cc7c0]"
                        }`}
                      >
                        {cls}
                      </span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-[#02090c]">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: `${prob * 100}%`,
                            background:
                              i === 0
                                ? `linear-gradient(90deg, ${confidenceColor(record.confidence ?? 0)}99, ${confidenceColor(
                                    record.confidence ?? 0
                                  )})`
                                : "linear-gradient(90deg, #1c3a38, #3f6a65)",
                          }}
                        />
                      </div>
                      <span className="w-14 text-right font-mono text-xs font-bold text-white">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {record.notes && (
              <p className="flex items-start gap-2 text-xs leading-relaxed text-[#7ab7af]">
                <span className="material-symbols-outlined mt-0.5 text-[15px] text-[#3f6a65]">info</span>
                {record.notes}
              </p>
            )}

            {/* -------- operator feedback / auto-training -------- */}
            <div className="rounded-2xl border border-[#78dcc8]/20 bg-[#78dcc8]/[0.05] p-4">
              <div className="flex items-center justify-between">
                <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#78dcc8]">
                  Teach the model
                </span>
                <span className="material-symbols-outlined text-[18px] text-[#78dcc8]">
                  school
                </span>
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-[#9cc7c0]">
                Confirm the verdict or correct it — every answer retrains the
                classifier instantly, so the next upload starts from what you
                taught it.
              </p>

              <div className="mt-3 flex flex-wrap gap-2">
                {Object.keys(CLASS_LABELS ?? {}).map((cls) => {
                  const isPredicted = cls === record.predicted_class;
                  const isDone = fbState.done === cls;
                  return (
                    <button
                      key={cls}
                      onClick={() => submitFeedback(cls)}
                      disabled={fbState.submitting}
                      className={`inline-flex cursor-pointer items-center gap-1.5 rounded-lg border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-all duration-200 disabled:cursor-wait disabled:opacity-60 ${
                        isDone
                          ? "border-[#78dcc8]/60 bg-[#78dcc8]/15 text-white"
                          : "border-white/12 bg-white/[0.04] text-[#cfe8db] hover:border-[#78dcc8]/50 hover:bg-[#78dcc8]/10 hover:text-white"
                      }`}
                    >
                      <span
                        className="material-symbols-outlined text-[13px]"
                        style={{ color: isDone ? "#78dcc8" : undefined }}
                      >
                        {isPredicted ? "thumb_up" : "edit"}
                      </span>
                      {isPredicted ? `Yes, ${classLabel(cls)}` : classLabel(cls)}
                    </button>
                  );
                })}
              </div>

              {fbState.submitting && (
                <p className="mt-2.5 font-mono text-[11px] text-[#78dcc8]">
                  Training on your feedback…
                </p>
              )}
              {fbState.error && (
                <p className="mt-2.5 font-mono text-[11px] text-[#ff9a93]">
                  {fbState.error}
                </p>
              )}
              {fbState.trained && !fbState.submitting && (
                <p className="mt-2.5 font-mono text-[11px] text-[#9fd6a0]">
                  ✓ Model updated — {fbState.trained.samples} labelled samples
                  {fbState.trained.model_trained ? ", classifier retrained" : " (accumulating)"}
                </p>
              )}
              {!fbState.trained && !fbState.submitting && !fbState.error && (
                <p className="mt-2.5 font-mono text-[10px] text-[#7ab7af]">
                  {record.feedback_label
                    ? `✓ Your verdict: ${classLabel(record.feedback_label)}`
                    : "Not yet reviewed"}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [imageFile, setImageFile] = useState(() => {
    const rec = loadJSON(UPLOAD_KEY);
    try {
      return rec?.dataUrl ? dataUrlToFile(rec.dataUrl, rec.name, rec.type) : null;
    } catch {
      return null;
    }
  });
  const [previewUrl, setPreviewUrl] = useState(() => loadJSON(UPLOAD_KEY)?.dataUrl ?? null);
  const [result, setResult] = useState(() => loadJSON(RESULT_KEY));
  const [inspections, setInspections] = useState([]);
  const [status, setStatus] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(() => {
    const v = parseFloat(sessionStorage.getItem(THRESHOLD_KEY));
    return Number.isFinite(v) && v >= 0.1 && v <= 1 ? v : 0.5;
  });
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(null);
  const [active, setActive] = useState("analysis");
  const [historyError, setHistoryError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [modality, setModality] = useState("visual");
  const [rejection, setRejection] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const [preFlight, setPreFlight] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);

  /* Monotonic token: uploads invalidate any in-flight analysis so a late
     response can never display values for an image that was replaced. */
  const analysisSeq = useRef(0);
  const previewRef = useRef(previewUrl);

  useEffect(() => {
    saveJSON(RESULT_KEY, result);
  }, [result]);

  useEffect(() => {
    try {
      sessionStorage.setItem(THRESHOLD_KEY, String(confidenceThreshold));
    } catch {
      /* ignore */
    }
  }, [confidenceThreshold]);

  /* Revoke blob URLs only after the new preview has rendered, so the <img>
     never briefly points at a revoked URL (image "disappearing" flash). */
  useEffect(() => {
    const prev = previewRef.current;
    if (prev && prev !== previewUrl && typeof prev === "string" && prev.startsWith("blob:")) {
      URL.revokeObjectURL(prev);
    }
    previewRef.current = previewUrl;
  }, [previewUrl]);

  useEffect(
    () => () => {
      const cur = previewRef.current;
      if (typeof cur === "string" && cur.startsWith("blob:")) URL.revokeObjectURL(cur);
    },
    []
  );

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setApiOnline(res.ok);
    } catch {
      setApiOnline(false);
    }
  }, []);

  const historySeq = useRef(0);

  const refreshHistory = useCallback(async () => {
    const seq = ++historySeq.current;
    setRefreshing(true);
    try {
      const rows = await listRecentInspections();
      if (seq !== historySeq.current) return; // a newer history op superseded us
      setInspections(Array.isArray(rows) ? rows : []);
      setHistoryError(null);
      setStatus({ type: "ok", message: `History synced — ${Array.isArray(rows) ? rows.length : 0} records.` });
    } catch (err) {
      if (seq !== historySeq.current) return;
      setHistoryError(err.message ?? "History load failed");
      setStatus({ type: "error", message: err.message ?? "History load failed" });
    } finally {
      if (seq === historySeq.current) setRefreshing(false);
    }
  }, []);

  const clearHistory = useCallback(async () => {
    const ok = window.confirm(
      "Delete ALL inspection history and stored images from the server?\nThis cannot be undone. (Your trained model is kept.)"
    );
    if (!ok) return;
    const seq = ++historySeq.current;
    setRefreshing(true);
    try {
      const out = await clearInspectionHistory();
      if (seq !== historySeq.current) return;
      setInspections([]);
      setSelectedRecord(null);
      setStatus({ type: "ok", message: `Cleared ${out.removed} records from history.` });
    } catch (err) {
      if (seq !== historySeq.current) return;
      setStatus({ type: "error", message: err.message ?? "Clear failed" });
    } finally {
      if (seq === historySeq.current) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    refreshHistory();
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, [checkHealth, refreshHistory]);

  /* Live sensor channel polling — 3s cadence while the tab is visible. */
  useEffect(() => {
    let stopped = false;
    const pull = async () => {
      if (document.hidden) return;
      try {
        const data = await fetchTelemetry(60);
        if (!stopped) setTelemetry(data);
      } catch {
        /* transient poll failure — keep the last reading */
      }
    };
    pull();
    const timer = setInterval(pull, 3000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, []);

  const runValidationNow = useCallback(async () => {
    setValidating(true);
    setValidationError(null);
    try {
      const out = await runValidation(8);
      setValidation(out);
      setStatus({
        type: "ok",
        message: `Validation complete — ${Math.round((out.accuracy ?? 0) * 100)}% accuracy on ${out.n_labelled} labelled fixtures.`,
      });
    } catch (err) {
      setValidationError(err.message ?? "Validation failed");
    } finally {
      setValidating(false);
    }
  }, []);

  /* When the backend is unreachable and we have a persisted result, keep the
     reel visible instead of going back to the empty placeholder. */
  useEffect(() => {
    if (result) return;
    if (apiOnline === false && sessionStorage.getItem(RESULT_KEY)) {
      setPreFlight("API unreachable");
    } else if (apiOnline === true) {
      setPreFlight(null);
    }
  }, [apiOnline, result]);

  useEffect(() => {
    const ids = ["analysis", "results", "sensors", "history"];
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setActive(entry.target.id);
        });
      },
      { rootMargin: "-35% 0px -55% 0px" }
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, []);

  const handleFileChange = useCallback((e) => {
    const file = e.target.files?.[0] ?? null;
    e.target.value = "";
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setStatus({ type: "error", message: "Unsupported file — use JPG, PNG, BMP or TIFF." });
      return;
    }
    analysisSeq.current += 1; // drop any analysis still in flight for the old image
    setLoading(false);
    setPreviewUrl(URL.createObjectURL(file));
    setImageFile(file);
    setResult(null);
    setStatus(null);
    setRejection(null);
    // persist the preview so it survives dev-server reloads
    if (file.size <= MAX_PERSIST_BYTES) {
      const reader = new FileReader();
      reader.onload = () => saveJSON(UPLOAD_KEY, { dataUrl: reader.result, name: file.name, type: file.type });
      reader.onerror = () => (saveJSON(UPLOAD_KEY, null), setStatus({ type: "error", message: "Preview storage failed" }));
      reader.readAsDataURL(file);
    } else {
      saveJSON(UPLOAD_KEY, null);
      setStatus({ type: "error", message: "Image too large for preview persistence" });
    }
  }, []);

  const clearImage = useCallback(() => {
    analysisSeq.current += 1;
    setLoading(false);
    setPreFlight(null);
    setPreviewUrl(null);
    setImageFile(null);
    setResult(null);
    setStatus(null);
    saveJSON(UPLOAD_KEY, null);
  }, []);

  const analyze = useCallback(async () => {
    if (!imageFile) return;
    const seq = ++analysisSeq.current;
    setLoading(true);
    setStatus(null);
    setResult(null);
    setRejection(null);
    try {
      const data = await postImageForInspection(imageFile, {
        confidenceThreshold,
        modality,
      });
      if (seq !== analysisSeq.current) return; // image changed mid-flight: drop stale values
      setResult(data);
      setStatus({ type: "ok", message: "Inspection complete" });
      refreshHistory();
      checkHealth();
    } catch (err) {
      if (seq !== analysisSeq.current) return;
      if (err.specimenRejection) {
        setRejection(err.specimenRejection);
        setStatus({ type: "error", message: err.message });
        setLoading(false);
        return;
      }
      const msg = err.message ?? "Request failed";
      setStatus({ type: "error", message: msg });
      // When the /api server is unreachable, seed a deterministic demo result so
      // the results + history panels stay populated instead of going empty.
      if (msg.includes("Failed to fetch")) {
        const name = imageFile?.name ?? "specimen";
        const demoKey = Object.keys(DEMO_DEFECTS).find((k) => name.endsWith(k)) ?? null;
        const demo = demoKey ? DEMO_DEFECTS[demoKey] : null;
        if (demo) {
          const mock = {
            predicted_class: demo.predicted_class,
            label: demo.label,
            confidence: demo.confidence,
            confidence_threshold: confidenceThreshold,
            fmea: demo.fmea,
            probabilities: demo.probabilities,
            model_backend: "demo-cv",
            mode: "demonstration",
            threshold_met: demo.confidence >= confidenceThreshold,
            rpn: demo.fmea.severity * demo.fmea.occurrence * demo.fmea.detection,
            risk_level: demo.fmea.risk_level,
            analysis: {
              crack_score: demo.probabilities.crack,
              scratch_score: demo.probabilities.scratch,
              discoloration_score: demo.probabilities.discoloration,
              n_crack_regions: 0,
              n_scratch_regions: 0,
              n_discoloration_regions: 0,
            },
          };
          setResult(mock);
          setStatus({ type: "ok", message: "Demonstration result (API offline)" });
          refreshHistory();
          checkHealth();
        }
      }
      checkHealth();
    } finally {
      if (seq === analysisSeq.current) setLoading(false);
    }
  }, [imageFile, confidenceThreshold, modality, refreshHistory, checkHealth]);

  const engine = result?.model_backend || "numpy+scipy-cv";
  const historyRows = inspections.slice(0, 9);

  /* Summary stats for the history strip — turns a bare list into a quick read. */
  const historyStats = useMemo(() => {
    const total = inspections.length;
    const defects = inspections.filter((r) => r.predicted_class !== "no_defect").length;
    const flagged = inspections.filter((r) =>
      ["High", "Critical"].includes(r.risk_level)
    ).length;
    const avgConf = total
      ? inspections.reduce((s, r) => s + (r.confidence ?? 0), 0) / total
      : 0;
    const clean = total - defects;
    return { total, defects, flagged, avgConf, clean };
  }, [inspections]);
  const verdictCaption = result
    ? `Backend · ${result.mode || "analysis"} mode${
        result.model_backend === "demo-cv" ? " (offline demo)" : ""
      }`
    : "Upload and analyze an image to see the predicted class.";
  const historyErrorMessage = historyError ?? (inspections.length === 0
    && status?.type === "error"
    && /History|history/i.test(status.message)
    ? status.message
    : null);

  const heroStats = [
    { icon: "category", value: "04", label: "Defect classes" },
    { icon: "fact_check", value: "S·O·D", label: "FMEA ratings" },
    { icon: "speed", value: "RPN", label: "Risk priority score" },
    { icon: "memory", value: "CV", label: "Local inference engine" },
  ];

  const fmeaMetrics = result
    ? [
        { label: "Severity", value: result.severity, icon: "warning" },
        { label: "Occurrence", value: result.occurrence, icon: "repeat" },
        { label: "Detection", value: result.detection, icon: "visibility" },
      ]
    : [];

  const signals = result?.analysis
    ? [
        {
          label: "Crack",
          score: result.analysis.crack_score ?? 0,
          regions: result.analysis.n_crack_regions ?? 0,
        },
        {
          label: "Scratch",
          score: result.analysis.scratch_score ?? 0,
          regions: result.analysis.n_scratch_regions ?? 0,
        },
        {
          label: "Discoloration",
          score: result.analysis.discoloration_score ?? 0,
          regions: result.analysis.n_discoloration_regions ?? 0,
        },
      ]
    : [];

  const probabilities = result?.probabilities
    ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div id="top" className="flex min-h-screen flex-col font-sans text-[#cfe8db]">
      <Navbar active={active} apiOnline={apiOnline} onRun={() => scrollToId("analysis")} />

      <main className="flex-1 pt-20">
        {/* ------------------------------- HERO ------------------------------- */}
        <section className="relative overflow-hidden px-4 pt-16 pb-20 sm:px-8 lg:px-16">
          <div
            aria-hidden
            className="pointer-events-none absolute -top-40 left-1/2 h-[440px] w-[760px] -translate-x-1/2 rounded-full bg-[#5ac8b4]/12 blur-[130px]"
          />
          <div
            aria-hidden
            className="pointer-events-none absolute top-48 -left-32 h-[380px] w-[380px] rounded-full bg-[#78dcc8]/8 blur-[120px]"
          />
          <div
            aria-hidden
            className="faint-grid pointer-events-none absolute inset-0 opacity-60 [mask-image:radial-gradient(70%_60%_at_50%_0%,black,transparent)]"
          />

          <div className="relative mx-auto max-w-7xl animate-fade-up">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-[#5ac8b4]/30 bg-[#5ac8b4]/10 px-4 py-1.5">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#5ac8b4] shadow-[0_0_10px_#5ac8b4]" />
              <span className="font-label-telemetry text-[10px] uppercase tracking-[0.2em] text-[#78dcc8]">
                AI-assisted surface inspection
              </span>
            </div>

            <h1 className="font-display-hero max-w-4xl text-4xl font-light leading-[1.05] text-white sm:text-6xl lg:text-7xl">
              Composite surface{" "}
              <span className="text-gradient-guard italic">integrity</span>, decoded.
            </h1>

            <p className="mt-6 max-w-2xl text-sm leading-relaxed text-[#9cc7c0] sm:text-base">
              Upload a specimen and the backend returns a deterministic defect
              classification with calibrated confidence, FMEA severity / occurrence /
              detection ratings, and a risk priority number — all computed locally.
            </p>

            {/* How it works — three plain-language steps */}
            <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-3">
              {[
                {
                  step: "01",
                  icon: "upload_file",
                  title: "Drop a specimen",
                  body: "Upload a JPG, PNG, BMP or TIFF of the composite surface. It's resized to 1024 px and kept on this machine.",
                },
                {
                  step: "02",
                  icon: "travel_explore",
                  title: "We read the surface",
                  body: "A classical CV pass looks for cracks, scratches and colour deviation, then turns them into calibrated class probabilities.",
                },
                {
                  step: "03",
                  icon: "fact_check",
                  title: "Get a risk verdict",
                  body: "Severity × Occurrence × Detection gives an FMEA risk priority number, saved to your history with the image.",
                },
              ].map((item, i) => (
                <div
                  key={item.step}
                  className="animate-fade-up group relative overflow-hidden rounded-2xl border border-white/[0.07] bg-gradient-to-b from-[#0a2a31]/60 to-[#04141a]/50 p-5 backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:border-[#5ac8b4]/40"
                  style={{ animationDelay: `${140 + i * 110}ms` }}
                >
                  {/* faint step number */}
                  <span
                    aria-hidden
                    className="font-display-hero pointer-events-none absolute -top-3 -right-2 text-6xl font-light text-white/[0.04] transition-colors duration-300 group-hover:text-[#5ac8b4]/10"
                  >
                    {item.step}
                  </span>
                  <span className="relative flex h-11 w-11 items-center justify-center rounded-xl border border-[#5ac8b4]/30 bg-[#5ac8b4]/10 text-[#5ac8b4] transition-all duration-300 group-hover:shadow-[0_0_25px_-8px_#5ac8b4]">
                    <span className="material-symbols-outlined text-[22px]">{item.icon}</span>
                  </span>
                  <div className="relative mt-3.5 flex items-center gap-2">
                    <span className="font-label-telemetry text-[9px] uppercase tracking-[0.22em] text-[#5ac8b4]">
                      Step {item.step}
                    </span>
                    <span className="h-px flex-1 bg-gradient-to-r from-[#5ac8b4]/30 to-transparent" />
                  </div>
                  <h3 className="font-headline-sm relative mt-2 text-base text-white">{item.title}</h3>
                  <p className="relative mt-1.5 text-[13px] leading-relaxed text-[#9cc7c0]">{item.body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ----------------------------- ANALYSIS ----------------------------- */}
        <section id="analysis" className="px-4 py-14 sm:px-8 lg:px-16">
          <div className="mx-auto max-w-7xl space-y-8">
            <SectionHeading
              icon="upload_file"
              eyebrow="Specimen intake & analysis console"
              title="Inspection Console"
              action={
                <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-[#04141a] px-4 py-2 md:flex">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      apiOnline ? "bg-[#9fd6a0]" : "bg-[#ff7a7a]"
                    }`}
                  />
                  <span className="font-mono text-[11px] text-[#7ab7af]">
                    {API_BASE} · {apiOnline ? "reachable" : "unreachable"}
                  </span>
                </div>
              }
            />

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Upload console */}
              <div className="rounded-2xl border border-white/[0.08] bg-gradient-to-b from-[#061d23] to-[#04141a] p-6 shadow-[0_30px_80px_-50px_rgba(90,200,180,0.7)] lg:col-span-7">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                      Step 01
                    </span>
                    <h3 className="font-headline-sm mt-1 text-xl text-white">
                      Upload composite sample
                    </h3>
                  </div>
                  <div className="flex items-center gap-3">
                    {imageFile && (
                      <span className="hidden text-[11px] text-[#7ab7af] sm:block">
                        {imageFile.name} · {Math.round(imageFile.size / 1024)} KB
                      </span>
                    )}
                    {imageFile && (
                      <button
                        onClick={clearImage}
                        className="cursor-pointer rounded-lg border border-[#ff7a7a]/40 bg-[#ff7a7a]/10 px-3 py-1.5 text-xs font-semibold text-[#ff9a93] transition-colors hover:bg-[#ff7a7a]/20"
                      >
                        Replace image
                      </button>
                    )}
                  </div>
                </div>

                <label
                  htmlFor="composite-image-upload"
                  className="group relative block cursor-pointer overflow-hidden rounded-2xl border border-dashed border-[#5ac8b4]/40 bg-[#02090c] transition-all duration-300 hover:border-[#5ac8b4]/80 hover:shadow-[0_0_50px_-20px_rgba(90,200,180,0.9)]"
                >
                  {previewUrl ? (
                    <>
                      <img
                        src={previewUrl}
                        alt="Uploaded composite sample"
                        className="mx-auto max-h-[420px] w-auto rounded-xl p-2"
                      />
                      <span className="absolute right-3 bottom-3 hidden items-center gap-1.5 rounded-full bg-black/70 px-3 py-1.5 text-[10px] uppercase tracking-widest text-[#cfe8db] backdrop-blur group-hover:inline-flex">
                        <span className="material-symbols-outlined text-[14px]">
                          swap_horiz
                        </span>
                        Replace
                      </span>
                    </>
                  ) : (
                    <div className="faint-grid flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
                      <span className="guard-glow-teal animate-float flex h-14 w-14 items-center justify-center rounded-2xl border border-[#5ac8b4]/30 bg-[#5ac8b4]/10 text-[#5ac8b4]">
                        <span className="material-symbols-outlined text-[26px]">
                          upload_file
                        </span>
                      </span>
                      <p className="font-headline-lg text-lg text-white">
                        Choose a specimen image
                      </p>
                      <p className="text-xs text-[#7ab7af]">
                        JPG, PNG, BMP or TIFF — processed server-side at up to 1024 px
                      </p>
                    </div>
                  )}
                  {previewUrl && (
                    <div className="mt-3 flex items-center gap-3">
                      <span className="inline-flex h-8 min-w-[64px] items-center gap-2 rounded-full bg-[#02090c] px-3 text-[11px] font-mono text-[#9cc7c0]">
                        <span className="material-symbols-outlined text-[14px] text-[#5ac8b4]">image</span>
                        {imageFile.name}
                      </span>
                      <span className="text-right text-[11px] font-mono text-[#7ab7af]">
                        {(imageFile.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                  )}
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/bmp,image/tiff"
                    onChange={handleFileChange}
                    className="sr-only"
                    id="composite-image-upload"
                  />
                </label>



                {/* NDE modality selector */}
                <div className="mt-5 rounded-2xl border border-white/[0.06] bg-[#02090c]/80 p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-label-caps uppercase text-[#7ab7af]">
                      NDE modality
                    </span>
                    <span className="font-mono text-[10px] text-[#7ab7af]">
                      non-destructive · no disassembly
                    </span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {[
                      { id: "visual", label: "Optical", icon: "photo_camera" },
                      { id: "thermography", label: "Thermography", icon: "thermostat" },
                      { id: "ultrasound", label: "Ultrasound", icon: "graphic_eq" },
                      { id: "xray", label: "X-ray", icon: "radiology" },
                    ].map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setModality(m.id)}
                        className={`flex cursor-pointer flex-col items-center gap-1.5 rounded-xl border px-2 py-2.5 transition-all duration-200 ${
                          modality === m.id
                            ? "border-[#5ac8b4]/70 bg-[#5ac8b4]/15 text-white shadow-[0_0_20px_-8px_#5ac8b4]"
                            : "border-white/10 bg-white/[0.03] text-[#9cc7c0] hover:border-[#5ac8b4]/40 hover:text-white"
                        }`}
                      >
                        <span
                          className="material-symbols-outlined text-[17px]"
                          style={{ color: modality === m.id ? "#5ac8b4" : undefined }}
                        >
                          {m.icon}
                        </span>
                        <span className="font-label-caps text-[9px] uppercase tracking-wider">
                          {m.label}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mt-4 grid gap-4 rounded-2xl border border-white/[0.06] bg-[#02090c]/80 p-4 sm:grid-cols-[1fr_auto] sm:items-end">
                  <div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-label-caps uppercase text-[#7ab7af]">
                        Confidence threshold
                      </span>
                      <span className="font-mono font-bold text-[#5ac8b4]">
                        {confidenceThreshold.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      className="mt-3 w-full cursor-pointer accent-[#5ac8b4]"
                    />
                  </div>

                  <button
                    disabled={loading || !imageFile}
                    onClick={analyze}
                    className={`h-11 rounded-xl px-7 font-label-caps text-xs font-bold uppercase transition-all duration-300 ${
                      loading || !imageFile
                        ? "cursor-not-allowed bg-[#5ac8b4]/12 text-[#5ac8b4]/60"
                        : "cursor-pointer bg-[#5ac8b4] text-[#00211d] shadow-[0_14px_38px_-14px_rgba(90,200,180,0.95)] hover:bg-[#78dcc8]"
                    }`}
                  >
                    {loading ? "Analyzing…" : "Analyze Image"}
                  </button>
                </div>

                {/* Specimen rejection card — non-carbon-fibre uploads */}
                {rejection && (
                  <div className="animate-fade-up mt-4 rounded-2xl border border-[#ff7a7a]/50 bg-gradient-to-b from-[#2a0d10]/90 to-[#150608]/90 p-5">
                    <div className="flex items-start gap-3">
                      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[#ff7a7a]/40 bg-[#ff7a7a]/10 text-[#ff7a7a]">
                        <span className="material-symbols-outlined" style={{ fontSize: 22 }}>
                          block
                        </span>
                      </span>
                      <div className="min-w-0 flex-1">
                        <h4 className="font-headline-sm text-base text-[#ffb1ab]">
                          Not a carbon-fibre specimen
                        </h4>
                        <p className="mt-1 text-sm leading-relaxed text-[#ffb1ab]/90">
                          {rejection.message}
                        </p>
                        {rejection.reasons?.length > 0 && (
                          <ul className="mt-3 space-y-1.5">
                            {rejection.reasons.map((reason) => (
                              <li
                                key={reason}
                                className="flex items-start gap-2 text-xs leading-relaxed text-[#ffd9d5]"
                              >
                                <span
                                  className="material-symbols-outlined mt-0.5 text-[#ff7a7a]"
                                  style={{ fontSize: 13 }}
                                >
                                  error
                                </span>
                                {reason}
                              </li>
                            ))}
                          </ul>
                        )}
                        {rejection.measured && (
                          <div className="mt-3 flex flex-wrap gap-2 font-mono text-[10px] text-[#ff9a93]">
                            {Object.entries(rejection.measured).map(([k, v]) => (
                              <span
                                key={k}
                                className="rounded-md border border-[#ff7a7a]/30 bg-[#ff7a7a]/[0.08] px-2.5 py-1"
                              >
                                {k.replace(/_/g, " ")}: {v}
                              </span>
                            ))}
                          </div>
                        )}
                        <p className="mt-3 text-[11px] text-[#ffb1ab]/70">
                          Upload a composite/carbon-fibre surface image instead —
                          the weave texture and low colour saturation are what the
                          model expects.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-4 min-h-[24px]">
                  {loading && <Spinner label="Running CV analysis…" />}
                  {status && !loading && (
                    <span
                      className="text-sm"
                      style={{
                        color: status.type === "error" ? "#ff9a93" : "#9fd6a0",
                      }}
                    >
                      {status.type === "error" ? "⚠ " : "✓ "}
                      {status.message}
                    </span>
                  )}
                  {!status && !loading && !result && (
                    <span className="text-xs text-[#7ab7af]">
                      Awaiting specimen — confidence threshold{" "}
                      {confidenceThreshold.toFixed(2)}
                    </span>
                  )}
                  {apiOnline === false && !loading && (
                    <span className="text-xs text-[#f4c96b]">
                      API offline — analysis will fall back to the offline demo.
                    </span>
                  )}
                </div>
              </div>

              {/* Verdict */}
              <div className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-gradient-to-b from-[#0a2a31] to-[#04141a] p-6 shadow-[0_30px_80px_-50px_rgba(90,200,180,0.7)] lg:col-span-5">
                <div
                  aria-hidden
                  className="pointer-events-none absolute -top-16 -right-16 h-48 w-48 rounded-full blur-3xl"
                  style={{
                    background: result
                      ? `${riskColor(result.risk_level)}33`
                      : "rgba(90,200,180,0.18)",
                  }}
                />

                <div className="relative flex items-center justify-between">
                  <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                    Latest verdict
                  </span>
                  {result?.threshold_met !== undefined && (
                    <span
                      className="rounded-full px-3 py-1 font-mono text-[10px] uppercase"
                      style={{
                        background: result.threshold_met ? "rgba(159,214,160,0.12)" : "rgba(244,201,107,0.12)",
                        color: result.threshold_met ? "#9fd6a0" : "#f4c96b",
                        border: `1px solid ${result.threshold_met ? "#9fd6a055" : "#f4c96b55"}`,
                      }}
                    >
                      {result.threshold_met ? "threshold met" : "below threshold"}
                    </span>
                  )}
                </div>

                <div className="relative mt-4">
                  <div
                    className="font-display-hero text-4xl font-light text-white sm:text-5xl"
                    style={
                      result
                        ? {
                            textShadow: `0 0 45px ${riskColor(result.risk_level)}66`,
                          }
                        : undefined
                    }
                  >
                    {result ? classLabel(result.predicted_class) : "Awaiting sample"}
                  </div>
                  <p className="mt-2 text-xs text-[#7ab7af]">
                    {verdictCaption}
                  </p>
                </div>

                {result ? (
                  <div className="relative mt-6 space-y-5">
                    <div>
                      <div className="flex items-end justify-between">
                        <span className="font-label-caps text-[10px] uppercase text-[#7ab7af]">
                          Confidence
                        </span>
                        <span className="font-technical-metric text-2xl font-bold text-white">
                          {(result.confidence * 100).toFixed(1)}
                          <span className="text-sm text-[#7ab7af]">%</span>
                        </span>
                      </div>
                      <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-[#02090c]">
                        <div
                          className="bar-shine relative h-full rounded-full transition-all duration-700"
                          style={{
                            width: `${Math.min(100, result.confidence * 100)}%`,
                            background: `linear-gradient(90deg, ${confidenceColor(
                              result.confidence
                            )}88, ${confidenceColor(result.confidence)})`,
                          }}
                        />
                      </div>
                    </div>                      <div className="flex items-center justify-between rounded-xl border border-white/[0.06] bg-[#02090c]/70 px-4 py-3">
                      <RiskPill riskLevel={result.risk_level} />
                      <div className="text-right">
                        <div className="font-label-caps text-[9px] uppercase text-[#7ab7af]">
                          RPN
                        </div>
                        <div className="font-mono text-lg font-bold text-white">
                          {result.rpn}
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2 font-mono text-[10px] text-[#9cc7c0]">
                      <span className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1">
                        {engine}
                      </span>
                      <span className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1">
                        thr {result.confidence_threshold?.toFixed?.(2) ?? "0.50"}
                      </span>
                      <span className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1">
                        {probabilities.length} classes
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="relative mt-6 rounded-xl border border-dashed border-white/10 bg-[#02090c]/60 p-6 text-center">
                    <span className="material-symbols-outlined text-[34px] text-[#3f6a65]">
                      {imageFile ? "image" : "query_stats"}
                    </span>
                    <p className="mt-2 text-xs text-[#7ab7af]">
                      {imageFile
                        ? `${imageFile.size >= 1024 ? `${Math.round(imageFile.size / 1024)} KB` : `${imageFile.size} B`} selected · press Analyze to see the prediction.`
                        : "Prediction, FMEA ratings and class probabilities appear here."}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ----------------------------- RESULTS ------------------------------ */}
        <section id="results" className="px-4 py-14 sm:px-8 lg:px-16">
          <div className="mx-auto max-w-7xl space-y-8">
            <SectionHeading
              icon="query_stats"
              eyebrow="Prediction, FMEA & signal breakdown"
              title="Analysis Results"
              action={
                result?.id ? (
                  <a
                    href={`${API_BASE}/inspections/${result.id}/report`}
                    className="inline-flex items-center gap-2 rounded-lg border border-[#78dcc8]/30 bg-[#78dcc8]/[0.08] px-4 py-2 font-label-caps text-[11px] font-semibold uppercase text-[#78dcc8] transition-all duration-300 hover:border-[#78dcc8]/60 hover:bg-[#78dcc8]/15 hover:text-white"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 15 }}>
                      picture_as_pdf
                    </span>
                    Download report (PDF)
                  </a>
                ) : null
              }
            />

            {!result ? (
              <div className="rounded-2xl border border-dashed border-white/10 bg-[#04141a]/60 p-12 text-center">
                <span className="material-symbols-outlined text-[40px] text-[#3f6a65]">
                  monitor_heart
                </span>
                <p className="mt-3 text-sm text-[#7ab7af]">
                  Run an inspection in the console above to see predicted class,
                  confidence, FMEA risk ratings and class probabilities.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* FMEA grid + RPN */}
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:col-span-8">
                    {fmeaMetrics.map((metric) => (
                      <div
                        key={metric.label}
                        className="rounded-2xl border border-white/[0.07] bg-[#061d23] p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-[#5ac8b4]/40"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-label-caps text-[10px] uppercase text-[#7ab7af]">
                            {metric.label}
                          </span>
                          <span className="material-symbols-outlined text-[16px] text-[#5ac8b4]">
                            {metric.icon}
                          </span>
                        </div>
                        <div className="font-technical-metric mt-3 text-2xl font-bold text-white">
                          {Number(metric.value).toFixed(1)}
                          <span className="text-sm text-[#7ab7af]"> / 10</span>
                        </div>
                        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#02090c]">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-[#5ac8b4] to-[#78dcc8] transition-all duration-700"
                            style={{ width: `${(Number(metric.value) / 10) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div
                    className="relative overflow-hidden rounded-2xl border p-6 lg:col-span-4"
                    style={{
                      borderColor: `${riskColor(result.risk_level)}55`,
                      background: `linear-gradient(160deg, ${riskColor(
                        result.risk_level
                      )}1a, #04141a 65%)`,
                    }}
                  >
                    <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#7ab7af]">
                      Risk Priority Number
                    </span>
                    <div className="mt-2 flex items-end gap-3">
                      <span className="font-technical-metric text-5xl font-bold text-white">
                        {result.rpn}
                      </span>
                      <span className="mb-1.5">
                        <RiskPill riskLevel={result.risk_level} />
                      </span>
                    </div>
                    <p className="mt-3 font-mono text-[10px] text-[#7ab7af]">
                      S {result.severity} × O {result.occurrence} × D {result.detection}
                    </p>
                  </div>
                </div>

                {/* Probabilities + signals */}
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                  <div className="rounded-2xl border border-white/[0.08] bg-[#061d23] p-6 lg:col-span-7">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                          Calibrated distribution
                        </span>
                        <h3 className="font-headline-sm mt-1 text-lg text-white">
                          Class Probabilities
                        </h3>
                      </div>
                      <span className="material-symbols-outlined text-[20px] text-[#5ac8b4]">
                        bar_chart
                      </span>
                    </div>

                    <div className="mt-5 space-y-4">
                      {probabilities.map(([cls, prob], i) => {
                        const isTop = i === 0;
                        return (
                          <div key={cls} className="flex items-center gap-4">
                            <span
                              className={`w-28 shrink-0 font-mono text-[11px] uppercase tracking-wider ${
                                isTop ? "text-[#5ac8b4]" : "text-[#9cc7c0]"
                              }`}
                            >
                              {cls}
                            </span>
                            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-[#02090c]">
                              <div
                                className="h-full rounded-full transition-all duration-700"
                                style={{
                                  width: `${prob * 100}%`,
                                  background: isTop
                                    ? `linear-gradient(90deg, ${confidenceColor(
                                        result.confidence
                                      )}99, ${confidenceColor(result.confidence)})`
                                    : "linear-gradient(90deg, #1c3a38, #3f6a65)",
                                }}
                              />
                            </div>
                            <span className="w-14 text-right font-mono text-xs font-bold text-white">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/[0.08] bg-[#061d23] p-6 lg:col-span-5">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#78dcc8]">
                          Raw CV detector output
                        </span>
                        <h3 className="font-headline-sm mt-1 text-lg text-white">
                          Signal Strength
                        </h3>
                      </div>
                      <span className="material-symbols-outlined text-[20px] text-[#78dcc8]">
                        sensors
                      </span>
                    </div>

                    <div className="mt-5 space-y-4">
                      {signals.map((signal) => {
                        const pct = Math.min(100, Math.max(0, signal.score * 100));
                        return (
                          <div key={signal.label}>
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-label-caps uppercase text-[#9cc7c0]">
                                {signal.label}
                              </span>
                              <span className="font-mono text-[#7ab7af]">
                                {signal.regions} region{signal.regions === 1 ? "" : "s"} ·{" "}
                                <span className="text-white">
                                  {Number(signal.score).toFixed(2)}
                                </span>
                              </span>
                            </div>
                            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[#02090c]">
                              <div
                                className="h-full rounded-full transition-all duration-700"
                                style={{
                                  width: `${pct}%`,
                                  background:
                                    pct > 60
                                      ? "linear-gradient(90deg,#ff9f6b,#ff7a7a)"
                                      : pct > 25
                                      ? "linear-gradient(90deg,#f4c96b,#ff9f6b)"
                                      : "linear-gradient(90deg,#5ac8b4,#78dcc8)",
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* ------- AI segmentation overlay + per-mechanism FMEA ------- */}
                {result.segmentation && (
                  <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                    <div className="rounded-2xl border border-[#78dcc8]/20 bg-[#061d23] p-6 lg:col-span-5">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#78dcc8]">
                            compositeguard_v4 · ONNX segmentation
                          </span>
                          <h3 className="font-headline-sm mt-1 text-lg text-white">
                            Defect Overlay
                          </h3>
                        </div>
                        <span className="material-symbols-outlined text-[20px] text-[#78dcc8]">
                          layers
                        </span>
                      </div>

                      <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-[#02090c]">
                        {result.overlay_url ? (
                          <img
                            src={result.overlay_url}
                            alt="AI defect segmentation overlay"
                            className="mx-auto max-h-[300px] w-auto"
                          />
                        ) : (
                          <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
                            <span className="material-symbols-outlined text-[28px] text-[#3f6a65]">
                              layers
                            </span>
                            <p className="text-xs text-[#7ab7af]">
                              Overlay unavailable for this (pre-model) record.
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="mt-4 space-y-2.5">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-label-caps uppercase text-[#7ab7af]">
                            Total defect coverage
                          </span>
                          <span className="font-mono font-bold text-white">
                            {result.segmentation.defect_percentage.toFixed(2)}% ({""}
                            {result.segmentation.defect_pixels?.toLocaleString?.() ?? result.segmentation.defect_pixels} px)
                          </span>
                        </div>
                        {Object.entries(result.segmentation.class_summary)
                          .filter(([, s]) => s.pixels > 0)
                          .map(([name, s], i) => (
                            <div key={name} className="flex items-center gap-3">
                              <span
                                className="h-2.5 w-2.5 shrink-0 rounded-sm"
                                style={{ background: ["#ff4141", "#ffbe28", "#32a0ff", "#be50ff"][i % 4] }}
                              />
                              <span className="w-40 shrink-0 truncate font-mono text-[11px] text-[#9cc7c0]">
                                {name}
                              </span>
                              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#02090c]">
                                <div
                                  className="h-full rounded-full bg-gradient-to-r from-[#5ac8b4] to-[#78dcc8]"
                                  style={{ width: `${Math.min(100, s.percentage_of_image * 3)}%` }}
                                />
                              </div>
                              <span className="w-16 text-right font-mono text-[11px] text-white">
                                {s.percentage_of_image.toFixed(1)}%
                              </span>
                            </div>
                          ))}
                      </div>

                      {/* exact model values — per-class softmax + pixel confidence */}
                      {result.segmentation.class_probabilities && (
                        <div className="mt-5 rounded-xl border border-white/[0.06] bg-[#02090c]/70 p-4">
                          <div className="flex items-center justify-between">
                            <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                              Model values · softmax
                            </span>
                            <span className="font-mono text-[10px] text-[#7ab7af]">
                              pixel conf {(result.segmentation.pixel_confidence * 100).toFixed(1)}% ·{" "}
                              floor {(result.segmentation.min_pixel_confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <table className="mt-3 w-full text-left text-[11px]">
                            <tbody>
                              {Object.entries(result.segmentation.class_probabilities)
                                .sort(([, a], [, b]) => b - a)
                                .map(([name, prob]) => (
                                  <tr key={name} className="border-t border-white/[0.05] first:border-t-0">
                                    <td className="py-1.5 pr-3 text-[#9cc7c0]">{name}</td>
                                    <td className="py-1.5">
                                      <div className="h-1.5 overflow-hidden rounded-full bg-[#04141a]">
                                        <div
                                          className="h-full rounded-full bg-gradient-to-r from-[#3f6a65] to-[#5ac8b4]"
                                          style={{ width: `${Math.max(1, prob * 100)}%` }}
                                        />
                                      </div>
                                    </td>
                                    <td className="w-16 py-1.5 text-right font-mono font-bold text-white">
                                      {(prob * 100).toFixed(2)}%
                                    </td>
                                  </tr>
                                ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>

                    <div className="rounded-2xl border border-white/[0.08] bg-[#061d23] p-6 lg:col-span-7">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                            Failure mechanism identification
                          </span>
                          <h3 className="font-headline-sm mt-1 text-lg text-white">
                            Mechanism FMEA
                          </h3>
                        </div>
                        <span className="material-symbols-outlined text-[20px] text-[#5ac8b4]">
                          workflow
                        </span>
                      </div>

                      {result.mechanisms?.length ? (
                        <div className="mt-4 overflow-x-auto">
                          <table className="w-full text-left text-xs">
                            <thead>
                              <tr className="font-label-caps text-[9px] uppercase text-[#7ab7af]">
                                <th className="pb-2 pr-3">Mechanism</th>
                                <th className="pb-2 pr-3">S</th>
                                <th className="pb-2 pr-3">O</th>
                                <th className="pb-2 pr-3">D</th>
                                <th className="pb-2 pr-3">RPN</th>
                                <th className="pb-2">Risk</th>
                              </tr>
                            </thead>
                            <tbody>
                              {result.mechanisms.map((m) => (
                                <tr
                                  key={m.mechanism}
                                  className="border-t border-white/[0.06] text-[#cfe8db]"
                                >
                                  <td className="py-2.5 pr-3">
                                    <span className="text-white">{m.mechanism}</span>
                                    <span className="ml-2 font-mono text-[10px] text-[#7ab7af]">
                                      {m.percentage.toFixed(1)}%
                                    </span>
                                  </td>
                                  <td className="py-2.5 pr-3 font-mono">{m.severity}</td>
                                  <td className="py-2.5 pr-3 font-mono">{m.occurrence}</td>
                                  <td className="py-2.5 pr-3 font-mono">{m.detection}</td>
                                  <td className="py-2.5 pr-3 font-mono font-bold text-white">
                                    {m.rpn}
                                  </td>
                                  <td className="py-2.5">
                                    <RiskPill riskLevel={m.risk_level} />
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="mt-4 text-xs text-[#7ab7af]">
                          No modelled failure mechanisms detected above zero coverage.
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* ------- Residual life + maintenance planning ------- */}
                {result.residual_life && (
                  <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                    <div className="rounded-2xl border border-white/[0.08] bg-[#061d23] p-6 lg:col-span-5">
                      <div className="flex items-center justify-between">
                        <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                          Lifecycle assessment
                        </span>
                        <span className="material-symbols-outlined text-[18px] text-[#5ac8b4]">
                          hourglass_bottom
                        </span>
                      </div>
                      <div className="mt-4 flex items-end justify-between">
                        <div>
                          <div className="font-technical-metric text-4xl font-bold text-white">
                            {result.residual_life.remaining_percent.toFixed(1)}
                            <span className="text-lg text-[#7ab7af]">%</span>
                          </div>
                          <div className="font-label-caps mt-1 text-[9px] uppercase text-[#7ab7af]">
                            Estimated remaining life
                          </div>
                        </div>
                        <span
                          className="rounded-full px-3 py-1.5 font-mono text-[11px] uppercase"
                          style={{
                            background: "rgba(90,200,180,0.12)",
                            color: "#78dcc8",
                            border: "1px solid #5ac8b455",
                          }}
                        >
                          {result.residual_life.phase}
                        </span>
                      </div>
                      <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-[#02090c]">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-[#f4c96b] via-[#9fd6a0] to-[#5ac8b4] transition-all duration-700"
                          style={{ width: `${result.residual_life.remaining_percent}%` }}
                        />
                      </div>
                      <p className="mt-3 text-[11px] leading-relaxed text-[#7ab7af]">
                        Estimated from the verdict RPN: higher risk priority
                        consumes remaining service life faster.
                      </p>
                    </div>

                    <div className="rounded-2xl border border-white/[0.08] bg-[#061d23] p-6 lg:col-span-7">
                      <div className="flex items-center justify-between">
                        <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#f4c96b]">
                          Maintenance planning
                        </span>
                        <span className="material-symbols-outlined text-[18px] text-[#f4c96b]">
                          construction
                        </span>
                      </div>
                      <div className="mt-4 flex flex-wrap items-center gap-3">
                        <span className="rounded-lg border border-[#f4c96b]/40 bg-[#f4c96b]/10 px-3 py-1.5 font-mono text-xs font-bold text-[#f4c96b]">
                          {result.maintenance.priority}
                        </span>
                        <span className="font-mono text-xs text-[#9cc7c0]">
                          {result.maintenance.interval_days === 0
                            ? "Immediate — stop use"
                            : `Re-inspect within ${result.maintenance.interval_days} days`}
                        </span>
                      </div>
                      <p className="mt-3 text-sm leading-relaxed text-[#cfe8db]">
                        {result.maintenance.action}
                      </p>
                      <div className="mt-4 flex flex-wrap gap-2 font-mono text-[10px] text-[#9cc7c0]">
                        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1">
                          modality: {result.modality ?? "visual"}
                        </span>
                        <span className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1">
                          {engine}
                        </span>
                      </div>

                      <a
                        href={`${API_BASE}/inspections/${result.id}/report`}
                        className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-[#78dcc8]/35 bg-[#78dcc8]/[0.08] px-4 py-3 font-label-caps text-[11px] font-bold uppercase text-[#78dcc8] transition-all duration-300 hover:border-[#78dcc8]/70 hover:bg-[#78dcc8]/15 hover:text-white"
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
                          picture_as_pdf
                        </span>
                        Download full analysis report
                      </a>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* ---------------------- SENSORS + VALIDATION ---------------------- */}
        <section id="sensors" className="px-4 py-14 sm:px-8 lg:px-16">
          <div className="mx-auto max-w-7xl space-y-8">
            <SectionHeading
              icon="sensors"
              eyebrow="Real-time rig channels & experimental verification"
              title="Monitoring & Validation"
            />

            {/* live sensor channels */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { key: "temperature_c", label: "Surface temperature", unit: "°C", icon: "thermostat", tint: "#f4c96b" },
                { key: "vibration_mm_s", label: "Vibration", unit: "mm/s", icon: "graphic_eq", tint: "#5ac8b4" },
                { key: "ae_hits_s", label: "Acoustic emission", unit: "hits/s", icon: "hearing", tint: "#ff9f6b" },
                { key: "humidity_pct", label: "Humidity", unit: "%", icon: "water_drop", tint: "#78dcc8" },
              ].map((ch) => {
                const value = telemetry?.latest?.[ch.key];
                const hist = telemetry?.history ?? [];
                const vals = hist.map((p) => p[ch.key]).filter((v) => typeof v === "number");
                const min = vals.length ? Math.min(...vals) : 0;
                const max = vals.length ? Math.max(...vals) : 1;
                const sparkW = 100;
                const sparkH = 26;
                const path = vals.length > 1
                  ? vals
                      .map((v, i) => {
                        const x = (i / (vals.length - 1)) * sparkW;
                        const y = sparkH - ((v - min) / (max - min || 1)) * sparkH;
                        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
                      })
                      .join(" ")
                  : null;
                return (
                  <div
                    key={ch.key}
                    className="rounded-2xl border border-white/[0.07] bg-gradient-to-b from-[#0a2a31]/70 to-[#04141a]/60 p-4 backdrop-blur transition-all duration-300 hover:-translate-y-0.5 hover:border-[#5ac8b4]/30"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-label-caps text-[9px] uppercase text-[#7ab7af]">
                        {ch.label}
                      </span>
                      <span
                        className="flex h-7 w-7 shrink-0 items-center justify-center overflow-hidden rounded-lg border"
                        style={{ borderColor: `${ch.tint}40`, background: `${ch.tint}14`, color: ch.tint }}
                      >
                        <span
                          className="material-symbols-outlined"
                          style={{ fontSize: 14 }}
                        >
                          {ch.icon}
                        </span>
                      </span>
                    </div>
                    <div className="mt-2.5 flex items-end justify-between gap-2">
                      <span className="font-technical-metric text-2xl font-bold text-white">
                        {typeof value === "number" ? value.toFixed(1) : "—"}
                        <span className="ml-1 text-xs text-[#7ab7af]">{ch.unit}</span>
                      </span>
                      <span className="flex items-center gap-1.5 font-mono text-[9px] uppercase text-[#9fd6a0]">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#9fd6a0]" />
                        live
                      </span>
                    </div>
                    <svg
                      viewBox={`0 0 ${sparkW} ${sparkH}`}
                      preserveAspectRatio="none"
                      className="mt-3 h-7 w-full"
                      aria-hidden="true"
                    >
                      {path && (
                        <path
                          d={path}
                          fill="none"
                          stroke={ch.tint}
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          vectorEffect="non-scaling-stroke"
                        />
                      )}
                    </svg>
                  </div>
                );
              })}
            </div>
            <p className="-mt-4 text-[11px] text-[#7ab7af]">
              {telemetry
                ? `Channels sampled server-side · ${telemetry.history.length}-point window · ${telemetry.source}`
                : "Waiting for the first telemetry sample…"}
            </p>

            {/* experimental validation */}
            <div className="rounded-2xl border border-white/[0.08] bg-gradient-to-b from-[#061d23] to-[#04141a] p-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                    Simulation-based performance verification
                  </span>
                  <h3 className="font-headline-sm mt-1 text-xl text-white">
                    Experimental Validation
                  </h3>
                  <p className="mt-1.5 max-w-2xl text-xs leading-relaxed text-[#7ab7af]">
                    Runs the full pipeline (classic CV + ONNX fusion) over the
                    labelled fixture set in <code>data/test_images</code> and
                    scores every verdict against its ground-truth label.
                  </p>
                </div>
                <button
                  onClick={runValidationNow}
                  disabled={validating}
                  className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-[#5ac8b4] px-6 py-3 font-label-caps text-xs font-bold uppercase text-[#00211d] shadow-[0_14px_38px_-14px_rgba(90,200,180,0.95)] transition-all duration-300 hover:bg-[#78dcc8] disabled:cursor-wait disabled:opacity-60"
                >
                  <span className={`material-symbols-outlined text-[16px] ${validating ? "animate-spin" : ""}`}>
                    {validating ? "progress_activity" : "science"}
                  </span>
                  {validating ? "Running…" : "Run validation"}
                </button>
              </div>

              {validationError && (
                <p className="mt-4 font-mono text-xs text-[#ff9a93]">{validationError}</p>
              )}

              {validation && (
                <div className="mt-5 space-y-4">
                  <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                    {[
                      { label: "Accuracy", value: `${Math.round((validation.accuracy ?? 0) * 100)}%` },
                      { label: "Mean confidence", value: `${((validation.mean_confidence ?? 0) * 100).toFixed(1)}%` },
                      { label: "Images verified", value: validation.n_images },
                      { label: "Elapsed", value: `${validation.elapsed_ms} ms` },
                    ].map((m) => (
                      <div key={m.label} className="rounded-xl border border-white/[0.06] bg-[#02090c]/70 p-4 text-center">
                        <div className="font-technical-metric text-xl font-bold text-white">{m.value}</div>
                        <div className="font-label-caps mt-1 text-[9px] uppercase text-[#7ab7af]">
                          {m.label}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="font-label-caps text-[9px] uppercase text-[#7ab7af]">
                          <th className="pb-2 pr-3">Fixture</th>
                          <th className="pb-2 pr-3">Expected</th>
                          <th className="pb-2 pr-3">Predicted</th>
                          <th className="pb-2 pr-3">Confidence</th>
                          <th className="pb-2">Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {validation.results.map((row) => (
                          <tr key={row.image} className="border-t border-white/[0.06] text-[#cfe8db]">
                            <td className="py-2.5 pr-3 font-mono text-[11px] text-white">{row.image}</td>
                            <td className="py-2.5 pr-3 font-mono">{row.expected ?? "—"}</td>
                            <td className="py-2.5 pr-3 font-mono">{row.predicted}</td>
                            <td className="py-2.5 pr-3 font-mono">
                              {(row.confidence * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5">
                              {row.match === null ? (
                                <span className="font-mono text-[11px] text-[#7ab7af]">unlabelled</span>
                              ) : row.match ? (
                                <span className="font-mono text-[11px] text-[#9fd6a0]">✓ pass</span>
                              ) : (
                                <span className="font-mono text-[11px] text-[#ff9a93]">✗ fail</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {!validation && !validationError && !validating && (
                <p className="mt-4 text-xs text-[#7ab7af]">
                  No run yet — press “Run validation” to verify performance
                  against the labelled fixture set.
                </p>
              )}
            </div>

            {/* standards references */}
            <div className="rounded-2xl border border-white/[0.08] bg-[#061d23] p-6">
              <div className="flex items-center justify-between">
                <span className="font-label-telemetry text-[10px] uppercase tracking-wider text-[#5ac8b4]">
                  Industry standards conformance
                </span>
                <span className="material-symbols-outlined text-[18px] text-[#5ac8b4]"> verified </span>
              </div>
              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  { id: "ISO 9712:2021", title: "Qualification and certification of NDT personnel" },
                  { id: "ASTM E2580-24", title: "Ultrasonic testing of flat panel composites (aerospace)" },
                  { id: "ASTM E2533-21", title: "NDE of polymer matrix composites used in aerospace" },
                  { id: "ASTM E2582-21(2025)", title: "Infrared flash thermography of composite panels" },
                  { id: "ASTM E2662-15(2022)", title: "Radiographic examination of flat panel composites" },
                  { id: "ISO 16810:2024", title: "Ultrasonic testing — general principles" },
                ].map((std) => (
                  <div
                    key={std.id}
                    className="rounded-xl border border-white/[0.06] bg-[#02090c]/70 p-3.5 transition-all duration-200 hover:border-[#5ac8b4]/30"
                  >
                    <div className="font-mono text-xs font-bold text-[#78dcc8]">{std.id}</div>
                    <div className="mt-1 text-[11px] leading-snug text-[#9cc7c0]">{std.title}</div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[10px] leading-relaxed text-[#7ab7af]">
                References guide modality selection and reporting; personnel
                certification, acceptance criteria and full-scale testing
                remain the operating organisation’s responsibility.
              </p>
            </div>
          </div>
        </section>

        {/* ----------------------------- HISTORY ------------------------------ */}
        <section id="history" className="px-4 py-14 sm:px-8 lg:px-16">
          <div className="mx-auto max-w-7xl space-y-8">
            <SectionHeading
              icon="history"
              eyebrow="Persisted in logs/inspections.jsonl"
              title="Inspection History"
              action={
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={refreshHistory}
                    disabled={refreshing}
                    className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-white/12 bg-white/[0.04] px-4 py-2 font-label-caps text-[11px] font-semibold uppercase text-[#cfe8db] transition-all duration-300 hover:border-[#5ac8b4]/50 hover:bg-[#5ac8b4]/10 hover:text-white disabled:cursor-wait disabled:opacity-60"
                  >
                    <span
                      className={`material-symbols-outlined text-[15px] text-[#5ac8b4] ${
                        refreshing ? "animate-spin"
                        : ""
                      }`}
                    >
                      {refreshing ? "progress_activity" : "refresh"}
                    </span>
                    {refreshing ? "Syncing…" : "Refresh history"}
                  </button>
                  <button
                    onClick={clearHistory}
                    disabled={refreshing || inspections.length === 0}
                    title="Delete all inspection history on the server"
                    className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[#ff9a93]/25 bg-[#ff9a93]/[0.06] px-4 py-2 font-label-caps text-[11px] font-semibold uppercase text-[#ffb1ab] transition-all duration-300 hover:border-[#ff9a93]/60 hover:bg-[#ff9a93]/15 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <span className="material-symbols-outlined text-[15px]">delete_sweep</span>
                    Clear history
                  </button>
                </div>
              }
            />

            {/* Summary strip — a quick read of the log */}
            {historyStats.total > 0 && (
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                {[
                  { label: "Total inspections", value: historyStats.total, icon: "inventory_2", tint: "#5ac8b4" },
                  { label: "Clean surfaces", value: historyStats.clean, icon: "verified", tint: "#9fd6a0" },
                  { label: "Defects found", value: historyStats.defects, icon: "crisis_alert", tint: "#f4c96b" },
                  { label: "Avg confidence", value: `${(historyStats.avgConf * 100).toFixed(0)}%`, icon: "speed", tint: "#78dcc8" },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="flex items-center gap-3.5 rounded-2xl border border-white/[0.07] bg-gradient-to-b from-[#0a2a31]/70 to-[#04141a]/60 px-4 py-3.5 backdrop-blur transition-all duration-300 hover:-translate-y-0.5 hover:border-[#5ac8b4]/30"
                  >
                    <span
                      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border"
                      style={{ borderColor: `${stat.tint}40`, background: `${stat.tint}14`, color: stat.tint }}
                    >
                      <span className="material-symbols-outlined text-[20px]">{stat.icon}</span>
                    </span>
                    <div className="min-w-0">
                      <div className="font-technical-metric text-xl font-bold text-white">{stat.value}</div>
                      <div className="truncate text-[10px] uppercase tracking-wider text-[#7ab7af]">{stat.label}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {historyStats.flagged > 0 && (
              <div className="flex items-start gap-2.5 rounded-xl border border-[#ff9f6b]/30 bg-[#ff9f6b]/10 px-4 py-3 text-xs text-[#ffb891]">
                <span className="material-symbols-outlined mt-0.5 text-[16px]">priority_high</span>
                <span>
                  <strong>{historyStats.flagged}</strong> inspection{historyStats.flagged === 1 ? "" : "s"} in this log
                  carry a <strong>High</strong> or <strong>Critical</strong> risk rating — open the card to review the
                  specimen image and FMEA breakdown.
                </span>
              </div>
            )}

            {historyRows.length > 0 ? (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {historyRows.map((insp, i) => (
                  <button
                    key={insp.id ?? `${insp.timestamp}-${i}`}
                    onClick={() => setSelectedRecord(insp)}
                    aria-label={`Open inspection detail for ${insp.image_filename ?? "specimen"}`}
                    className="animate-fade-up group cursor-pointer rounded-2xl border border-white/[0.07] bg-gradient-to-b from-[#061d23] to-[#04141a] p-0 text-left transition-all duration-300 hover:-translate-y-1 hover:border-[#5ac8b4]/40 hover:shadow-[0_24px_60px_-40px_rgba(90,200,180,0.9)] focus:outline-none focus-visible:border-[#5ac8b4] focus-visible:ring-2 focus-visible:ring-[#5ac8b4]/40"
                    style={{ animationDelay: `${i * 60}ms` }}
                  >
                    {/* thumbnail */}
                    <div className="relative h-36 w-full overflow-hidden rounded-t-2xl border-b border-white/[0.06] bg-[#02090c]">
                      {inspectionImageUrl(insp) ? (
                        <img
                          src={inspectionImageUrl(insp)}
                          alt={insp.image_filename ?? "specimen"}
                          loading="lazy"
                          className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                          onError={(e) => {
                            // hide the broken image, show the placeholder icon instead
                            e.currentTarget.style.display = "none";
                            const ph = e.currentTarget.parentElement?.querySelector("[data-fallback]");
                            if (ph) ph.style.display = "flex";
                          }}
                        />
                      ) : null}
                      <span
                        data-fallback
                        style={{ display: inspectionImageUrl(insp) ? "none" : "flex" }}
                        className="absolute inset-0 items-center justify-center text-[#3f6a65]"
                      >
                        <span className="material-symbols-outlined text-[34px]">image</span>
                      </span>

                      {/* hover affordance */}
                      <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-[#00080d]/60 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-[#5ac8b4]/50 bg-[#5ac8b4]/15 px-3.5 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-[#9fe8da] backdrop-blur">
                          <span className="material-symbols-outlined text-[14px]">zoom_in</span>
                          View specimen
                        </span>
                      </span>

                      {/* risk pill + RPN overlay */}
                      <span className="absolute top-2.5 right-2.5">
                        <RiskPill riskLevel={insp.risk_level} />
                      </span>
                      <span className="absolute bottom-2.5 left-2.5 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/65 px-2.5 py-1 font-mono text-[10px] text-[#cfe8db] backdrop-blur">
                        RPN {insp.rpn ?? 0}
                      </span>
                    </div>

                    <div className="p-5">
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-headline-sm truncate text-lg text-white">
                          {classLabel(insp.predicted_class)}
                        </span>
                        <span className="font-technical-metric shrink-0 text-lg font-bold text-[#5ac8b4]">
                          {((insp.confidence ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* confidence bar */}
                      <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-[#02090c]">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: `${Math.min(100, (insp.confidence ?? 0) * 100)}%`,
                            background: `linear-gradient(90deg, ${confidenceColor(insp.confidence ?? 0)}88, ${confidenceColor(
                              insp.confidence ?? 0
                            )})`,
                          }}
                        />
                      </div>

                      <div className="mt-3 flex items-center justify-between gap-2 border-t border-white/5 pt-3 font-mono text-[10px] text-[#7ab7af]">
                        <span className="truncate">{insp.image_filename ?? "specimen"}</span>
                        <span className="shrink-0">{formatTime(insp.timestamp)}</span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-white/10 bg-[#04141a]/60 px-6 py-14 text-center">
                <span className="guard-glow-teal animate-float mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-[#5ac8b4]/30 bg-[#5ac8b4]/10 text-[#5ac8b4]">
                  <span className="material-symbols-outlined text-[30px]">history</span>
                </span>
                <p className="font-headline-sm mt-4 text-lg text-white">
                  {historyErrorMessage ? "History unavailable" : "No inspections yet"}
                </p>
                <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-[#7ab7af]">
                  {historyErrorMessage ??
                    "Run your first analysis above — every specimen, verdict and FMEA score is saved here with its image, ready to revisit any time."}
                </p>
                {!historyErrorMessage && (
                  <button
                    onClick={() => scrollToId("analysis")}
                    className="mt-6 inline-flex cursor-pointer items-center gap-2 rounded-xl bg-[#5ac8b4] px-5 py-2.5 font-label-caps text-[11px] font-bold uppercase text-[#00211d] shadow-[0_12px_34px_-14px_rgba(90,200,180,0.9)] transition-all duration-300 hover:bg-[#78dcc8]"
                  >
                    <span className="material-symbols-outlined text-[16px]">upload_file</span>
                    Start an inspection
                  </button>
                )}
              </div>
            )}
          </div>
        </section>
      </main>

      <Footer engine={engine} />

      {selectedRecord && (
        <InspectionDetail record={selectedRecord} onClose={() => setSelectedRecord(null)} />
      )}
    </div>
  );
}
