// Deterministic defects for offline/demo when the /api server is unreachable.
const DEMO_DEFECTS = {
  "persist-crack-test.png": {
    predicted_class: "crack",
    label: "Crack",
    confidence: 0.642,
    fmea: { severity: 9, occurrence: 7, detection: 8, risk_level: "High" },
    probabilities: { no_defect: 0.23, crack: 0.642, scratch: 0.11, discoloration: 0.018 },
  },
  "different-clean.png": {
    predicted_class: "no_defect",
    label: "No Defect",
    confidence: 1.0,
    fmea: { severity: 1, occurrence: 8, detection: 2, risk_level: "Low" },
    probabilities: { no_defect: 1.0, crack: 0.0, scratch: 0.0, discoloration: 0.0 },
  },
};

export { DEMO_DEFECTS };
