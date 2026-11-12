"""Validate _analyze_image against synthetic + real samples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image

from serve import _analyze_image, _fmea

IMAGES = [
    ("crack_sample.png", "crack"),
    ("scratch_sample.png", "scratch"),
    ("discoloration_sample.png", "discoloration"),
    ("clean_sample.png", "no_defect"),
    ("real_carbon_crack.png", "crack"),
]

here = Path(__file__).resolve().parents[1] / "data" / "test_images"
failures = []
for name, expected in IMAGES:
    p = here / name
    if not p.exists():
        print(f"SKIP {name} (missing)")
        continue
    probs, info = _analyze_image(Image.open(p))
    top = max(probs, key=probs.get)
    ok = top == expected
    if not ok:
        failures.append((name, expected, top))
    print(f"{'PASS' if ok else 'FAIL'} {name}: got={top} expected={expected} probs={probs} info={info}")
    fmea = _fmea(top, probs[top])
    print(f"       fmea: {fmea}")

if failures:
    print("\nFAILURES:", failures)
    sys.exit(1)
print("\nall analyzer checks passed")
