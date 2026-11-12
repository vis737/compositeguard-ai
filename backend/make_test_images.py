"""Generate synthetic + real-crop test images for validating the analyzer."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "test_images"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)


def weave_background(w=480, h=360):
    """Simulate a carbon-fiber weave: alternating light/dark tow bands + noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    band = 26
    warp = ((xx // band) % 2).astype(float)
    weft = ((yy // band) % 2).astype(float)
    base = 0.18 + 0.16 * warp * (1 - weft) + 0.16 * weft * (1 - warp)
    # add tow texture
    base += 0.05 * np.sin(xx / 2.0) * np.sin(yy / 2.0)
    base += rng.normal(0, 0.012, base.shape)
    g = np.clip(base, 0.03, 0.9)
    img = np.stack([g, g, g * 1.02], axis=2)
    return Image.fromarray((img * 255).astype(np.uint8))


def make_crack():
    img = weave_background()
    d = ImageDraw.Draw(img)
    # jagged dark crack across the surface
    pts = [(20, 300), (90, 270), (150, 240), (210, 200), (280, 175), (350, 120), (420, 95), (470, 60)]
    d.line(pts, fill=(8, 8, 10), width=6)
    d.line([(150, 240), (170, 300)], fill=(10, 10, 12), width=3)  # branch
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(OUT / "crack_sample.png")


def make_scratch():
    img = weave_background()
    d = ImageDraw.Draw(img)
    # thin bright straight scratches
    d.line([(40, 60), (430, 130)], fill=(235, 235, 240), width=2)
    d.line([(80, 250), (400, 300)], fill=(220, 222, 225), width=1)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(OUT / "scratch_sample.png")


def make_discoloration():
    img = weave_background()
    arr = np.asarray(img).astype(float)
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    blob = ((xx - 160) ** 2 / 90**2 + (yy - 180) ** 2 / 70**2) < 1.0
    arr[blob, 0] = np.clip(arr[blob, 0] * 1.5 + 40, 0, 255)  # reddish stain
    arr[blob, 2] = arr[blob, 2] * 0.7
    Image.fromarray(arr.astype(np.uint8)).save(OUT / "discoloration_sample.png")


def make_clean():
    weave_background().save(OUT / "clean_sample.png")


def crop_real_sample():
    """Crop the carbon-fiber photo region from the user's screenshot."""
    src = Path(r"C:\Users\Vishnu\AppData\Local\Temp\freebuff-desktop-pastes\paste-1792758050526-9044.png")
    if not src.exists():
        print("real screenshot not found, skipping crop")
        return
    im = Image.open(src).convert("RGB")
    w, h = im.size
    # the photo occupies roughly the left 62% and vertical band per the screenshot layout
    box = (int(0.075 * w), int(0.26 * h), int(0.615 * w), int(0.81 * h))
    im.crop(box).save(OUT / "real_carbon_crack.png")
    print("cropped real sample:", box)


if __name__ == "__main__":
    make_crack()
    make_scratch()
    make_discoloration()
    make_clean()
    crop_real_sample()
    print("test images written to", OUT)
