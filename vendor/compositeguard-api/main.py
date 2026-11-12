
import base64
import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Configuration
MODEL_PATH = Path(__file__).parent / "compositeguard_v4.onnx"
IMAGE_SIZE = (256, 256)

CLASS_NAMES = {
    0: "Background / Clean",
    1: "Fiber Breakage",
    2: "Porosity / Cracks",
    3: "Foreign Inclusions",
    4: "Delamination",
}

# Load the trained ONNX model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Place compositeguard_v4.onnx beside main.py."
    )

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

app = FastAPI(
    title="CompositeGuard AI API",
    description="AI-assisted composite material defect segmentation",
    version="1.0.0",
)

# Allow a local frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "CompositeGuard AI API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_PATH.name,
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, WEBP, or BMP image.",
        )

    try:
        contents = await file.read()

        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 15 MB.",
            )

        original = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        )

    # Preserve the original dimensions for the returned mask.
    original_size = original.size

    # Model training preprocessing: RGB, [0, 1], ImageNet normalization.
    resized = original.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized = (image_array - mean) / std
    input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

    # Run ONNX inference.
    logits = session.run(
        [output_name],
        {input_name: input_tensor.astype(np.float32)},
    )[0]

    # Expected model output: [1, 5, 256, 256].
    if logits.ndim != 4 or logits.shape[1] != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected model output shape: {logits.shape}",
        )

    mask = np.argmax(logits[0], axis=0).astype(np.uint8)

    # Resize the categorical mask using nearest-neighbor interpolation.
    mask_image = Image.fromarray(mask, mode="L")
    mask_image = mask_image.resize(
        original_size, Image.Resampling.NEAREST
    )
    mask = np.asarray(mask_image, dtype=np.uint8)

    # Colors: background, fiber, porosity, inclusions, delamination.
    palette = np.array([
        [0, 0, 0],
        [255, 65, 65],
        [255, 190, 40],
        [50, 160, 255],
        [190, 80, 255],
    ], dtype=np.uint8)

    color_mask = palette[mask]

    # Blend the predicted mask over the original image.
    original_array = np.asarray(original, dtype=np.uint8)
    overlay = original_array.copy()
    defect_pixels = mask != 0
    overlay[defect_pixels] = (
        0.55 * original_array[defect_pixels]
        + 0.45 * color_mask[defect_pixels]
    ).astype(np.uint8)

    def encode_png(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    total_pixels = int(mask.size)
    defect_pixel_count = int(np.count_nonzero(defect_pixels))

    defect_summary = {}
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue

        count = int(np.count_nonzero(mask == class_id))
        defect_summary[class_name] = {
            "pixels": count,
            "percentage_of_image": round(
                100.0 * count / total_pixels, 4
            ),
        }

    detected = [
        name for name, info in defect_summary.items()
        if info["pixels"] > 0
    ]

    return {
        "filename": file.filename,
        "image_width": original_size[0],
        "image_height": original_size[1],
        "predicted_defects": detected,
        "defect_pixels": defect_pixel_count,
        "defect_percentage": round(
            100.0 * defect_pixel_count / total_pixels, 4
        ),
        "class_summary": defect_summary,
        "mask_png_base64": encode_png(mask),
        "overlay_png_base64": encode_png(overlay),
        "note": (
            "AI prediction from a model trained on synthetic data. "
            "Not validated for industrial inspection."
        ),
    }
