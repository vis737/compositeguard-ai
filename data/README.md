# Data Directory

## Folder Structure

```
data/
├── train/                  # Training images (organized by class)
│   ├── class_0/            # e.g., "crack-free"
│   ├── class_1/            # e.g., "crack"
│   └── ...
├── val/                    # Validation images
│   ├── class_0/
│   └── class_1/
└── test/                   # Test images (held-out, never seen during training)
    ├── class_0/
    └── class_1/
```

## Recommended Dataset Format

Each class directory should contain images of that class. Supported formats:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

## License & Attribution

**Important:** Only use images you have permission to use. Public domain or CC-0 datasets are recommended.

### Suggested Public Datasets (research only)

1. **Crack datasets for concrete/asphalt** - Search on Kaggle or Google Dataset Search
2. **Material surface defect datasets** - Verify licenses before use

**Do not** use datasets without confirming usage rights.

## Creating Your Own Dataset

1. Collect images of composite material samples with various defect types
2. Label each image with its defect class
3. Organize into the folder structure above
4. Aim for at least 50-100 images per class for reasonable training
5. Include both "good" (no defect) and "defective" samples

## Demonstration Mode

If you don't have a dataset, the system includes a **demonstration mode** that:
- Uses a pretrained model trained on synthetic composite defect data
- Marks predictions as "demonstration - not trained on your data"
- Allows you to upload any image and see model output

To use demonstration mode, simply upload an image - the system will fall back to the demo model if no custom weights are found.