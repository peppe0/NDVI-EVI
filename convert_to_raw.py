#!/usr/bin/env python3
"""
Convert JP2/TIFF images to raw binary format that CUDA can read directly.
"""
import argparse
import numpy as np
import os

try:
    import rasterio
except ImportError:
    rasterio = None

from PIL import Image

def load_image_first_band(input_path):
    """Load first band from JP2/TIFF as uint16 2D array."""
    ext = os.path.splitext(input_path)[1].lower()

    # Rasterio handles JP2/TIFF robustly (especially Sentinel-2 JP2).
    if rasterio is not None and ext in {".jp2", ".tif", ".tiff"}:
        with rasterio.open(input_path) as src:
            arr = src.read(1)
    else:
        # Pillow fallback for environments without rasterio.
        img = Image.open(input_path)
        arr = np.array(img)

        if arr.ndim == 3:
            # If multi-channel image, keep first channel only.
            arr = arr[:, :, 0]

    return arr.astype(np.uint16, copy=False)


def image_to_raw(input_path, output_path):
    """Convert JP2/TIFF image to raw 16-bit unsigned binary."""
    print(f"Converting {input_path}...")
    arr = load_image_first_band(input_path)
    height, width = arr.shape

    print(f"  Dimensions: {width} x {height}")
    print(f"  Data type: {arr.dtype}, min={arr.min()}, max={arr.max()}")

    # Save as raw binary (row-major order, C-style)
    arr.ravel().tofile(output_path)

    # Save dimensions to text file
    meta_path = output_path.replace(".raw", "_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"{width} {height}\n")

    print(f"  Saved {output_path} ({os.path.getsize(output_path)} bytes)")
    print(f"  Saved {meta_path}")
    return width, height


def default_output_path(input_path):
    base = os.path.splitext(input_path)[0]
    return f"{base}.raw"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JP2/TIFF images to RAW uint16.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input image paths (.jp2/.tif/.tiff). If omitted, uses default Sentinel-2 files.",
    )
    parser.add_argument(
        "--outputs",
        nargs="*",
        help="Optional output .raw paths (must match number of inputs).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Converting JP2/TIFF images to raw binary format...\n")

    inputs = args.inputs
    if not inputs:
        inputs = [
            "images/B04_10m_2.jp2",
            "images/B08_10m_2.jp2",
            "images/B02_10m_2.jp2",
        ]

    if args.outputs and len(args.outputs) != len(inputs):
        raise ValueError("--outputs count must match number of input files.")

    outputs = args.outputs if args.outputs else [default_output_path(p) for p in inputs]

    dims = []
    for in_path, out_path in zip(inputs, outputs):
        w, h = image_to_raw(in_path, out_path)
        dims.append((w, h, in_path))

    print("\n" + "=" * 60)
    first_w, first_h, _ = dims[0]
    if all((w == first_w and h == first_h) for w, h, _ in dims):
        print("All images have matching dimensions.")
        print(f"  Size: {first_w} x {first_h} = {first_w * first_h} pixels")
    else:
        print("Warning: Images have different dimensions:")
        for w, h, name in dims:
            print(f"  {name}: {w}x{h}")

    print("\nConversion complete! Update main.cu to read .raw files.")
