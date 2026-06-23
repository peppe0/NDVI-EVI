import time
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from scipy import ndimage

try:
    import rasterio
except ImportError:
    rasterio = None

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()
        elapsed = end - start

        print(f"{func.__name__} -> {elapsed:.6f} s ({elapsed*1000:.3f} ms)")

        return result

    return wrapper

@timer
def calculate_ndvi(red_band, nir_band):
    """
    Calculates Normalized Difference Vegetation Index.
    Formula: (NIR - Red) / (NIR + Red)
    """
    # Use float to avoid integer division issues
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    
    # Avoid division by zero
    denominator = nir + red
    denominator[denominator == 0] = 0.0001 
    
    ndvi = (nir - red) / denominator
    return ndvi

@timer
def calculate_evi(red_band, nir_band, blue_band):
    """
    Calculates Enhanced Vegetation Index.
    Formula: G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L))
    Standard Coefficients: G=2.5, C1=6, C2=7.5, L=1
    """
    # Coefficients (Standard MODIS/Landsat values)
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    blue = blue_band.astype(float)
    
    numerator = nir - red
    denominator = nir + (C1 * red) - (C2 * blue) + L
    
    # Avoid division by zero
    denominator[denominator == 0] = 0.0001
    
    evi = G * (numerator / denominator)
    
    # EVI can sometimes go out of bounds -1 to 1 due to noise, clamp it
    evi = np.clip(evi, -1.0, 1.0)
    
    return evi

@timer
def generate_health_map(index_map, threshold=0.66):
    """
    Simple thresholding to create a mask.
    If Index > 0.66 -> Healthy (1)
    Else -> Non-Healthy (0)
    """
    # Create an empty mask
    health_mask = np.zeros_like(index_map)
    
    # [cite_start]Apply threshold [cite: 11-12]
    health_mask[index_map >= threshold] = 1 
    
    return health_mask

@timer
def load_band(path):
    """Load first band from JP2/TIFF using rasterio."""
    if rasterio is None:
        raise RuntimeError(
            "rasterio is not installed in the active environment. "
            "Activate your virtualenv before running with image inputs."
        )

    with rasterio.open(path) as src:
        return src.read(1)


@timer
def report_band_quality(band, band_name, black_ratio_threshold=0.95):
    """Print basic quality diagnostics for a single band."""
    total_pixels = band.size
    if total_pixels == 0:
        raise ValueError(f"{band_name}: band is empty.")

    band_float = band.astype(np.float64)
    finite_mask = np.isfinite(band_float)
    finite_count = np.count_nonzero(finite_mask)
    invalid_count = total_pixels - finite_count
    invalid_ratio = invalid_count / total_pixels

    # Treat very small values as black to account for nodata encoded as 0/1.
    black_pixels = np.count_nonzero(band_float <= 1)
    black_ratio = black_pixels / total_pixels

    if finite_count == 0:
        raise ValueError(f"{band_name}: all pixels are invalid (NaN/Inf).")

    finite_values = band_float[finite_mask]
    band_min = float(np.min(finite_values))
    band_max = float(np.max(finite_values))
    band_std = float(np.std(finite_values))

    print(
        f"{band_name} quality | min={band_min:.2f}, max={band_max:.2f}, "
        f"std={band_std:.2f}, invalid={invalid_ratio:.2%}, black={black_ratio:.2%}"
    )

    if invalid_count > 0:
        print(f"WARNING: {band_name} contains {invalid_count} invalid pixels (NaN/Inf).")
    if black_ratio >= black_ratio_threshold:
        print(
            f"WARNING: {band_name} is mostly black ({black_ratio:.2%} <= 1 value pixels). "
            "Possible nodata/black band."
        )
    if band_max == band_min or band_std < 1e-9:
        print(f"WARNING: {band_name} appears constant. Possible corrupted or empty information band.")


# Define scenes with fixed paths
SCENES = [
    # Uncomment or add more scenes below

    {
        "red": "images/2025-04-30/20250430_B04_10m.jp2",
        "nir": "images/2025-04-30/20250430_B08_10m.jp2",
        "blue": "images/2025-04-30/20250430_B02_10m.jp2",
        "name": "2025-04-30"
    },

    {
        "red": "images/corrupted_2025-04-30/20250430_B04_10m_corrupted_all.jp2",
        "nir": "images/2025-04-30/20250430_B08_10m_corrupted_all.jp2",
        "blue": "images/2025-04-30/20250430_B02_10m_corrupted_all.jp2",
        "name": "corrupted_2025-04-30"
    },

    {
        "red": "images/2025-05-10/20250510_B04_10m.jp2",
        "nir": "images/2025-05-10/20250510_B08_10m.jp2",
        "blue": "images/2025-05-10/20250510_B02_10m.jp2",
        "name": "2025-05-10"
    },
    {
        "red": "images/2025-05-20/20250520_B04_10m.jp2",
        "nir": "images/2025-05-20/20250520_B08_10m.jp2",
        "blue": "images/2025-05-20/20250520_B02_10m.jp2",
        "name": "2025-05-20"
    },
    {
        "red": "images/2025-05-30/20250530_B04_10m.jp2",
        "nir": "images/2025-05-30/20250530_B08_10m.jp2",
        "blue": "images/2025-05-30/20250530_B02_10m.jp2",
        "name": "2025-05-30"
    },

    {
        "red": "images/corrupted_2025-05-30/20250530_B04_10m_corrupted_all.jp2",
        "nir": "images/corrupted_2025-05-30/20250530_B08_10m_corrupted_all.jp2",
        "blue": "images/corrupted_2025-05-30/20250530_B02_10m_corrupted_all.jp2",
        "name": "corrupted_2025-05-30"
    },

    {
        "red": "images/2025-06-09/20250609_B04_10m.jp2",
        "nir": "images/2025-06-09/20250609_B08_10m.jp2",
        "blue": "images/2025-06-09/20250609_B02_10m.jp2",
        "name": "2025-06-09"
    },

    {
        "red": "images/corrupted_2025-06-09/20250609_B04_10m_corrupted_all.jp2",
        "nir": "images/corrupted_2025-06-09/20250609_B08_10m_corrupted_all.jp2",
        "blue": "images/corrupted_2025-06-09/20250609_B02_10m_corrupted_all.jp2",
        "name": "corrupted_2025-06-09"
    },

    {
        "red": "images/2025-06-19/20250619_B04_10m.jp2",
        "nir": "images/2025-06-19/20250619_B08_10m.jp2",
        "blue": "images/2025-06-19/20250619_B02_10m.jp2",
        "name": "2025-06-19"
    },

    {
        "red": "images/corrupted_2025-06-19/20250619_B04_10m_corrupted_all.jp2",
        "nir": "images/corrupted_2025-06-19/20250619_B08_10m_corrupted_all.jp2",
        "blue": "images/corrupted_2025-06-19/20250619_B02_10m_corrupted_all.jp2",
        "name": "corrupted_2025-06-19"
    },

    {
        "red": "images/2025-07-09/20250709_B04_10m.jp2",
        "nir": "images/2025-07-09/20250709_B08_10m.jp2",
        "blue": "images/2025-07-09/20250709_B02_10m.jp2",
        "name": "2025-07-09"
    },
    {
        "red": "images/2025-07-19/20250719_B04_10m.jp2",
        "nir": "images/2025-07-19/20250719_B08_10m.jp2",
        "blue": "images/2025-07-19/20250719_B02_10m.jp2",
        "name": "2025-07-19"
    },
    {
        "red": "images/2025-07-29/20250729_B04_10m.jp2",
        "nir": "images/2025-07-29/20250729_B08_10m.jp2",
        "blue": "images/2025-07-29/20250729_B02_10m.jp2",
        "name": "2025-07-29"
    },

    {
        "red": "images/2025-08-08/20250808_B04_10m.jp2",
        "nir": "images/2025-08-08/20250808_B08_10m.jp2",
        "blue": "images/2025-08-08/20250808_B02_10m.jp2",
        "name": "2025-08-08"
    },
    {
        "red": "images/2025-08-18/20250818_B04_10m.jp2",
        "nir": "images/2025-08-18/20250818_B08_10m.jp2",
        "blue": "images/2025-08-18/20250818_B02_10m.jp2",
        "name": "2025-08-18"
    },
    {
        "red": "images/2025-08-28/20250828_B04_10m.jp2",
        "nir": "images/2025-08-28/20250828_B08_10m.jp2",
        "blue": "images/2025-08-28/20250828_B02_10m.jp2",
        "name": "2025-08-28"
    },
    {
        "red": "images/2025-09-07/20250907_B04_10m.jp2",
        "nir": "images/2025-09-07/20250907_B08_10m.jp2",
        "blue": "images/2025-09-07/20250907_B02_10m.jp2",
        "name": "2025-09-07"
    },

    {
        "red": "images/2025-09-17/20250917_B04_10m.jp2",
        "nir": "images/2025-09-17/20250917_B08_10m.jp2",
        "blue": "images/2025-09-17/20250917_B02_10m.jp2",
        "name": "2025-09-17"
    },
    {
        "red": "images/2025-09-27/20250927_B04_10m.jp2",
        "nir": "images/2025-09-27/20250927_B08_10m.jp2",
        "blue": "images/2025-09-27/20250927_B02_10m.jp2",
        "name": "2025-09-27"
    },
    {
        "red": "images/2025-10-07/20251007_B04_10m.jp2",
        "nir": "images/2025-10-07/20251007_B08_10m.jp2",
        "blue": "images/2025-10-07/20251007_B02_10m.jp2",
        "name": "2025-10-07"
    },
    {
        "red": "images/2025-10-17/20251017_B04_10m.jp2",
        "nir": "images/2025-10-17/20251017_B08_10m.jp2",
        "blue": "images/2025-10-17/20251017_B02_10m.jp2",
        "name": "2025-10-17"
    },

    {
        "red": "images/2025-10-27/20251027_B04_10m.jp2",
        "nir": "images/2025-10-27/20251027_B08_10m.jp2",
        "blue": "images/2025-10-27/20251027_B02_10m.jp2",
        "name": "2025-10-27"
    },
    {
        "red": "images/2025-11-06/20251106_B04_10m.jp2",
        "nir": "images/2025-11-06/20251106_B08_10m.jp2",
        "blue": "images/2025-11-06/20251106_B02_10m.jp2",
        "name": "2025-11-06"
    },
    {
        "red": "images/2025-11-16/20251116_B04_10m.jp2",
        "nir": "images/2025-11-16/20251116_B08_10m.jp2",
        "blue": "images/2025-11-16/20251116_B02_10m.jp2",
        "name": "2025-11-16"
    },
    {
        "red": "images/2025-11-26/20251126B04_10m.jp2",
        "nir": "images/2025-11-26/20251126_B08_10m.jp2",
        "blue": "images/2025-11-26/20251126_B02_10m.jp2",
        "name": "2025-11-26"
    },

    {
        "red": "images/2025-12-06/20251206_B04_10m.jp2",
        "nir": "images/2025-12-06/20251206_B08_10m.jp2",
        "blue": "images/2025-12-06/20251206_B02_10m.jp2",
        "name": "2025-12-06"
    },
    {
        "red": "images/2025-12-16/20251216_B04_10m.jp2",
        "nir": "images/2025-12-16/20251216_B08_10m.jp2",
        "blue": "images/2025-12-16/20251216_B02_10m.jp2",
        "name": "2025-12-16"
    },
    {
        "red": "images/2025-12-26/20251226_B04_10m.jp2",
        "nir": "images/2025-12-26/20251226_B08_10m.jp2",
        "blue": "images/2025-12-26/20251226_B02_10m.jp2",
        "name": "2025-12-26"
    },
    {
        "red": "images/2026-01-05/20260105_B04_10m.jp2",
        "nir": "images/2026-01-05/20260105_B08_10m.jp2",
        "blue": "images/2026-01-05/20260105_B02_10m.jp2",
        "name": "2026-01-05"
    },

    {
        "red": "images/2026-01-15/20260115_B04_10m.jp2",
        "nir": "images/2026-01-15/20260115_B08_10m.jp2",
        "blue": "images/2026-01-15/20260115_B02_10m.jp2",
        "name": "2026-01-15"
    },
    {
        "red": "images/2026-01-25/20260125_B04_10m.jp2",
        "nir": "images/2026-01-25/20260125_B08_10m.jp2",
        "blue": "images/2026-01-25/20260125_B02_10m.jp2",
        "name": "2026-01-25"
    },
    {
        "red": "images/2026-02-04/20260204_B04_10m.jp2",
        "nir": "images/2026-02-04/20260204_B08_10m.jp2",
        "blue": "images/2026-02-04/20260204_B02_10m.jp2",
        "name": "2026-02-04"
    },
    {
        "red": "images/2026-02-14/20260214_B04_10m.jp2",
        "nir": "images/2026-02-14/20260214_B08_10m.jp2",
        "blue": "images/2026-02-14/20260214_B02_10m.jp2",
        "name": "2026-02-14"
    },

    {
        "red": "images/2026-02-24/20260224_B04_10m.jp2",
        "nir": "images/2026-02-24/20260224_B08_10m.jp2",
        "blue": "images/2026-02-24/20260224_B02_10m.jp2",
        "name": "2026-02-24"
    },
    {
        "red": "images/2026-03-06/20260306_B04_10m.jp2",
        "nir": "images/2026-03-06/20260306_B08_10m.jp2",
        "blue": "images/2026-03-06/20260306_B02_10m.jp2",
        "name": "2026-03-06"
    },
    {
        "red": "images/2026-03-16/20260316_B04_10m.jp2",
        "nir": "images/2026-03-16/20260316_B08_10m.jp2",
        "blue": "images/2026-03-16/20260316_B02_10m.jp2",
        "name": "2026-03-16"
    },
    {
        "red": "images/2026-03-26/20260326_B04_10m.jp2",
        "nir": "images/2026-03-26/20260326_B08_10m.jp2",
        "blue": "images/2026-03-26/20260326_B02_10m.jp2",
        "name": "2026-03-26"
    },
        
    {
        "red": "images/2026-04-05/20260405_B04_10m.jp2",
        "nir": "images/2026-04-05/20260405_B08_10m.jp2",
        "blue": "images/2026-04-05/20260405_B02_10m.jp2",
        "name": "2026-04-05"
    },
    {
        "red": "images/2026-04-15/20260415_B04_10m.jp2",
        "nir":  "images/2026-04-15/20260415_B08_10m.jp2",
        "blue": "images/2026-04-15/20260415_B02_10m.jp2",
        "name":  "2026-04-15"
    },

    {
        "red": "images/2026-04-25/20260425_B04_10m.jp2",
        "nir": "images/2026-04-25/20260425_B08_10m.jp2",
        "blue": "images/2026-04-25/20260425_B02_10m.jp2",
        "name": "2026-04-25"
    },
    {
        "red": "images/2026-05-05/20260505_B04_10m.jp2",
        "nir": "images/2026-05-05/20260505_B08_10m.jp2",
        "blue": "images/2026-05-05/20260505_B02_10m.jp2",
        "name": "2026-05-05"
    },


]

@timer
def compute_statistics(input, flag):
    index = input.astype(float)

    # -------------------------
    # 1. Local mean (3x3)
    # -------------------------
    kernel_size = 3
    local_mean = ndimage.uniform_filter(index, size=kernel_size)

    avg_local_mean = np.mean(local_mean)

    # -------------------------
    # 2. Local variance (3x3)
    # -------------------------
    local_sq_mean = ndimage.uniform_filter(index**2, size=kernel_size)
    local_var = local_sq_mean - local_mean**2

    avg_local_var = np.mean(local_var)

    # -------------------------
    # 3. Sobel-like gradient magnitude
    # -------------------------
    gx = ndimage.sobel(index, axis=1)
    gy = ndimage.sobel(index, axis=0)
    grad = np.sqrt(gx**2 + gy**2)

    avg_grad = np.mean(grad)

    # -------------------------
    # 4. Quality flags
    # -------------------------
    flat_pixels = np.sum(np.abs(grad) < 1e-6)

    invalid_pixels = np.sum(~np.isfinite(index))
    black_pixels = np.sum(index <= 0)

    saturated_pixels = np.sum(index >= 0.9)  # puoi adattare soglia

    # outlier semplice (robusto ma veloce)
    mean = np.mean(index)
    std = np.std(index)
    outliers = np.sum((index < mean - 3*std) | (index > mean + 3*std))

    # stripe-like (euristica semplice: righe con varianza bassa)
    row_var = np.var(index, axis=1)
    stripe_like = np.sum(row_var < np.mean(row_var) * 0.1)

    total = index.size

    quality_score = 1.0 - (
        (invalid_pixels + outliers + flat_pixels) / total
    )
    quality_score = np.clip(quality_score, 0, 1)

    # -------------------------
    # 5. Health classification
    # -------------------------
    dead = np.sum(index < 0.2)
    diseased = np.sum((index >= 0.2) & (index < 0.5))
    healthy = np.sum(index >= 0.5)
    uncertain = invalid_pixels

    value = ""
    if flag == 1:
        value += "EVI"
    else:
        value += "NDVI"

    print(f"\n{value} Advanced Statistics:")
    print(f"Average {value} local mean (3x3): {avg_local_mean:.6f}")
    print(f"Average {value} local var  (3x3): {avg_local_var:.6f}")
    print(f"Average {value} Sobel-like grad: {avg_grad:.6f}")
    print(f"Average Quality Score: {quality_score:.6f}")

    print("Quality Flags Summary:")
    print(f"  Black pixels:     {black_pixels} ({black_pixels/total:.2%})")
    print(f"  Saturated pixels: {saturated_pixels} ({saturated_pixels/total:.2%})")
    print(f"  Outlier pixels:   {outliers} ({outliers/total:.2%})")
    print(f"  Flat pixels:      {flat_pixels} ({flat_pixels/total:.2%})")
    print(f"  Invalid pixels:   {invalid_pixels} ({invalid_pixels/total:.2%})")
    print(f"  Stripe-like:      {stripe_like} ({stripe_like/total:.2%})")

    print("Health Class Summary:")
    print(f"  Dead:      {dead} ({dead/total:.2%})")
    print(f"  Diseased:  {diseased} ({diseased/total:.2%})")
    print(f"  Healthy:   {healthy} ({healthy/total:.2%})")
    print(f"  Uncertain: {uncertain} ({uncertain/total:.2%})")


@timer
def compute_output(ndvi_result, evi_result, ndvi_health, evi_health, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax0 = axes[0, 0]
    im0 = ax0.imshow(ndvi_result, cmap='RdYlGn', vmin=-1, vmax=1)
    ax0.set_title("NDVI Index")
    plt.colorbar(im0, ax=ax0)

    ax1 = axes[0, 1]
    im1 = ax1.imshow(evi_result, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title("EVI Index")
    plt.colorbar(im1, ax=ax1)

    ax2 = axes[1, 0]
    ax2.imshow(ndvi_health, cmap='gray')
    ax2.set_title("NDVI Health Mask (>0.66)")

    ax3 = axes[1, 1]
    ax3.imshow(evi_health, cmap='gray')
    ax3.set_title("EVI Health Mask (>0.66)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


@timer
def process_scene(red_path, nir_path, blue_path, out_path):
    start_time = time.time()

    red_band = load_band(red_path)
    nir_band = load_band(nir_path)
    blue_band = load_band(blue_path)

    if red_band.shape != nir_band.shape or red_band.shape != blue_band.shape:
        raise ValueError("Input bands must have identical dimensions.")

    report_band_quality(red_band, "RED")
    report_band_quality(nir_band, "NIR")
    report_band_quality(blue_band, "BLUE")

    print("Calculating Indices...")

    ndvi_result = calculate_ndvi(red_band, nir_band)
    compute_statistics(ndvi_result, 0)

    evi_result = calculate_evi(red_band, nir_band, blue_band)
    compute_statistics(evi_result, 1)

    ndvi_health = generate_health_map(ndvi_result)
    evi_health = generate_health_map(evi_result)

    print(f"Average NDVI: {np.mean(ndvi_result):.4f}")
    print(f"Average EVI:  {np.mean(evi_result):.4f}")

    compute_output(ndvi_result, evi_result, ndvi_health, evi_health, out_path)

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.6f} seconds")

@timer
def main():
    if not SCENES:
        print("ERROR: No scenes defined in SCENES list.")
        return

    # Create output directory if it doesn't exist
    output_dir = "cpu_results"
    os.makedirs(output_dir, exist_ok=True)

    for index, scene in enumerate(SCENES, start=1):
        red_path = scene["red"]
        nir_path = scene["nir"]
        blue_path = scene["blue"]
        scene_name = scene.get("name", f"scene_{index}")
        
        out_path = os.path.join(output_dir, f"result_{scene_name}.png")
        
        print(f"\nProcessing scene {index} ({scene_name}): {red_path}, {nir_path}, {blue_path}")
        try:
            process_scene(red_path, nir_path, blue_path, out_path)
            print(f"✓ Scene {index} completed. Output saved to {out_path}\n")
        except Exception as e:
            print(f"✗ Scene {index} failed: {e}\n")


if __name__ == "__main__":
    main()
