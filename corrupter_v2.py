import argparse
import numpy as np
import rasterio


def corrupt_jp2_pixels(
    input_path,
    output_path,
    gaussian_std=5.0,
    atm_bias=0.0,
    atm_scale=1.0,
    sp_ratio=0.01,
    seed=42
):
    rng = np.random.default_rng(seed)

    with rasterio.open(input_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    bands, h, w = data.shape

    # -----------------------------
    # 1. Atmospheric model
    # -----------------------------
    # Simula haze + offset radiometrico
    # (tipico effetto atmosfera: shift + scaling leggero)
    data = data * atm_scale + atm_bias

    # -----------------------------
    # 2. Gaussian noise
    # -----------------------------
    noise = rng.normal(0, gaussian_std, size=data.shape)
    data = data + noise

    # -----------------------------
    # 3. Salt & pepper (dead pixels)
    # -----------------------------
    total_pixels = h * w
    n_sp = int(sp_ratio * total_pixels)

    if nodata is None:
        nodata_value = 0
    else:
        nodata_value = nodata

    # scegli pixel casuali
    sp_idx = rng.choice(total_pixels, size=n_sp, replace=False)

    flat = data.reshape(bands, -1)

    half = n_sp // 2

    # salt (hot pixels)
    flat[:, sp_idx[:half]] = np.nanmax(flat)

    # pepper (dead / NaN-like)
    flat[:, sp_idx[half:]] = np.nanmin(flat)

    data = flat.reshape(bands, h, w)

    # -----------------------------
    # 4. Handle NaN / nodata
    # -----------------------------
    if nodata is not None:
        data = np.nan_to_num(data, nan=nodata)

    # cast back
    if np.issubdtype(src.dtypes[0], np.integer):
        data = np.clip(data, np.iinfo(src.dtypes[0]).min, np.iinfo(src.dtypes[0]).max)
        data = data.astype(src.dtypes[0])

    profile.update(driver="JP2OpenJPEG")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--gaussian_std", type=float, default=5.0)
    p.add_argument("--atm_bias", type=float, default=0.0)
    p.add_argument("--atm_scale", type=float, default=1.0)

    p.add_argument("--sp_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    corrupt_jp2_pixels(
        args.input,
        args.output,
        args.gaussian_std,
        args.atm_bias,
        args.atm_scale,
        args.sp_ratio,
        args.seed
    )