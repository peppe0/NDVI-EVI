# Optimized Comparative Analysis of NDVI and EVI Indices

GPU-accelerated pipeline (CUDA) for computing and comparing the **NDVI** and **EVI**
vegetation indices on high-resolution **Sentinel-2** imagery, with on-device quality
control and a sequential CPU baseline for comparison.

> Master's Degree in Computer Engineering — GPU Programming project
> Authors: Daniele Cecconata, Giuseppe Abbatiello

---

## Overview

Each Sentinel-2 scene is a 10,980 × 10,980 px tile (~120.56 M pixels per band).
Processing these large, JPEG2000-compressed bands sequentially on a CPU is a major
bottleneck. This project parallelizes the whole per-pixel workload on the GPU:

- **NDVI** = (NIR − Red) / (NIR + Red)
- **EVI**  = G · (NIR − Red) / (NIR + C₁·Red − C₂·Blue + L), with G=2.5, C₁=6.0, C₂=7.5, L=1.0
- Bands used: **B4** (Red), **B8** (NIR), **B2** (Blue), all at 10 m resolution.

On top of the indices the pipeline computes, entirely on-device:
- 3×3 local statistics (mean, variance, Sobel-like gradient),
- a per-pixel **quality control** (black pixels, sensor saturation, NaN/Inf,
  statistical outliers, flat regions, stripe noise) with a weighted quality score,
- a four-class **vegetation health classification** (DEAD / DISEASED / HEALTHY / UNCERTAIN),
- global statistics via a two-level **parallel reduction** (warp-level `__shfl_down_sync`
  + shared-memory block reduction + shared-memory histograms).

JPEG2000 bands are decoded directly on the GPU with **nvJPEG2000**.

---

## Repository layout

| File | Description |
|------|-------------|
| `cpu_example.py` | Sequential CPU baseline (NumPy/SciPy): NDVI, EVI, stats, quality flags, health map. |
| `corrupter_v2.py` | Perturbation/robustness layer: injects atmospheric bias+scaling, Gaussian noise, salt-and-pepper into JP2 bands. |
| `main_nvJ2000_fixed.cu` | GPU pipeline — synchronous baseline (blocking D2H copies). |
| `main_nvJ2000_stream.cu` | GPU pipeline — CUDA streams (overlap decode/compute/async copies). |
| `main_nvJ2000_reduction.cu` | GPU pipeline — on-device parallel reduction (statistics computed on GPU, minimal D2H). |
| `launcher_cpu.sbatch` | SLURM launcher for the CPU baseline. |
| `launcher_nvJ2000_{fixed,stream,reduction}.sbatch` | SLURM launchers for the three GPU variants (build + run + optional profiling). |
| `stb_image.h` | Single-header image library used for PNG output. |
| `architetture.txt` | GPU compute-capability notes (sm_70/80/86/90). |
| `test_nvJ2000/` | Minimal nvJPEG2000 decode test. |

The three GPU variants are successive optimizations of the same algorithm:
`fixed` → `stream` → `reduction`.

---

## Requirements

**GPU side**
- NVIDIA GPU (tested on Tesla V100 `sm_70`, A40 `sm_86`, H200 `sm_90`)
- CUDA Toolkit 12.6 / 13.0 with the **nvJPEG2000** library and `zlib`
- A C++/CUDA toolchain (`nvcc`, `gcc`)

**CPU baseline**
- Python 3 with `numpy`, `scipy`, `matplotlib`, `rasterio`

**Data**
- Sentinel-2 scenes (bands B02/B04/B08, 10 m) from the
  [Copernicus Data Space Ecosystem](https://browser.dataspace.copernicus.eu/),
  arranged as `images/<YYYY-MM-DD>/<YYYYMMDD>_B0{2,4,8}_10m.jp2`.

The `images/`, `cpu_results/`, `gpu_results/`, profiling files and SLURM logs are
git-ignored (see `.gitignore`).

---

## Build & run

### GPU pipeline (example: `reduction`)

Compile for all target architectures:

```bash
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     main_nvJ2000_reduction.cu -o main_nvJ2000_reduction \
     -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
     -lnvjpeg2k -lz -Xlinker -rpath,$CONDA_PREFIX/lib
```

Run (the integer argument is the temporal sliding-window size):

```bash
./main_nvJ2000_reduction 3
```

Output PNGs (index colormaps and health masks) are written to `gpu_results/`.

### On an HPC cluster (SLURM)

```bash
sbatch launcher_nvJ2000_reduction.sbatch          # plain run
sbatch --export=ALL,PROFILE_MODE=nsys launcher_nvJ2000_reduction.sbatch   # Nsight Systems
sbatch --export=ALL,PROFILE_MODE=ncu  launcher_nvJ2000_reduction.sbatch   # Nsight Compute
```

The same pattern applies to `fixed` and `stream`. Check status with
`squeue -u <user>`; stdout/stderr go to `nvJ2000_*.log` / `nvJ2000_*.err`.

### CPU baseline

```bash
python cpu_example.py          # or: sbatch launcher_cpu.sbatch
```

Results are written to `cpu_results/`.

### Robustness / corruption layer

```bash
python corrupter_v2.py --input  images/<date>/<band>.jp2 \
                       --output images/<date>/<band>_corrupted.jp2 \
                       --atm_scale 0.7 --atm_bias 1000 \
                       --gaussian_std 300 --sp_ratio 0.05
```

---

## Results (summary)

On the H200, the per-scene compute is accelerated by ~3 orders of magnitude over the
CPU baseline, but the **end-to-end** runtime is dominated by JPEG2000 decoding and disk
I/O — a textbook case of **Amdahl's Law**. The three GPU variants progressively remove
the Device-to-Host transfer bottleneck:

- `fixed` → `stream`: overlaps decode/compute/copies via CUDA streams.
- `stream` → `reduction`: computes statistics on-device, replacing multi-megabyte D2H
  copies of full pixel grids with a few aggregated scalars.

See the project report for the full breakdown, profiling tables (NVIDIA Nsight Systems)
and the multi-GPU (V100 / A40 / H200) comparison.

---

## Profiling

The GPU launchers can emit Nsight traces (`profile_main_nvJ2000_*.nsys-rep`). They can
be inspected with the Nsight Systems GUI, with `nsys stats`, or by exporting to SQLite
and querying directly, e.g.:

```sql
-- per-kernel launches (≈ scenes processed: ndvi_kernel runs once per scene)
SELECT s.value AS kernel, COUNT(*) AS launches
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
GROUP BY s.value ORDER BY launches DESC;
```
