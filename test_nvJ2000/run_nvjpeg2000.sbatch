#!/bin/bash
#SBATCH --job-name=nvjpeg2k_test
#SBATCH --output=nvjpeg2k.log
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -e

echo "=============================="
echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "=============================="

# 1. Clean the environment
module purge

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || true

# 2. Activate Conda FIRST
module load miniconda3/3.13.25 2>/dev/null || true

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "/share/apps/miniconda3/bin/conda" ]; then
    eval "$(/share/apps/miniconda3/bin/conda shell.bash hook)"
else
    echo "ERROR: conda not available in PATH"; exit 1
fi
conda activate nvjpeg_modern || { echo "ERROR: failed to activate nvjpeg_modern"; exit 1; }

# 3. Load Compilers SECOND (Overrides any conflicting Conda packages)
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1

echo "CUDA loaded from:"
which nvcc
nvcc --version
nvidia-smi

# 4. Compilation
echo "Using nvJPEG2000 from Conda environment at ${CONDA_PREFIX}"

rm -f test_nvjpeg2k
# Note: Ensure your file is named test_nvjpeg2000.cu (you used test_nvjpeg2k.cpp previously)
nvcc -arch=sm_90 test_nvjpeg2000.cu -o test_nvjpeg2k \
    -I${CONDA_PREFIX}/include \
    -L${CONDA_PREFIX}/lib \
    -lnvjpeg2k \
    -Xlinker -rpath,${CONDA_PREFIX}/lib \
    || { echo "Compilation failed"; exit 1; }

# 5. Execution
echo "Running test..."
./test_nvjpeg2k images/B02_10m.jp2

echo "=============================="
echo "Job finished on $(date)"
echo "=============================="