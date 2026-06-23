#!/bin/bash
#SBATCH --job-name=proj3_a40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=nvJ2000_a40.log
#SBATCH --error=nvJ2000_a40.err
#SBATCH --mem-per-cpu=32000M
#SBATCH --gres=gpu:1

module load gcc/12.4.0
module load nvhpc-hpcx-2.20-cuda12/25.1
module load miniconda3/3.13.25 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || true

if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
elif [ -f "/share/apps/miniconda3/bin/conda" ]; then
	eval "$(/share/apps/miniconda3/bin/conda shell.bash hook)"
else
	echo "ERROR: conda not available in PATH"; exit 1
fi
conda activate nvjpeg_modern || { echo "ERROR: failed to activate nvjpeg_modern"; exit 1; }

echo "Loaded NVIDIA SDK !!!"
nvidia-smi
nvcc --version

echo "Compiling main_nvJ2000_fixed.cu for sm_86!"
nvcc -arch=sm_86 main_nvJ2000_fixed.cu -o main_nvJ2000_a40 \
    -I${CONDA_PREFIX}/include \
    -L${CONDA_PREFIX}/lib \
    -lnvjpeg2k -lz \
    -Xlinker -rpath,${CONDA_PREFIX}/lib \
    || { echo "Compilation failed"; exit 1; }

echo "Running main_nvJ2000_a40..."
srun --unbuffered ./main_nvJ2000_a40 3
echo "Done!"
