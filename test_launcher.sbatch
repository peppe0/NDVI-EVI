#!/bin/bash
#SBATCH --job-name=nvjpeg2k_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=nvjpeg2k_test.log
#SBATCH --error=nvjpeg2k_test.err
#SBATCH --mem-per-cpu=4000M
#SBATCH --gres=gpu:1

module load gcc/12.4.0
module load nvhpc-hpcx-2.20-cuda12/25.1
module load miniconda3/3.13.25 2>/dev/null || true

# Vai nella directory di submit
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1

# Init conda
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "/share/apps/miniconda3/bin/conda" ]; then
    eval "$(/share/apps/miniconda3/bin/conda shell.bash hook)"
else
    echo "ERROR: conda non disponibile"
    exit 1
fi

# Attiva environment
conda activate nvjpeg_modern || {
    echo "ERROR: impossibile attivare nvjpeg_modern"
    exit 1
}

echo "GPU INFO"
nvidia-smi

echo "NVCC VERSION"
nvcc --version

echo "CUDA PATH"
which nvcc
echo "CUDA_HOME=${CUDA_HOME}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo "Checking nvJPEG2000 headers"
ls ${CONDA_PREFIX}/include/nvjpeg2k.h || {
    echo "ERROR: nvjpeg2k.h non trovato"
    exit 1
}

echo "Conda packages"
conda list | grep -i nvjpeg

echo "nvJPEG libraries in Conda env"
find ${CONDA_PREFIX}/lib -name "*nvjpeg*"

echo "Checking CUDA strings inside libnvjpeg2k"
strings ${CONDA_PREFIX}/lib/libnvjpeg2k.so | grep CUDA || true

echo "Compilazione"

nvcc -arch=sm_86 test_nvjpeg2k.cpp -o test_nvjpeg2k \
    -I${CONDA_PREFIX}/include \
    -L${CONDA_PREFIX}/lib \
    -lnvjpeg2k \
    -Xlinker -rpath,${CONDA_PREFIX}/lib \
    || {
        echo "Compilation failed"
        exit 1
    }

echo "Dynamic linked libraries"
ldd ./test_nvjpeg2k

echo "Filtered nvjpeg libraries"
ldd ./test_nvjpeg2k | grep -i nvjpeg || true

echo "Filtered cuda libraries"
ldd ./test_nvjpeg2k | grep -i cuda || true

echo "Esecuzione"

srun --unbuffered ./test_nvjpeg2k

RET=$?

echo "Program exit code = ${RET}"
echo "Fine"