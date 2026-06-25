#!/bin/bash
#SBATCH --job-name=proj3_red
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_h200
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=nvJ2000_reduction.log
#SBATCH --error=nvJ2000_reduction.err

# Host memory limits can be lower now due to parallel reduction on GPU, 
# but we will keep 8GB to be completely safe.
#SBATCH --mem-per-cpu=8000M
#SBATCH --gres=gpu:1

# ************************************************
# command to launch the job in the scheduler:
# sbatch launcher_nvJ2000_reduction.sbatch
#
# Command to check the job status:
# squeue -u gpu_group5
# ************************************************

module load gcc/12.4.0
module load nvhpc-hpcx-2.20-cuda12/25.1
# Optional: load system Conda module if available on this cluster.
module load miniconda3/3.13.25 2>/dev/null || true

# Ensure we run from the submission directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || true

# Initialize conda
if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
elif [ -f "/share/apps/miniconda3/bin/conda" ]; then
	eval "$(/share/apps/miniconda3/bin/conda shell.bash hook)"
else
	echo "ERROR: conda not available in PATH"; exit 1
fi
conda activate nvjpeg_modern || { echo "ERROR: failed to activate nvjpeg_modern"; exit 1; }

grep -n "nvjpeg2kDecodeStateCreate" ${CONDA_PREFIX}/include/*.h || echo "nvjpeg2kDecodeStateCreate not found in Conda headers"

echo "Loaded NVIDIA SDK !!!"
nvidia-smi
nvcc --version

echo "Compiling main_nvJ2000_reduction.cu!"
if [ ! -f main_nvJ2000_reduction.cu ]; then
	echo "ERROR: main_nvJ2000_reduction.cu not found in $(pwd)"; exit 1
fi

echo "Using nvJPEG2000 from Conda environment at ${CONDA_PREFIX}"
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     main_nvJ2000_reduction.cu -o main_nvJ2000_reduction \
     -I${CONDA_PREFIX}/include \
     -L${CONDA_PREFIX}/lib \
     -lnvjpeg2k -lz \
     -Xlinker -rpath,${CONDA_PREFIX}/lib \
     || { echo "Compilation failed"; exit 1; }

PROFILE_MODE=${PROFILE_MODE:-none}

# Nsight tools must resolve the real NVIDIA driver libcuda, not a conda stub.
if [ "$PROFILE_MODE" = "nsys" ] || [ "$PROFILE_MODE" = "ncu" ]; then
	if [ -n "${CONDA_PREFIX:-}" ]; then
		LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "${CONDA_PREFIX}/lib/stubs" | grep -v "${CONDA_PREFIX}/lib" | paste -sd: -)
		export LD_LIBRARY_PATH
	fi

	CUDA_DRIVER_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda.so.1/{print $NF; exit}')
	if [ -n "$CUDA_DRIVER_LIB" ]; then
		CUDA_DRIVER_DIR=$(dirname "$CUDA_DRIVER_LIB")
		export LD_LIBRARY_PATH="${CUDA_DRIVER_DIR}:${LD_LIBRARY_PATH}"
	fi
fi

if [ "$PROFILE_MODE" = "nsys" ]; then
	echo "Running with Nsight Systems profiling..."
	if ! command -v nsys >/dev/null 2>&1; then
		echo "ERROR: nsys not found in current environment."
		exit 1
	fi
	nsys profile --stats=true --force-overwrite true -o profile_main_nvJ2000_reduction ./main_nvJ2000_reduction 3
elif [ "$PROFILE_MODE" = "ncu" ]; then
	echo "Running with Nsight Compute profiling..."
	if ! command -v ncu >/dev/null 2>&1; then
		echo "ERROR: ncu not found in current environment."
		exit 1
	fi
	ncu --set full --target-processes all --force-overwrite true -o profile_kernels_nvJ2000_reduction ./main_nvJ2000_reduction 3
else
	srun --unbuffered ./main_nvJ2000_reduction 3

fi

echo "Done!"