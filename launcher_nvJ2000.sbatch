#!/bin/bash
#SBATCH --job-name=tentativo_proj1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_a100
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=nvJ2000_2.log
#SBATCH --mem-per-cpu=32000M
#SBATCH --gres=gpu:1

# ************************************************
# command to launch the job in the scheduler:
# sbatch launcher.sbatch

# Command to check the job status:
# squeue -u jrodriguez
# ************************************************

# load cuda module
# module load nvidia/cudasdk/10.1

module load gcc/12.4.0
#module load hpcx
#module load nvhpc-hpcx-cuda12/25.1
module load nvhpc-hpcx-2.20-cuda12/25.1 
# Optional: load system Conda module if available on this cluster.
module load miniconda3/3.13.25 2>/dev/null || true

# Ensure we run from the submission directory (where you ran `sbatch`)
# SLURM sets SLURM_SUBMIT_DIR to the directory where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || true

# In non-interactive SLURM shells, initialize conda then activate the desired env
if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
elif [ -f "/share/apps/miniconda3/bin/conda" ]; then
	eval "$(/share/apps/miniconda3/bin/conda shell.bash hook)"
else
	echo "ERROR: conda not available in PATH"; exit 1
fi
conda activate nvjpeg_modern || { echo "ERROR: failed to activate nvjpeg_modern"; exit 1; }

echo "Loaded NVIDIA SDK !!!"

# 1. Check driver version vs library expectations
nvidia-smi | head -4
echo "CUDA runtime: $(nvcc --version | grep release)"
# 2. Check which libnvjpeg2k is actually being loaded
echo "libnvjpeg2k location:"
ls -la ${CONDA_PREFIX}/lib/libnvjpeg2k*
# 3. Check for libcuda.so stub conflicts
echo "libcuda.so resolution:"
ldconfig -p 2>/dev/null | grep libcuda
ls -la ${CONDA_PREFIX}/lib/stubs/libcuda* 2>/dev/null || echo "No stubs"
# 4. Check LD_LIBRARY_PATH for conflicts
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
# 5. Check the actual linked libraries at runtime
ldd ./main_nvJ2000_fixed 2>&1 | grep -E "nvjpeg|cuda"


echo "Compiling main_nvJ2000.cu!"

# Ensure source exists
if [ ! -f main_nvJ2000.cu ]; then
	echo "ERROR: main_nvJ2000.cu not found in $(pwd)"; exit 1
fi

echo "Using nvJPEG2000 from Conda environment at ${CONDA_PREFIX}"
nvcc -arch=sm_86 main_nvJ2000.cu -o main_nvJ2000 \
    -I${CONDA_PREFIX}/include \
    -L${CONDA_PREFIX}/lib \
    -lnvjpeg2k \
    -Xlinker -rpath,${CONDA_PREFIX}/lib \
    || { echo "Compilation failed"; exit 1; }

PROFILE_MODE=${PROFILE_MODE:-none}

# Nsight tools must resolve the real NVIDIA driver libcuda, not a conda stub.
if [ "$PROFILE_MODE" = "nsys" ] || [ "$PROFILE_MODE" = "ncu" ]; then
	if [ -n "${CONDA_PREFIX:-}" ]; then
		# Remove common conda stub locations that break profiler-driver connection.
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
	stdbuf -oL -eL \
	nsys profile --stats=true --force-overwrite true -o profile_main_nvJ2000 ./main_nvJ2000 3
elif [ "$PROFILE_MODE" = "ncu" ]; then
	echo "Running with Nsight Compute profiling..."
	if ! command -v ncu >/dev/null 2>&1; then
		echo "ERROR: ncu not found in current environment."
		exit 1
	fi
	# Profile all launched kernels to avoid filter mismatch on demangled signatures.
	stdbuf -oL -eL \
	ncu --set full --target-processes all --force-overwrite true -o profile_kernels_nvJ2000 ./main_nvJ2000 3
else
	stdbuf -oL -eL ./main_nvJ2000 3
fi

echo "Done!"





# module avail  # check all the available packages in the system.













# nvc++ -cuda CUDA_add/add.cu -o CUDA_add/add.o


# Copia dei file da storage HOME a storage locale del nodo

#cp -r /home/myusername/data $TMPDIR/data

# myexecutable -in $TMPDIR/data -out $TMPDIR/results

# Copia dei risultati qualora generati nella cartella temporanea
#cp -r $TMPDIR/results /home/myusername/results

