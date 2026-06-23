#!/bin/bash
#SBATCH --job-name=proj1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=gpu_h200
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=main_gpu.log
#SBATCH --mem-per-cpu=4000M
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
	conda activate gdal_cpp_env || { echo "ERROR: failed to activate gdal_cpp_env"; exit 1; }
else
	echo "ERROR: conda not available in PATH"; exit 1
fi

echo "Loaded NVIDIA SDK !!!"
nvidia-smi

# check the current version of the compiler:
#nvc --version              #for C
#nvc++ --version            #for C++
#nvfortran --version        #for Fortran

nvcc --version

echo "Compiling!"

# Ensure source exists
if [ ! -f main.cu ]; then
	echo "ERROR: main.cu not found in $(pwd)"; exit 1
fi

# Use gdal-config from the active conda env if available to get correct flags
if [ -x "${CONDA_PREFIX}/bin/gdal-config" ]; then
	CFLAGS=$(${CONDA_PREFIX}/bin/gdal-config --cflags)
	LIBS=$(${CONDA_PREFIX}/bin/gdal-config --libs)
	echo "Using ${CONDA_PREFIX}/bin/gdal-config -> CFLAGS='$CFLAGS' LIBS='$LIBS'"
	nvcc main.cu -o main ${CFLAGS} ${LIBS} -lpcre2-8 -Xlinker -rpath,${CONDA_PREFIX}/lib || { echo "Compilation failed"; exit 1; }
else
	echo "gdal-config not found in ${CONDA_PREFIX}; using fallback include/lib flags"
	nvcc main.cu -o main -I${CONDA_PREFIX}/include -L${CONDA_PREFIX}/lib -lgdal -lpcre2-8 -Xlinker -rpath,${CONDA_PREFIX}/lib || { echo "Compilation failed"; exit 1; }
fi

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
	nsys profile --stats=true --force-overwrite true -o profile_main ./main
elif [ "$PROFILE_MODE" = "ncu" ]; then
	echo "Running with Nsight Compute profiling..."
	if ! command -v ncu >/dev/null 2>&1; then
		echo "ERROR: ncu not found in current environment."
		exit 1
	fi
	# Profile all launched kernels to avoid filter mismatch on demangled signatures.
	ncu --set full --target-processes all --force-overwrite true -o profile_kernels ./main
else
	./main
fi

echo "Done!"





# module avail  # check all the available packages in the system.













# nvc++ -cuda CUDA_add/add.cu -o CUDA_add/add.o


# Copia dei file da storage HOME a storage locale del nodo

#cp -r /home/myusername/data $TMPDIR/data

# myexecutable -in $TMPDIR/data -out $TMPDIR/results

# Copia dei risultati qualora generati nella cartella temporanea
#cp -r $TMPDIR/results /home/myusername/results

