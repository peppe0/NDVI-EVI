#!/bin/bash
#SBATCH --job-name=cpu_example_run
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpu_group5@hpc-legionlogin.polito.it
#SBATCH --partition=cpu_sapphire
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=log_cpu_with_corrupted_images.log
#SBATCH --mem-per-cpu=8000M
#SBATCH --cpus-per-task=1


# ************************************************
# Submit job:   sbatch launcher_cpu.sbatch
# Check status: squeue -u $USER
# Cancel job:   scancel <job_id>
# ************************************************

module load gcc/12.4.0
module load hpcx

# Activate virtual environment
source /home/gpu_group5/peppe/project/venv/bin/activate

echo "=== Python: $(which python) ==="

echo "=== Starting Image Processing ==="
cd /home/gpu_group5/peppe/project

python cpu_example.py

echo "=== Processing complete ==="
