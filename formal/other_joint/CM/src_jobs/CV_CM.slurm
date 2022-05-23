#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/CM/out/CM_CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/CM/out/CM_CVNUM.e
#SBATCH -J CM_CVNUM

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/run_rep.py \
--num_cv 1 \
--rep CM \
--results_save_dir $base/CM/saved_scores \
--start_fold CVNUM \
--standard_scale
