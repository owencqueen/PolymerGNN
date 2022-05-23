#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/PI/out/PI_CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/PI/out/PI_CVNUM.e
#SBATCH -J PI_CVNUM

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/run_rep.py \
--num_cv 1 \
--rep PI \
--results_save_dir $base/PI/saved_scores \
--start_fold CVNUM \
--standard_scale
