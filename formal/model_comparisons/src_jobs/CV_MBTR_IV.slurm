#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons/MBTR/out_iv/MBTRIV_CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons/MBTR/out_iv/MBTRIV_CVNUM.e
#SBATCH -J MBTRIV_CVNUM

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/run_rep.py \
--num_cv 1 \
--rep MBTR \
--target IV \
--results_save_dir $base/MBTR/iv \
--start_fold CVNUM \
