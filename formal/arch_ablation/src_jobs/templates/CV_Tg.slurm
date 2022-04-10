#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/arch_ablation/outputs/tg/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/arch_ablation/outputs/tg/fold=CVNUM.e
#SBATCH -J CV_Tg_CVNUM

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/arch_ablation"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/mono_graph.py \
--num_cv 1 \
--Tg \
--properties default \
--results_save_dir $base/saved_scores/tg \
--start_fold CVNUM \
--cv_verbose 0