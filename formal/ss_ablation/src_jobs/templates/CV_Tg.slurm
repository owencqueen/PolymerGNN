#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/ss_ablation/outputs/tg/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/ss_ablation/outputs/tg/fold=CVNUM.e
#SBATCH -J CVNUM_Tgss

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/ss_ablation"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/cv_gnn.py \
--num_cv 1 \
--Tg \
--properties Mw \
--results_save_dir $base/saved_scores/tg \
--start_fold CVNUM \
--standard_scale \
--cv_verbose 0
