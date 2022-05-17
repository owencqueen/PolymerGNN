#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
##SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/sthatigo/PolymerGNN/formal/performance/outputs/tg/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/sthatigo/PolymerGNN/formal/performance/outputs/tg/fold=CVNUM.e
#SBATCH -J CV_Tg_CVNUM

base="/lustre/isaac/scratch/sthatigo/PolymerGNN/formal/performance"

cd $base
conda activate eastman

python3 src/cv_gnn.py \
--num_cv 1 \
--Tg \
--Kelvin \
--properties default \
--results_save_dir $base/saved_scores/tg \
--save_history \
--history_loc $base/history/tg \
--start_fold CVNUM \
--cv_verbose 0
