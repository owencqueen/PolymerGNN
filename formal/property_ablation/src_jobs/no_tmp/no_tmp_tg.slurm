#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
##SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation/src_jobs/no_tmp/outputs_tg/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation/src_jobs/no_tmp/outputs_tg/fold=CVNUM.e
#SBATCH -J CVNUM_Tgnotmp

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/cv_gnn.py \
--num_cv 1 \
--Tg \
--properties logMw logAN logOHN \
--results_save_dir $base/saved_scores/no_tmp/tg \
--save_history \
--history_loc $base/history/no_tmp/tg \
--start_fold CVNUM \
--cv_verbose 0
