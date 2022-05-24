#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
##SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/performance/outputs/joint_opt/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/performance/outputs/joint_opt/fold=CVNUM.e
#SBATCH -J CV_CVNUM_joint_opt

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/performance"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/cv_gnn.py \
--num_cv 1 \
--IV \
--Tg \
--properties Mw AN OHN %TMP \
--results_save_dir $base/saved_scores/joint_opt \
--standard_scale \
--start_fold CVNUM \
--cv_verbose 0
