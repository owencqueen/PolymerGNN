#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
##SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation/src_jobs/no_an_ohn/outputs_joint/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation/src_jobs/no_an_ohn/outputs_joint/fold=CVNUM.e
#SBATCH -J CVNUM_Jnoan_ohn

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/cv_gnn.py \
--num_cv 1 \
--IV \
--Tg \
--properties logMw %TMP \
--results_save_dir $base/saved_scores/no_an_ohn/joint \
--start_fold CVNUM \
--cv_verbose 0
