#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/evidential/outputs/iv/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/evidential/outputs/iv/fold=CVNUM.e
#SBATCH -J CVNUM_IVevidential

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/evidential"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/mono_graph.py \
--num_cv 1 \
--IV \
--properties Mw AN OHN %TMP \
--results_save_dir $base/saved_scores/iv \
--start_fold CVNUM \
--standard_scale \
--cv_verbose 0
