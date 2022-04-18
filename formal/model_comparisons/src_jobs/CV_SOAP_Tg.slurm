#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons/SOAP/out_tg/SOAP_CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons/SOAP/out_tg/SOAP_CVNUM.e
#SBATCH -J STg_CVNUM

base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons"

cd $base
conda activate /lustre/isaac/scratch/oqueen/polymergnn_env

python3 src/run_rep.py \
--num_cv 50 \
--rep SOAP \
--target Tg \
--results_save_dir $base/SOAP/tg \
--start_fold CVNUM \
