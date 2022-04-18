base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons"

for i in {0..49}
    do
        cp $base/src_jobs/CV_MBTR_IV.slurm $base/src_jobs/parallel/CV_MBTR_IV_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/parallel/CV_MBTR_IV_$i.slurm

        sbatch $base/src_jobs/parallel/CV_MBTR_IV_$i.slurm

        rm $base/src_jobs/parallel/CV_MBTR_IV_$i.slurm

    done