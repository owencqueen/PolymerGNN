base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

for i in {0..10}
    do
        cp $base/src_jobs/no_log/no_log.slurm $base/src_jobs/no_log_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/no_log_$i.slurm

        sbatch $base/src_jobs/no_log_$i.slurm

        rm $base/src_jobs/no_log_$i.slurm

    done