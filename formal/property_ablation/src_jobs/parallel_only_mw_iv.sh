base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

for i in {0..49}
    do
        cp $base/src_jobs/only_mw/only_mw_iv.slurm $base/src_jobs/only_mw_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/only_mw_$i.slurm

        sbatch $base/src_jobs/only_mw_$i.slurm

        rm $base/src_jobs/only_mw_$i.slurm

    done