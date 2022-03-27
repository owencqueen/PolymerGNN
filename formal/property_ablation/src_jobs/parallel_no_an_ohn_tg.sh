base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

for i in {0..49}
    do
        cp $base/src_jobs/no_an_ohn/no_an_ohn_tg.slurm $base/src_jobs/no_an_ohn_tg_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/no_an_ohn_tg_$i.slurm

        sbatch $base/src_jobs/no_an_ohn_tg_$i.slurm

        rm $base/src_jobs/no_an_ohn_tg_$i.slurm

    done