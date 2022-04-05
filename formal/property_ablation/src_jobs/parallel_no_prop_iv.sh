base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

for i in {11..49}
    do
        cp $base/src_jobs/no_prop/no_prop_iv.slurm $base/src_jobs/no_prop_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/no_prop_$i.slurm

        sbatch $base/src_jobs/no_prop_$i.slurm

        rm $base/src_jobs/no_prop_$i.slurm

    done