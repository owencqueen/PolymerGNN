base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/property_ablation"

for i in {0..49}
    do
        cp $base/src_jobs/no_tmp/no_tmp_iv.slurm $base/src_jobs/no_tmp_iv_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/no_tmp_iv_$i.slurm

        sbatch $base/src_jobs/no_tmp_iv_$i.slurm

        rm $base/src_jobs/no_tmp_iv_$i.slurm

    done