base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/performance"

for i in {11..49}
    do
        cp $base/src_jobs/templates/CV_IV.slurm $base/src_jobs/CV_IV_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_IV_$i.slurm

        sbatch $base/src_jobs/CV_IV_$i.slurm

        rm $base/src_jobs/CV_IV_$i.slurm

    done