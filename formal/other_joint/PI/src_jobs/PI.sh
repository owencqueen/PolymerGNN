base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/PI"

for i in {0..49}
    do
        cp $base/src_jobs/CV_PI.slurm $base/src_jobs/CV_PI_$i.slurm
 
        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_PI_$i.slurm

        sbatch $base/src_jobs/CV_PI_$i.slurm

        rm $base/src_jobs/CV_PI_$i.slurm

    done