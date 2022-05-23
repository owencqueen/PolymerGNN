base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/MBTR"

for i in {0..2}
    do
        cp $base/src_jobs/CV_MBTR.slurm $base/src_jobs/CV_MBTR_$i.slurm
 
        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_MBTR_$i.slurm

        sbatch $base/src_jobs//CV_MBTR_$i.slurm

        rm $base/src_jobs/CV_MBTR_$i.slurm

    done
