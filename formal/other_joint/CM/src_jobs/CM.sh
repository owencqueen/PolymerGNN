base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/CM"

for i in {0..49}
    do
        cp $base/src_jobs/CV_CM.slurm $base/src_jobs/CV_CM_$i.slurm
 
        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_CM_$i.slurm

        sbatch $base/src_jobs/CV_CM_$i.slurm

        rm $base/src_jobs/CV_CM_$i.slurm

    done