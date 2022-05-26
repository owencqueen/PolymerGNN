base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/other_joint/OHP"

for i in {0..2}
    do
        cp $base/src_jobs/CV_OHP.slurm $base/src_jobs/CV_OHP_$i.slurm
 
        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_OHP_$i.slurm

        sbatch $base/src_jobs//CV_OHP_$i.slurm

        rm $base/src_jobs/CV_OHP_$i.slurm

    done
