base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/performance"

for i in {11..49}
    do
        cp $base/src_jobs/templates/CV_joint.slurm $base/src_jobs/CV_joint_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_joint_$i.slurm

        sbatch $base/src_jobs/CV_joint_$i.slurm

        rm $base/src_jobs/CV_joint_$i.slurm

    done