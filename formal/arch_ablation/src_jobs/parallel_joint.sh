base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/arch_ablation"

for i in {0..49}
    do
        cp $base/src_jobs/templates/CV_joint.slurm $base/src_jobs/CV_joint_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_joint_$i.slurm

        sbatch $base/src_jobs/CV_joint_$i.slurm

        rm $base/src_jobs/CV_joint_$i.slurm

    done