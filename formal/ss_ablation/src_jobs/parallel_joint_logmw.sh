base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/ss_ablation"

for i in {0..49}
    do
        cp $base/src_jobs/templates/CV_joint_logmw.slurm $base/src_jobs/CV_joint_logmw_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/CV_joint_logmw_$i.slurm

        sbatch $base/src_jobs/CV_joint_logmw_$i.slurm

        rm $base/src_jobs/CV_joint_logmw_$i.slurm

    done