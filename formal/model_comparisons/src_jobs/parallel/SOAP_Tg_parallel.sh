base="/lustre/isaac/scratch/oqueen/PolymerGNN/formal/model_comparisons"

for i in {5..49}
    do
        cp $base/src_jobs/CV_SOAP_Tg.slurm $base/src_jobs/parallel/CV_SOAP_Tg_$i.slurm

        sed -i "s/CVNUM/$i/" $base/src_jobs/parallel/CV_SOAP_Tg_$i.slurm

        sbatch $base/src_jobs/parallel/CV_SOAP_Tg_$i.slurm

        rm $base/src_jobs/parallel/CV_SOAP_Tg_$i.slurm

    done