base="/lustre/haven/proj/UTK0022/PolymerGNN/formal/performance"

for i in {1..49}
    do
        cp $base/src_jobs/templates/CV_IV.pbs .

        sed -i "s/NUM/$i/" CV_IV.pbs

        qsub CV_IV.pbs

        rm $base/src_jobs/CV_IV.pbs

    done