# Guide to Experiments

This directory contains all the code for formal experiments that appear within the results of the paper. They include slurm scripts to submit jobs on the UTK clusters and cross-validate every experiment.


| Syntax      | Description |
| ----------- | ----------- |
| **arch_ablation**      | Ablation study exp where one GNN is used to embed both acids and glycols.       |
| **explainability**     | Explainability experiments, namely the barplot        |
| **log_ablation**       | Not sure?        |
| **model_comparisons**  | All model comparisons, including CM, SOAP, etc. (excluding Gavin's results)  |
| **onepool_ablation**   | Ablation study exp. where max pooling is performed over both acid and glycol embedding.        |
| **performance**        | Best model variants.        |
| **property_ablation**  | Ablation study of removing input resin properties.        |

## How to set up an experiment directory
Each experiment should be set up with four sub-directories. One example is shown below with `onepool_ablation`:
```
onepool_ablation
├── outputs
│   ├── iv  [100 entries exceeds filelimit, not opening dir]
│   ├── joint  [100 entries exceeds filelimit, not opening dir]
│   └── tg  [100 entries exceeds filelimit, not opening dir]
├── saved_scores
│   ├── iv  [50 entries exceeds filelimit, not opening dir]
│   ├── joint  [50 entries exceeds filelimit, not opening dir]
│   └── tg  [50 entries exceeds filelimit, not opening dir]
├── src
│   └── mono_graph.py
└── src_jobs
    ├── parallel_IV.sh
    ├── parallel_Tg.sh
    ├── parallel_joint.sh
    └── templates
        ├── CV_IV.slurm
        ├── CV_Tg.slurm
        └── CV_joint.slurm
```
Here are the steps to set this up:
1. Create all directories: `src`, `src_jobs`, `saved_scores`, `outputs`
2. Create all subdirectories for the outputs and saved scores of each experiment, such as `outputs/iv` and `saved_scores/iv`
3. Write the script to replace `src/mono_graph.py`, which is the main driver code. Try to keep all variations of the script as command-line arguments.
4. Write the templates for the cross validation trials. These are scripts (`*.slurm`). I have been putting them in `src_jobs/templates`
5. Ensure paths are correct for template scripts. These should involve the most editing. These edits will decide where the saved scores and outputs are stored, which is very important.
6. Ensure experiment is ran correctly in the template script for each experiment.
7. Copy the `parallel*` scripts from previous experiment directories, put in `src_jobs`. These are scripts such as `parallel_IV.sh` above.
8. Ensure the correct paths are used in the `parallel*` scripts.

After setting up all of the above, you should be good to run the experiments and generate scores. These scores can be used by the `formal/summarize_results.py` script to show the proper scores and statistics needed to report on the paper.

**Note**: It is important to keep all of our scripts in case we need to re-run experiments later. This setup should make that process simpler.

Please reference the old directory structures as a guide for creating this setup. If you see any ways to optimize on this, please do so! Thanks.
