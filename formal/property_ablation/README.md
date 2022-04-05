# How to run property ablation

To set up a new experiment:
1. Make sure the options are supported in src/cv_gnn.py
2. Set up a folder in src_jobs:
    ```
    exp_name/
        outputs_iv/
        outputs_tg/
        outputs_joint/
    ```
3. Add .slurm scripts to the `exp_name/` directory
4. Add parallel running script to `src_jobs/` directory
5. Add corresponding location in `saved_scores` directory
    ```
    saved_scores/
        exp_name/
            iv/
            tg/
            joint/
    ```
    