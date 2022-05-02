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
