# PolymerGNN
<img src="https://github.com/owencqueen/PolymerGNN/blob/main/img/whole_arch.jpg" data-canonical-src="https://github.com/owencqueen/PolymerGNN/blob/main/img/whole_arch.jpg" width="700" height="700" />
Graph neural networks for polymer property prediction.

## PolymerLearn Package Structure
`polymerlearn` is the main component of the repository. This contains all of the driver code for `PolymerGNN`, explainability tools, and utilities for proprocessing data in the format provided.

```
polymerlearn
├── __init__.py
├── explain
│   ├── __init__.py
│   ├── custom_gcam.py
│   ├── explain_gnn.py
│   └── modified_gnns.py
├── models
│   └── gnn
│       ├── __init__.py
│       ├── iv.py
│       ├── iv_evidential.py
│       ├── joint.py
│       └── tg.py
└── utils
    ├── __init__.py
    ├── graph_prep.py
    ├── losses.py
    ├── train_evidential.py
    ├── train_graphs.py
    ├── uncertainty.py
    └── xyz2mol.py
```

## Setting up the Environment

We recommend using Python >3.8 to run this code and setting up a virtual environment into which the packages will be installed. If you use Anaconda, you can set up all the requirements from the yml file:
```
conda env create --name your_env_name --file polymergnn_env.yml
```
This sets up all of the packages, and Owen has verified that it can be used on the ISAAC cluster.

For more general setup instructions for external packages, please reference [conda-instructions.md](https://github.com/owencqueen/PolymerGNN/blob/main/conda-instructions.md).

To install the polymerlearn package, use the following command:
```
cd /path/to/PolymerGNN
pip install -e .
```
If using conda, ensure you have `pip` installed into the conda environment (`conda install pip`).

**note**: As of Mar 24, DONT USE the requirements.txt file. Will update this later. Leaving the below bit for use later:

This should also install all dependencies for the package. Most dependencies are not strict, but ensure that you're running [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) >2.0.0. Please see `requirements.txt` for a list of Python packages required to run our code.

## Relevant files
To view a general API overview of the package, see the notebooks in the `demo` directory. These include demo's on predicting IV, Tg, and running the joint model. Examples of both cross validation and regular training are shown. 

To modify model architecture, please see the models in the `polymerlearn/models/gnn` directory. This contains models to perform IV, Tg, and joint prediction, including some experimental models with which the team has worked with.

## Questions?
If you have any questions, please reach out to Owen (owen_queen@hms.harvard.edu).
