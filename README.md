# PolymerGNN
GNNs for polymer property prediction.

## Setting up the Environment

We recommend using Python >3.8 to run this code and setting up a virtual environment into which the packages will be installed. To install the polymerlearn package, use the following command:
```
cd /path/to/PolymerGNN
pip install -e .
```
This should also install all dependencies for the package. Most dependencies are not strict, but ensure that you're running [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) >2.0.0. Please see `requirements.txt` for a list of Python packages required to run our code.

## Relevant files
To view a general API overview of the package, see the notebooks in the `demo` directory. These include demo's on predicting IV, Tg, and running the joint model. Examples of both cross validation and regular training are shown. 

To modify model architecture, please see the models in the `polymerlearn/models/gnn` directory. This contains models to perform IV, Tg, and joint prediction, including some experimental models with which the team has worked with.

## Questions?
If you have any questions, please reach out to Owen (oqueen@vols.utk.edu).
