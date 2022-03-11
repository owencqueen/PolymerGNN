# PolymerGNN
GNNs for polymer property prediction.

## Setting up the Environment

We recommend using Python >3.8 to run this code. To install the polymerlearn package, use the following command:
```
cd /path/to/PolymerGNN
pip install -e .
```
This should also install all dependencies for the package. Most dependencies are not strict, but ensure that you're running [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) >2.0.0.

## Relevant files
To view a general API overview of the package, see the notebooks in the `demo` directory. These include demo's on predicting IV, Tg, and running the joint model. Examples of both cross validation and regular training are shown.
