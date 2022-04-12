# Intstructions to Install Conda Packages

In this file, I'll list out all of the packages that you need to install via conda. For each package, I'll give it in the form "X:`x`", where X is the package name and `x` is what you type in the conda command to install, i.e. `conda install x`.

1. PyTorch: `torch` ([installation instructions](https://pytorch.org/get-started/locally/))
- Note, make sure you install for your correct CPU/GPU version. This only really matters for matching cuda versions on the GPU, CPU install instructions should be straightforward.
2. PyTorch Geometric and corresponding packages: `pyg` ([installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))
 - Using `conda install pyg` should take care of version conflicts, so long as you first install `torch`
3. Rdkit: `rdkit` ([installation instructions](https://anaconda.org/rdkit/rdkit), [see here also](https://www.rdkit.org/docs/Install.html))


Additional pip requirements (latest version of each should work):
1. `pandas`
2. `numpy`
3. `tqdm` (progress bar)
4. `matplotlib`
5. `sklearn`
6. `scipy`
7. `captum` (tools for explainability)
8. `networkx`

Finally, you'll need to make sure that the reference to the `polymerlearn` package is installed locally. From the main PolymerGNN directory, please run
```
pip install -e .
```
