import os, argparse
import torch
import pandas as pd
from polymerlearn.utils import get_IV_add, GraphDataset
from polymerlearn.models.gnn import PolymerGNN_IV
from polymerlearn.utils import train

# Load data from local path:
data = pd.read_csv(os.path.join('/Users/owenqueen/Desktop/eastman_project-confidential/Eastman_Project/CombinedData', 
            'pub_data.csv'))

add = get_IV_add(data)

dataset = GraphDataset(
    data = data,
    Y_target=['IV'],
    test_size = 0.2,
    add_features=add
)

