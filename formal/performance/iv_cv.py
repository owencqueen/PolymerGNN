import os, argparse
import torch
import pandas as pd
from polymerlearn.utils import get_IV_add, GraphDataset
from polymerlearn.models.gnn import PolymerGNN_IV
from polymerlearn.utils import train

from polymerlearn.models.gnn import PolymerGNN_IV
from polymerlearn.utils import CV_eval

# Load data from local path:
data = pd.read_csv(os.path.join('/Users/owenqueen/Desktop/eastman_project-confidential/Eastman_Project/CombinedData', 
            'pub_data.csv'))

parser = argparse.ArgumentParser()
parser.add_argument('--num_cv', type = int, required = True,
    help = 'Number of cross validations')

args = parser.parse_args()

add = get_IV_add(data)

dataset = GraphDataset(
    data = data,
    Y_target=['IV'],
    test_size = 0.2,
    add_features=add
)

model_generator_kwargs = {
    'input_feat': 6,         # How many input features on each node; don't change this
    'hidden_channels': 32,   # How many intermediate dimensions to use in model
                            # Can change this ^^
    'num_additional': 4      # How many additional resin properties to include in the prediction
                            # Corresponds to the number in get_IV_add
}

optimizer_gen = torch.optim.AdamW
criterion = torch.nn.MSELoss()

for i in range(args.num_cv):
    pass