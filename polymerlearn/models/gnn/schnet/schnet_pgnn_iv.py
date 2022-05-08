import torch
import torch_geometric
from torch_geometric.nn import SAGEConv, GATConv, Sequential, BatchNorm
from torch_geometric.nn import SAGPooling

from .model import SchNet

default_schnet = {
    'hidden_channels': 32 * 2, 
    'num_filters': 64,             
    'num_interactions': 6, 
    'num_gaussians': 50,
    'cutoff': 10.0, 
    'max_num_neighbors': 32,
    'get_embedding': True
}

class PolymerGNN_SchNet_IV(torch.nn.Module):
    '''
    Args:
        input_feat (int): Number of input features on each node.
        hidden_channels (int): Number of neurons in hidden layers throughout
            the neural network.
        num_additional (int, optional): Number of additional resin properties
            to be used during the training/prediction.
    '''
    def __init__(self, input_feat, hidden_channels, num_additional = 0, 
            pretrained_schnet = None, schnet_params = default_schnet):
        super(PolymerGNN_SchNet_IV, self).__init__()
        self.hidden_channels = hidden_channels

        if pretrained_schnet is None: 
            self.gnn = SchNet(**schnet_params)
        else:
            self.gnn = pretrained_schnet

        self.fc1 = torch.nn.Linear(hidden_channels * 2 + num_additional, hidden_channels)
        self.leaky1 = torch.nn.PReLU()
        self.fc2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, Abatch: torch_geometric.data.Batch, Gbatch: torch_geometric.data.Batch, 
            add_features: torch.Tensor):
        '''
        Args:
            Abatch (torch_geometric.data.Batch): Batch object representing all acids in
                the input. See make_like_batch for transforming to this.
            Gbatch (torch_geometric.data.Batch): Batch object representing all glycols in
                the input. See make_like_batch for transforming to this.
            add_features (torch.Tensor): Additional features for this sample.
        '''
        # Decompose X into acid and glycol

        # Use the same gnn layer (based on previous experiments):
        Aembeddings = self.gnn(Abatch.z, Abatch.pos, Abatch.batch)[0]
        Gembeddings = self.gnn(Gbatch.z, Gbatch.pos, Gbatch.batch)[0]

        Aembed, Gembed = Aembeddings, Gembeddings

        # print('Aembed shape', Aembeddings.shape)
        # print('Gembed shape', Gembeddings.shape)
        
        # Aembed, _ = torch.max(Aembeddings, dim=0)
        # Gembed, _ = torch.max(Gembeddings, dim=0)

        # Aggregate pooled vectors
        if add_features is not None:
            poolAgg = torch.cat([Aembed, Gembed, add_features])
        else:
            poolAgg = torch.cat([Aembed, Gembed])

        x = self.leaky1(self.fc1(poolAgg))
        x = self.fc2(x)

        # Because we're predicting log:
        return torch.exp(x)