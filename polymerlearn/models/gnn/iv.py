import torch
import torch_geometric
from torch_geometric.nn import SAGEConv, GATConv, Sequential, BatchNorm
from torch_geometric.nn import SAGPooling

class PolymerGNN_IV(torch.nn.Module):
    '''
    Args:
        input_feat (int): Number of input features on each node.
        hidden_channels (int): Number of neurons in hidden layers throughout
            the neural network.
        num_additional (int, optional): Number of additional resin properties
            to be used during the training/prediction.
    '''
    def __init__(self, input_feat, hidden_channels, num_additional = 0):
        super(PolymerGNN_IV, self).__init__()
        self.hidden_channels = hidden_channels

        self.Asage = Sequential('x, edge_index, batch', [
            (GATConv(input_feat, hidden_channels, aggr = 'max'), 'x, edge_index -> x'),
            BatchNorm(hidden_channels, track_running_stats=False),
            torch.nn.PReLU(),
            (SAGEConv(hidden_channels, hidden_channels, aggr = 'max'), 'x, edge_index -> x'),
            BatchNorm(hidden_channels, track_running_stats=False),
            torch.nn.PReLU(),
            (SAGPooling(hidden_channels), 'x, edge_index, batch=batch -> x'),
        ])

        self.Gsage = Sequential('x, edge_index, batch', [
            (GATConv(input_feat, hidden_channels, aggr = 'max'), 'x, edge_index -> x'),
            BatchNorm(hidden_channels, track_running_stats=False),
            torch.nn.PReLU(),
            (SAGEConv(hidden_channels, hidden_channels, aggr = 'max'), 'x, edge_index -> x'),
            BatchNorm(hidden_channels, track_running_stats=False),
            torch.nn.PReLU(),
            (SAGPooling(hidden_channels), 'x, edge_index, batch=batch -> x'),
        ])

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

        Aembeddings = self.Asage(Abatch.x, Abatch.edge_index, Abatch.batch)[0]
        Gembeddings = self.Gsage(Gbatch.x, Gbatch.edge_index, Gbatch.batch)[0]
        
        Aembed, _ = torch.max(Aembeddings, dim=0)
        Gembed, _ = torch.max(Gembeddings, dim=0)

        # Aggregate pooled vectors
        if add_features is not None:
            poolAgg = torch.cat([Aembed, Gembed, add_features])
        else:
            poolAgg = torch.cat([Aembed, Gembed])

        x = self.leaky1(self.fc1(poolAgg))
        x = self.fc2(x)

        # Because we're predicting log:
        return torch.exp(x)