import torch
import torch_geometric as pyg
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, Sequential, global_max_pool, global_mean_pool, BatchNorm
from torch_geometric.nn import SAGPooling, Set2Set, GlobalAttention

class PolymerGNN_TgMono(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, num_additional = 0):
        super(PolymerGNN_TgMono, self).__init__()
        self.hidden_channels = hidden_channels

        self.sage = Sequential('x, edge_index, batch', [
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

    def forward(self, Abatch: torch.Tensor, Gbatch: torch.Tensor, add_features: torch.Tensor):
        '''
        
        '''
        # Decompose X into acid and glycol

        Aembeddings = self.sage(Abatch.x, Abatch.edge_index, Abatch.batch)[0]
        Gembeddings = self.sage(Gbatch.x, Gbatch.edge_index, Gbatch.batch)[0]
        
        Aembed, _ = torch.max(Aembeddings, dim=0)
        Gembed, _ = torch.max(Gembeddings, dim=0)

        # Aggregate pooled vectors
        if add_features is not None:
            poolAgg = torch.cat([Aembed, Gembed, add_features])
        else:
            poolAgg = torch.cat([Aembed, Gembed])

        x = self.leaky1(self.fc1(poolAgg))
        pred = self.fc2(x)
        factor = self.mult_factor(x).tanh()

        # Because we're predicting log:
        return torch.exp(pred) * factor
