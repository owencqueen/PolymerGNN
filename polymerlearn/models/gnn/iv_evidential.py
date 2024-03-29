import torch
from torch_geometric.nn import SAGEConv, GATConv, Sequential, BatchNorm
from torch_geometric.nn import SAGPooling

class PolymerGNN_IV_evidential(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, num_additional = 0):
        super(PolymerGNN_IV_evidential, self).__init__()
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

        # self.Gsage = Sequential('x, edge_index, batch', [
        #     (GATConv(input_feat, hidden_channels, aggr = 'max'), 'x, edge_index -> x'),
        #     BatchNorm(hidden_channels, track_running_stats=False),
        #     torch.nn.PReLU(),
        #     (SAGEConv(hidden_channels, hidden_channels, aggr = 'max'), 'x, edge_index -> x'),
        #     BatchNorm(hidden_channels, track_running_stats=False),
        #     torch.nn.PReLU(),
        #     (SAGPooling(hidden_channels), 'x, edge_index, batch=batch -> x'),
        # ])

        self.fc1 = torch.nn.Linear(hidden_channels * 2 + num_additional, hidden_channels)
        self.leaky1 = torch.nn.PReLU()
        self.fc2 = torch.nn.Linear(hidden_channels, 4) # Output 4 parameters of NIG distribution

        self.evidence = torch.nn.Softplus()

    def forward(self, Abatch: torch.Tensor, Gbatch: torch.Tensor, add_features: torch.Tensor):
        '''
        
        '''
        # Decompose X into acid and glycol

        Aembeddings = self.Asage(Abatch.x, Abatch.edge_index, Abatch.batch)[0]
        Gembeddings = self.Asage(Gbatch.x, Gbatch.edge_index, Gbatch.batch)[0]

        # self.saveA_embed = Aembed.clone()
        # self.saveG_embed = Gembed.clone()
        
        Aembed, _ = torch.max(Aembeddings, dim=0)
        Gembed, _ = torch.max(Gembeddings, dim=0)

        # Aggregate pooled vectors
        if add_features is not None:
            poolAgg = torch.cat([Aembed, Gembed, add_features])
        else:
            poolAgg = torch.cat([Aembed, Gembed])

        x = self.leaky1(self.fc1(poolAgg))

        # Unpack last layer:
        gamma, logv, logalpha, logbeta = self.fc2(x).squeeze()

        # Because we predict log, exp transform gamma:
        #gamma = torch.exp(loggamma)
        # KEEP IN MIND: may need to alter distribution b/c exp of gamma

        # Activate all other parameters:
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)

        #yhat = torch.exp(self.fc2(x))

        multi_dict = {
            'gamma': gamma, 
            'v': v,
            'alpha': alpha,
            'beta': beta
        }
        
        return multi_dict