import torch
from torch_geometric.nn import SAGEConv, GATConv, Sequential, BatchNorm
from torch_geometric.nn import SAGPooling

class PolymerGNN_jointMono_SinglePool(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, num_additional = 0):
        super(PolymerGNN_jointMono_SinglePool, self).__init__()
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

        self.fc1 = torch.nn.Linear(hidden_channels + num_additional, hidden_channels)
        self.leaky1 = torch.nn.PReLU()
        self.fc2 = torch.nn.Sequential( # IV model
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        self.fc3 = torch.nn.Linear(hidden_channels, 1) # Tg

        self.mult_factor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Tanh(),
        )

    def forward(self, Abatch: torch.Tensor, Gbatch: torch.Tensor, add_features: torch.Tensor):
        # Decompose X into acid and glycol

        Aembeddings = self.sage(Abatch.x, Abatch.edge_index, Abatch.batch)[0]
        Gembeddings = self.sage(Gbatch.x, Gbatch.edge_index, Gbatch.batch)[0]

        AGembed, _ = torch.max(torch.cat([Aembeddings, Gembeddings]), dim = 0)
        
        # Aembed, _ = torch.max(Aembeddings, dim=0)
        # Gembed, _ = torch.max(Gembeddings, dim=0)

        # Aggregate pooled vectors
        if add_features is not None:
            poolAgg = torch.cat([AGembed, add_features])
        else:
            poolAgg = AGembed

        x = self.leaky1(self.fc1(poolAgg))

        # Run IV:
        x_IV = torch.exp(self.fc2(x))

        # Run Tg (+ multiplying factor):
        x_Tg = torch.exp(self.fc3(x))
        m = self.mult_factor(x)
        x_Tg = x_Tg * m

        return {'IV': x_IV, 'Tg': x_Tg}