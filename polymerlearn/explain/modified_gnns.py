import torch
from torch_geometric.nn import SAGEConv, GATConv, Sequential, BatchNorm
from torch_geometric.nn import SAGPooling

class PolymerGNN_IV_EXPLAIN(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, num_additional = 0):
        super(PolymerGNN_IV_EXPLAIN, self).__init__()
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

    def forward(self, 
            Abatch_X: torch.Tensor, 
            Abatch_edge_index: torch.Tensor,
            Abatch_batch: torch.Tensor,
            Gbatch_X: torch.Tensor, 
            Gbatch_edge_index: torch.Tensor,
            Gbatch_batch: torch.Tensor, 
            add_features: torch.Tensor):
        '''
        Only thing that's different is the forward method
        '''
        # Decompose X into acid and glycol

        Aembeddings = self.Asage(Abatch_X, Abatch_edge_index, Abatch_batch)[0]
        Gembeddings = self.Gsage(Gbatch_X, Gbatch_edge_index, Gbatch_batch)[0]

        # self.saveAembed = Aembeddings.clone()
        # self.saveGembed = Gembeddings.clone()
        
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

class PolymerGNN_Tg_EXPLAIN(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, num_additional = 0):
        super(PolymerGNN_Tg_EXPLAIN, self).__init__()
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

        self.mult_factor = torch.nn.Linear(hidden_channels, 1)

    def forward(self, 
            Abatch_X: torch.Tensor, 
            Abatch_edge_index: torch.Tensor,
            Abatch_batch: torch.Tensor,
            Gbatch_X: torch.Tensor, 
            Gbatch_edge_index: torch.Tensor,
            Gbatch_batch: torch.Tensor, 
            add_features: torch.Tensor):
        '''
        
        '''
        # Decompose X into acid and glycol

        Aembeddings = self.Asage(Abatch_X, Abatch_edge_index, Abatch_batch)[0]
        Gembeddings = self.Gsage(Gbatch_X, Gbatch_edge_index, Gbatch_batch)[0]
        
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