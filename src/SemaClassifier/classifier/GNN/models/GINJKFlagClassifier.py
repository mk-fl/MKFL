import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing
#from utils import get_graph_info, build_graph

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "mean") # can change to add
        self.emb_dim = emb_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim), 
            # torch.nn.BatchNorm1d(2*emb_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = torch.nn.Linear(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # import pdb; pdb.set_trace()
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

class GINJKFlag_node(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, num_layers=4, drop_ratio=0.5, residual=False):
        super(GINJKFlag_node, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = hidden
        self.num_tasks = num_classes
        self.num_features = num_features
        self.drop_ratio = drop_ratio
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINConv(hidden))

    def forward(self, x, edge_index, edge_attr, batch, perturb=None):
        tmp = x + perturb if perturb is not None else x
        h_list = []
        h = tmp
        xs = []
        for i in range(self.num_layers):
            h = F.relu(self.convs[i](h, edge_index, edge_attr))
            h_list.append(h)
        node_representation = torch.cat(h_list, dim=1)  # Concatenate representations from all layers

        return node_representation

class GINJKFlag(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, num_layers=4, drop_ratio=0.5, residual=False):
        super(GINJKFlag, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = hidden
        self.num_tasks = num_classes
        self.num_features = num_features

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.gnn_node = GINJKFlag_node(num_features, hidden, num_classes, num_layers, drop_ratio, residual)

        self.pool = global_mean_pool
        # self.pool = global_add_pool

        self.graph_pred_linear = torch.nn.Linear(hidden * num_layers, num_classes)  # Adjust the output dimension
        # self.graph_pred_linear = torch.nn.Linear(hidden, num_classes)



    def forward(self, x, edge_index, edge_attr, batch, perturb=None):
        
        node_rep = self.gnn_node(x, edge_index, edge_attr, batch, perturb)
        # import pdb; pdb.set_trace()
        pooled_rep = self.pool(node_rep, batch)
        graph_rep = self.graph_pred_linear(pooled_rep)

        return F.log_softmax(graph_rep, dim=-1)