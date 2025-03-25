from models.gcl import E_GCL, unsorted_segment_sum, GCL
import torch
from torch import nn
from torch import cdist


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.SiLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        # del self.coord_mlp    # pay attention here
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask = None):
        row, col = edge_index
        if edge_mask is None:
            trans = coord_diff * self.coord_mlp(edge_feat)
        else:
            trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask = None, edge_mask = None, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # for us the edge_mask is located on the diagonal, where the row_index == col_index

        if edge_mask is None:
            edge_feat = edge_feat
        else:
            edge_feat = edge_feat * edge_mask # TO DO: edge_feat = edge_feat * edge_mask

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf,  out_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.ReLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()   # act_fn = nn.SiLU()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        # self.act_fcn = act_fn

        ### Encoder
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)

        self.x_embedding = nn.Sequential(nn.ReLU(),
                                         nn.Linear(in_node_nf, int(in_node_nf/2)),
                                         nn.ReLU(),
                                         nn.Linear(int(in_node_nf/2), int(in_node_nf/4)),
                                         nn.ReLU(),
                                         nn.Linear(int(in_node_nf/4), int(in_node_nf/8)),
                                         nn.ReLU(),
                                         nn.Linear(int(in_node_nf / 8), 3),
                                         )


        #self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.embedding_out = nn.Sequential(nn.ReLU(),
                                         nn.Linear(hidden_nf, int(hidden_nf / 2)),
                                         nn.ReLU(),
                                         nn.Linear(int(hidden_nf / 2), int(hidden_nf / 4)),
                                         nn.ReLU(),
                                         nn.Linear(int(hidden_nf / 4), int(hidden_nf / 8)),
                                         nn.ReLU(),
                                         nn.Linear(int(hidden_nf / 8), 3),
                                         )

        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn = act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding_in(h0)
        # x = self.x_embedding(h0)

        for i in range(0, self.n_layers):
            if self.node_attr:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
                #x = (self.embedding_out(h)+x)/2  # model_2 settings: 4
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)
                #x = (self.embedding_out(h)+x)/2  # model_2 new settings


        # h = self.embedding_out(h)  # model_2 settings: 1
        # distance = cdist(h, h, p=2)  # model_2 settings: 1
        distance = cdist(x, x, p=2)
        return h, x, distance

class GENN(nn.Module):
    def __init__(self, in_node_nf,  out_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(GENN, self).__init__()   # act_fn = nn.SiLU()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        # self.act_fcn = act_fn

        ### Encoder
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)

        self.x_embedding = nn.Sequential(nn.ReLU(),
                                         nn.Linear(in_node_nf, int(in_node_nf / 2)),
                                         nn.ReLU(),
                                         nn.Linear(int(in_node_nf / 2), int(in_node_nf / 4)),
                                         nn.ReLU(),
                                         nn.Linear(int(in_node_nf / 4), int(in_node_nf / 8)),
                                         nn.ReLU(),
                                         nn.Linear(int(in_node_nf / 8), 3),
                                         )

        #self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.embedding_out = nn.Sequential(nn.ReLU(),
                                         nn.Linear(hidden_nf, int(hidden_nf / 2)),
                                         nn.ReLU(),
                                         nn.Linear(int(hidden_nf / 2), int(hidden_nf / 4)),
                                         nn.ReLU(),
                                         nn.Linear(int(hidden_nf / 4), int(hidden_nf / 8)),
                                         nn.ReLU(),
                                         nn.Linear(int(hidden_nf / 8), 3),
                                         )

        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn = act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.to(self.device)

    def forward(self, h0, pos, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding_in(h0)
        x = pos

        for i in range(0, self.n_layers):
            if self.node_attr:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
                x = (self.embedding_out(h)+x)/2  # model_2 settings: 4
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)
                x = (self.embedding_out(h)+x)/2  # model_2 new settings


        # h = self.embedding_out(h)  # model_2 settings: 1
        # distance = cdist(h, h, p=2)  # model_2 settings: 1
        distance = cdist(x, x, p=2)
        return h, x, distance


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        # self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i,
                            GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn,
                                attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                                     act_fn,
                                     nn.Linear(hidden_nf, 3),
                                     #nn.Tanh()
                                     )
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)

    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        # h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)

        edges_feature = None
        for i in range(0, self.n_layers):
            h, edges_feature = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        # return h

        h = self.decoder(h)
        distance =  cdist(h, h, p=2)
        return edges_feature, h, distance






