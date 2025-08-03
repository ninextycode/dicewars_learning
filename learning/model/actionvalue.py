import torch
from torch import nn
from .gat import GATLayer, GATEdgeValue, GATGraphSummary
from learning.data.process_utils import DWNorm, get_active_edges_mask



class DWActionValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        # there are 8 teams - each node has 8 corresponding values
        # for each node, there are 1-8 dice, all belonging to a single team
        # value for this team = # of dice, for other teams value = 0
        self.norm_fn = DWNorm()
        self.gat_layers = nn.ModuleList([
            GATLayer(8, 8, p_msg_dropout=0),  # avoid dropout for sparse input
            GATLayer(8, 8),
            GATLayer(8, 8)
        ])
        # model outputs prediction for the next action value for the current active player
        self.attack_value_fn = GATEdgeValue(8, 1)
        self.end_turn_value_fn = GATGraphSummary(8, 1)

    def forward(
            self, in_states: torch.Tensor, edges: torch.Tensor,
            node_graph_ids: torch.Tensor, edges_graph_ids: torch.Tensor
        ):
        # model expects input state features to be sorted in the following order
        #  first - active player's number of dice
        #  following - players' # of dice in a given node 
        #  in decreasing order of their total number of dice in all nodes
        # in_states - [n_nodes, 8]
        # edges - [2, n_edges]
        # graph_node_ids - [n_nodes]
        # graph_edges_ids - [n_edges]
        n_graphs = node_graph_ids.max() + 1
        x = self.norm_fn(in_states)

        for gat in self.gat_layers:
            x = gat(x, edges)

        active_edges_mask = get_active_edges_mask(in_states, edges)
        # edge_attack_val - [n_active_edges, 1]
        edge_attack_val = self.attack_value_fn(x, edges[:, active_edges_mask])
        # edge_attack_val - [n_active_edges]
        edge_attack_val = edge_attack_val.squeeze(-1)

        # end_turn_val - [n_graphs, 8]
        end_turn_val = self.end_turn_value_fn(x, node_graph_ids)

        graph_action_vals = []
        for i_graph in range(n_graphs):
            # reindex edges from merged batch graph to individual graphs
            edge_mask = edges_graph_ids[active_edges_mask] == i_graph
            graph_action_vals.append(dict(
                end_turn_val=end_turn_val[i_graph],
                edge_attack_val=edge_attack_val[edge_mask]
            ))

        return graph_action_vals

