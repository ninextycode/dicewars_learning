import torch
from torch import nn
from learning.model.gat import GATLayer, GATEdgeValue, GATGraphSummary
from learning.data.process_utils import get_active_edges_mask, DWNorm

    

class DWStateChange(nn.Module):
    def __init__(self, in_states, out_states, p_dropout=0.35):
        super().__init__()
        self.input_dropout = nn.Dropout(p_dropout)
        self.transform_fn = nn.Linear(in_states, out_states, bias=True)

    def forward(self, x):
        # x - [n_nodes, in_states]
        x = self.transform_fn(self.input_dropout(x))
        # x - [n_nodes, out_states]
        return x
    

class DWStateFlowModel(nn.Module):
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
        # model outputs prediction for the next action and the final winner
        self.winner_logit_fn = GATGraphSummary(8, 8)
        self.attack_logit_fn = GATEdgeValue(8, 1)
        self.end_turn_logit_fn = GATGraphSummary(8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Reset GAT layers
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()
        # Reset output modules
        self.winner_logit_fn.reset_parameters()
        self.attack_logit_fn.reset_parameters()
        self.end_turn_logit_fn.reset_parameters()

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
        # state corresponding to active player is in the first feature
        # create a mask of attack edges of an active player
        active_edges_mask = get_active_edges_mask(in_states, edges)

        n_graphs = node_graph_ids.max() + 1
        x = self.norm_fn(in_states)
        for gat in self.gat_layers:
            x = gat(x, edges)

        # winner_login - [n_graphs, 8]
        winner_logit = self.winner_logit_fn(x, node_graph_ids)

        # edges_attack_logit - [n_active_edges]
        # mask out impossible attack edges
        edges_attack_logit = \
            self.attack_logit_fn(x, edges[:, active_edges_mask]).squeeze(-1)

        # end_turn_logit - [n_graphs, 1]
        end_turn_logit = self.end_turn_logit_fn(x, node_graph_ids)

        graph_out_vals = []
        for i_graph in range(n_graphs):
            # reindex edges from merged batch graph to individual graphs
            edge_mask = edges_graph_ids[active_edges_mask] == i_graph
            # node_mask = node_graph_ids == i_graph
            graph_out_vals.append(dict(
                winner_logit=winner_logit[i_graph],
                edges_attack_logit=edges_attack_logit[edge_mask],
                end_turn_logit=end_turn_logit[i_graph]
            ))

        return graph_out_vals

