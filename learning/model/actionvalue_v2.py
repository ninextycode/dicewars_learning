import torch
from torch import nn
from learning.model.gat import GATLayer, GATEdgeValue, GATGraphSummary
from learning.model.utils import DWNorm
from learning.data.process_utils import get_active_edges_mask



class ActionValueModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # input is the number of dice on nodes irrespective of the team
        n_deg_embed = 1
        n_xy_embed = 2
        n_lap_embed = 4
        n_rw_embed = 4
        n_dice_scatter = 8
        in_features = n_deg_embed + n_xy_embed + n_rw_embed + n_lap_embed + n_dice_scatter
        self.gat_layers = nn.ModuleList([
            # avoid dropout for sparse input
            GATLayer(in_features, 20, n_output_heads=2, p_msg_dropout=0), 
            GATLayer(40, 20, n_output_heads=2),
            GATLayer(40, 20, n_output_heads=2),
        ])
        self.attack_value_fn = GATEdgeValue(40, 1)
        self.end_turn_value_fn = GATGraphSummary(40, 1)

        # Create the mean and std parameters based on whether dice_scatter is included
        mean_input = [
            3.8,  # degrees
            0,    # laplacians
            0,
            0,
            0,

            # random walk
            0.271,  # 2
            0.069,  # 3
            0.076,  # 5
            0.084,  # 8
                
            13.3,  # x-y coordinates
            15.3,
            
            # dice values
            1.284, 1.645, 0.648, 0.185,
            0.079, 0.044, 0.025, 0.014
        ]
        
        std_input = [
            1.36,    # degrees
            0.182,  # laplacians
            0.182,
            0.182,
            0.182,

            # random walk
            0.073,  # 2
            0.035,  # 3
            0.034,  # 5
            0.036,  # 8
            
            7.63,  # x-y coordinates
            8.74,

            # dice values
            2.458, 2.680, 1.761, 0.854,
            0.500, 0.351, 0.265, 0.197
        ]
        
        self.in_states_mean = nn.Parameter(
            torch.tensor(mean_input, dtype=torch.float32),
            requires_grad=False
        )
        
        self.in_states_std = nn.Parameter(
            torch.tensor(std_input, dtype=torch.float32),
            requires_grad=False
        )

        self.mean_val = torch.tensor(-820, requires_grad=False)
        self.std_val = torch.tensor(1870, requires_grad=False)


    def normalize_output(self, values):
        return (values - self.mean_val) / self.std_val
    
    def denormalize_output(self, normalized_values):
        return normalized_values * self.std_val + self.mean_val

    def normalize_input(self, values):
        return (values - self.in_states_mean) / self.in_states_std

    def forward(
            self, in_states: torch.Tensor, edges: torch.Tensor,
            node_graph_ids: torch.Tensor, edges_graph_ids: torch.Tensor
        ):
        # in_states - [n_nodes, n_features]
        #  this should include graph structure embeddings,
        #  and dice numbers, scattered by power and active player
        # edges - [2, n_edges]
        # graph_node_ids - [n_nodes]
        # edges_graph_ids - [n_edges]
        n_graphs = node_graph_ids.max() + 1
        x = self.normalize_input(in_states)

        for gat in self.gat_layers:
            x = gat(x, edges)
        # x - [n_nodes, 40]

        # a bunch of embedding data is saved in first columns, player dice data is the last 8
        active_edges_mask = get_active_edges_mask(in_states, edges, active_player=-8)
        # edge_attack_val - [n_active_edges, 1]
        edge_attack_val = self.attack_value_fn(x, edges[:, active_edges_mask])
        # edge_attack_val - [n_active_edges]
        edge_attack_val = edge_attack_val.squeeze(-1)

        # end_turn_val - [n_graphs, 1]
        end_turn_val = self.end_turn_value_fn(x, node_graph_ids)
        # end_turn_val - [n_graphs]
        end_turn_val = end_turn_val.squeeze(-1)

        graph_action_vals = []
        for i_graph in range(n_graphs):
            # reindex edges from merged batch graph to individual graphs
            edge_mask = edges_graph_ids[active_edges_mask] == i_graph
            graph_action_vals.append(dict(
                end_turn_val=end_turn_val[i_graph],
                edge_attack_val=edge_attack_val[edge_mask]
            ))

        return graph_action_vals

