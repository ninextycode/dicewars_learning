import torch
from torch import nn
from learning.model.gat import GATLayer



class NodeValueModel(nn.Module):
    def __init__(self, with_dice_scatter=True):
        super().__init__()
        # input is the number of dice on nodes irrespective of the team
        self.with_dice_scatter = with_dice_scatter
        n_deg_embed = 1
        n_xy_embed = 2
        n_lap_embed = 4
        n_rw_embed = 4
        n_dice_scatter = 8 if with_dice_scatter else 0
        in_features = n_deg_embed + n_xy_embed + n_rw_embed + n_lap_embed + n_dice_scatter
        self.gat_layers = nn.ModuleList([
            # avoid dropout for sparse input
            GATLayer(in_features, 20, n_output_heads=2, p_msg_dropout=0), 
            GATLayer(40, 20, n_output_heads=2),
            GATLayer(40, 20, n_output_heads=2),
        ])
        self.final_dropout = nn.Dropout(p=0.35)
        self.node_value_fn = nn.Linear(40, 1)

        # Create the mean and std parameters based on whether dice_scatter is included
        mean_values = [
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
        ]
        
        std_values = [
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
        ]
        
        # Add dice scatter values if needed
        if with_dice_scatter:
            mean_values.extend([1.284, 1.645, 0.648, 0.185, 0.079, 0.044, 0.025, 0.014])
            std_values.extend([2.458, 2.680, 1.761, 0.854, 0.500, 0.351, 0.265, 0.197])
        
        self.in_states_mean = nn.Parameter(
            torch.tensor(mean_values, dtype=torch.float32),
            requires_grad=False
        )
        
        self.in_states_std = nn.Parameter(
            torch.tensor(std_values, dtype=torch.float32),
            requires_grad=False
        )

    def normalize_in_states(self, in_states):
        return (in_states - self.in_states_mean) / self.in_states_std

    def augment_in_states(self, in_states):
        expected_features = self.in_states_mean.size(0)
        actual_features = in_states.size(-1)
        
        if actual_features < expected_features:
            # If we have fewer features than expected, pad with mean values
            # For input [n_nodes, features]
            padding = self.in_states_mean[torch.newaxis, actual_features:]
            padding = padding.expand(in_states.shape[0], -1).to(in_states.device)
            return torch.cat([in_states, padding], dim=1)
    
        return in_states

    def forward(
        self,
        in_states: torch.Tensor,
        node_player_ids: torch.Tensor,
        edges: torch.Tensor,
        node_graph_ids: torch.Tensor,
        edges_graph_ids: torch.Tensor,
        gather_logits = True
    ):
        # in_states - [n_nodes, n_features]
        #  this should include graph structure embeddings,
        #  and dice numbers, scattered by power and active player
        # node_player_ids - [n_nodes]
        # edges - [2, n_edges]
        # graph_node_ids - [n_nodes]
        # edges_graph_ids - [n_edges]
        n_graphs = node_graph_ids.max() + 1
        x = self.normalize_in_states(in_states)

        for gat in self.gat_layers:
            x = gat(x, edges)
        # x - [n_nodes, 40]
        
        # node_value - [n_nodes]
        node_values = self.node_value_fn(x).squeeze(-1)

        if not gather_logits:
            return node_values

        n_teams = 8
        # winner_logits - [n_graphs, 8]
        winner_logits = torch.zeros(
            n_graphs, n_teams, dtype=node_values.dtype, device=node_values.device
        )
        winner_logits.index_put_(
            (node_graph_ids.to(torch.int32), node_player_ids.to(torch.int32)),
            node_values,
            accumulate=True,
        )

        return winner_logits

