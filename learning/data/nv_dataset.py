from torch.utils.data import Dataset
from learning.data.load_utils import load_dicewars_data
from learning.data.process_utils import get_adj_matrix_embeddings
from learning.data.process_utils import get_power_order
import torch


class NodeValueDataset(Dataset):
    def __init__(self, json_data, mix_players=True):
        if isinstance(json_data, str):
            json_data = load_dicewars_data(json_data)
        self.graph_id = json_data["graph_id"]
        self.dice_numbers_scatter = json_data["nodes_states"]
        self.player_ids = json_data["player_ids"]
        self.edges = json_data["edges"]
        self.active_players = json_data["active_players"]
        self.n_states = self.dice_numbers_scatter.shape[0]

        adj_matrix = json_data["adj_matrix"]
        adj_graph_embed = get_adj_matrix_embeddings(adj_matrix)
        node_positions = json_data["node_positions"]
        self.graph_embed = torch.cat(
            [adj_graph_embed, node_positions], dim=1
        )      

        if mix_players:
            self.dice_numbers_scatter = self.dice_numbers_scatter.clone()
            self.player_ids = self.player_ids.clone()
            self.active_players = self.active_players.clone()

            original_winner = json_data["winner"]
            self.winner = torch.empty(self.n_states, dtype=torch.int64)
            # Apply randomization
            # Generate a random permutation of indices 0-7 for each state
            for i in range(self.n_states):
                # self.winner[i] = original_winner
                # Create a random permutation
                permutation = torch.randperm(8, dtype=torch.int64)  # old -> new
                inv_perm = torch.argsort(permutation)               # new -> old
                # Apply the permutation to dice_numbers_scatter
                self.dice_numbers_scatter[i] = self.dice_numbers_scatter[i, :, inv_perm]
                # Apply the permutation to player_ids
                self.player_ids[i] = permutation[self.player_ids[i]]
                # Apply the permutation to active_players
                self.active_players[i] = permutation[self.active_players[i]]
                # Apply the permutation to winner
                self.winner[i] = permutation[original_winner]
        else: 
            self.winner = torch.full((self.n_states,), json_data["winner"].item(), dtype=torch.int64)


    def __len__(self):
        return self.n_states
    
    def __getitem__(self, index):
        # Get the power order for the current state
        desc_power_idx = get_power_order(
            self.dice_numbers_scatter[index], self.active_players[index]
        )
        
        # Reorder dice numbers according to the power order
        dice_numbers_scatter = self.dice_numbers_scatter[index, :, desc_power_idx]
        node_states = torch.cat([self.graph_embed, dice_numbers_scatter], dim=1)

        return {
            # scalar
            "graph_id": self.graph_id,
            # [n_nodes, n_embed + 8 (dice value scatter))]
            "node_states": node_states,
            # [n_nodes]
            "player_ids": self.player_ids[index],
            # [2, n_edges]
            "edges": self.edges,
            # scalar
            "winner": self.winner[index],
            # [8]
            "reindex": desc_power_idx,
            # [8]
            "active_players": self.active_players[index]
        }