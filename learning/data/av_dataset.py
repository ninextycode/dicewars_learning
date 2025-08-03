from torch.utils.data import Dataset
from learning.data.load_utils import load_dicewars_data
from learning.data.process_utils import get_power_order
import torch 


class ActionValueDataset(Dataset):
    def __init__(self, json_data):
        if isinstance(json_data, str):
            json_data = load_dicewars_data(json_data, drop_terminal_state=True)
        self.graph_id = json_data["graph_id"]
        self.nodes_states = json_data["nodes_states"]
        self.attack_edges = json_data["attack_edges"]
        self.end_turn_players = json_data["end_turn_players"]
        self.action_values = json_data["action_values"]
        self.edges = json_data["edges"]
        self.n_states = self.nodes_states.shape[0]
        self.active_players = json_data["active_players"]
    
    def __len__(self):
        return self.n_states
    
    def __getitem__(self, index):
        # [n_players]
        desc_power_idx = get_power_order(
            self.nodes_states[index], self.active_players[index]
        )
        return {
            # scalar
            "graph_id": self.graph_id, 
            # [n_nodes, n_features]
            "nodes_state": self.nodes_states[index, :, desc_power_idx],
            # [2]
            "attack_edge": self.attack_edges[index],
            # scalar
            "action_value": self.action_values[index],
            # [2, n_edges]
            "edges": self.edges,
            # [8]
            "reindex": desc_power_idx
        }
    