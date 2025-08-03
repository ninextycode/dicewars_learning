from torch.utils.data import Dataset
from learning.data.load_utils import load_dicewars_data
from learning.data.process_utils import get_power_order
import torch 



class DWStateFlowDataset(Dataset):
    def __init__(self, json_data):
        if isinstance(json_data, str):
            json_data = load_dicewars_data(json_data, drop_terminal_state=False)
        self.graph_id = json_data["graph_id"]
        self.nodes_states = json_data["nodes_states"][:-1]
        self.attack_edges = json_data["attack_edges"]
        self.active_players = json_data["active_players"]
        self.edges = json_data["edges"]
        self.n_states = self.nodes_states.shape[0]
        # find index where there are non-zero forces in a terminal state - this is the winner
        terminal_state_sum = torch.sum(json_data["nodes_states"][-1], dim=0)
        self.winner = torch.where(terminal_state_sum > 0)[0].item()
        

    def __len__(self):
        return self.n_states
    
    def __getitem__(self, index):
        # [n_players]
        desc_power_idx = get_power_order(
            self.nodes_states[index], self.active_players[index]
        )

        winner = torch.where(desc_power_idx == self.winner)[0]
        
        if len(winner) != 1:
            raise ValueError("Winner index is not unique")

        return {
            # scalar
            "graph_id": self.graph_id, 
            # [n_nodes, n_players]
            "nodes_state": self.nodes_states[index, :, desc_power_idx],
            # [2], has value -1 if no attack but end of turn
            "attack_edge": self.attack_edges[index],
            # [2, n_edges]
            "edges": self.edges,
            # [8]
            "reindex": desc_power_idx,
            # scalar - make sure the index is being re-mapped to point to the winner after re-indexing
            "winner": winner
        }