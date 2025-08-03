import torch
from torch import nn


class DWNorm(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, x):
        # downscale dice on a node
        return x / 8
    

class DWDeNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # downscale dice on a node
        return x * 8
    

def get_active_edges_mask(game_node_states, edges, active_player=0):
    active_player_nodes = torch.where(game_node_states[:, active_player] > 0)[0]
    active_edges_map = torch.isin(edges[0], active_player_nodes) & ~torch.isin(edges[1], active_player_nodes)
    return active_edges_map


def get_power_order(nodes_state, active_player):
    # reindex players by their power, active player at the front
    player_power = torch.sum(nodes_state, dim=0)
    # make sure active player goes to index 0
    player_power[active_player] = torch.max(player_power) + 1
    # [n_players]
    desc_power_idx = torch.argsort(player_power, descending=True)
    
    return desc_power_idx


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        # non_blocking lets the copy overlap with compute if source is pinned
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    else:
        return obj       
        