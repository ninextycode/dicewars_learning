import glob
import torch
import os
import logging
from learning.data.load_utils import adj_matrix_to_edges, scatter_dice_values
from learning.data.process_utils import get_adj_matrix_embeddings
from learning.data.process_utils import get_active_edges_mask


def add_winners_info(history_data):
    if "winner" in history_data \
        and "players_out" in history_data \
        and "player_out_which_state" in history_data:
        return

    players_alive = [set(s["teams"]) for s in history_data["states"]]
    players_out = []
    player_out_which_state = []

    for i_next_state, (pa_state, pa_next_state) in enumerate(zip(players_alive[:-1], players_alive[1:]), 1):
        diff = pa_state - pa_next_state
        if len(diff) > 0:
            players_out.extend(diff)
            player_out_which_state.extend([i_next_state] * len(diff))

    winner = pa_next_state.pop()

    # Create a dictionary with the winner info
    winner_info = {}
    if "winner" not in history_data:
        winner_info["winner"] = winner
    if "players_out" not in history_data:
        winner_info["players_out"] = players_out
    if "player_out_which_state" not in history_data:
        winner_info["player_out_which_state"] = player_out_which_state
    
    # Update history_data to have winner info at the front
    if winner_info:
        temp_data = history_data.copy()
        history_data.clear()
        history_data.update(winner_info)
        history_data.update(temp_data)


def load_latest_model_state(model_name):
    model_state = None
    latest_epoch = -1

    models_dir = f"learning/{model_name}_checkpoints" 

    # Find the latest checkpoint file (with highest epoch number)
    checkpoint_files = glob.glob(os.path.join(models_dir, f"{model_name}_*.pt"))

    # Extract epoch numbers from filenames
    epoch_nums = [int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files]
    latest_epoch = max(epoch_nums)
    latest_checkpoint = os.path.join(models_dir, f"{model_name}_{latest_epoch:06}.pt")
    logging.info(f"Loading latest checkpoint: {latest_checkpoint} (epoch {latest_epoch})")
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
    model_state = checkpoint["model_state"]
    
    return model_state


def extract_input(game_state_data):    
    field_state_data = game_state_data["state"]
    dice_values = torch.tensor(field_state_data["dice"]).unsqueeze(0)
    player_ids = torch.tensor(field_state_data["teams"]).unsqueeze(0)
    # the function expects multiple graph states for the whole game
    # we only send the current one - with top level dimension added - then removed
    # to match function expected input shape
    scattered_dice_values = scatter_dice_values(dice_values, player_ids)[0]
    
    adj_matrix = torch.tensor(game_state_data["adjacency"])
    adj_graph_embed = get_adj_matrix_embeddings(adj_matrix)
    node_positions = torch.tensor(game_state_data["node_positions"])
    graph_embed = torch.cat(
        [adj_graph_embed, node_positions], dim=1
    )

    node_states = torch.cat([graph_embed, scattered_dice_values], dim=1)
    edges  = adj_matrix_to_edges(adj_matrix, add_self_edges=True)
    node_graph_ids = torch.zeros(node_states.shape[0], dtype=torch.int32)
    edges_graph_ids = torch.zeros(edges.shape[1], dtype=torch.int32)

    return node_states, edges, node_graph_ids, edges_graph_ids


def extract_action(model_output, node_states, edges):
    end_turn_val = model_output[0]["end_turn_val"]
    edge_attack_val = model_output[0]["edge_attack_val"]
    active_edges_mask = get_active_edges_mask(node_states, edges, active_player=-8)

    if len(edge_attack_val) > 0 \
        and edge_attack_val.max() > end_turn_val:

        active_edges = edges[:, active_edges_mask]
        attack_edge = active_edges[:, torch.argmax(edge_attack_val)]
        return attack_edge.cpu().tolist()
    else:
        return [-1, -1]
