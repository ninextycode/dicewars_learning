import json
import torch
import numpy as np


class GameValues:
    win_value = 2000
    lose_value = -2000
    move_value = -1


def add_self_edges(edges):
    n_nodes = torch.max(edges) + 1
    all_nodes = torch.arange(n_nodes)
    start_node = edges[0]
    end_node = edges[1]
    has_self_edge = torch.unique(start_node[start_node == end_node])
    
    has_self_edge_map = torch.zeros(n_nodes, dtype=torch.bool)
    has_self_edge_map[has_self_edge] = True
    nodes_without_self_edge = all_nodes[~has_self_edge_map]
    
    self_edges = torch.vstack([nodes_without_self_edge, nodes_without_self_edge])
    return torch.hstack([edges, self_edges])


def adj_matrix_to_edges(adj_mat, add_self_edges):
    n_nodes = len(adj_mat)
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            is_self_edge = (i == j)
            if adj_mat[i][j] > 0 or (is_self_edge and add_self_edges):
                edges.append([i, j])

    return torch.tensor(edges, dtype=torch.int32).T


def extract_edges(history_data):
    return adj_matrix_to_edges(history_data["adjacency"], add_self_edges=True)


def scatter_dice_values(dice_values, player_ids):
    n_states = dice_values.shape[-2]
    n_nodes = dice_values.shape[-1]
    n_players = 8
    # nodes_states - [n_states, n_nodes, n_players]
    nodes_states = torch.zeros(n_states, n_nodes, n_players, dtype=torch.float32)

    state_idx, node_idx = np.ogrid[:n_states, :n_nodes]
    state_idx = torch.tensor(state_idx)
    node_idx = torch.tensor(node_idx)
    idx = (state_idx, node_idx, player_ids)
    nodes_states[idx] = dice_values.to(torch.float32)
    return nodes_states


def extract_nodes_states(history_data):
    states_data = history_data["states"]
    dice_values = torch.tensor([s["dice"] for s in states_data])
    player_ids = torch.tensor([s["teams"] for s in states_data])

    adj_matrix = history_data["adjacency"]

    n_nodes = len(adj_matrix)
    n_states = len(states_data)
    
    # nodes_states - [n_states, n_nodes, n_players]
    nodes_states = scatter_dice_values(dice_values, player_ids)

    state_idx, node_idx = np.ogrid[:n_states, :n_nodes]
    state_idx = torch.tensor(state_idx)
    node_idx = torch.tensor(node_idx)
    idx = (state_idx, node_idx, player_ids)
    nodes_states[idx] = dice_values.to(torch.float32)

    winner = player_ids[-1].unique().squeeze()
    if winner.ndim != 0:
        raise ValueError("Multiple winners")
    
    return player_ids, nodes_states, winner


def extract_action_values(history_data):
    last_state = history_data["states"][-1]
    action_count = {i: 0 for i in range(8)}
    attack_edges = []
    end_turn_players = []
    action_values = []
    active_players = []

    winning_team = set(last_state["teams"])
    if len(winning_team) > 1:
        raise ValueError("There are multiple winning teams in the last state.")
    winning_team = winning_team.pop()

    for action in history_data["actions"]:
        player_id = action["player"]
        action_count[player_id] += 1

    for action in history_data["actions"]:
        player_id = action["player"] 
        active_players.append(player_id)
        
        if player_id == winning_team:
            player_values = GameValues.win_value
        else:
            player_values = GameValues.lose_value
        player_values += GameValues.move_value * action_count[player_id]
        action_values.append(player_values)

        if action["move_made"]:
            attack_edges.append([action["from"], action["to"]])
        else:
            attack_edges.append([-1, -1])
        
        if action["turn_end"]:
            end_turn_players.append(player_id)
        else:
            end_turn_players.append(-1)

        if action["move_made"] and action["turn_end"]:
            raise ValueError("An action cannot be both a move and an end turn at the same time.")
    
    active_players = torch.tensor(active_players, dtype=torch.int32)
    attack_edges = torch.tensor(attack_edges, dtype=torch.int32)
    end_turn_players = torch.tensor(end_turn_players, dtype=torch.int32)
    action_values = torch.tensor(action_values, dtype=torch.float32)

    return active_players, attack_edges, end_turn_players, action_values


def process_json_data(json_data):
    if isinstance(json_data, str):
        try:
            graph_id = int(json_data.split("_")[1])
        except:
            uint64_limit = torch.iinfo(torch.uint64).max + 1
            graph_id = hash(json_data) % uint64_limit
        graph_id = torch.tensor(graph_id, dtype=torch.uint64)
        json_data = json.load(open(json_data, "r"))
    else:
        graph_id = torch.randint(low=0, high=2**63-1, size=(1,), dtype=torch.uint64)[0]
    return graph_id, json_data


def load_dicewars_data(json_data, drop_terminal_state=True):
    graph_id, json_data = process_json_data(json_data)
    # nodes_states - [n_states, n_nodes, n_players]
    player_ids, nodes_states, winner = extract_nodes_states(json_data)

    if drop_terminal_state:
        nodes_states = nodes_states[:-1]  # drop the last "terminal" state

    edges = extract_edges(json_data)
    active_players, attack_edges, end_turn_players, action_values = \
        extract_action_values(json_data)
    
    adj_matrix = torch.tensor(json_data["adjacency"])
    node_positions = torch.tensor(json_data["node_positions"])

    return dict(
        graph_id=graph_id,
        player_ids=player_ids,
        nodes_states=nodes_states, 
        attack_edges=attack_edges,  
        active_players=active_players,
        end_turn_players=end_turn_players, 
        action_values=action_values, 
        edges=edges,
        adj_matrix=adj_matrix,
        node_positions=node_positions,
        winner=winner
    )
