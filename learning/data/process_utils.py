import torch
import tqdm


def get_active_edges_mask(game_node_states, edges, active_player=0):
    # active player node with 1 dice cannot attack, should be considered inactive
    active_player_nodes = torch.where(game_node_states[:, active_player] > 1)[0]
    other_players_nodes = torch.where(game_node_states[:, active_player] == 0)[0]
    active_edges_map = torch.isin(edges[0], active_player_nodes) \
        & torch.isin(edges[1], other_players_nodes)
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


def calculate_mean(train_loader):
    # Calculate statistics properly with variable-sized batches
    sum_of_values = 0.0
    count_of_values = 0
    feature_sums = None
    batch_count = 0

    for batch in tqdm.tqdm(train_loader, desc="Calculating average node states"):
        current_states = batch["nodes_states"]
        
        # Add to running statistics
        sum_of_values += current_states.sum().item()
        count_of_values += current_states.shape[0]
        
        # For per-feature statistics
        if feature_sums is None:
            # Initialize with zeros matching the feature dimension
            feature_sums = torch.zeros(current_states.shape[1], dtype=torch.float32)
        
        # Sum across batch dimension for feature-wise statistics
        feature_sums += current_states.sum(dim=0)
        
        batch_count += 1

    # Calculate the averages
    feature_averages = feature_sums / count_of_values 

    print(f"Statistics calculated over {batch_count} batches")
    print(f"Shape of feature dimension: {feature_sums.shape}")
    print(f"Feature-wise averages:")
    for i, avg in enumerate(feature_averages):
        print(f"  Feature {i}: {avg.item():.6f}")
    return feature_averages


def calculate_std(train_loader, feature_averages):
    # Calculate statistics for standard deviation with variable-sized batches
    sum_squared_diff = 0.0
    count_of_values = 0
    feature_sum_squared_diff = None
    batch_count = 0

    for batch in tqdm.tqdm(train_loader, desc="Calculating std dev of node states"):
        current_states = batch["nodes_states"]
        
        # Add to running statistics
        sum_squared_diff += ((current_states - feature_averages) ** 2).sum().item()
        count_of_values += current_states.shape[0]
        
        # For per-feature statistics
        if feature_sum_squared_diff is None:
            # Initialize with zeros matching the feature dimension
            feature_sum_squared_diff = torch.zeros(current_states.shape[1], dtype=torch.float32)
        
        # Sum squared differences across batch dimension for feature-wise statistics
        feature_sum_squared_diff += ((current_states - feature_averages) ** 2).sum(dim=0)
        
        batch_count += 1

    # Calculate the standard deviations
    overall_std = torch.sqrt(torch.tensor(sum_squared_diff / count_of_values))
    feature_std = torch.sqrt(feature_sum_squared_diff / count_of_values)

    print(f"Standard deviation statistics calculated over {batch_count} batches")
    print(f"Overall standard deviation: {overall_std.item():.6f}")
    print(f"Feature-wise standard deviations:")
    for i, std in enumerate(feature_std):
        print(f"  Feature {i}: {std.item():.6f}")
    return feature_std



def get_adj_matrix_embeddings(adj_matrix, n_lap_eg = 4, rw_steps = (2,3,5,8)):
    deg = torch.sum(adj_matrix, dim=1).to(torch.float32)
    deg_inv_sqrt = deg.rsqrt()

    L_sym = torch.eye(adj_matrix.shape[0]) \
        - deg_inv_sqrt[:, torch.newaxis] * adj_matrix * deg_inv_sqrt[torch.newaxis, :]

    eg_val, eg_vec = torch.linalg.eigh(L_sym)
    # drop zero eigenvalue
    if eg_val[0] < 1e-6:
        eg_val = eg_val[1:]
        eg_vec = eg_vec[:, 1:]

    ev_node_embedding = eg_vec[:, :n_lap_eg]
    # eigenvalue ambiguity is resolved by making the largest component positive
    max_idx = torch.argmax(torch.abs(ev_node_embedding), dim=0)
    # Get the sign of the largest component
    sign = torch.sign(ev_node_embedding[max_idx, torch.arange(n_lap_eg)])
    ev_node_embedding = ev_node_embedding * sign[torch.newaxis, :]

    rw_prop_step = 1 / deg[:, torch.newaxis] * adj_matrix
    rw_node_embedding = torch.empty(
        (adj_matrix.shape[0], len(rw_steps)),
        dtype=rw_prop_step.dtype
    )
    for i, s in enumerate(rw_steps):
        rw_prob_s_steps = torch.linalg.matrix_power(rw_prop_step, s)
        rw_node_embedding[:, i] = torch.diag(rw_prob_s_steps)

    return torch.cat([
        deg.unsqueeze(1),
        ev_node_embedding,
        rw_node_embedding
    ], dim=1)