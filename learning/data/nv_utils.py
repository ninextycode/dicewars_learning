import torch


def collate_nv_data(sample_list):
    """
    Collate function that combines multiple node value data samples into a single batch.
    
    Node indices are shifted to become the merged super-graph indices, allowing
    batch processing of multiple graphs as a single large graph.
    
    Parameters:
        sample_list (list): List of dictionaries containing individual graph data samples
        
    Returns:
        dict: Combined batch with the following keys:
            - nodes_states: Combined node states - graph positional embeddings and numbers
            - player_ids: Combined player IDs
            - node_graph_ids: Graph ID for each node
            - edge_graph_ids: Graph ID for each edge
            - all_edges: Combined edge indices with adjusted offsets
            - winners: Winner information for each graph
    """
    node_states = []
    player_ids = []
    all_edges = []
    per_graph_edges = []
    node_graph_ids = []
    edge_graph_ids = []
    winners = []
    
    node_id_offset = 0
    for i, sample in enumerate(sample_list):
        node_states.append(sample["node_states"])
        player_ids.append(sample["player_ids"])
        
        all_edges.append(sample["edges"] + node_id_offset)
        per_graph_edges.append(sample["edges"])
        
        n_nodes = sample["node_states"].shape[0]
        n_edges = sample["edges"].shape[1]
        
        node_graph_ids.append(torch.full((n_nodes,), i, dtype=torch.int32))
        edge_graph_ids.append(torch.full((n_edges,), i, dtype=torch.int32))
        
        winners.append(sample["winner"])
        
        node_id_offset += n_nodes

    node_states = torch.cat(node_states)
    player_ids = torch.cat(player_ids)
    all_edges = torch.cat(all_edges, dim=1)
    node_graph_ids = torch.cat(node_graph_ids)
    edge_graph_ids = torch.cat(edge_graph_ids)
    winners = torch.stack(winners)

    return dict(
        nodes_states=node_states,
        player_ids=player_ids,
        node_graph_ids=node_graph_ids,
        edge_graph_ids=edge_graph_ids,
        all_edges=all_edges,
        per_graph_edges=per_graph_edges,
        winners=winners
    )


def extract_input(batch, with_dice_scatter=True):
    """
    Extract input tensors needed for model forward pass.
    
    Parameters:
        batch (dict): Batch dictionary from collate_nv_data
        with_dice_scatter (bool): Whether to include dice scatter features (default: True)
        
    Returns:
        tuple: Contains the following elements:
            - nodes_states: Combined features for all nodes
            - player_ids: Player, index 0-7, that controls the corresponding node
            - all_edges: Combined edge indices with adjusted offsets
            - node_graph_ids: Graph ID for each node
            - edge_graph_ids: Graph ID for each edge
    """
    # Create node features by combining dice numbers and player IDs
    nodes_states = batch["nodes_states"]
    
    # If not using dice scatter, remove the last 8 features
    if not with_dice_scatter and nodes_states.shape[1] > 11:
        nodes_states = nodes_states[:, :-8]
        
    player_ids = batch["player_ids"]
    
    all_edges = batch["all_edges"]
    node_graph_ids = batch["node_graph_ids"]
    edge_graph_ids = batch["edge_graph_ids"]
    
    return nodes_states, player_ids, all_edges, node_graph_ids, edge_graph_ids


def extract_output_target(batch, model_output):
    """
    Extract output and target values for loss computation.
    
    Parameters:
        batch (dict): Batch dictionary from collate_nv_data
        model_output (dict): Output from model forward pass
        
    Returns:
        tuple: Contains the following elements:
            - output: Model predictions, logits for graphs, length 8
            - target: Ground truth targets, values 0-7
    """
    # [n_graphs]
    winners = batch["winners"]
    
    return model_output, winners
