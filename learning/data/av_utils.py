import torch
from learning.data.process_utils import get_active_edges_mask


def collate_av_data(sample_list):
    """
    Collate function that combines multiple graph states into a single graph state.
    
    Node indices are shifted to become the merged super-graph indices, allowing
    batch processing of multiple graphs as a single large graph.
    
    Parameters:
        sample_list (list): List of dictionaries containing individual graph data samples
        
    Returns:
        dict: Combined batch with the following keys:
            - nodes_states: Combined node features
            - attack_edges: Attack edge indices with adjusted offsets
            - original_attack_edges: Original attack edge indices (un-shifted)
            - action_values: Action values
            - node_graph_ids: Graph ID for each node
            - edge_graph_ids: Graph ID for each edge
            - all_edges: Combined edge indices with adjusted offsets
            - graph_reindexes: Re-indexing applied to node states to achieve the order
                                where the first feature corresponds to active player,
                                others - to players in descending order of their # of dice
    """
    # graph data collate function that combines multiple graph staters into a single graph state
    # node indices are shifted to become the merged super-graph indices
    nodes_states = []
    shifted_attack_edges = []
    original_attack_edges = []
    action_values = []
    all_edges = []
    per_graph_edges = []
    node_graph_ids = []
    edge_graph_ids = []
    graph_reindexes = []
    
    node_id_offset = 0
    for i, sample in enumerate(sample_list):
        nodes_states.append(sample["nodes_state"])
        
        # Store the original attack edge before shifting
        original_attack_edges.append(sample["attack_edge"].clone())
        
        # Apply offset for the shifted attack edge
        sample_attack_edge = sample["attack_edge"].clone() 
        if sample_attack_edge[0] != -1:
            sample_attack_edge += node_id_offset
        shifted_attack_edges.append(sample_attack_edge)
        
        action_values.append(sample["action_value"])
        all_edges.append(sample["edges"] + node_id_offset)
        per_graph_edges.append(sample["edges"])
        n_nodes = sample["nodes_state"].shape[0]
        n_edges = sample["edges"].shape[1]
        node_graph_ids.append(torch.full((n_nodes,), i, dtype=torch.int32))
        edge_graph_ids.append(torch.full((n_edges,), i, dtype=torch.int32))
        node_id_offset += n_nodes
        graph_reindexes.append(sample["reindex"])

    nodes_states = torch.cat(nodes_states)
    shifted_attack_edges = torch.stack(shifted_attack_edges)
    original_attack_edges = torch.stack(original_attack_edges)
    action_values = torch.tensor(action_values, dtype=torch.float32)
    all_edges = torch.cat(all_edges, dim=1)
    node_graph_ids = torch.cat(node_graph_ids)
    edge_graph_ids = torch.cat(edge_graph_ids)

    return dict(
        nodes_states=nodes_states,
        shifted_attack_edges=shifted_attack_edges,
        original_attack_edges=original_attack_edges,
        action_values=action_values,
        node_graph_ids=node_graph_ids,
        edge_graph_ids=edge_graph_ids,
        all_edges=all_edges,
        per_graph_edges=per_graph_edges,
        graph_reindexes=graph_reindexes
    )


def extract_input(batch):
    nodes_states = batch["nodes_states"]
    all_edges = batch["all_edges"]
    node_graph_ids = batch["node_graph_ids"]
    edge_graph_ids = batch["edge_graph_ids"]
    return nodes_states, all_edges, node_graph_ids, edge_graph_ids 


def extract_output_target(batch, model_output):
    nodes_states = batch["nodes_states"]
    node_graph_ids = batch["node_graph_ids"]
    all_attack_edges = batch["original_attack_edges"]
    action_values = batch["action_values"]
    per_graph_edges = batch["per_graph_edges"]
    output = []
    target = []

    for graph_i, graph_output in enumerate(model_output):
        end_value = graph_output["end_turn_val"]
        attack_edge_value = graph_output["edge_attack_val"]

        graph_node_states = nodes_states[graph_i == node_graph_ids]
        graph_edges = per_graph_edges[graph_i]
        active_edges_mask = get_active_edges_mask(graph_node_states, graph_edges)
        active_edges = graph_edges[:, active_edges_mask]
        
        graph_attack_edge = all_attack_edges[graph_i]

        if graph_attack_edge[0] != -1:  # attack
            attack_edge_idx = torch.where(torch.all(
                active_edges == graph_attack_edge[..., torch.newaxis], dim=0
            ))[0]
            if len(attack_edge_idx) != 1:
                raise ValueError(
                    f"Attack edge {graph_attack_edge}"
                    f" not found for graph_i = {graph_i}"
                )
            predicted_value = attack_edge_value[attack_edge_idx][0]
        else: # end turn
            # action = "end turn"
            predicted_value = end_value     

        output.append(predicted_value)
        target.append(action_values[graph_i])
    
    return torch.stack(output), torch.stack(target)
