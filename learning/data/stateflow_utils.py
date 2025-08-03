import torch
from learning.data.process_utils import get_active_edges_mask


def collate_stateflow_data(sample_list):
    """
    Collate function that combines multiple graph states into a single graph state for stateflow model
    
    Node indices are shifted to become the merged super-graph indices, allowing
    batch processing of multiple graphs as a single large graph.
    
    Parameters:
        sample_list (list): List of dictionaries containing individual graph data samples
        
    Returns:
        dict: Combined batch with the following keys:
            - nodes_states: Combined node features - vectors of dice numbers, 
                            where indices correspond to players
            - shifted_attack_edges: Attack edge indices with adjusted offsets, 
                                    where value -1 corresponds to end of turn instead of an attack
            - original_attack_edges: Original attack edge indices (without node offset adjustment),
                                    preserving the original graph structure
            - node_graph_ids: ordinal graph index for each node, indicating which sample/graph it belongs to
            - edge_graph_ids: ordinal graph index for each edge, indicating which sample/graph it belongs to
            - all_edges: Combined edge indices with adjusted offsets for the merged super-graph
            - all_edges_original: Combined edge indices without adjusted offsets
            - graph_edges: List of original edge indices for each graph (without offset adjustment),
                              preserving individual graph connectivity
            - graph_reindexes: Re-indexing applied to node states to achieve the order
                            where the first feature corresponds to active player,
                            others - to players in descending order of their # of dice
            - winners: Indices of the final winning player of the game whose state was sampled
                        winner's index corresponds to the power-descending order, just like nodes_states
            - graph_file_id: Id of the corresponding json data file
    """
    # graph data collate function that combines multiple graph staters into a single graph state
    # node indices are shifted to become the merged super-graph indices
    nodes_states = [] 
    shifted_attack_edges = []
    original_attack_edges = []
    all_edges = []
    all_edges_original = []
    graph_edges = []
    node_graph_ids = []
    edge_graph_ids = []
    graph_reindexes = []
    graph_file_ids = []
    winners = []
    
    node_id_offset = 0
    for i, sample in enumerate(sample_list):
        nodes_states.append(sample["nodes_state"])
        winners.append(sample["winner"])
        # Store the original attack edge before shifting
        original_attack_edges.append(sample["attack_edge"].clone())
        # Apply offset for the shifted attack edge
        sample_attack_edge = sample["attack_edge"].clone() 
        if sample_attack_edge[0] != -1:
            sample_attack_edge += node_id_offset
        shifted_attack_edges.append(sample_attack_edge)
        all_edges.append(sample["edges"] + node_id_offset)
        all_edges_original.append(sample["edges"])
        graph_edges.append(sample["edges"])
        n_nodes = sample["nodes_state"].shape[0]
        n_edges = sample["edges"].shape[1]
        node_graph_ids.append(torch.full((n_nodes,), i, dtype=torch.int32))
        edge_graph_ids.append(torch.full((n_edges,), i, dtype=torch.int32))
        graph_reindexes.append(sample["reindex"])
        graph_file_ids.append(sample["graph_id"])  # Store the graph ID
        node_id_offset += n_nodes

    nodes_states = torch.cat(nodes_states)
    shifted_attack_edges = torch.stack(shifted_attack_edges)
    original_attack_edges = torch.stack(original_attack_edges)
    all_edges = torch.cat(all_edges, dim=1)
    all_edges_original = torch.cat(all_edges_original, dim=1)
    node_graph_ids = torch.cat(node_graph_ids)
    edge_graph_ids = torch.cat(edge_graph_ids)
    winners = torch.cat(winners)

    return dict(
        nodes_states=nodes_states,
        shifted_attack_edges=shifted_attack_edges,
        original_attack_edges=original_attack_edges,
        node_graph_ids=node_graph_ids,
        edge_graph_ids=edge_graph_ids,
        all_edges=all_edges,
        all_edges_original=all_edges_original,
        graph_edges=graph_edges,
        graph_reindexes=graph_reindexes,
        graph_file_ids=graph_file_ids,
        winners=winners
    )


def extract_input(batch):
    nodes_states = batch["nodes_states"]
    all_edges = batch["all_edges"]
    node_graph_ids = batch["node_graph_ids"]
    edge_graph_ids = batch["edge_graph_ids"]
    return nodes_states, all_edges, node_graph_ids, edge_graph_ids 


def extract_winners_output_target(batch, model_output):
    target_winners = batch["winners"]
    output_winners_logits = torch.stack([
        graph_output_dict["winner_logit"]
        for graph_output_dict in model_output
    ])
    return output_winners_logits, target_winners


def extract_action_output_target(batch, model_output):
    nodes_states = batch["nodes_states"]
    node_graph_ids = batch["node_graph_ids"]

    all_attack_edges = batch["original_attack_edges"]
    per_graph_edges = batch["graph_edges"]

    output = []
    target = []

    for graph_i, graph_output_dict in enumerate(model_output):
        graph_node_states = nodes_states[graph_i == node_graph_ids]
        graph_edges = per_graph_edges[graph_i]
        active_edges_mask = get_active_edges_mask(graph_node_states, graph_edges)
        active_edges = graph_edges[:, active_edges_mask]

        output_edges_attack_logit = graph_output_dict["edges_attack_logit"]
        output_end_turn_logit = graph_output_dict["end_turn_logit"] 
        per_graph_output = torch.cat([output_edges_attack_logit, output_end_turn_logit])
        output.append(per_graph_output)

        graph_attack_edge = all_attack_edges[graph_i]

        action_idx = None

        if graph_attack_edge[0] != -1:  # attack
            attack_edge_idx = torch.where(torch.all(
                active_edges == graph_attack_edge[..., torch.newaxis], dim=0
            ))[0]
            if len(attack_edge_idx) != 1:
                raise ValueError(
                    f"Attack edge {graph_attack_edge}"
                    f" not found for graph_i = {graph_i}"
                )
            action_idx = attack_edge_idx[0]  
        else: # end turn
            # end turn action index is the last index after all active edges
            action_idx = torch.tensor(
                active_edges.shape[1], 
                dtype=torch.int64, device=per_graph_output.device
            )
                                      
        target.append(action_idx)
    
    # the cross-entropy loss should be applied 
    return output, target
