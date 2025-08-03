import torch
from torch import nn


# adding bias in intermediate layers is an unnecessary complication
# same shift is a message will be reduced by softmax exponent ratio

class GATMessage(nn.Module):
    def __init__(self, in_features, out_features, p_dropout=0.35):
        super().__init__()
        self.input_dropout = nn.Dropout(p_dropout)
        self.transform_fn = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.transform_fn.reset_parameters()
    
    def forward(self, x):
        x = self.transform_fn(self.input_dropout(x))
        return x
    

class GATAttentionCoefficients(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.transform_fn = nn.Linear(n_features, 1, bias=False)
        self.activation_fn = nn.LeakyReLU()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.transform_fn.reset_parameters()

    def forward(self, x):
        # x has dimensions [n_inputs, n_features]
        x = self.transform_fn(x)
        x = self.activation_fn(x)
        # output has dimensions [n_inputs]
        x = x.squeeze(-1)
        return x
    

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, p_msg_dropout=0.35):
        # edges = 2 x |E| numpy array
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.message_fn = GATMessage(in_features, out_features, p_msg_dropout)
        self.attention_coef_fn = GATAttentionCoefficients(2 * out_features)
        self.attention_dropout = nn.Dropout(p=0.35)
        self.activation_fn = nn.ReLU()

        self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.eye_(self.residual_proj.weight) 
        # Exclude residual projection layer from learning
        # it should merely change input shape to be added to output
        self.residual_proj.weight.requires_grad = False
        self.residual_dropout = nn.Dropout(p=0.35)
        self.passthrough_coef = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters of sub-modules
        self.message_fn.reset_parameters()
        self.attention_coef_fn.reset_parameters()
        # Initialize passthrough coefficient
        nn.init.constant_(self.passthrough_coef, 1.0)

    def forward(self, in_states, edges):
        # when multiple graphs are batched together into one, this function can avoid using explicit graph_id for nodes
        # because the nodes in different graphs are never each other neighbors 
        # in_states has dimensions [n_nodes, in_features]
        n_nodes = in_states.shape[0]    
        messages = self.message_fn(in_states)
        # messages has dimensions [n_edges, out_features]
        m_from = messages[edges[0], :]
        m_to = messages[edges[1], :]
        # m_stack has dimensions [n_edges, 2 * out_features]
        m_stack = torch.cat([m_from, m_to], dim=-1)
        # attention_coef has dimensions [n_edges]
        attention_coef = self.attention_coef_fn(m_stack)
        
        softmax_attention_coef = indexed_softmax(
            attention_coef, edges[1]
        )
        softmax_attention_coef = self.attention_dropout(softmax_attention_coef)
        
        # edges[0] - source/input end of the edge
        # scaled_messages_along_input_edges - [n_edges, out_features]
        scaled_messages_along_input_edges = \
            softmax_attention_coef[..., torch.newaxis] * messages[edges[0]]

        # merged_msgs - [n_nodes, out_features]
        merged_msgs = torch.zeros_like(messages)
        # (n_edges) are scattered into (n_nodes) 
        merged_msgs.index_add_(
            dim=0,
            index=edges[1],  # edges[1] - target/output end of the edge
            source=scaled_messages_along_input_edges
        )

        # merged_msgs = torch.empty(
        #     n_nodes, self.out_features,
        #     dtype=messages.dtype, device=messages.device
        # )
        # for i_node in range(n_nodes):
        #     in_edge_map_idx = edges[1] == i_node
        #     neighbors = edges[0][in_edge_map_idx]
        #     # neighb_msgs - [n_nbh, out_f]
        #     neighb_msgs = messages[neighbors, :]
        #     # neighb_att - [n_nbh]
        #     neighb_att_coef = attention_coef[in_edge_map_idx]
        #     neighb_att = nn.functional.softmax(neighb_att_coef, dim=0)
        #     neighb_att = self.attention_dropout(neighb_att)
        #     # broadcast attention coefficients over the last dim = same over all instance features
        #     # merged_msg - [out_f]
        #     merged_msg = torch.sum(neighb_msgs * neighb_att[..., torch.newaxis], dim=0)
        #     merged_msgs[i_node, :] = merged_msg

        new_states_res = self.activation_fn(merged_msgs)
        new_states_res = self.residual_dropout(new_states_res)
        old_states_passthrough = self.residual_proj(in_states) * torch.sigmoid(self.passthrough_coef)
        new_states = old_states_passthrough + new_states_res
        return new_states


def indexed_softmax(values: torch.Tensor, indices: torch.Tensor):
    # values - [n_items]
    # sorted_indices - [n_items] 
    n_indices = indices.max() + 1  
    # max_per_index - [n_indices]
    max_per_index = torch.full(
        (n_indices,), -float('inf'),
        device=values.device,
        dtype=values.dtype
    )

    max_per_index.scatter_reduce_(
        dim=0,
        index=indices,
        src=values,
        reduce="amax"
    )
    values_exp = torch.exp(values - max_per_index[indices])

    sum_per_index = torch.zeros(
        size=max_per_index.shape,
        dtype=values_exp.dtype,
        device=values_exp.device
    )
    sum_per_index.scatter_reduce_(
        dim=0,
        index=indices,
        src=values_exp,
        reduce="sum"
    )
    values_softmax = values_exp / sum_per_index[indices]
    # values_softmax - [n_items]
    # groups corresponding to indices are mapped to their softmax map
    return values_softmax


class GATGraphSummary(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.input_dropout = nn.Dropout(p=0.35)
        self.attention_coef_fn = GATAttentionCoefficients(in_features)
        self.attention_dropout = nn.Dropout(p=0.2)
        self.summary_fn = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters of sub-modules
        self.attention_coef_fn.reset_parameters()
        # Use Glorot (Xavier) uniform initialization for the summary function
        nn.init.xavier_uniform_(self.summary_fn.weight)
        if self.summary_fn.bias is not None:
            nn.init.zeros_(self.summary_fn.bias)

    def forward(self, in_states: torch.Tensor, graph_ids: torch.Tensor):
        """
        Compute graph summary using attention mechanism.
        
        Parameters:
            in_states (torch.Tensor): Node features with shape [n_nodes, in_features]
            graph_ids (torch.Tensor): Graph IDs for each node with shape [n_nodes]
            
        Returns:
            torch.Tensor: Graph summaries with shape [n_graphs, out_features]
        """
        n_graphs = graph_ids.max() + 1
        # att_coeffs/attention - [n_nodes]
        in_states = self.input_dropout(in_states)
        att_coeffs = self.attention_coef_fn(in_states)
        attention = indexed_softmax(att_coeffs, graph_ids)
        attention = self.attention_dropout(attention)
        in_states_weighted = in_states * attention[..., torch.newaxis]

        # per_graph_sum - [n_graphs, in_features]
        per_graph_sum = torch.zeros(
            n_graphs, self.in_features,
            device=in_states_weighted.device,
            dtype=in_states_weighted.dtype
        )
        per_graph_sum.index_add_(0, graph_ids, in_states_weighted)

        # summary - [n_graphs, out_features]
        summary = self.summary_fn(per_graph_sum)
        return summary
    

class GATEdgeValue(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.edge_value_fn = nn.Linear(2 * in_features, out_features)
        self.edge_state_dropout = nn.Dropout(0.35)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Use Glorot (Xavier) uniform initialization for weights
        nn.init.xavier_uniform_(self.edge_value_fn.weight)
        if self.edge_value_fn.bias is not None:
            nn.init.zeros_(self.edge_value_fn.bias)

    def forward(self, in_states: torch.Tensor, edges: torch.Tensor):
        """
        Compute edge values based on connected node features.
        
        Parameters:
            in_states (torch.Tensor): Node features with shape [n_nodes, in_features]
            edges (torch.Tensor): Edge indices with shape [2, n_edges]
            
        Returns:
            torch.Tensor: Edge values with shape [n_edges, out_features]
        """
        state_from = in_states[edges[0], :]
        state_to = in_states[edges[1], :]
        # edge_state - [n_edges, 2*in_features]
        edge_state = torch.cat([state_from, state_to], dim=-1) 
        edge_state = self.edge_state_dropout(edge_state)
        # value - [n_edges, out_features]
        value = self.edge_value_fn(edge_state)
        return value