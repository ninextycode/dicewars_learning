import torch
from torch import nn
from learning.model.utils import indexed_softmax

# https://arxiv.org/pdf/1710.10903

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
    def __init__(
            self, 
            in_features, 
            out_features,
            n_output_heads=1,
            concatenate_outputs=True,
            p_msg_dropout=0.35
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.message_heads = nn.ModuleList([
            GATMessage(in_features, out_features, p_msg_dropout)
            for _ in range(n_output_heads)
        ])

        self.attention_coef_heads = nn.ModuleList([
            GATAttentionCoefficients(2 * out_features)
            for _ in range(n_output_heads)
        ])

        self.attention_dropout = nn.Dropout(p=0.35)
        self.activation_fn = nn.ReLU()

        self.concatenate_outputs = concatenate_outputs
        if self.concatenate_outputs:
            full_out_features = out_features * n_output_heads
        else:
            full_out_features = out_features
        self.in_states_pass = nn.Linear(in_features, full_out_features, bias=False)
        torch.nn.init.eye_(self.in_states_pass.weight) 
        # Exclude residual projection layer from learning
        # it should merely change input shape to be added to output
        self.in_states_pass.weight.requires_grad = False
        self.residual_dropout = nn.Dropout(p=0.35)
        self.passthrough_coef = nn.Parameter(torch.tensor(-1.0))

        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters of sub-modules
        for head in self.message_heads:
            head.reset_parameters()
        for head in self.attention_coef_heads:
            head.reset_parameters()
        # Initialize passthrough coefficient
        nn.init.constant_(self.passthrough_coef, -1.0)

    def forward(self, in_states, edges):
        # when multiple graphs are batched together into one, this function can avoid using explicit graph_id for nodes
        # because the nodes in different graphs are never each other neighbors 
        # in_states has dimensions [n_nodes, in_features]
        n_nodes = in_states.shape[0]

        messages = []
        for message_head in self.message_heads:
            messages.append(message_head(in_states))
        # messages has dimensions [n_nodes, n_heads, out_features]
        messages = torch.stack(messages, dim=1) 

        # m_from / m_to have dimensions [n_edges, n_heads, out_features]
        msg_from = messages[edges[0], ...]
        msg_to = messages[edges[1], ...]
        # m_stack have dimensions [n_edges, n_heads, 2 * out_features]
        msg_stack = torch.cat([msg_from, msg_to], dim=-1)

        attention_coeffs = []
        for i, attention_head in enumerate(self.attention_coef_heads):
            # output has 1-d dimension [n_edges]
            attention_coeffs.append(attention_head(msg_stack[:, i, :]))
        # attention_coeffs has dimensions [n_edges, n_heads]
        attention_coeffs = torch.stack(attention_coeffs, dim=1)

        # softmax_attention_coef has dimensions [n_edges, n_heads]
        softmax_attention_coef = indexed_softmax(
            attention_coeffs, edges[1], indexed_dim=0
        )
        softmax_attention_coef = self.attention_dropout(softmax_attention_coef)
        
        # edges[0] - source/input end of the edge
        # softmax_attention_coef[:, :, torch.newaxis] - [n_edges, n_heads, 1]
        # scaled_input_messages - [n_edges, n_heads, out_features]
        scaled_input_messages = \
            softmax_attention_coef[:, :, torch.newaxis] * msg_from

        # merged_msgs - [n_nodes, n_heads, out_features]
        merged_msgs = torch.zeros_like(messages)
        # scattered inputs (n_edges) are pooled into (n_nodes) outputs 
        merged_msgs.index_add_(
            dim=0,
            index=edges[1],  # edges[1] - target/output end of the edge
            source=scaled_input_messages
        )

        # old_states_pass - [n_nodes, out_features / out_features * n_heads]
        old_states_pass = self.in_states_pass(in_states) * torch.sigmoid(self.passthrough_coef)

        if not self.concatenate_outputs:
            # average head outputs
            # merged_msgs - [n_nodes, out_features]
            merged_msgs = torch.mean(merged_msgs, dim=0)

        # new_states_residual - [n_nodes, n_heads, out_features]
        new_states_residual = self.activation_fn(merged_msgs)
        new_states_residual = self.residual_dropout(new_states_residual)

        if self.concatenate_outputs:
            #  [n_nodes, n_heads, out_features] -> [n_nodes, (n_heads * out_features)]
            new_states_residual = new_states_residual.reshape((n_nodes, -1))
        
        new_states = old_states_pass + new_states_residual
        return new_states


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