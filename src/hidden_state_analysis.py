"""
This module provides analysis of hidden states across layers.
"""
import torch
import torch.nn.functional as F

class HiddenStateAnalyzer:
    def __init__(self, model):
        self.model = model

    def get_hidden_states(self, input_ids):
        """
        Get hidden states from all layers of the model.
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states

    def analyze_token_evolution(self, input_ids, token_position):
        """
        Analyze how a token's representation evolves across layers by
        calculating the cosine similarity of its hidden state at each
        layer with its hidden state at the final layer.
        """
        if token_position is None:
            return []

        hidden_states = self.get_hidden_states(input_ids)
        final_layer_state = hidden_states[-1][0, token_position]

        similarities = []
        for layer_state in hidden_states:
            current_layer_state = layer_state[0, token_position]
            similarity = F.cosine_similarity(final_layer_state, current_layer_state, dim=0)
            similarities.append(similarity.item())
            
        return similarities
