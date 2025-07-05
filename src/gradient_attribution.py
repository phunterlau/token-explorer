"""
Gradient-Based Attribution Analysis

This module implements various gradient-based attribution methods to understand
which input tokens contribute most to the model's predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union


class GradientAttribution:
    """
    Implements gradient-based attribution methods for transformer models.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the GradientAttribution analyzer.
        
        Args:
            model: The transformer model
            tokenizer: The tokenizer associated with the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def _get_embeddings(self, input_ids):
        """
        Get the input embeddings for the given token IDs.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            
        Returns:
            Embeddings tensor [batch_size, seq_len, hidden_size]
        """
        # Get embeddings from the model's embedding layer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            # For models like Llama, Qwen
            embeddings = self.model.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            # For GPT-style models
            embeddings = self.model.transformer.wte(input_ids)
        elif hasattr(self.model, 'embeddings'):
            # For BERT-style models
            embeddings = self.model.embeddings.word_embeddings(input_ids)
        else:
            # Fallback: try to find embedding layer
            for name, module in self.model.named_modules():
                if 'embed' in name.lower() and isinstance(module, torch.nn.Embedding):
                    embeddings = module(input_ids)
                    break
            else:
                raise ValueError("Could not find embedding layer in the model")
        
        return embeddings
    
    def calculate_saliency(self, input_ids, target_token_id):
        """
        Calculate saliency maps using gradient of output logit w.r.t. input embeddings.
        
        Args:
            input_ids: List of token IDs representing the input sequence
            target_token_id: The token ID to calculate attribution for
            
        Returns:
            List of attribution scores for each input token
        """
        # Convert to tensor if needed
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids]).to(self.device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Get embeddings and enable gradients
        embeddings = self._get_embeddings(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Ensure gradients are retained
        
        # Forward pass with custom embeddings
        outputs = self._forward_with_embeddings(embeddings)
        
        # Get logits for the last position (next token prediction)
        logits = outputs.logits[0, -1, :]
        target_logit = logits[target_token_id]
        
        # Backward pass
        target_logit.backward()
        
        # Get gradients and calculate attribution scores
        if embeddings.grad is None:
            # Fallback: return zeros if gradients are not available
            return [0.0] * len(input_ids[0])
        gradients = embeddings.grad[0]  # [seq_len, hidden_size]
        
        # Calculate L2 norm of gradients for each token
        attribution_scores = torch.norm(gradients, dim=1).float().detach().cpu().numpy()
        
        # Normalize to [0, 1] range
        if attribution_scores.max() > 0:
            attribution_scores = attribution_scores / attribution_scores.max()
        
        return attribution_scores.tolist()
    
    def calculate_integrated_gradients(self, input_ids, target_token_id, baseline="zero", n_steps=50):
        """
        Calculate Integrated Gradients attribution.
        
        Args:
            input_ids: List of token IDs representing the input sequence
            target_token_id: The token ID to calculate attribution for
            baseline: Baseline for integration ("zero" or "pad")
            n_steps: Number of integration steps
            
        Returns:
            List of attribution scores for each input token
        """
        # Convert to tensor if needed
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids]).to(self.device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Get original embeddings
        original_embeddings = self._get_embeddings(input_ids)
        
        # Create baseline embeddings
        if baseline == "zero":
            baseline_embeddings = torch.zeros_like(original_embeddings)
        elif baseline == "pad":
            pad_token_id = self.tokenizer.pad_token_id or 0
            pad_ids = torch.full_like(input_ids, pad_token_id)
            baseline_embeddings = self._get_embeddings(pad_ids)
        else:
            raise ValueError("Baseline must be 'zero' or 'pad'")
        
        # Calculate integrated gradients
        integrated_gradients = torch.zeros_like(original_embeddings)
        
        for step in range(n_steps):
            # Interpolate between baseline and original
            alpha = step / (n_steps - 1) if n_steps > 1 else 1.0
            interpolated_embeddings = baseline_embeddings + alpha * (original_embeddings - baseline_embeddings)
            interpolated_embeddings.requires_grad_(True)
            
            # Forward pass
            outputs = self._forward_with_embeddings(interpolated_embeddings)
            logits = outputs.logits[0, -1, :]
            target_logit = logits[target_token_id]
            
            # Backward pass
            if interpolated_embeddings.grad is not None:
                interpolated_embeddings.grad.zero_()
            target_logit.backward(retain_graph=True)
            
            # Accumulate gradients
            integrated_gradients += interpolated_embeddings.grad
        
        # Average the gradients and multiply by input difference
        integrated_gradients = integrated_gradients / n_steps
        attribution = integrated_gradients * (original_embeddings - baseline_embeddings)
        
        # Calculate L2 norm for each token
        attribution_scores = torch.norm(attribution[0], dim=1).float().detach().cpu().numpy()
        
        # Normalize to [0, 1] range
        if attribution_scores.max() > 0:
            attribution_scores = attribution_scores / attribution_scores.max()
        
        return attribution_scores.tolist()
    
    def calculate_input_x_gradient(self, input_ids, target_token_id):
        """
        Calculate Input × Gradient attribution.
        
        Args:
            input_ids: List of token IDs representing the input sequence
            target_token_id: The token ID to calculate attribution for
            
        Returns:
            List of attribution scores for each input token
        """
        # Convert to tensor if needed
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids]).to(self.device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Get embeddings and enable gradients
        embeddings = self._get_embeddings(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Ensure gradients are retained
        
        # Forward pass
        outputs = self._forward_with_embeddings(embeddings)
        logits = outputs.logits[0, -1, :]
        target_logit = logits[target_token_id]
        
        # Backward pass
        target_logit.backward()
        
        # Get gradients and calculate attribution scores
        if embeddings.grad is None:
            # Fallback: return zeros if gradients are not available
            return [0.0] * len(input_ids[0])
        
        # Calculate input × gradient
        input_x_grad = embeddings[0] * embeddings.grad[0]
        
        # Calculate L2 norm for each token
        attribution_scores = torch.norm(input_x_grad, dim=1).float().detach().cpu().numpy()
        
        # Normalize to [0, 1] range
        if attribution_scores.max() > 0:
            attribution_scores = attribution_scores / attribution_scores.max()
        
        return attribution_scores.tolist()
    
    def _forward_with_embeddings(self, embeddings):
        """
        Forward pass using custom embeddings instead of token IDs.
        
        Args:
            embeddings: Custom embeddings tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Model outputs
        """
        # This is a bit tricky as we need to bypass the embedding layer
        # We'll use a hook to replace the embeddings
        
        batch_size, seq_len = embeddings.shape[:2]
        
        # Create dummy input_ids (they won't be used due to our hook)
        dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        
        # Store the custom embeddings
        self._custom_embeddings = embeddings
        
        # Register a hook to replace embeddings
        def embedding_hook(module, input, output):
            return self._custom_embeddings
        
        # Find and hook the embedding layer
        embedding_layer = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embedding_layer = self.model.model.embed_tokens
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            embedding_layer = self.model.transformer.wte
        elif hasattr(self.model, 'embeddings'):
            embedding_layer = self.model.embeddings.word_embeddings
        else:
            # Fallback: find first embedding layer
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    embedding_layer = module
                    break
        
        if embedding_layer is None:
            raise ValueError("Could not find embedding layer to hook")
        
        # Register hook and run forward pass
        hook = embedding_layer.register_forward_hook(embedding_hook)
        try:
            outputs = self.model(dummy_input_ids)
        finally:
            hook.remove()
            delattr(self, '_custom_embeddings')
        
        return outputs
