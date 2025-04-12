"""
This code is used to process an LLM one token at at time.

The Explorer class manages the prompt internally and handles all interactions with the LLM.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
import tomli
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from functools import lru_cache

# Import Gemma3 specific classes
try:
    from transformers import Gemma3ForCausalLM
    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False

class Explorer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", use_bf16=False, enable_layer_prob=False, seed=None):
        """
        Initialize the Explorer with a model name.
        
        Args:
            model_name: Name of the model to load (default "Qwen/Qwen2.5-0.5B")
            use_bf16: Whether to load model in bf16 precision (default False)
            enable_layer_prob: Whether to enable layer probability and correlation calculations (default False)
            seed: Random seed for generation (default None)
        """
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Auto select device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # Create a PyTorch generator with the provided seed for deterministic generation
        if self.seed is not None:
            # MPS doesn't support custom generators, so use CPU for MPS
            generator_device = 'cpu' if self.device.type == 'mps' else self.device
            self.generator = torch.Generator(device=generator_device)
            self.generator.manual_seed(self.seed)
        else:
            self.generator = None
            
        # Load model with bf16 if specified and force eager attention implementation
        if use_bf16:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"  # Force attention weights
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager"  # Force attention weights
            )
        self.model = self.model.to(self.device)
        
        # Initialize with empty prompt
        self.prompt_text = ""
        self.prompt_tokens = []
        
        # Load cache size from config
        try:
            with open("config.toml", "rb") as f:
                config = tomli.load(f)
                self.cache_size = config["display"]["cache_size"]
        except:
            self.cache_size = 100  # Default cache size
            
        # Store layer probability setting
        self.enable_layer_prob = enable_layer_prob
            
        # Constants for token influence calculation
        # Estimate total heads based on model architecture
        try:
            self.TOTAL_HEADS = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        except:
            # Fallback if config structure is different
            self.TOTAL_HEADS = 24  # Default assumption
        
        self.TOP_K_HEADS = min(32, self.TOTAL_HEADS)  # Can't exceed total heads
        
        # Cache for layer probabilities, correlations, and token influence using OrderedDict for FIFO
        # Key: tuple of prompt tokens
        # Value: (layer_probs, token_correlations, token_influences)
        # where layer_probs is list of probability tensors
        # token_correlations is dict mapping token_id to correlation list
        # token_influences is dict mapping token_id to influence list
        self._cache: OrderedDict[Tuple[int, ...], Tuple[List[torch.Tensor], Dict[int, List[float]], Dict[int, List[float]]]] = OrderedDict()
    
    def _manage_cache(self, key: Tuple[int, ...], value: Tuple[List[torch.Tensor], Dict[int, List[float]], Dict[int, List[float]]]):
        """
        Add item to cache, removing oldest if at capacity.
        
        Args:
            key: Cache key (token sequence)
            value: Cache value (layer probabilities and correlations)
        """
        if len(self._cache) >= self.cache_size:
            # Remove oldest item (first item in OrderedDict)
            self._cache.popitem(last=False)
        self._cache[key] = value
        
    def set_prompt(self, prompt_text):
        """
        Set the current prompt text and update the encoded tokens.
        
        Args:
            prompt_text: The prompt text to set
        """
        self.prompt_text = prompt_text
        self.prompt_tokens = self.tokenizer.encode(prompt_text)
        return self
    
    def _calculate_layer_correlation(self, token_id, layer_probs):
        """
        Calculate correlation score for each layer for a given token.
        Uses same approach as score_example.py's toxic_scores, but for a single token.
        Higher score means the layer more strongly predicts this token.
        
        Args:
            token_id: The token ID to calculate correlation for
            layer_probs: List of probability distributions from each layer
            
        Returns:
            List of correlation scores for each layer
        """
        # Calculate correlation as probability of token at each layer
        # This matches score_example.py's approach of summing probabilities
        # for target tokens, but for a single token
        correlations = []
        for layer_prob in layer_probs:
            # Get probability for this token at this layer
            layer_token_prob = layer_prob[token_id].item()
            correlations.append(layer_token_prob)
        
        # Normalize correlations to [0,1] range
        max_corr = max(correlations)
        if max_corr > 0:
            correlations = [c/max_corr for c in correlations]
            
        return correlations

    def get_prompt_token_probabilities(self):
        """
        Calculate the probability of each token in the sequence given its preceding context,
        using a single forward pass.
        
        Args:
            self: The Explorer object
        Returns:
            list: A list of probabilities for each token in the sequence
        """
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get the model's output in a single forward pass, using the generator if available
        with torch.no_grad():
            if self.generator is not None:
                outputs = self.model(input_ids, generation_config={"generator": self.generator})
            else:
                outputs = self.model(input_ids)
            logits = outputs.logits[0]  # Shape: [sequence_length, vocab_size]
        
        # Calculate probabilities for each position
        token_probabilities = []
        
        # First token has no context, so we'll use None or some default
        token_probabilities.append(0.5)
        
        # For each position after the first
        for pos in range(len(self.prompt_tokens) - 1):
            # The logits at position 'pos' predict the token at position 'pos+1'
            position_logits = logits[pos]
            position_probs = torch.softmax(position_logits, dim=-1)
            
            # Convert to float32 for probability calculation
            position_probs_float = position_probs.to(torch.float32)
            
            # Get probability of the actual next token
            next_token_id = self.prompt_tokens[pos + 1]
            next_token_prob = position_probs_float[next_token_id].item()
            
            token_probabilities.append(next_token_prob)
        return token_probabilities
    
    def get_prompt_token_normalized_entropies(self):
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get the model's output in a single forward pass, using the generator if available
        with torch.no_grad():
            if self.generator is not None:
                outputs = self.model(input_ids, generation_config={"generator": self.generator})
            else:
                outputs = self.model(input_ids)
            logits = outputs.logits[0]  # Shape: [sequence_length, vocab_size]
        
        # Calculate normalized entropy for each position
        normalized_entropies = []
        
        # First token has no context, so we'll use None or some default
        normalized_entropies.append(0.5)
        
        # For each position after the first
        for pos in range(len(self.prompt_tokens) - 1):
            # The logits at position 'pos' predict the token at position 'pos+1'
            position_logits = logits[pos]
            position_probs = torch.softmax(position_logits, dim=-1)
            
            # Calculate entropy: -sum(p * log(p))
            # Convert to float32 for entropy calculation
            position_probs_float = position_probs.to(torch.float32)
            # We filter out zeros to avoid log(0) issues
            probs_np = position_probs_float.cpu().numpy()
            non_zero_probs = probs_np[probs_np > 0]
            entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
            
            # Normalize by maximum possible entropy (log2 of vocabulary size)
            max_entropy = np.log2(len(position_probs))
            normalized_entropy = entropy / max_entropy
            
            normalized_entropies.append(normalized_entropy)
        
        return normalized_entropies
        
    def get_local_token_bias(self):
        """
        Computes cosine similarities between each token's hidden state and the hidden state
        of the last token (the next-token predictor).
        
        Returns:
            List of cosine similarities for each token in the prompt.
        """
        if not self.prompt_tokens:
            return []
            
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get model output with hidden states, using the generator if available
        with torch.no_grad():
            if self.generator is not None:
                outputs = self.model(input_ids, output_hidden_states=True, generation_config={"generator": self.generator})
            else:
                outputs = self.model(input_ids, output_hidden_states=True)
            
        # Get the last layer's hidden states. Shape: (seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1][0]
        
        # Get the hidden state of the last token (used for next-token prediction)
        final_state = hidden_states[-1]
        
        # Compute cosine similarity for each token with the final token
        cosine_similarities = []
        for token_state in hidden_states:
            # Calculate cosine similarity
            dot_product = torch.sum(token_state * final_state)
            norm_token = torch.norm(token_state)
            norm_final = torch.norm(final_state)
            
            # Avoid division by zero
            if norm_token > 0 and norm_final > 0:
                similarity = dot_product / (norm_token * norm_final)
                cosine_similarities.append(similarity.item())
            else:
                cosine_similarities.append(0.0)
        
        return cosine_similarities
        
    def get_token_energies(self, T=1.0):
        """
        Computes Helmholtz free energy for each token in the prompt.
        Higher energy values typically indicate out-of-distribution tokens.
        
        This implementation is optimized to:
        1. Use a single forward pass for all tokens
        2. Cache results for better performance when toggling visualization modes
        
        Args:
            T: Temperature parameter (default: 1.0)
            
        Returns:
            List of energy values for each token in the prompt.
        """
        if not self.prompt_tokens:
            return []
        
        # Check if we have a cached result
        cache_key = tuple(self.prompt_tokens)
        cache_attr = f"_energy_cache_{T}"
        
        # Check if we have this specific energy calculation cached
        if hasattr(self, cache_attr) and cache_key in getattr(self, cache_attr):
            return getattr(self, cache_attr)[cache_key]
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
            
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get the model's output in a single forward pass, using the generator if available
        with torch.no_grad():
            if self.generator is not None:
                outputs = self.model(input_ids, generation_config={"generator": self.generator})
            else:
                outputs = self.model(input_ids)
            logits = outputs.logits[0]  # Shape: [sequence_length, vocab_size]
        
        # Calculate energies for each position
        energies = []
        
        # First token has no context, so we'll use a default value
        energies.append(0.0)
        
        # For each position after the first
        for pos in range(len(self.prompt_tokens) - 1):
            # The logits at position 'pos' predict the token at position 'pos+1'
            position_logits = logits[pos]
            
            # Calculate energy: -T * logsumexp(logits / T)
            energy = -T * torch.logsumexp(position_logits / T, dim=-1)
            
            # Get the energy value
            token_energy = energy.item()
            energies.append(token_energy)
        
        # Cache the result
        getattr(self, cache_attr)[cache_key] = energies
        
        return energies

    def get_prompt(self):
        """
        Get the current prompt text.
        
        Returns:
            The current prompt text
        """
        return self.prompt_text
    
    def get_prompt_tokens(self):
        """
        Get the current encoded prompt tokens.
        
        Returns:
            List of token ids representing the current prompt
        """
        return self.prompt_tokens
    
    def get_prompt_tokens_strings(self):
        """
        Get the current prompt tokens as a string.
        Returns a list of strings with special tokens (newline, tab, etc.) made visible.
        """
        return [self._format_special_token(self.tokenizer.decode(token)) for token in self.prompt_tokens]
    
    def pop_token(self):
        """
        Remove and return the last token from the prompt tokens.
        If the prompt is empty, return None.
        
        Returns:
            The removed token id, or None if prompt was empty
        """
        if not self.prompt_tokens:
            return None
            
        # Pop last token and update prompt text
        last_token = self.prompt_tokens.pop()
        self.prompt_text = self.tokenizer.decode(self.prompt_tokens)
        return last_token
    
    def append_token(self, token_id):
        """
        Append a token to the current prompt tokens and update prompt text.
        
        Args:
            token_id: The token id to append
        """
        # Add token to prompt tokens
        self.prompt_tokens.append(token_id)
        
        # Update prompt text to match new tokens
        self.prompt_text = self.tokenizer.decode(self.prompt_tokens)
        return self
    
    def _format_special_token(self, token):
        """
        Format special/invisible tokens into readable strings.
        
        Args:
            token: The token string to format
        Returns:
            Formatted string with special tokens made visible
        """
        # Common special tokens mapping
        special_tokens = {
            '\n': '\\n',  # newline
            '\t': '\\t',  # tab
            '\r': '\\r',  # carriage return
            ' ': '\\s',   # space
        }
        
        # If token is in special_tokens, return its visible representation
        if token in special_tokens:
            return special_tokens[token]
        return token

    def get_token_influence(self, token_id: int) -> List[float]:
        """
        Calculate the influence of each token in the prompt on the given token.
        Uses cached values if available.
        
        Args:
            token_id: The token ID to calculate influence for
            
        Returns:
            List of influence scores for each token in the prompt
        """
        if not self.prompt_tokens:
            return []
        
        # Check cache first
        cache_key = tuple(self.prompt_tokens)
        if cache_key in self._cache:
            # Cache hit for this prompt
            _, _, token_influences = self._cache[cache_key]
            if token_id in token_influences:
                # Cache hit for this token
                return token_influences[token_id]
        else:
            # Cache miss for this prompt, initialize empty cache entry
            token_influences = {}
            
        # Calculate influence scores
        # Convert token IDs to tensor and create input
        input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
        
        # Get model output with attention, using the generator if available
        with torch.no_grad():
            if self.generator is not None:
                outputs = self.model(input_ids, output_attentions=True, generation_config={"generator": self.generator})
            else:
                outputs = self.model(input_ids, output_attentions=True)
            
        # Stack attention from all layers [num_layers, batch, heads, seq, seq]
        attentions = torch.stack(outputs.attentions)
        
        # Get dimensions
        num_layers, _, num_heads, seq_len, _ = attentions.shape
        
        # Calculate head importance via attention variance
        head_variance = attentions.var(dim=-1).mean(dim=(0, 3))  # [layers, heads]
        flattened_variance = head_variance.flatten()
        
        # Safety check for TOP_K_HEADS
        valid_k = min(self.TOP_K_HEADS, len(flattened_variance))
        top_heads = torch.topk(flattened_variance, valid_k).indices
        
        # Filter and aggregate attention
        filtered_attentions = []
        for idx in top_heads:
            layer_idx = idx // num_heads
            head_idx = idx % num_heads
            filtered_attentions.append(attentions[layer_idx, :, head_idx])
        
        # Aggregate attention for the last token
        if filtered_attentions:
            aggregated = torch.stack(filtered_attentions).mean(dim=0)[0, -1]  # [seq_len]
            
            # Get the influence scores - convert to float32 first to avoid BFloat16 error
            influence_scores = aggregated.to(torch.float32).cpu().numpy()
            
            # Normalize to [0,1] range
            if influence_scores.max() > 0:
                influence_scores = influence_scores / influence_scores.max()
                
            result = influence_scores.tolist()
        else:
            # Return zeros if no filtered attentions
            result = [0.0] * len(self.prompt_tokens)
        
        # Cache the result
        token_influences[token_id] = result
        
        # Update cache if needed
        if cache_key in self._cache:
            layer_probs, token_correlations, _ = self._cache.pop(cache_key)
            self._cache[cache_key] = (layer_probs, token_correlations, token_influences)
        
        return result
    
    def calculate_z_scores(self, next_token_logits):
        """
        Calculate z-scores for all tokens based on the provided logits.
        
        Args:
            next_token_logits: Logits tensor for next token prediction
            
        Returns:
            Tensor of z-scores for all tokens
        """
        # Compute mean and std over entire vocabulary
        mean = next_token_logits.mean().item()
        std = next_token_logits.std().item()
        
        # Calculate z-score for each token
        z_scores = (next_token_logits - mean) / (std + 1e-12)
        
        return z_scores
    
    def get_top_n_tokens(self, n=5, search=""):
        """
        Get the top n most likely next tokens given the current prompt.
        Optionally filter tokens by a search string.
        
        Args:
            n: Number of top tokens to return (default 5)
            search: Optional string to filter tokens (default "")
            
        Returns:
            List of dicts containing token info and probabilities, sorted by probability
        """
        # Check cache first
        cache_key = tuple(self.prompt_tokens)
        layer_probs = []
        token_correlations = {}
        token_influences = {}
        
        if cache_key in self._cache:
            # Cache hit - Move accessed item to end (most recently used)
            print(f"Cache hit for token sequence of length {len(cache_key)}")
            layer_probs, token_correlations, token_influences = self._cache.pop(cache_key)
            self._cache[cache_key] = (layer_probs, token_correlations, token_influences)
            next_token_probs = layer_probs[-1] if layer_probs else None  # Final layer probabilities
        
        # If not in cache or layer_probs is empty (when enable_layer_prob was previously False)
        if not cache_key in self._cache or not layer_probs:
            # Convert token IDs to tensor and create input
            input_ids = torch.tensor([self.prompt_tokens]).to(self.device)
            
            # Determine if we need hidden states based on layer_prob setting
            output_hidden_states = self.enable_layer_prob
            
            # Get model output, using the generator if available for deterministic generation
            with torch.no_grad():
                if output_hidden_states:
                    if self.generator is not None:
                        outputs = self.model(input_ids, output_hidden_states=True, generation_config={"generator": self.generator})
                    else:
                        outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                else:
                    if self.generator is not None:
                        outputs = self.model(input_ids, generation_config={"generator": self.generator})
                    else:
                        outputs = self.model(input_ids)
                
            # Get logits for next token prediction
            next_token_logits = outputs.logits[0, -1, :]
            
            # Calculate z-scores
            z_scores = self.calculate_z_scores(next_token_logits)
            
            # Get probabilities for final layer using softmax
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0).to(torch.float32)
            
            # Calculate per-layer probabilities only if enabled
            if self.enable_layer_prob:
                # Skip first element as it's the embeddings, not a layer output
                layer_probs = []
                for layer_output in hidden_states[1:]:
                    # Get last token's hidden state
                    last_hidden = layer_output[0, -1, :]
                    # Project to vocab size using model's lm_head
                    layer_logits = self.model.lm_head(last_hidden.unsqueeze(0)).squeeze(0)
                    # Get probabilities
                    layer_probs.append(torch.nn.functional.softmax(layer_logits, dim=0).to(torch.float32))
                
                # Cache miss - Store in cache
                print(f"Cache miss for token sequence of length {len(cache_key)}")
                self._manage_cache(cache_key, (layer_probs, token_correlations, token_influences))
            else:
                # If layer probability calculation is disabled, use empty placeholders
                # This ensures the returned token objects have the expected structure
                layer_probs = []
        
        # Create placeholder values for when layer_prob is disabled
        empty_layer_probs = [0.0]  # Single value placeholder
        empty_correlations = [0.0]  # Single value placeholder
        
        if search:
            # Filter tokens that contain the search string
            matching_tokens = []
            for idx, prob in enumerate(next_token_probs):
                token = self._format_special_token(self.tokenizer.decode(idx))
                if search.lower() in token.lower():
                    # Only calculate correlations if layer_prob is enabled
                    if self.enable_layer_prob:
                        # Get correlations from cache or calculate and cache them
                        if idx not in token_correlations:
                            correlations = self._calculate_layer_correlation(idx, layer_probs)
                            token_correlations[idx] = correlations
                        else:
                            correlations = token_correlations[idx]
                        
                        # Calculate token influence
                        influence = self.get_token_influence(idx)
                    else:
                        # Use placeholder values when disabled
                        correlations = empty_correlations
                        influence = [0.0] * len(self.prompt_tokens) if self.prompt_tokens else []
                    
                    # Create token object with appropriate values
                    token_obj = {
                        "token_id": idx,
                        "token": token,
                        "probability": prob.item(),
                        "z_score": z_scores[idx].item(),
                        "token_influence": influence
                    }
                    
                    # Add layer-specific data only if enabled
                    if self.enable_layer_prob and layer_probs:
                        token_obj["layer_probs"] = [layer_prob[idx].item() for layer_prob in layer_probs]
                        token_obj["layer_correlations"] = correlations
                    else:
                        # Use placeholder values when disabled
                        token_obj["layer_probs"] = empty_layer_probs
                        token_obj["layer_correlations"] = empty_correlations
                        
                    matching_tokens.append(token_obj)
            
            # Sort by probability and take top n
            matching_tokens.sort(key=lambda x: x["probability"], reverse=True)
            return matching_tokens[:n]
        else:
            # Original behavior for no search string
            top_probs, top_indices = torch.topk(next_token_probs, n)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                token = self._format_special_token(self.tokenizer.decode(idx))
                
                # Only calculate correlations if layer_prob is enabled
                if self.enable_layer_prob:
                    # Get correlations from cache or calculate and cache them
                    if idx not in token_correlations:
                        correlations = self._calculate_layer_correlation(idx, layer_probs)
                        token_correlations[idx] = correlations
                    else:
                        correlations = token_correlations[idx]
                    
                    # Calculate token influence
                    influence = self.get_token_influence(idx)
                else:
                    # Use placeholder values when disabled
                    correlations = empty_correlations
                    influence = [0.0] * len(self.prompt_tokens) if self.prompt_tokens else []
                
                # Create token object with appropriate values
                token_obj = {
                    "token": token,
                    "token_id": idx.item(),
                    "probability": prob.item(),
                    "z_score": z_scores[idx].item(),
                    "token_influence": influence
                }
                
                # Add layer-specific data only if enabled
                if self.enable_layer_prob and layer_probs:
                    token_obj["layer_probs"] = [layer_prob[idx].item() for layer_prob in layer_probs]
                    token_obj["layer_correlations"] = correlations
                else:
                    # Use placeholder values when disabled
                    token_obj["layer_probs"] = empty_layer_probs
                    token_obj["layer_correlations"] = empty_correlations
                    
                results.append(token_obj)
                
            return results


# Example usage
if __name__ == "__main__":
    explorer = Explorer()
    explorer.set_prompt("Once upon a time, there was a")
    
    print("Prompt:", explorer.get_prompt())
    print("Encoded prompt:", explorer.get_prompt_tokens())
    print("-----")
    print("Top tokens:", explorer.get_top_n_tokens())
    print("-----")
    print("Filtered tokens:", explorer.get_top_n_tokens(search="man"))
    print("-----")
    print("Appending token:", explorer.get_top_n_tokens(search="man")[0])
    explorer.append_token(explorer.get_top_n_tokens(search="man")[0]["token_id"])
    print("-----")
    print("Prompt:", explorer.get_prompt())
    print("Encoded prompt:", explorer.get_prompt_tokens())
    print("-----")
    print("Popping token:", explorer.pop_token())
    print("-----")
    print("Prompt:", explorer.get_prompt()) 
    print("Token probabilities:", explorer.get_prompt_token_probabilities())
    print("-----")
    print("Token entropies:", explorer.get_prompt_token_normalized_entropies())
