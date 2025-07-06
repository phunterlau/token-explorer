from dataclasses import dataclass
import torch
from huggingface_hub import hf_hub_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SAE:
    """Holds the loaded SAE model and its metadata."""
    model: torch.nn.Module
    layer: int
    repo_id: str
    filename: str

class SAEManager:
    """Manages loading and interacting with Sparse Autoencoders (SAEs)."""

    def __init__(self, sae_configs):
        """
        Initializes the manager with a list of SAE configurations.
        
        Args:
            sae_configs (list): A list of dictionaries, where each dictionary
                                contains 'layer', 'repo_id', and 'filename'.
        """
        self.sae_configs = sae_configs
        self.saes = {}  # Maps layer_index -> SAE object

    def load_saes(self):
        """
        Iterates through configs, downloads models from Hugging Face Hub,
        and populates the saes dictionary. This is called once at startup.
        """
        if not self.sae_configs:
            logger.info("No SAE configurations found. Skipping SAE loading.")
            return

        logger.info(f"Loading {len(self.sae_configs)} SAEs...")
        for config in self.sae_configs:
            layer = config.get('layer')
            repo_id = config.get('repo_id')
            filename = config.get('filename')
            revision = config.get('revision', 'main')  # Default to 'main' branch

            if not all([layer is not None, repo_id, filename]):
                logger.warning(f"Skipping invalid SAE config: {config}")
                continue

            try:
                logger.info(f"Loading SAE for layer {layer} from {repo_id}/{filename} (revision: {revision})...")
                model_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
                
                # Handle both .pt and .safetensors files
                if filename.endswith('.safetensors'):
                    from safetensors import safe_open
                    sae_model = {}
                    with safe_open(model_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            sae_model[key] = f.get_tensor(key)
                else:
                    sae_model = torch.load(model_path)
                
                self.saes[layer] = SAE(
                    model=sae_model,
                    layer=layer,
                    repo_id=repo_id,
                    filename=filename
                )
                logger.info(f"Successfully loaded SAE for layer {layer}.")
            except Exception as e:
                logger.error(f"Failed to load SAE for layer {layer} from {repo_id}: {e}")

    def get_feature_activations(self, layer_index, hidden_state):
        """
        Takes a hidden state, finds the correct SAE for the layer, runs the
        forward pass, and returns a sorted list of top-activating features.
        
        Args:
            layer_index (int): The layer to get feature activations from.
            hidden_state (torch.Tensor): The hidden state from the model.
            
        Returns:
            list: A sorted list of (feature_id, activation_value) tuples,
                  or an empty list if no SAE is found for the layer.
        """
        sae_obj = self.saes.get(layer_index)
        if not sae_obj:
            return []

        try:
            with torch.no_grad():
                # Handle different SAE formats
                if hasattr(sae_obj.model, 'encode'):
                    # If it's a proper SAE object with encode method
                    feature_activations = sae_obj.model.encode(hidden_state)
                elif isinstance(sae_obj.model, dict):
                    # If it's a dictionary of tensors (from safetensors)
                    # Assume standard SAE structure: W_enc, b_enc for encoding
                    if 'W_enc' in sae_obj.model and 'b_enc' in sae_obj.model:
                        W_enc = sae_obj.model['W_enc']
                        b_enc = sae_obj.model['b_enc']
                        
                        # Debug: Log tensor shapes
                        logger.debug(f"Layer {layer_index} - Hidden state shape: {hidden_state.shape}")
                        logger.debug(f"Layer {layer_index} - W_enc shape: {W_enc.shape}")
                        logger.debug(f"Layer {layer_index} - b_enc shape: {b_enc.shape}")
                        
                        # Ensure hidden_state is 2D [batch_size, hidden_dim]
                        if hidden_state.dim() == 1:
                            hidden_state = hidden_state.unsqueeze(0)
                        
                        # Check if dimensions match for matrix multiplication
                        if hidden_state.shape[-1] != W_enc.shape[1]:
                            logger.error(f"Layer {layer_index} - Dimension mismatch: hidden_state {hidden_state.shape} vs W_enc {W_enc.shape}")
                            return []
                        
                        # Move SAE tensors to the same device as the hidden state
                        W_enc = W_enc.to(hidden_state.device)
                        b_enc = b_enc.to(hidden_state.device)
                        
                        # Standard SAE encoding: ReLU(x @ W_enc.T + b_enc)
                        feature_activations = torch.relu(hidden_state @ W_enc.T + b_enc)
                    else:
                        logger.warning(f"Unknown SAE structure for layer {layer_index}. Available keys: {list(sae_obj.model.keys())}")
                        return []
                else:
                    logger.warning(f"Unsupported SAE model type for layer {layer_index}: {type(sae_obj.model)}")
                    return []
                
                # Get the top activating features
                # We'll sort by activation value in descending order
                if feature_activations.dim() == 1:
                    feature_activations = feature_activations.unsqueeze(0)
                
                sorted_indices = torch.argsort(feature_activations[0], descending=True)
                
                # Create a list of (feature_id, activation_value)
                activations = []
                for idx in sorted_indices:
                    activation_value = feature_activations[0, idx].item()
                    if activation_value > 0: # Only show positive activations
                        activations.append((idx.item(), activation_value))
            
            return activations
        except Exception as e:
            logger.error(f"Error computing feature activations for layer {layer_index}: {e}")
            return []

    def get_available_layers(self):
        """Returns a sorted list of layers that have a loaded SAE."""
        return sorted(self.saes.keys())
