"""
UI Data Adapter - Converts Explorer data to UI-ready format.

This module separates UI formatting logic from core analysis logic,
making the code more maintainable and testable.
"""

from src.utils import entropy_to_color, probability_to_color


class UIDataAdapter:
    """Converts Explorer data to UI-ready format"""
    
    def __init__(self, explorer):
        self.explorer = explorer
    
    def format_special_token(self, token):
        """
        Format special/invisible tokens into readable strings.
        Moved from Explorer class to separate UI concerns.
        
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

    def get_prompt_tokens_display(self):
        """Get formatted token strings for display"""
        tokens = self.explorer.get_prompt_tokens()
        return [self.format_special_token(self.explorer.tokenizer.decode(token)) 
                for token in tokens]
    
    def prob_to_color(self, prob, max_prob):
        """Convert probability to a color (red to blue)"""
        # Scale probability by max value (no extra scaling factor)
        scaled_prob = min(prob / max_prob, 1.0)
        # Red (low) to blue (high)
        return f"#{int(255 * (1 - scaled_prob)):02x}00{int(255 * scaled_prob):02x}"

    def get_max_layer_value(self, tokens, layer_mode):
        """Get maximum layer value across all tokens"""
        if layer_mode == "corr":
            return max(max(token["layer_correlations"]) for token in tokens)
        elif layer_mode == "prob":
            return max(max(token["layer_probs"]) for token in tokens)
        return 0

    def layer_values_to_heatmap(self, token, max_val, layer_mode):
        """Convert layer values to a heatmap string"""
        if layer_mode == "none":
            return ""
            
        blocks = []
        values = token["layer_correlations"] if layer_mode == "corr" else token["layer_probs"]
        for val in values:
            color = self.prob_to_color(val, max_val)
            blocks.append(f"[on {color}] ")
        # Combine all blocks into a single string
        return "".join(blocks)

    def get_table_rows(self, tokens, show_z_scores, layer_mode):
        """Convert token data to table format"""
        # Get global max value for consistent scaling
        max_val = self.get_max_layer_value(tokens, layer_mode)
        
        # Include layers column only if layer display is enabled
        headers = ["token_id", "token", "prob"]
        if show_z_scores:
            headers.append("z-score")
        if layer_mode != "none":
            headers.append("layers")
        
        rows = []
        for token in tokens:
            row = [
                token["token_id"],
                token["token"],
                f"{token['probability']:.4f}"
            ]
            if show_z_scores:
                row.append(f"{token['z_score']:.4f}")
            if layer_mode != "none":
                row.append(self.layer_values_to_heatmap(token, max_val, layer_mode))
            rows.append(tuple(row))
        
        return [tuple(headers)] + rows
    
    def influence_to_color(self, influence):
        """Convert influence score to a color (green to purple)"""
        # Green (low) to purple (high)
        return f"#{0:02x}{int(255 * (1 - influence)):02x}{int(255 * influence):02x}"
        
    def bias_to_color(self, bias):
        """Convert local token bias score to a color (orange to blue)"""
        # Orange (low) to blue (high)
        scaled_bias = min(max(bias, 0.0), 1.0)  # Ensure bias is in [0,1]
        return f"#{int(255 * (1 - scaled_bias)):02x}{int(128 * (1 - scaled_bias)):02x}{int(255 * scaled_bias):02x}"
        
    def energy_to_color(self, energy, min_energy, max_energy):
        """Convert energy score to a color (green to red)
        
        Lower energy (green) = in-distribution
        Higher energy (red) = out-of-distribution
        """
        # Normalize energy to [0,1] range
        if max_energy == min_energy:
            normalized = 0.5
        else:
            normalized = (energy - min_energy) / (max_energy - min_energy)
            
        # Green (low energy, in-distribution) to red (high energy, out-of-distribution)
        return f"#{int(255 * normalized):02x}{int(255 * (1 - normalized)):02x}00"
    
    def render_entropy_display(self, cursor_position=None):
        """Render entropy visualization with optional cursor"""
        entropy_legend = "".join([
            f"[on {entropy_to_color(i/10)}] {i/10:.2f} [/on]"
            for i in range(11)
        ])
        prompt_legend = f"[bold]Token entropy:[/bold]{entropy_legend}"
        
        token_entropies = self.explorer.get_prompt_token_normalized_entropies()
        token_strings = self.get_prompt_tokens_display()
        
        # Build prompt text with cursor indicator
        prompt_parts = []
        for i, (token, entropy) in enumerate(zip(token_strings, token_entropies)):
            if cursor_position is not None and i == cursor_position:
                # Add cursor brackets around current token - escape brackets to prevent markup conflicts
                prompt_parts.append(f"[on {entropy_to_color(entropy)}]\\[{token}\\][/on]")
            else:
                prompt_parts.append(f"[on {entropy_to_color(entropy)}]{token}[/on]")
        
        prompt_text = "".join(prompt_parts)
        
        return prompt_text, prompt_legend
    
    def render_probability_display(self, cursor_position=None):
        """Render probability visualization with optional cursor"""
        prob_legend = "".join([
            f"[on {probability_to_color(i/10)}] {i/10:.2f} [/on]"
            for i in range(11)
        ])
        prompt_legend = f"[bold]Token prob:[/bold]{prob_legend}"
        
        token_probs = self.explorer.get_prompt_token_probabilities()
        token_strings = self.get_prompt_tokens_display()
        
        # Build prompt text with cursor indicator
        prompt_parts = []
        for i, (token, prob) in enumerate(zip(token_strings, token_probs)):
            if cursor_position is not None and i == cursor_position:
                # Add cursor brackets around current token - escape brackets to prevent markup conflicts
                prompt_parts.append(f"[on {probability_to_color(prob)}]\\[{token}\\][/on]")
            else:
                prompt_parts.append(f"[on {probability_to_color(prob)}]{token}[/on]")
        
        prompt_text = "".join(prompt_parts)
        
        return prompt_text, prompt_legend
    
    def render_influence_display(self, cursor_position=None):
        """Render influence visualization with optional cursor"""
        # Get the top token to analyze influence
        top_tokens = self.explorer.get_top_n_tokens(n=1)
        if top_tokens:
            top_token = top_tokens[0]
            influence_scores = top_token["token_influence"]
            
            # Create influence legend
            influence_legend = "".join([
                f"[on {self.influence_to_color(i/10)}] {i/10:.2f} [/on]"
                for i in range(11)
            ])
            prompt_legend = f"[bold]Token attention influence:[/bold]{influence_legend}"
            
            # Create influence heatmap with cursor
            token_strings = self.get_prompt_tokens_display()
            prompt_parts = []
            for i, (token, score) in enumerate(zip(token_strings, influence_scores)):
                if cursor_position is not None and i == cursor_position:
                    prompt_parts.append(f"[on {self.influence_to_color(score)}]\\[{token}\\][/on]")
                else:
                    prompt_parts.append(f"[on {self.influence_to_color(score)}]{token}[/on]")
            
            prompt_text = "".join(prompt_parts)
        else:
            prompt_text = self.explorer.get_prompt()
            prompt_legend = "[bold]No tokens to analyze influence[/bold]"
        
        return prompt_text, prompt_legend
    
    def render_local_bias_display(self, cursor_position=None):
        """Render local bias visualization with optional cursor"""
        # Get local token bias scores
        bias_scores = self.explorer.get_local_token_bias()
        token_strings = self.get_prompt_tokens_display()
        
        if bias_scores:
            # Normalize bias scores to [0,1] range for visualization
            max_bias = max(bias_scores)
            if max_bias > 0:
                normalized_bias = [b/max_bias for b in bias_scores]
            else:
                normalized_bias = bias_scores
            
            # Create bias legend
            bias_legend = "".join([
                f"[on {self.bias_to_color(i/10)}] {i/10:.2f} [/on]"
                for i in range(11)
            ])
            prompt_legend = f"[bold]Local token bias:[/bold]{bias_legend}"
            
            # Create bias heatmap with cursor
            prompt_parts = []
            for i, (token, bias) in enumerate(zip(token_strings, normalized_bias)):
                if cursor_position is not None and i == cursor_position:
                    prompt_parts.append(f"[on {self.bias_to_color(bias)}]\\[{token}\\][/on]")
                else:
                    prompt_parts.append(f"[on {self.bias_to_color(bias)}]{token}[/on]")
            
            prompt_text = "".join(prompt_parts)
        else:
            prompt_text = self.explorer.get_prompt()
            prompt_legend = "[bold]No tokens to analyze local bias[/bold]"
        
        return prompt_text, prompt_legend
    
    def render_energy_display(self, cursor_position=None):
        """Render energy visualization with optional cursor"""
        # Get token energies (Helmholtz free energy)
        energy_scores = self.explorer.get_token_energies()
        token_strings = self.get_prompt_tokens_display()
        
        if energy_scores:
            # Get min and max energy for normalization
            min_energy = min(energy_scores)
            max_energy = max(energy_scores)
            
            # Create energy legend
            # Lower energy (green) = in-distribution
            # Higher energy (red) = out-of-distribution
            energy_legend = "".join([
                f"[on {self.energy_to_color(min_energy + i*(max_energy-min_energy)/10, min_energy, max_energy)}] {i/10:.1f} [/on]"
                for i in range(11)
            ])
            prompt_legend = f"[bold]Token energy (OOD score):[/bold]{energy_legend}\n[bold]Green = in-distribution, Red = out-of-distribution[/bold]"
            
            # Create energy heatmap with cursor
            prompt_parts = []
            for i, (token, energy) in enumerate(zip(token_strings, energy_scores)):
                if cursor_position is not None and i == cursor_position:
                    prompt_parts.append(f"[on {self.energy_to_color(energy, min_energy, max_energy)}]\\[{token}\\][/on]")
                else:
                    prompt_parts.append(f"[on {self.energy_to_color(energy, min_energy, max_energy)}]{token}[/on]")
            
            prompt_text = "".join(prompt_parts)
        else:
            prompt_text = self.explorer.get_prompt()
            prompt_legend = "[bold]No tokens to analyze energy[/bold]"
        
        return prompt_text, prompt_legend
    
    def render_layer_legend(self, layer_mode, tokens_to_show):
        """Render layer heatmap legend if enabled"""
        layer_legend = ""
        if layer_mode != "none":
            try:
                num_layers = len(self.explorer.get_top_n_tokens(n=1)[0]["layer_probs"])
                # Add layer numbers
                layer_numbers = "".join([
                    f"[bold]{i+1}[/bold] " for i in range(num_layers)
                ])
                # Get max value across all tokens
                tokens = self.explorer.get_top_n_tokens(n=tokens_to_show)
                max_val = self.get_max_layer_value(tokens, layer_mode)
                # Add scale legend with dynamic range (no extra scaling factor)
                scale_points = [i * max_val / 10 for i in range(11)]
                scale = "".join([
                    f"[on {self.prob_to_color(p, max_val)}] [/on]"
                    for p in scale_points
                ])
                value_type = "Layer correlation" if layer_mode == "corr" else "Layer prob"
                layer_legend = f"[bold]Layers:[/bold] {layer_numbers}\n[bold]{value_type}:[/bold] {scale} (0.0 → {max_val:.3f})"
            except (IndexError, KeyError):
                # Handle case where no tokens or layer data is available
                layer_legend = ""
        
        return layer_legend
