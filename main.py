from itertools import cycle
from src.explorer import Explorer
from src.utils import entropy_to_color, probability_to_color
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static, DataTable
from textwrap import dedent
import sys
import argparse
import tomli
from datetime import datetime
import random
import time
import torch
import numpy as np

# Replace the constants with config
def load_config(config_file="config.toml"):
    try:
        with open(config_file, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        print(f"Config file '{config_file}' not found, using default values")
        return {
            "model": {
                "name": "Qwen/Qwen2.5-0.5B"
            },
            "prompt": {
                "example_prompt": "Once upon a time, there was a",
                "max_prompts": 9
            },
            "display": {
                "tokens_to_show": 30
            }
        }

# Initialize with default values that will be overridden in __main__
config = load_config()
MODEL_NAME = config["model"]["name"]
EXAMPLE_PROMPT = config["prompt"]["example_prompt"]
TOKENS_TO_SHOW = config["display"]["tokens_to_show"]
MAX_PROMPTS = config["prompt"]["max_prompts"]

class TokenExplorer(App):
    """Main application class."""

    display_modes = cycle(["prompt", "prob", "entropy", "influence", "local_bias", "energy"])
    display_mode = reactive(next(display_modes))
    layer_display_mode = cycle(["none", "prob", "corr"])  # None, probabilities, correlations
    current_layer_mode = reactive(next(layer_display_mode))
    show_z_scores = reactive(False)  # Toggle for z-score display

    BINDINGS = [("e", "change_display_mode", "Mode"),
                ("left,h", "pop_token", "Back"),
                ("right,l", "append_token", "Add"),
                ("d", "add_prompt", "New"),
                ("a", "remove_prompt", "Del"),
                ("w", "increment_prompt", "Next"),
                ("s", "decrement_prompt", "Prev"),
                ("x", "save_prompt", "Save"),
                ("j", "select_next", "Down"),
                ("k", "select_prev", "Up"),
                ("p", "toggle_layer_display", "Layer"),
                ("z", "toggle_z_scores", "Z-Score"),
                ("ctrl+r", "reset_prompt", "Reset")]
    
    # No custom commands class needed
    
    def __init__(self, prompt=EXAMPLE_PROMPT, use_bf16=False, enable_layer_prob=False, seed=None):
        super().__init__()
        self.seed = seed
        self.title = f"TokenExplorer - {MODEL_NAME} - Seed: {self.seed}"
        # Add support for multiple prompts.
        self.prompts = [prompt]
        self.original_prompt = prompt  # Store original prompt for reset
        self.prompt_index = 0
        self.explorer = Explorer(MODEL_NAME, use_bf16=use_bf16, enable_layer_prob=enable_layer_prob)
        self.explorer.set_prompt(prompt)
        self.rows = self._top_tokens_to_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            )
        self.selected_row = 0  # Track currently selected token row
        # Store layer probability enabled state for UI
        self.layer_prob_enabled = enable_layer_prob
    
    def _prob_to_color(self, prob, max_prob):
        """Convert probability to a color (red to blue)"""
        # Scale probability by max value (no extra scaling factor)
        scaled_prob = min(prob / max_prob, 1.0)
        # Red (low) to blue (high)
        return f"#{int(255 * (1 - scaled_prob)):02x}00{int(255 * scaled_prob):02x}"

    def _get_max_layer_value(self, tokens):
        """Get maximum layer value across all tokens"""
        if self.current_layer_mode == "corr":
            return max(max(token["layer_correlations"]) for token in tokens)
        elif self.current_layer_mode == "prob":
            return max(max(token["layer_probs"]) for token in tokens)
        return 0

    def _layer_values_to_heatmap(self, token, max_val):
        """Convert layer values to a heatmap string"""
        if self.current_layer_mode == "none":
            return ""
            
        blocks = []
        values = token["layer_correlations"] if self.current_layer_mode == "corr" else token["layer_probs"]
        for val in values:
            color = self._prob_to_color(val, max_val)
            blocks.append(f"[on {color}] ")
        # Combine all blocks into a single string
        return "".join(blocks)

    def _top_tokens_to_rows(self, tokens):
        # Get global max value for consistent scaling
        max_val = self._get_max_layer_value(tokens)
        # Include layers column only if layer display is enabled
        headers = ["token_id", "token", "prob"]
        if self.show_z_scores:
            headers.append("z-score")
        if self.current_layer_mode != "none":
            headers.append("layers")
        
        rows = []
        for token in tokens:
            row = [
                token["token_id"],
                token["token"],
                f"{token['probability']:.4f}"
            ]
            if self.show_z_scores:
                row.append(f"{token['z_score']:.4f}")
            if self.current_layer_mode != "none":
                row.append(self._layer_values_to_heatmap(token, max_val))
            rows.append(tuple(row))
        
        return [tuple(headers)] + rows
        
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(id="results")
            yield DataTable(id="table")
        yield Footer()

    def _refresh_table(self):
        table = self.query_one(DataTable)
        self.rows = self._top_tokens_to_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            )
        # Clear both columns and rows
        table.clear()
        table.columns.clear()
        # Add new columns and rows
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        # Reset cursor to top
        self.selected_row = 0
        table.move_cursor(row=self.selected_row)
        self.query_one("#results", Static).update(self._render_prompt())
            
    def _influence_to_color(self, influence):
        """Convert influence score to a color (green to purple)"""
        # Green (low) to purple (high)
        return f"#{0:02x}{int(255 * (1 - influence)):02x}{int(255 * influence):02x}"
        
    def _bias_to_color(self, bias):
        """Convert local token bias score to a color (orange to blue)"""
        # Orange (low) to blue (high)
        scaled_bias = min(max(bias, 0.0), 1.0)  # Ensure bias is in [0,1]
        return f"#{int(255 * (1 - scaled_bias)):02x}{int(128 * (1 - scaled_bias)):02x}{int(255 * scaled_bias):02x}"
        
    def _energy_to_color(self, energy, min_energy, max_energy):
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
    
    def _render_prompt(self):
        if self.display_mode == "entropy":
            entropy_legend = "".join([
                f"[on {entropy_to_color(i/10)}] {i/10:.2f} [/on]"
                for i in range(11)
                ])
            prompt_legend = f"[bold]Token entropy:[/bold]{entropy_legend}"
            token_entropies = self.explorer.get_prompt_token_normalized_entropies()
            token_strings = self.explorer.get_prompt_tokens_strings()
            prompt_text = "".join(f"[on {entropy_to_color(entropy)}]{token}[/on]" for token, entropy in zip(token_strings, token_entropies))
        elif self.display_mode == "prob":
            prob_legend = "".join([
                f"[on {probability_to_color(i/10)}] {i/10:.2f} [/on]"
                for i in range(11)
                ])
            prompt_legend = f"[bold]Token prob:[/bold]{prob_legend}"
            token_probs = self.explorer.get_prompt_token_probabilities()
            token_strings = self.explorer.get_prompt_tokens_strings()
            prompt_text = "".join(f"[on {probability_to_color(prob)}]{token}[/on]" for token, prob in zip(token_strings, token_probs))
        elif self.display_mode == "influence":
            # Get the top token to analyze influence
            top_tokens = self.explorer.get_top_n_tokens(n=1)
            if top_tokens:
                top_token = top_tokens[0]
                influence_scores = top_token["token_influence"]
                
                # Create influence legend
                influence_legend = "".join([
                    f"[on {self._influence_to_color(i/10)}] {i/10:.2f} [/on]"
                    for i in range(11)
                    ])
                prompt_legend = f"[bold]Token attention influence:[/bold]{influence_legend}"
                
                # Create influence heatmap
                token_strings = self.explorer.get_prompt_tokens_strings()
                prompt_text = "".join(f"[on {self._influence_to_color(score)}]{token}[/on]" 
                                     for token, score in zip(token_strings, influence_scores))
            else:
                prompt_text = self.explorer.get_prompt()
                prompt_legend = "[bold]No tokens to analyze influence[/bold]"
        elif self.display_mode == "local_bias":
            # Get local token bias scores
            bias_scores = self.explorer.get_local_token_bias()
            token_strings = self.explorer.get_prompt_tokens_strings()
            
            if bias_scores:
                # Normalize bias scores to [0,1] range for visualization
                max_bias = max(bias_scores)
                if max_bias > 0:
                    normalized_bias = [b/max_bias for b in bias_scores]
                else:
                    normalized_bias = bias_scores
                
                # Create bias legend
                bias_legend = "".join([
                    f"[on {self._bias_to_color(i/10)}] {i/10:.2f} [/on]"
                    for i in range(11)
                    ])
                prompt_legend = f"[bold]Local token bias:[/bold]{bias_legend}"
                
                # Create bias heatmap
                prompt_text = "".join(f"[on {self._bias_to_color(bias)}]{token}[/on]" 
                                     for token, bias in zip(token_strings, normalized_bias))
            else:
                prompt_text = self.explorer.get_prompt()
                prompt_legend = "[bold]No tokens to analyze local bias[/bold]"
        elif self.display_mode == "energy":
            # Get token energies (Helmholtz free energy)
            energy_scores = self.explorer.get_token_energies()
            token_strings = self.explorer.get_prompt_tokens_strings()
            
            if energy_scores:
                # Get min and max energy for normalization
                min_energy = min(energy_scores)
                max_energy = max(energy_scores)
                
                # Create energy legend
                # Lower energy (green) = in-distribution
                # Higher energy (red) = out-of-distribution
                energy_legend = "".join([
                    f"[on {self._energy_to_color(min_energy + i*(max_energy-min_energy)/10, min_energy, max_energy)}] {i/10:.1f} [/on]"
                    for i in range(11)
                    ])
                prompt_legend = f"[bold]Token energy (OOD score):[/bold]{energy_legend}\n[bold]Green = in-distribution, Red = out-of-distribution[/bold]"
                
                # Create energy heatmap
                prompt_text = "".join(f"[on {self._energy_to_color(energy, min_energy, max_energy)}]{token}[/on]" 
                                     for token, energy in zip(token_strings, energy_scores))
            else:
                prompt_text = self.explorer.get_prompt()
                prompt_legend = "[bold]No tokens to analyze energy[/bold]"
        else:
            prompt_text = self.explorer.get_prompt()
            prompt_legend = ""
        # Add layer heatmap legend if enabled
        layer_legend = ""
        if self.current_layer_mode != "none" and len(self.rows) > 1:
            num_layers = len(self.explorer.get_top_n_tokens(n=1)[0]["layer_probs"])
            # Add layer numbers
            layer_numbers = "".join([
                f"[bold]{i+1}[/bold] " for i in range(num_layers)
            ])
            # Get max value across all tokens
            tokens = self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            max_val = self._get_max_layer_value(tokens)
            # Add scale legend with dynamic range (no extra scaling factor)
            scale_points = [i * max_val / 10 for i in range(11)]
            scale = "".join([
                f"[on {self._prob_to_color(p, max_val)}] [/on]"
                for p in scale_points
            ])
            value_type = "Layer correlation" if self.current_layer_mode == "corr" else "Layer prob"
            layer_legend = f"[bold]Layers:[/bold] {layer_numbers}\n[bold]{value_type}:[/bold] {scale} (0.0 → {max_val:.3f})"

        return dedent(f"""
{prompt_text}





{prompt_legend}
{layer_legend}
[bold]Prompt[/bold] {self.prompt_index+1}/{len(self.prompts)} tokens: {len(self.explorer.prompt_tokens)}
""")
    
    def on_mount(self) -> None:
        
        self.query_one("#results", Static).update(self._render_prompt())
        table = self.query_one(DataTable)
        table.add_columns(*self.rows[0])
        table.add_rows(self.rows[1:])
        table.cursor_type = "row"
    
    def action_add_prompt(self):
        if len(self.prompts) < MAX_PROMPTS:
            self.prompts.append(self.explorer.get_prompt())
            self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
            self.explorer.set_prompt(self.prompts[self.prompt_index])
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()

    def action_remove_prompt(self):
        if len(self.prompts) > 1:
            self.prompts.pop(self.prompt_index)
            self.prompt_index = (self.prompt_index - 1) % len(self.prompts)
            self.explorer.set_prompt(self.prompts[self.prompt_index])
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()
    
    def action_increment_prompt(self):
        self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
        self.explorer.set_prompt(self.prompts[self.prompt_index])
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

    def action_decrement_prompt(self):
        self.prompt_index = (self.prompt_index - 1) % len(self.prompts)
        self.explorer.set_prompt(self.prompts[self.prompt_index])
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

    def action_change_display_mode(self):
        self.display_mode = next(self.display_modes)
        self.query_one("#results", Static).update(self._render_prompt())

    def action_toggle_layer_display(self):
        """Toggle layer display mode (none -> probabilities -> correlations)"""
        if not self.layer_prob_enabled:
            # If layer probability calculations are disabled, show a message
            self.query_one("#results", Static).update(
                dedent(f"""
                Layer probability calculations are disabled.
                Run with --layer_prob flag to enable this feature.
                
                Example: python main.py --layer_prob
                """)
            )
            return
            
        # If enabled, toggle as normal
        self.current_layer_mode = next(self.layer_display_mode)
        self._refresh_table()
        
    def action_toggle_z_scores(self):
        """Toggle display of z-scores for tokens"""
        self.show_z_scores = not self.show_z_scores
        self._refresh_table()

    def action_pop_token(self):
        if len(self.explorer.get_prompt_tokens()) > 1:
            self.explorer.pop_token()
            self.prompts[self.prompt_index] = self.explorer.get_prompt()
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()
            
    def action_save_prompt(self):
        with open(f"prompts/prompt_{self.prompt_index}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "w") as f:
            f.write(self.explorer.get_prompt())

    def action_select_next(self):
        """Move selection down one row"""
        if self.selected_row < len(self.rows) - 2:  # -2 for header row
            self.selected_row += 1
            table = self.query_one(DataTable)
            table.move_cursor(row=self.selected_row)
            
    def action_select_prev(self):
        """Move selection up one row"""
        if self.selected_row > 0:
            self.selected_row -= 1
            table = self.query_one(DataTable)
            table.move_cursor(row=self.selected_row)

    def action_append_token(self):
        """Append currently selected token"""
        table = self.query_one(DataTable)
        if table.cursor_row is not None:
            self.explorer.append_token(self.rows[table.cursor_row+1][0])
            self.prompts[self.prompt_index] = self.explorer.get_prompt()
            self._refresh_table()  # This will reset cursor position
            
    def action_reset_prompt(self):
        """Reset prompt to original state"""
        self.explorer.set_prompt(self.original_prompt)
        self.prompts[self.prompt_index] = self.original_prompt
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Token Explorer Application')
    parser.add_argument('--input', '-i', type=str, help='Path to input text file')
    parser.add_argument('--config', type=str, default='config.toml', help='Path to configuration file (default: config.toml)')
    parser.add_argument('--bf16', action='store_true', help='Load model in bf16 precision')
    parser.add_argument('--layer_prob', action='store_true', help='Enable layer probability and correlation calculations (may be computationally expensive)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for generation')
    args = parser.parse_args()
    
    # Load configuration from the specified file and override global constants
    config = load_config(args.config)
    # Update the global variables with values from the config file
    MODEL_NAME = config["model"]["name"]
    EXAMPLE_PROMPT = config["prompt"]["example_prompt"]
    TOKENS_TO_SHOW = config["display"]["tokens_to_show"]
    MAX_PROMPTS = config["prompt"]["max_prompts"]

    # Determine and set the random seed
    if args.seed is None:
        seed = int(time.time()) # Use current time as a random seed if none provided
    else:
        seed = args.seed
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = EXAMPLE_PROMPT
    if args.input:
        try:
            with open(args.input, 'r') as f:
                prompt = f.read()
        except FileNotFoundError:
            print(f"Error: Could not find input file '{args.input}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
        
    app = TokenExplorer(prompt=prompt, use_bf16=args.bf16, enable_layer_prob=args.layer_prob, seed=seed)
    app.run()
