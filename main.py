from itertools import cycle
from src.explorer import Explorer
from src.ui_adapter import UIDataAdapter
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

    display_modes = cycle(["prompt", "prob", "entropy", "influence", "local_bias", "energy", "hidden_state_similarity", "residual_stream"])
    display_mode = reactive(next(display_modes))
    layer_display_mode = cycle(["none", "prob", "corr"])  # None, probabilities, correlations
    current_layer_mode = reactive(next(layer_display_mode))
    show_z_scores = reactive(False)  # Toggle for z-score display
    # Note: cursor_position is now a regular instance variable, not reactive

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
                ("ctrl+r", "reset_prompt", "Reset"),
                ("c", "toggle_cursor", "Cursor"),
                # New VIM-inspired token navigation
                ("ctrl+j", "cursor_next", "Token→"),
                ("ctrl+k", "cursor_prev", "←Token"),
                ("ctrl+w", "cursor_word_forward", "Word→"),
                ("ctrl+b", "cursor_word_back", "←Word"),
                ("ctrl+0", "cursor_start", "Start"),
                ("ctrl+dollar", "cursor_end", "End"),
                # Layer navigation for cross-layer feature view
                ("n", "next_layer", "Next Layer"),
                ("b", "prev_layer", "Prev Layer")]
    
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
        self.ui_adapter = UIDataAdapter(self.explorer)  # Add UI adapter
        self.rows = self.ui_adapter.get_table_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW),
            show_z_scores=False,
            layer_mode="none"
            )
        self.selected_row = 0  # Track currently selected token row
        # Store layer probability enabled state for UI
        self.layer_prob_enabled = enable_layer_prob
        # Initialize token cursor position as regular instance variable (renamed to avoid Textual conflict)
        self.token_cursor_position = 0
        # Initialize cursor visibility (on by default)
        self.cursor_visible = True
        self.current_layer = 0
    
        
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(id="results")
            yield DataTable(id="table")
        yield Footer()

    def _refresh_table(self):
        table = self.query_one(DataTable)
        self.rows = self.ui_adapter.get_table_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW),
            show_z_scores=self.show_z_scores,
            layer_mode=self.current_layer_mode
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
            
    
    def _render_prompt(self):
        # Use UIDataAdapter to render different display modes with cursor position
        # Pass cursor position only if cursor is visible
        cursor_pos = self.token_cursor_position if self.cursor_visible else None
        
        if self.display_mode == "entropy":
            prompt_text, prompt_legend = self.ui_adapter.render_entropy_display(cursor_pos)
        elif self.display_mode == "prob":
            prompt_text, prompt_legend = self.ui_adapter.render_probability_display(cursor_pos)
        elif self.display_mode == "influence":
            prompt_text, prompt_legend = self.ui_adapter.render_influence_display(cursor_pos)
        elif self.display_mode == "local_bias":
            prompt_text, prompt_legend = self.ui_adapter.render_local_bias_display(cursor_pos)
        elif self.display_mode == "energy":
            prompt_text, prompt_legend = self.ui_adapter.render_energy_display(cursor_pos)
        elif self.display_mode == "hidden_state_similarity":
            prompt_text, prompt_legend = self.ui_adapter.render_hidden_state_similarity_display(cursor_pos)
        elif self.display_mode == "residual_stream":
            prompt_text, prompt_legend = self.ui_adapter.render_residual_stream_display(cursor_pos, self.current_layer)
        else:
            # For plain prompt mode, show cursor only if visible
            tokens = self.ui_adapter.get_prompt_tokens_display()
            prompt_parts = []
            for i, token in enumerate(tokens):
                if self.cursor_visible and i == self.token_cursor_position:
                    # Escape brackets to prevent Textual markup conflicts
                    prompt_parts.append(f"\\[{token}\\]")
                else:
                    prompt_parts.append(token)
            
            prompt_text = "".join(prompt_parts)
            prompt_legend = ""
        
        # Add layer heatmap legend if enabled
        layer_legend = self.ui_adapter.render_layer_legend(self.current_layer_mode, TOKENS_TO_SHOW)

        # Show current token in status line if cursor is visible
        if self.cursor_visible:
            tokens = self.ui_adapter.get_prompt_tokens_display()
            current_token = tokens[self.token_cursor_position] if self.token_cursor_position < len(tokens) else ""
            cursor_info = f" | [bold]Cursor:[/bold] {self.token_cursor_position+1}/{len(self.explorer.prompt_tokens)} \"{current_token}\""
        else:
            cursor_info = ""

        return dedent(f"""
{prompt_text}





{prompt_legend}
{layer_legend}
[bold]Prompt[/bold] {self.prompt_index+1}/{len(self.prompts)} tokens: {len(self.explorer.prompt_tokens)}{cursor_info}
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
            self.token_cursor_position = 0  # Reset cursor for new prompt
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()

    def action_remove_prompt(self):
        if len(self.prompts) > 1:
            self.prompts.pop(self.prompt_index)
            self.prompt_index = (self.prompt_index - 1) % len(self.prompts)
            self.explorer.set_prompt(self.prompts[self.prompt_index])
            self.token_cursor_position = 0  # Reset cursor for different prompt
            self.query_one("#results", Static).update(self._render_prompt())
            self._refresh_table()
    
    def action_increment_prompt(self):
        self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
        self.explorer.set_prompt(self.prompts[self.prompt_index])
        self.token_cursor_position = 0  # Reset cursor for different prompt
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

    def action_decrement_prompt(self):
        self.prompt_index = (self.prompt_index - 1) % len(self.prompts)
        self.explorer.set_prompt(self.prompts[self.prompt_index])
        self.token_cursor_position = 0  # Reset cursor for different prompt
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
            # Ensure cursor position is still valid after removing token
            max_pos = len(self.explorer.get_prompt_tokens()) - 1
            self.token_cursor_position = min(self.token_cursor_position, max_pos)
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
        self.token_cursor_position = 0  # Reset cursor position
        self.query_one("#results", Static).update(self._render_prompt())
        self._refresh_table()

    # New VIM-inspired token cursor navigation methods
    def action_cursor_next(self):
        """Move cursor to next token (VIM-inspired)"""
        max_pos = len(self.explorer.get_prompt_tokens()) - 1
        if self.token_cursor_position < max_pos:
            self.token_cursor_position += 1
            self.query_one("#results", Static).update(self._render_prompt())

    def action_cursor_prev(self):
        """Move cursor to previous token (VIM-inspired)"""
        if self.token_cursor_position > 0:
            self.token_cursor_position -= 1
            self.query_one("#results", Static).update(self._render_prompt())

    def action_cursor_word_forward(self):
        """Move cursor to next word boundary (VIM w)"""
        tokens = self.explorer.get_prompt_tokens_strings()
        current_pos = self.token_cursor_position
        max_pos = len(tokens) - 1
        
        # Find next word boundary (skip current word, find next non-space)
        while current_pos < max_pos:
            current_pos += 1
            token = tokens[current_pos]
            # Consider a new word if token doesn't start with space or punctuation
            if not token.startswith((' ', '\t', '\n')) and token.isalnum():
                break
        
        self.token_cursor_position = min(current_pos, max_pos)
        self.query_one("#results", Static).update(self._render_prompt())

    def action_cursor_word_back(self):
        """Move cursor to previous word boundary (VIM b)"""
        tokens = self.explorer.get_prompt_tokens_strings()
        current_pos = self.token_cursor_position
        
        # Find previous word boundary
        while current_pos > 0:
            current_pos -= 1
            token = tokens[current_pos]
            # Consider a word start if token doesn't start with space or punctuation
            if not token.startswith((' ', '\t', '\n')) and token.isalnum():
                break
        
        self.token_cursor_position = max(current_pos, 0)
        self.query_one("#results", Static).update(self._render_prompt())

    def action_cursor_start(self):
        """Move cursor to start of prompt (VIM 0)"""
        self.token_cursor_position = 0
        self.query_one("#results", Static).update(self._render_prompt())

    def action_cursor_end(self):
        """Move cursor to end of prompt (VIM $)"""
        self.token_cursor_position = len(self.explorer.get_prompt_tokens()) - 1
        self.query_one("#results", Static).update(self._render_prompt())

    def action_toggle_cursor(self):
        """Toggle cursor visibility on/off"""
        self.cursor_visible = not self.cursor_visible
        self.query_one("#results", Static).update(self._render_prompt())

    def action_next_layer(self):
        """Go to the next layer in cross-layer feature view."""
        if self.display_mode in ["residual_stream", "hidden_state_similarity"]:
            num_layers = self.explorer.model.config.num_hidden_layers
            self.current_layer = (self.current_layer + 1) % num_layers
            self.query_one("#results", Static).update(self._render_prompt())

    def action_prev_layer(self):
        """Go to the previous layer in cross-layer feature view."""
        if self.display_mode in ["residual_stream", "hidden_state_similarity"]:
            num_layers = self.explorer.model.config.num_hidden_layers
            self.current_layer = (self.current_layer - 1 + num_layers) % num_layers
            self.query_one("#results", Static).update(self._render_prompt())

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
