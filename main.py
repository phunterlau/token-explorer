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

# Replace the constants with config
def load_config():
    try:
        with open("config.toml", "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        print("Config file not found, using default values")
        return {
            "model": "Qwen/Qwen2.5-0.5B",
            "example_prompt": "Once upon a time, there was a",
            "tokens_to_show": 30,
            "max_prompts": 9
        }

config = load_config()
MODEL_NAME = config["model"]["name"]
EXAMPLE_PROMPT = config["prompt"]["example_prompt"]
TOKENS_TO_SHOW = config["display"]["tokens_to_show"]
MAX_PROMPTS = config["prompt"]["max_prompts"]

class TokenExplorer(App):
    """Main application class."""

    display_modes = cycle(["prompt", "prob", "entropy"])
    display_mode = reactive(next(display_modes))
    layer_display_mode = cycle(["none", "prob", "corr"])  # None, probabilities, correlations
    current_layer_mode = reactive(next(layer_display_mode))

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
                ("p", "toggle_layer_display", "Layer")]
    
    def __init__(self, prompt=EXAMPLE_PROMPT, use_bf16=False):
        super().__init__()
        # Add support for multiple prompts.
        self.prompts = [prompt]
        self.prompt_index = 0
        self.explorer = Explorer(MODEL_NAME, use_bf16=use_bf16)
        self.explorer.set_prompt(prompt)
        self.rows = self._top_tokens_to_rows(
            self.explorer.get_top_n_tokens(n=TOKENS_TO_SHOW)
            )
        self.selected_row = 0  # Track currently selected token row
    
    def _prob_to_color(self, prob, max_prob):
        """Convert probability to a color (red to blue)"""
        # Scale probability by max value
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
        if self.current_layer_mode != "none":
            headers.append("layers")
        
        rows = []
        for token in tokens:
            row = [
                token["token_id"],
                token["token"],
                f"{token['probability']:.4f}"
            ]
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
            # Add scale legend with dynamic range
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
        self.current_layer_mode = next(self.layer_display_mode)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Token Explorer Application')
    parser.add_argument('--input', '-i', type=str, help='Path to input text file')
    parser.add_argument('--bf16', action='store_true', help='Load model in bf16 precision')
    args = parser.parse_args()

    prompt = EXAMPLE_PROMPT
    if args.input:
        try:
            with open(args.input, 'r') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Could not find input file '{args.input}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    app = TokenExplorer(prompt=prompt, use_bf16=args.bf16)
    app.run()
