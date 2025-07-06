# Token Explorer

Token Explorer is an interactive tool designed to help researchers, developers, and AI enthusiasts understand how Large Language Models (LLMs) generate text one token at a time. By providing a "video game" style interface, it allows you to explore the inner workings of LLMs, visualize token probabilities, analyze model attention patterns, and gain insights into model behavior.

The project started with <https://github.com/willkurt/token-explorer> and I added much on token generation and probability functions to understand LLM behavior, now it is too different from the original branch and I can't merge, so let it be a standalone fork.

## Key Features

Token Explorer offers a comprehensive set of features for exploring and understanding LLM behavior:

- **Interactive Token Generation**: Step through text generation one token at a time, seeing exactly what the model predicts at each step
- **Multiple Visualization Modes**: Analyze token probabilities, entropy, attention patterns, semantic similarity, and out-of-distribution detection
- **Layer-by-Layer Analysis**: Examine how token predictions evolve through the model's layers (optional feature)
- **Prompt Management**: Create, save, and switch between multiple prompts to compare different approaches
- **Intuitive Interface**: Navigate using either arrow keys or vim-style navigation (h/j/k/l) along with WASD keys
- **Hardware Optimization**: Automatically uses the best available device (CUDA > MPS > CPU)
- **Customizable Configuration**: Adjust model settings, display options, and more through a simple configuration file

## Installation & Setup

Token Explorer uses `uv` for project management, a fast Python package installer and resolver.

1. **Install uv**: Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

2. **Run Token Explorer**: Once you have `uv` installed, you can run the app with:

   ```bash 
   uv run main.py
   ```

3. **Using your own prompts**: You can provide any text file as an argument:

   ```bash
   uv run main.py --input <path_to_file>
   ```

4. **Reducing memory usage**: Use the `--bf16` flag to load the model in bfloat16 precision:

   ```bash
   uv run main.py --bf16
   ```

### Command-line Parameters

The application supports several command-line parameters:

| Parameter | Description |
|-----------|-------------|
| `--input <path>`, `-i <path>` | Specify a text file to use as the initial prompt |
| `--bf16` | Load the model in bfloat16 precision to reduce memory usage |
| `--layer_prob` | Enable layer probability and correlation calculations |
| `--cross_layer` | Enable cross-layer feature analysis |
| `--seed <integer>` | Set the random seed for generation (uses time-based seed if not provided) |
| `--config <path>` | Path to configuration file (default: config.toml) |

**Note on Layer Analysis**: Layer probability and correlation calculations are **disabled by default** as they can be computationally expensive. When enabled with the `--layer_prob` flag, you can use the 'p' hotkey to toggle between different layer visualization modes.

```bash
# Enable layer probability and correlation calculations
uv run main.py --layer_prob

# Combine with other flags as needed
uv run main.py --input my_prompt.txt --bf16 --layer_prob --seed 42
```

## Usage Guide

When you start Token Explorer, you'll see your prompt and a table of the top tokens the model predicts next, along with their probabilities.

![Starting prompt](./imgs/starting_prompt.png)

### Keyboard Controls

Token Explorer uses a "video game" style interface. For the best experience, place your left hand on WASD and your right hand on the arrow keys.

| Key | Alternative | Action |
|-----|-------------|--------|
| ↑/↓ | j/k | Navigate up/down in the token table |
| → | l | Append selected token to prompt |
| ← | h | Remove last token from prompt |
| d | | Add current prompt as a new entry in prompt list |
| a | | Remove current prompt from list |
| w | | Go to next prompt in list |
| s | | Go to previous prompt in list |
| e | | Cycle through visualization modes |
| p | | Toggle layer visualization (when enabled) |
| z | | Toggle z-score display |
| x | | Save current prompt to file |
| c | | Toggle cursor visibility on/off |
| Ctrl+r | | Reset prompt to original state |
| Ctrl+q | | Quit application |

#### Token Cursor Navigation

Token Explorer includes a powerful cursor system for precise token-level navigation and analysis. The cursor allows you to focus on specific tokens within your prompt and see detailed information about them.

**Cursor Features:**
- **Visual Indicators**: Current token is highlighted with `[token]` brackets in all visualization modes
- **Status Bar Info**: Shows current cursor position and selected token (e.g., "Cursor: 4/14 'time'")
- **Toggle Visibility**: Press `c` to show/hide cursor indicators for clean viewing
- **Multi-Mode Support**: Works across all visualization modes (entropy, probability, influence, etc.)

**Cursor Navigation Keys:**
| Key | Action |
|-----|--------|
| Ctrl+j | Move cursor to next token |
| Ctrl+k | Move cursor to previous token |
| Ctrl+w | Move cursor to next word boundary |
| Ctrl+b | Move cursor to previous word boundary |
| Ctrl+0 | Move cursor to start of prompt |
| Ctrl+$ | Move cursor to end of prompt |

**Usage Examples:**
- **Analyze specific tokens**: Use cursor navigation to examine individual tokens in detail
- **Compare token properties**: Move cursor between tokens to see how entropy, probability, or influence changes
- **Precise exploration**: Navigate word-by-word or token-by-token through complex prompts
- **Clean viewing**: Toggle cursor off (press `c`) for distraction-free analysis
- **Status tracking**: Monitor current token and position in the status bar

The cursor system is inspired by VIM-style navigation, making it familiar and efficient for users who prefer keyboard-driven interfaces.

### Basic Workflow

1. **Navigate the token table** using up/down arrows or j/k keys to select tokens
   ![Navigating the table](./imgs/navigating_table.png)

2. **Append tokens** by pressing the right arrow key or 'l'
   ![Appending a token](./imgs/appending_token.png)

3. **Backtrack** by using the left arrow key or 'h' to remove tokens

4. **Save interesting prompts** by pressing 'd' to duplicate your current prompt
   ![Adding a prompt](./imgs/add_a_prompt.png)

5. **Compare different approaches** by cycling through prompts with 'w' and 's'

6. **Analyze model behavior** by toggling visualization modes with 'e'

## Visualization Modes

Token Explorer offers multiple visualization modes to help you understand different aspects of the model's behavior. Press `e` to cycle through these modes:

### 1. Default View
Shows the plain prompt text without any special visualization.

### 2. Token Probabilities
Displays the probability of each token in the prompt, helping you see where the model was confident or uncertain.

![Probabilities](./imgs/probabilities.png)

Color scale: Red (low probability) to Blue (high probability)

### 3. Token Entropies
Shows the entropy (uncertainty) of the token distribution at each position.

![Entropy](./imgs/entropies.png)

- High entropy (closer to 1.0): Model is uncertain about the next token
- Low entropy (closer to 0.0): Model is confident about the next token

### 4. Token Attention Influence
Visualizes how much each token in the prompt influences the prediction of the next token.

How it works:
1. Identifies the most informative attention heads based on attention variance
2. Extracts and aggregates attention weights from these heads
3. Normalizes the weights to create a heatmap

Color scale: Green (low influence) to Purple (high influence)

### 5. Local Token Bias
Shows the semantic similarity between each token and the current prediction context.

How it works:
1. Extracts hidden state representations from the last layer
2. Computes cosine similarity between each token and the last token
3. Normalizes these similarities for visualization

Color scale: Orange (low similarity) to Blue (high similarity)

### 6. Token Energy (Out-of-Distribution Detection)
Displays the Helmholtz free energy for each token, identifying tokens that may be out-of-distribution.

### 7. Hidden State Similarity
Visualizes how a token's representation evolves across the model's layers. It does this by calculating the cosine similarity of the token's hidden state at each layer with its hidden state at the final layer. This provides a real, data-driven view of how the model's understanding of a token converges as it passes through the network.

### 8. Residual Stream Analysis
Provides a comprehensive view of the model's residual stream, including:
- **Layer-wise Magnitude**: Shows the L2 norm of the residual stream for a selected token at each layer.
- **Information Flow Heatmap**: Visualizes the magnitude of the residual stream for all tokens across all layers.
- **Component Contributions**: Displays an estimated breakdown of how much the attention and MLP components contribute to the residual stream at a given layer.

How it works:
1. Computes energy as `-T * logsumexp(logits / T)` for each token position
2. Normalizes values to create an intuitive heatmap

Color scale: Green (in-distribution) to Red (out-of-distribution)

This visualization helps identify:
- Tokens that don't fit well with surrounding context
- Potential errors or inconsistencies in the model's understanding
- Points where the model might be uncertain or confused

### 9. Sparse Autoencoder (SAE) Feature Activations

This feature, inspired by the groundbreaking work on [circuit-tracer](https://github.com/safety-research/circuit-tracer) and the concept of Sparse Autoencoders (SAEs) from [Anthropic's research](https://transformer-circuits.pub/2023/sae/index.html), allows you to peer into the latent space of the model and observe the activation of individual SAE features. SAEs decompose the dense hidden states of an LLM into a sparse set of interpretable features, offering a deeper understanding of the model's internal representations.

**Key Features**:
- **Feature Visualization**: Displays the top-activating SAE features for the token currently under the cursor.
- **Layer Exploration**: Cycle through different SAE layers to see how features activate at various depths of the model.
- **Asynchronous Computation**: Feature activations are computed in a background thread, ensuring the UI remains responsive and preventing freezing.
- **Two-Level Caching**: Utilizes both in-memory and disk caching to store computed activations, significantly speeding up subsequent access to the same token/layer combinations.
- **Loading Indicator**: A "Loading features..." message provides clear feedback while computations are in progress.

**Usage**:
- When in "Feature Activations" mode (cycle with `e`), use `Ctrl+j`/`Ctrl+k` (or `Ctrl+w`/`Ctrl+b`, `Ctrl+0`/`Ctrl+$`) to move the cursor and see the SAE features for the selected token.
- Press `p` to cycle through the available SAE layers.

**Color scale**: Higher activation values indicate a stronger presence of that feature.

### 10. Gradient-Based Attribution Analysis
This feature, inspired by the work on `circuit-tracer`, provides a powerful way to understand which input tokens are most influential in predicting a subsequent token. It uses gradient-based methods to attribute the model's prediction to the input tokens.

**Key Features**:
- **Three Attribution Methods**:
  - **Saliency**: Simple gradient of the output with respect to the input.
  - **Input × Gradient**: Element-wise product of the input and its gradient.
  - **Integrated Gradients**: A more robust method that integrates gradients along a path from a baseline to the input.
- **Interactive Visualization**: The attribution scores are visualized as a heatmap over the input tokens, making it easy to see which tokens are most important.
- **Cursor Integration**: The analysis is performed for the token currently under the cursor, allowing for real-time exploration of the model's reasoning.

**Color scale**: Green (less important) to Blue (more important)

## Advanced Features

### Layer Analysis

Token Explorer provides layer-by-layer analysis of how predictions evolve through the model's layers. This feature is toggled with the `p` key and must be enabled with the `--layer_prob` flag when starting the application.

When enabled, pressing `p` cycles through:

1. **Layer Probabilities**:
   - Shows raw probability values for the token at each layer
   - Red (0.0) to blue (1.0) color scale
   - Helps visualize how strongly each layer predicts the token

2. **Layer Correlations**:
   - Shows normalized prediction values for each layer
   - Values scaled to [0,1] range for easier visualization
   - Red (0.0) to blue (1.0) color scale
   - Helps understand how prediction evolves through the model

This feature is particularly valuable for ML researchers as it reveals:
- How the model's confidence in a token changes through the network
- Which layers contribute most to the final prediction
- Patterns in how predictions evolve:
  * Early layers: Diffuse probabilities across many tokens
  * Middle layers: Emerging preferences
  * Final layers: Convergence on the chosen token

### Z-Score Display

Toggle the display of z-scores for tokens by pressing `z`. Z-scores show how many standard deviations a token's logit is from the mean, providing insight into the statistical significance of token predictions.

## Example Workflow: Analyzing Model Reasoning

Let's explore how Token Explorer can help us understand a model's reasoning process using a math problem from GSM8K:

```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Reasoning: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
Answer: 72
```

### Step 1: Analyze the model's confidence in its answer

First, we load the prompt and back up to the answer token:

![Backing up to the answer](./imgs/backing_up_to_answer.png)

We can see the model is extremely confident about the answer, assigning nearly 100% probability to the token "7" at the start of "72".

### Step 2: Identify interesting points for exploration

Switching to the entropy view, we notice the token 'Natalia' has higher entropy than other tokens:

![Entropy](./imgs/natalia.png)

This indicates the model is more uncertain at this point in the generation.

### Step 3: Create a branch for exploration

To preserve our place while exploring, we press 'd' to duplicate our prompt:

![Swapping prompts](./imgs/workflow1.png)

### Step 4: Investigate alternative paths

We rewind to the 'Natalia' token and examine the probability distribution:

![Probabilities](./imgs/workflow2.png)

The model is considering multiple options at this point.

### Step 5: Test model robustness

We create another branch and select a different token path:

![Exploring](./imgs/workflow3.png)

Interestingly, even when we choose a low-probability path in the middle of the reasoning, the model still arrives at the correct answer of 72!

This workflow demonstrates how Token Explorer can help you:
1. Identify points of uncertainty in the model's generation
2. Explore alternative generation paths
3. Test the robustness of the model's reasoning
4. Understand how the model arrives at its conclusions

## Configuration

Token Explorer can be configured through the `config.toml` file. Here are the key settings you can modify:

### Model Settings
```toml
[model]
name = "Qwen/Qwen2.5-0.5B"  # Model to use
enable_sae = false          # Master switch for SAE features (set to true to enable)
```

**Note**: Token Explorer works best with smaller models due to performance considerations. Larger models will work but may be slower.

### SAE Settings
```toml
# Example SAE configuration for Llama-3.2-1B, layer 8
# See: https://huggingface.co/mntss/skip-transcoder-Llama-3.2-1B-131k-nobos
[[sae]]
layer = 8
repo_id = "mntss/skip-transcoder-Llama-3.2-1B-131k-nobos"
filename = "layer_8.safetensors"
revision = "new-training"
```
**Performance Considerations for SAEs**:
SAE feature activation computation can be highly resource-intensive. For optimal performance:
- **GPU/MPS is Highly Recommended**: Running on CPU will be significantly slower and may lead to freezing. Ensure you have a CUDA-enabled GPU or an Apple Silicon Mac (which uses MPS).
- **Model Size**: Smaller models (e.g., 1B-2B parameters) are more suitable for interactive SAE analysis.
- **Number of Layers**: Loading fewer SAE layers (e.g., 1-3 layers) will drastically improve responsiveness. Loading many layers can consume significant VRAM and slow down the application.
- **Caching**: Token Explorer utilizes a two-level caching system (in-memory and disk) to speed up repeated access to the same feature activations. The first computation for a new token/layer will still incur a delay.

### Prompt Settings
```toml
[prompt]
example_prompt = "Once upon a time, there was a"  # Default prompt
max_prompts = 9  # Maximum number of prompts to store
```

### Display Settings
```toml
[display]
tokens_to_show = 30  # Number of tokens to display in the table
cache_size = 100  # Maximum number of cached token sequences
```

### Alternative Configuration Files

You can specify an alternative configuration file using the `--config` parameter:

```bash
uv run main.py --config config_gemma.toml
```

## Research Applications

Token Explorer is a powerful tool for understanding LLM behavior, with applications including:

1. **Interpretability Research**: Visualize and analyze how models process and generate text, including the activation of sparse autoencoder features.
2. **Model Debugging**: Identify where models make mistakes or exhibit unexpected behavior.
3. **Prompt Engineering**: Develop and test different prompting strategies.
4. **Educational Tool**: Learn about token-based generation, transformer architecture, and the role of SAEs.
5. **Out-of-Distribution Detection**: Identify when models encounter unfamiliar content.
6. **Attention Analysis**: Understand which parts of input text influence model predictions.
7. **Layer-wise Behavior**: Study how information flows through the model's layers and how SAE features contribute at each stage.

By providing an interactive, visual interface to explore these aspects of LLM behavior, Token Explorer bridges the gap between theoretical understanding and practical insights into how these models work.
