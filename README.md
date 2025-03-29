# Token Explorer

Token Explorer allows you to interactively explore the token generation process of an LLM, using a "video game" style interface. You can use either arrow keys or vim-style navigation (h/j/k/l) along with WASD keys.

Token explore allows you to:
- Choose a starting prompt, or provide your own text file.
- Step through generation one token at a time using either:
  * Arrow keys to navigate, pop and append tokens
  * Vim-style keys: h/l to pop/append, j/k to move up/down
- View the token probabilities and entropies.
- Add a copy of your current prompt to the list of prompts.
- Cycle through the prompts by pressing `w` and `s`.
- Add and remove prompts from the list with `a` and `d`.
- Automatically uses the best available device (CUDA > MPS > CPU).


## Running the app

Token Explore uses `uv` for project management. Please see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/) for more information.

Once you have `uv` installed, you can install the dependencies and run the app with:

```bash 
uv run main.py
```

In the model has a default prompt, but you can provide any text file as an argument to the app.

```bash
uv run main.py --input <path_to_file>
```

You can also use the `--bf16` flag to load the model in bfloat16 precision, which reduces memory usage:

```bash
uv run main.py --bf16
```

### Command-line Parameters

The application supports several command-line parameters:

- **`--input <path_to_file>`, `-i <path_to_file>`**: Specify a text file to use as the initial prompt
- **`--bf16`**: Load the model in bfloat16 precision to reduce memory usage
- **`--layer_prob`**: Enable layer probability and correlation calculations

Layer probability and correlation calculations are **disabled by default** as they can be computationally expensive. These calculations analyze how token predictions evolve through each layer of the model, providing valuable insights for ML researchers but requiring significant additional computation.

When enabled with the `--layer_prob` flag, you can use the 'p' hotkey to toggle between different layer visualization modes (none → probabilities → correlations → none).

```bash
# Enable layer probability and correlation calculations
uv run main.py --layer_prob

# Combine with other flags as needed
uv run main.py --input my_prompt.txt --bf16 --layer_prob
```

**Performance Impact**: Enabling this feature may significantly increase memory usage and slow down token generation, especially for larger models. The exact impact depends on your hardware and the model size.

## Usage

When you start the app you will see your prompt as well as a table of the top 30 tokens and their probabilities.

![Starting prompt](./imgs/starting_prompt.png)

The idea of Token Explorer is to make it very easy to explore the space of possible prompts and token generations from an LLM. To use Token Explorer it's best to treat your keyboard like you're playing a video game: put your left hand on WASD and your right hand on the arrow keys.

### Basic Usage

Use the up and down arrow keys to navigate the table. Use 'k'/'j' keys to select the current token so LLM can start generate the next one.

![Navigating the table](./imgs/navigating_table.png)

Here you can see I've highlighted the token "very" which has a probability of 0.03. Pressing the right arrow key or 'l' will append this token to the end of your prompt. Then it will display the next set of tokens.

![Appending a token](./imgs/appending_token.png)

If you want to go back and reselect the token, you can use the left arrow key or 'h' to pop the token back off the end of the prompt.

To **quit** the app, you can press `ctrl+q`.

You can also save your current prompt by pressing `x`. This will save the prompt to the `prompts` folder.

If you want to reset the current prompt to its original state, press `ctrl+r`. This is useful when you've made several token selections and want to start over from the beginning.

### Adding prompts

One of the goals of Token Explorer is to make it easy to play around with alternate methods of prompting. To faciliate this, Token Explorer allows you to duplicate your current prompt and add it to the list of prompts by pressing 'd'. In this image below we've added a copy of our current prompt to the list of prompts and are now at propmt 2 of 2:

![Adding a prompt](./imgs/add_a_prompt.png)

You can cycle through the prompts by pressing 'w' and 's', making it easy to try out different possible paths for your prompt, all while acting like you are the models sampler!

If you want to experiment with dramatically different prompts, you should write these out in a text file and pass them as an argument to the app.

## Visualization Layers

Token Explorer has a few different visualization layers that you can toggle on and off.

### Token Probabilities

It can be very helpful to see the probabilities of each token when generated, in part so we can see where our model might be going wrong. You can press `e` to toggle the probability view.

![Probabilities](./imgs/probabilities.png)

In the image above we've used the entire output of an LLM as our prompt. This allows us to understand better what the model was reasoning about when it generated the output. Notice for example that the model was basically certain the answer was 72.

### Token Entropies

Another way to understand the model's reasoning is to look at the entropy of the token distribution. Entropy represents how uncertain it is about the next token chosen. The highest (normalized) entropy is 1 (which means all tokens look like reasonable choices). The lowest is 0 (which means the model is certain about the next token).

You can simply press `e` again to enable the entropy view.

![Entropy](./imgs/entropies.png)

Pressing `e` again will cycle to the token influence view.

### Token Attention Influence

The token attention influence view shows how much each token in the prompt influences the prediction of the next token. This is calculated using attention patterns from the model's heads, filtered to focus on the most important attention heads.

How it works:
1. Calculates attention variance across all model heads
2. Selects top-K heads with highest variance (most informative)
3. Extracts attention weights from these heads for the last token
4. Aggregates and normalizes these weights to show influence scores

In this view:
- Each token is colored based on its attention influence
- Green indicates low influence
- Purple indicates high influence
- The legend shows the influence scale from 0.0 to 1.0

This visualization helps you understand which parts of the prompt are most important for the model's next token prediction. For example, you might see that the model pays more attention to recent tokens or to specific keywords in the prompt.

Pressing `e` again will return you back to the default view.

### Layer Analysis

Token Explorer provides two ways to analyze how predictions evolve through the model's layers, toggled with the `p` key. **Note: This feature is disabled by default** due to its computational expense. To enable it, run the app with the `--layer_prob` flag.

When enabled, pressing `p` will cycle through:

1. Layer Probabilities (first press):
   - Shows raw probability values for the token at each layer
   - Red (0.0) to blue (1.0) color scale
   - Helps visualize how strongly each layer predicts the token
   - Useful for understanding prediction strength across layers

2. Layer Correlations (second press):
   - Shows normalized prediction values for each layer
   - Values scaled to [0,1] range for easier visualization
   - Red (0.0) to blue (1.0) color scale
   - Helps understand how prediction evolves through the model

For ML researchers:
- Layer probabilities show the direct softmax outputs at each layer, revealing how the model's confidence in a token changes through the network
- Layer correlations are normalized to highlight relative changes in prediction strength, making it easier to identify which layers contribute most to the final prediction
- The progression through layers can reveal interesting patterns:
  * Early layers may show diffuse probabilities across many tokens
  * Middle layers often show emerging preferences
  * Final layers typically converge on the chosen token

Press `p` again to turn off layer visualization.

If you try to use this feature without enabling it with the `--layer_prob` flag, you'll see a message explaining how to enable it.

## Example Workflow

Let's try to understand our GSM8K prompt a bit better. The plaintext prompt is:

```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Reasoning: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
Answer: 72
```

First let's understand the model's answer. We'll start by loading the prompt into Token Explorer, and then backup using the `left` arrow key until we get to the answer token.

![Backing up to the answer](./imgs/backing_up_to_answer.png)

Here we can see that the model was basically certain about the answer, which makes sense given that the prompt is a simple arithmetic problem. As we can see, the model assigns a probability of essentially 1 to the answer starting with '7'. Recall that we could also see this visually be looking at the 'probabilities' layer.

It looks like our model is doing great, but let's go back to the entropy layer to see if we can find places to explore. Notice that the token 'Natalia' has higher entropy than the other tokens, which means the model is more uncertain about which token to choose next.

![Entropy](./imgs/natalia.png)

I'm curious what's happening there. I want to back up, but at the same time, don't want to lose my place in the prompt. I can use the `d` copy my prompt as a new prompt.

![Swapping prompts](./imgs/workflow1.png)

Now I can rewind until the token 'Natalia' and see if we can understand what's happening there, while still preserving my place in the prompt.

When we look at the probabilities for the next token, we can see that the model is confused between a range of choices.

![Probabilities](./imgs/workflow2.png)

I'm curious if this part was important for our getting the correct answer. To explore we'll:

- Create a copy of this point in the prompt with 'd'
- use 'right' arrow to fill until the end.

Here's the probability view for the new prompt:

![Exploring](./imgs/workflow3.png)

You can see both that we *did* still get the correct answer, and that the path I chose was fairly low probability for a bit. So we've learned something interesting! Even if we perturb the prompt to a low-probability path in the middle of it's reasoning, we still get the correct answer!

## Configuration

The configuration is done in the `config.toml` file. Here are the key settings you can modify:

- **Model**: The `model` section defaults to `Qwen/Qwen2.5-0.5B`. However, Token Explorer is *far* from optimized for performance, so it's best to use a smaller model for now.

- **Display Settings**: 
  - `tokens_to_show`: Number of tokens to display in the table (default: 30)
  - `cache_size`: Maximum number of cached token sequences (default: 100)

Layer probability and correlation calculations are controlled exclusively by the `--layer_prob` command-line flag.
