def probability_to_color(probability, alpha=1.0):
    """
    Maps a probability value (0.0-1.0) to a color on a blue-red scale.
    Blue represents high probability (1.0)
    Red represents low probability (0.0)
    
    Args:
        probability (float): Probability value between 0.0 and 1.0
        alpha (float, optional): Alpha/opacity value between 0.0 and 1.0. Defaults to 1.0.
    
    Returns:
        str: RGBA color string (format: 'rgba(r, g, b, a)')
    """
    # Ensure probability is in valid range
    probability = max(0, min(1, probability))
    
    # Red component (high when probability is low)
    red = int(255 * (1 - probability))
    
    # Blue component (high when probability is high)
    blue = int(255 * probability)
    
    # Green component (kept at 0 for a cleaner red-blue gradient)
    green = 0
    
    # Return rgba string
    return f"rgba({red}, {green}, {blue}, {alpha})"

def entropy_to_color(entropy, alpha=1.0):
    """
    Maps a normalized entropy value (0.0-1.0) to a grayscale color.
    White (255,255,255) represents highest entropy (1.0)
    Black (0,0,0) represents lowest entropy (0.0)
    
    Args:
        entropy (float): Normalized entropy value between 0.0 and 1.0
        alpha (float, optional): Alpha/opacity value between 0.0 and 1.0. Defaults to 1.0.
    
    Returns:
        str: RGBA color string (format: 'rgba(r, g, b, a)')
    """
    # Ensure entropy is in valid range
    entropy = max(0, min(1, entropy))
    
    # For grayscale, all RGB components have the same value
    # Higher entropy = lighter color (closer to white)
    value = int(255 * entropy)
    
    # Return rgba string
    return f"rgba({value}, {value}, {value}, {alpha})"

def influence_to_color(influence):
    """Convert influence score to a color (green to purple)"""
    # Green (low) to purple (high)
    return f"#{0:02x}{int(255 * (1 - influence)):02x}{int(255 * influence):02x}"

def bias_to_color(bias):
    """Convert local token bias score to a color (orange to blue)"""
    # Orange (low) to blue (high)
    scaled_bias = min(max(bias, 0.0), 1.0)  # Ensure bias is in [0,1]
    return f"#{int(255 * (1 - scaled_bias)):02x}{int(128 * (1 - scaled_bias)):02x}{int(255 * scaled_bias):02x}"

def energy_to_color(energy, min_energy, max_energy):
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

def gradient_attribution_to_color(attribution):
    """Convert gradient attribution score to a color (green to blue)
    
    Lower attribution (green) = less important
    Higher attribution (blue) = more important
    """
    # Ensure attribution is in [0,1] range
    attribution = min(max(attribution, 0.0), 1.0)
    
    # Green (low attribution) to blue (high attribution)
    return f"#{0:02x}{int(255 * (1 - attribution)):02x}{int(255 * attribution):02x}"
