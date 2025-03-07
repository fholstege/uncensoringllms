import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import DIRECTION_FOLDER
from utils import set_seed, str_to_bool


# Custom style configurations
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'axes.labelpad': 10,
    'axes.titlepad': 20,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold'
})

# If you want to use LaTeX rendering for better typography
plt.rcParams.update({
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
})

model_name= 'Qwen 1.8B'
colors = {model_name:'gray', 'After Removing\npolitical censorship direction':'darkorange', 'After Removing\nRefusal direction':'cornflowerblue'}


def compute_cosine_similarity(dir1, dir2, abs_val=False):
    """Compute cosine similarity between two direction vectors."""
    print('shape of dir1:', dir1.shape)
    print('shape of dir2:', dir2.shape)

    inproduct = torch.dot(dir1.squeeze(0), dir2.squeeze(0))
    norm1 = torch.norm(dir1)
    norm2 = torch.norm(dir2)
    sim = inproduct / (norm1 * norm2)
    
    if abs_val:
        return torch.abs(sim)
    return sim

def main(args):
    set_seed()
    
    # Get all direction files for the specified model
    model_prefix = args.model_id.replace("/", "_")
    dir_files = os.listdir(DIRECTION_FOLDER)
    
    # Filter only relevant files for this model
    model_files = [f for f in dir_files if f.startswith(model_prefix)]
    
    # Separate files by direction type
    type1_files = [f for f in model_files if f"_refusal_dir_{args.dir_type1}.pt" in f]
    type2_files = [f for f in model_files if f"_refusal_dir_{args.dir_type2}.pt" in f]
    
    # Sort files by layer number
    type1_files.sort(key=lambda f: int(f.split("_layer")[1].split("_")[0]))
    type2_files.sort(key=lambda f: int(f.split("_layer")[1].split("_")[0]))
    
    print(f"Found {len(type1_files)} {args.dir_type1} files and {len(type2_files)} {args.dir_type2} files")
    
    if not type1_files or not type2_files:
        print("Error: No direction files found for one or both types")
        return
    
    # Match files by layer
    layers = []
    similarities = []
    
    # Process only the specified number of layers or all if last_x_layers is 0
    for type1_file in type1_files:
        layer_num = int(type1_file.split("_layer")[1].split("_")[0])
        
        # Find corresponding file for type2
        matching_file = next((f for f in type2_files if f"_layer{layer_num}_" in f), None)
        if matching_file:
            # Load both directions
            dir1_data = torch.load(os.path.join(DIRECTION_FOLDER, type1_file))
            dir2_data = torch.load(os.path.join(DIRECTION_FOLDER, matching_file))
            
            dir1 = dir1_data['direction']
            dir2 = dir2_data['direction']
            
            # Compute cosine similarity with absolute value setting from args
            sim = compute_cosine_similarity(dir1, dir2, abs_val=args.abs).item()
            
            layers.append(layer_num)
            similarities.append(sim)
            
            print(f"Layer {layer_num}: Cosine similarity = {sim:.4f}")
    
    # Limit to last X layers if specified
    if args.last_x_layers > 0 and args.last_x_layers < len(layers):
        layers = layers[-args.last_x_layers:]
        similarities = similarities[-args.last_x_layers:]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(layers, similarities, marker='o', linestyle='-', linewidth=2, color='blue')
    
    # Update title to indicate if using absolute similarity
    abs_text = "Absolute cosine" if args.abs else "Cosine"
    plt.title(f'{abs_text} similarity between {args.dir_type1} and {args.dir_type2} direction \nfor: Qwen 1.8B')
    plt.xlabel('Layer')
    plt.ylabel(f'{abs_text}cosine similarity')
    plt.grid(True)
    plt.xticks(layers, rotation=45)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Set y-axis limits based on whether we're using absolute values
    if args.abs:
        plt.ylim(-0.1, 1.1)  # For absolute similarity: [0, 1]
    else:
        plt.ylim(-1.1, 1.1)  # For regular similarity: [-1, 1]
    
    # Save plot with indication of absolute calculation
    abs_suffix = "_abs" if args.abs else ""
    output_file = f"{model_prefix}_cosine_sim_{args.dir_type1}_vs_{args.dir_type2}{abs_suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show plot if requested
    if args.show_plot:
        plt.show()
    
    # Print overall statistics
    print(f"Average cosine similarity: {np.mean(similarities):.4f}")
    print(f"Max cosine similarity: {max(similarities):.4f} at layer {layers[similarities.index(max(similarities))]}")
    print(f"Min cosine similarity: {min(similarities):.4f} at layer {layers[similarities.index(min(similarities))]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute cosine similarity between two direction types')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen-1_8B-chat",
                        help='HuggingFace model ID')
    parser.add_argument('--dir_type1', type=str, default='refusal',
                        help='First direction type')
    parser.add_argument('--dir_type2', type=str, default='censorship',
                        help='Second direction type')
    parser.add_argument('--last_x_layers', type=int, default=23,
                        help='Process only the last X layers')
    parser.add_argument('--show_plot', action='store_true',
                        help='Display plot in addition to saving it')
    parser.add_argument('--abs', type=str,
                        help='Compute absolute cosine similarity')
                        
    args = parser.parse_args()
    args.abs = str_to_bool(args.abs)

    main(args)
