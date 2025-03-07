import random
import torch
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import get_model_layers, set_seed
from sklearn.decomposition import PCA
import os
import seaborn as sns

def visualize_hidden_states(model, tokenizer, harmful_inputs, harmless_inputs, layers_to_analyze, position):
    """Extract hidden states and visualize them using PCA."""
    all_hidden_states = []
    all_labels = []
    
    # Setup progress bar
    bar = tqdm(total=len(harmful_inputs) + len(harmless_inputs), desc="Extracting hidden states")
    
    # Extract hidden states for harmful inputs
    print("Processing harmful inputs...")
    for input_dict in harmful_inputs:
        bar.update(1)
        input_ids = input_dict['input_ids'].to(model.device)
        attention_mask = input_dict.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        for layer_idx in layers_to_analyze:
            # Get hidden state for this layer at the specified position using PyTorch
            hidden_state = outputs.hidden_states[layer_idx][:, position, :].cpu()
            all_hidden_states.append((hidden_state, layer_idx))
            all_labels.append(1)  # 1 for harmful
    
    # Extract hidden states for harmless inputs
    print("Processing harmless inputs...")
    for input_dict in harmless_inputs:
        bar.update(1)
        input_ids = input_dict['input_ids'].to(model.device)
        attention_mask = input_dict.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        for layer_idx in layers_to_analyze:
            # Get hidden state for this layer at the specified position using PyTorch
            hidden_state = outputs.hidden_states[layer_idx][:, position, :].cpu()
            all_hidden_states.append((hidden_state, layer_idx))
            all_labels.append(0)  # 0 for harmless
    
    bar.close()
    
    # Group hidden states by layer
    layer_to_hidden_states = {}
    layer_to_labels = {}
    
    for (hidden_state, layer_idx), label in zip(all_hidden_states, all_labels):
        if layer_idx not in layer_to_hidden_states:
            layer_to_hidden_states[layer_idx] = []
            layer_to_labels[layer_idx] = []
        
        layer_to_hidden_states[layer_idx].append(hidden_state.squeeze())
        layer_to_labels[layer_idx].append(label)
    
    # Create visualizations for each layer
    os.makedirs("activation_plots", exist_ok=True)
    
    for layer_idx in layers_to_analyze:
        # Stack tensors instead of numpy arrays
        hidden_states_tensor = torch.stack(layer_to_hidden_states[layer_idx])
        labels_tensor = torch.tensor(layer_to_labels[layer_idx])
        
        # Convert to numpy just for sklearn PCA (as it doesn't accept torch tensors)
        # We'll convert back to torch tensors after PCA
        hidden_states_np = hidden_states_tensor.numpy()
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_data_np = pca.fit_transform(hidden_states_np)
        reduced_data = torch.tensor(reduced_data_np)
        
        # Get boolean masks for plotting using PyTorch
        harmless_mask = labels_tensor == 0
        harmful_mask = labels_tensor == 1
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")
        
        # Plot data points using PyTorch tensors
        plt.scatter(
            reduced_data[harmless_mask, 0].numpy(), 
            reduced_data[harmless_mask, 1].numpy(), 
            c='blue', label='Censorship', alpha=0.7, s=50
        )
        plt.scatter(
            reduced_data[harmful_mask, 0].numpy(), 
            reduced_data[harmful_mask, 1].numpy(), 
            c='red', label='No censorship', alpha=0.7, s=50
        )
        
        # Add labels and title
        plt.xlabel(f'Principal Component 1 (Explained variance: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Principal Component 2 (Explained variance: {pca.explained_variance_ratio_[1]:.2f})')
        plt.title(f'PCA of Hidden States at Layer {layer_idx}')
        plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"activation_plots/layer_{layer_idx}_position_{position}_pca.png", dpi=300)
        plt.close()
        
        print(f"Plot for layer {layer_idx} saved to activation_plots/layer_{layer_idx}_position_{position}_pca.png")

        # Compute separation metrics using PyTorch
        harmless_points = reduced_data[harmless_mask]
        harmful_points = reduced_data[harmful_mask]
        
        if len(harmless_points) > 0 and len(harmful_points) > 0:
            harmless_centroid = harmless_points.mean(dim=0)
            harmful_centroid = harmful_points.mean(dim=0)
            
            centroid_distance = torch.norm(harmless_centroid - harmful_centroid).item()
            print(f"Layer {layer_idx} - Distance between centroids: {centroid_distance:.4f}")

def main(args):
    # Set up the device and seed
    torch.inference_mode()
    set_seed()
    
    print(f"Loading model {args.model_id}...")
    # Load the model and its tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # Get all model layers
    layers = get_model_layers(model)
    total_layers = len(layers)
    
    # Parse layers to analyze
    if args.layers:
        layers_to_analyze = [int(layer) for layer in args.layers.split("-")]
    else:
        # Default to last 4 layers
        layers_to_analyze = [total_layers//4, total_layers//2, 3*total_layers//4, total_layers-1]
    
    # Ensure layers are within valid range
    layers_to_analyze = [l for l in layers_to_analyze if 0 <= l < total_layers]
    
    pos = args.position
    print(f'Analyzing {len(layers_to_analyze)} layers: {layers_to_analyze} at position: {pos}')
    
    print(f"Loading instructions from {args.harmful_file} and {args.harmless_file}...")
    # Load the harmful and harmless instructions
    with open(args.harmful_file, "r") as f:
        harmful = [line.strip() for line in f.readlines() if line.strip()]

    with open(args.harmless_file, "r") as f:
        harmless = [line.strip() for line in f.readlines() if line.strip()]

    # Sample the instructions
    harmful_instructions = random.sample(harmful, min(len(harmful), args.num_instructions))
    harmless_instructions = random.sample(harmless, min(len(harmless), args.num_instructions))

    print("Tokenizing instructions...")
    # Tokenize instructions with fallback for models without chat template
    harmful_inputs = []
    harmless_inputs = []
    
    for insn in harmful_instructions:
        try:
            # Try using the chat template first
            input_text = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": insn}],
                add_generation_prompt=True,
                tokenize=False  # Return string instead of tokens
            )
            tokens = tokenizer(input_text, return_tensors="pt")
        except (ValueError, NotImplementedError, AttributeError):
            # Fallback for models without chat template
            tokens = tokenizer(f"USER: {insn}\nASSISTANT:", return_tensors="pt")
        harmful_inputs.append(tokens)
    
    for insn in harmless_instructions:
        try:
            # Try using the chat template first
            input_text = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": insn}],
                add_generation_prompt=True,
                tokenize=False  # Return string instead of tokens
            )
            tokens = tokenizer(input_text, return_tensors="pt")
        except (ValueError, NotImplementedError, AttributeError):
            # Fallback for models without chat template
            tokens = tokenizer(insn, return_tensors="pt")
        harmless_inputs.append(tokens)
    
    # Visualize the hidden states
    visualize_hidden_states(model, tokenizer, harmful_inputs, harmless_inputs, layers_to_analyze, pos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize hidden state activations for harmful vs harmless prompts')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen-1_8B-chat",
                        help='HuggingFace model ID')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to run model on (e.g., "cuda", "mps", "cpu")')
    parser.add_argument('--harmful_file', type=str, default="harmful.txt",
                        help='File containing harmful instructions')
    parser.add_argument('--harmless_file', type=str, default="harmless.txt",
                        help='File containing harmless instructions')
    parser.add_argument('--num_instructions', type=int, default=100,
                        help='Number of instructions to sample from each category')
    parser.add_argument('--layers', type=str, default="",
                        help='Layers to analyze (e.g., "8-12-16-20"). If empty, will use 4 evenly spaced layers.')
    parser.add_argument('--position', type=int, default=-1,
                        help='Which position in the sequence to extract embeddings from')

    args = parser.parse_args()
    main(args)
