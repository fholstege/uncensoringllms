import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from tqdm import tqdm
from datetime import datetime
import sys
from accelerate import disk_offload
from utils import set_seed, FileTextStreamer, get_model_layers
import random
from config import *
import torch.nn as nn
from typing import Optional, Tuple, List, Dict


def direction_ablation_hook(activation, p):

    # apply projection to activation (1, 1, 2048)
    proj = torch.matmul(activation, p)

    return activation - proj

class AblationDecoderLayer(nn.Module):
    def __init__(self, original_layer, refusal_direction):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer

        # Store the refusal direction in the correct device and dtype upfront
        self.r = refusal_direction.T

        # take the unit
        self.r_unit = self.r / torch.norm(self.r)

        self.projection =  torch.matmul(self.r_unit, self.r_unit.T)


    def forward(self, *args, **kwargs):
        # get the hidden states
        hidden_states = args[0]

        # apply the projection to all the hidden states
        proj = torch.matmul(hidden_states, self.projection)

        # remove the projection
        ablated = hidden_states - proj

        # apply to the first argument
        args = (ablated,) + args[1:]

        # return the forward pass of the original layer
        return self.original_layer.forward(*args, **kwargs)



def load_refusal_directions(model_id: str, dir_type: str) -> Dict[int, torch.Tensor]:
    """Load all available refusal directions for the model."""
    base_name = model_id.replace("/", "_")
    directions = {}

    # get the files in direction folder
    
    # Look for files matching the pattern
    for file in os.listdir(DIRECTION_FOLDER):
        if file.startswith(base_name) and "layer" in file and file.endswith("_refusal_dir_{}.pt".format(dir_type)):
            try:
                # Extract layer number from filename
                layer_str = file.split("layer")[1].split("_")[0]
                layer_idx = int(layer_str)
                print(f"Found direction for layer {layer_idx}")
                
                # Load the direction
                data = torch.load(DIRECTION_FOLDER + '/' + file)
                if isinstance(data, dict) and 'direction' in data:
                    directions[layer_idx] = data['direction']
                    print(f"Loaded direction for layer {layer_idx} from {file}")
                else:
                    # Handle legacy format
                    directions[layer_idx] = data
                    print(f"Loaded legacy direction for layer {layer_idx} from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return directions

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
   
    # Set up output file
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model_id.replace("/", "_")
        args.output_file = f"responses_{model_name}_{timestamp}.txt"
    
    print(f"Responses will be saved to {args.output_file}")
    
    # Set up the device and put the model in inference mode
    torch.inference_mode()
    #torch.set_default_device(args.device)
    
    print(f"Loading model {args.model_id}...")
    # Load the model and its tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        trust_remote_code=True, 
        device_map=args.device,
        torch_dtype=torch.float16
    ) # Set model to evaluation mode

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Load all available refusal directions
    print(f"Loading refusal directions for model {args.model_id} and type {args.dir_type}...")
    refusal_directions = load_refusal_directions(args.model_id, args.dir_type)
    
    if not refusal_directions:
        raise ValueError(f"No refusal directions found for model {args.model_id}. "
                         f"Run compute_dir.py first to generate directions.")
    
    # Determine which layers to apply ablation to based on last_x_layers
    available_layers = sorted(list(refusal_directions.keys()))
    
    if args.last_x_layers <= 0:
        target_layers = available_layers  # Use all available layers
    else:
        # Use only the last X layers for which we have directions
        target_layers = available_layers[-args.last_x_layers:] if args.last_x_layers <= len(available_layers) else available_layers
    
    print(f"Applying ablation to {len(target_layers)} layers: {target_layers}")
    # Get all model layers
    layers = get_model_layers(model)
    total_layers = len(layers)

    # Apply ablation to selected layers
    for layer_idx in target_layers:
        if layer_idx in refusal_directions and layer_idx < total_layers:
            # Convert to correct dtype once, not on every forward pass
            refusal_dir = refusal_directions[layer_idx].to(model.device).to(torch.float16)
            model.transformer.h[layer_idx] = AblationDecoderLayer(layers[layer_idx], refusal_dir)
            print(f"Applied ablation to layer {layer_idx}")
        else:
            print(f"Skipping layer {layer_idx}: direction not available or layer index out of bounds")
    
    
    print(f"Loading prompts from {args.test_file}...")
    # Load prompts
    with open(args.test_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    
    print(f"Processing {len(prompts)} prompts...")


    
    # Process each prompt and save to file
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        # Write header information
        output_file.write(f"# Responses from {args.model_id}\n")
        output_file.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"# Prompts file: {args.test_file}\n")
        
        # Process each prompt
        for i, prompt in enumerate(prompts, 1):
            separator = f"\n\n{'='*80}\n"
            prompt_header = f"PROMPT {i}/{len(prompts)}:\n{'-'*80}\n"
            response_header = f"{'-'*80}\nRESPONSE:\n"
            
            # Write prompt to file
            output_file.write(separator)
            output_file.write(prompt_header)
            output_file.write(f"{prompt}\n")
            output_file.write(response_header)
            
            # Also print to console
            print(separator)
            print(prompt_header)
            print(prompt)
            print(response_header)
            
            # Create a streamer that writes to both console and file
            streamer = FileTextStreamer(tokenizer, output_file)
            
            # Format prompt according to model's chat template with fallback
            try:
                # Try using the chat template first
                input_ids = tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                attention_mask = None
            except (ValueError, NotImplementedError, AttributeError):
                # Fallback for models without chat template (like Qwen/Qwen-1_8B-chat)
                tokenized = tokenizer(prompt, return_tensors="pt")
                input_ids = tokenized.input_ids.to(model.device)
                attention_mask = tokenized.attention_mask.to(model.device)
                
            # Generate response
            with torch.no_grad():
                model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    streamer=streamer,
                    temperature=TEMPERATURE,
                    do_sample=False if TEMPERATURE == 0.0 else True,
                    max_new_tokens=MAX_NEW_TOKENS
                )
    
    print(f"\n\n{'='*80}")
    print(f"Completed processing all prompts. Responses saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Examine model responses to specific prompts')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen-1_8B-chat",
                        help='HuggingFace model ID')
    parser.add_argument('--device', type=str, default="mps",
                        help='Device to run model on (e.g., "cuda", "mps", "cpu")')
    parser.add_argument('--test_file', type=str, default="data/censorship/censorship_test_split.txt",
                        help='File containing test prompts')
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save responses (default: responses_TIMESTAMP.txt)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
  
    parser.add_argument('--dir_type', type=str, default="censorship",
                        help='Type of direction to apply (e.g., refusal, censorship)')
    parser.add_argument('--last_x_layers', type=int, default=0,
                        help='Process only the last X layers (0 = all layers)')
    args = parser.parse_args()
    main(args)
