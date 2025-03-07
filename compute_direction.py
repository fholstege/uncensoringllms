import random
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import get_model_layers, set_seed
from config import DIRECTION_FOLDER
import numpy as np

def main(args):
    
    # Set up the device and put the model in inference mode
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
    
    # Determine which layers to process based on last_x_layers
    if args.last_x_layers <= 0:
        target_layers = list(range(total_layers))
    else:
        target_layers = list(range(max(0, total_layers - args.last_x_layers), total_layers))
    
    pos = args.position
    print(f'Computing for {len(target_layers)} layers: {target_layers} and position: {pos}')
    
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
            tokens = tokenizer(f"USER: {insn}\nASSISTANT:", return_tensors="pt")
        harmless_inputs.append(tokens)

    # Setup progress bar
    max_its = len(harmful_instructions) + len(harmless_instructions)
    bar = tqdm(total=max_its, desc="Generating outputs")

    # Function to generate outputs and extract hidden states
    def generate(token_dict):
        bar.update(n=1)
        input_ids = token_dict['input_ids'].to(model.device)
        attention_mask = token_dict['attention_mask'].to(model.device) if 'attention_mask' in token_dict else None
        
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False, 
            max_new_tokens=1, 
            return_dict_in_generate=True, 
            output_hidden_states=True
        )

    print("Generating outputs for harmful instructions...")
    harmful_outputs = [generate(input_dict) for input_dict in harmful_inputs]
    
    print("Generating outputs for harmless instructions...")
    harmless_outputs = [generate(input_dict) for input_dict in harmless_inputs]

    bar.close()

    print(f"Extracting hidden states from {len(target_layers)} layers...")
    # Process each target layer
    for layer_idx in target_layers:
        print(f"Processing layer {layer_idx}/{total_layers-1}...")
        
        # Extract hidden states at the specified layer and position
        harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
        harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]

        # Compute mean of hidden states for each category
        harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
        harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

        # Compute refusal direction as the normalized difference between harmful and harmless means
        refusal_dir = harmful_mean - harmless_mean
        refusal_dir = refusal_dir / refusal_dir.norm()

        # Save the computed direction along with metadata
        output_filename = args.model_id.replace("/", "_") + f"_layer{layer_idx}_refusal_dir_" + args.dir_type + ".pt"
        torch.save({
            'direction': refusal_dir,
            'layer_idx': layer_idx,
            'model_id': args.model_id,
            'position': pos,
            'total_layers': total_layers
        }, DIRECTION_FOLDER + '/' + output_filename)

if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute censorship direction for a language model')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen-1_8B-chat",
                        help='HuggingFace model ID')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to run model on (e.g., "cuda", "mps", "cpu")')
    parser.add_argument('--harmful_file', type=str, default="harmful.txt",
                        help='File containing harmful instructions')
    parser.add_argument('--harmless_file', type=str, default="harmless.txt",
                        help='File containing harmless instructions')
    parser.add_argument('--num_instructions', type=int, default=1000,
                        help='Number of instructions to sample')
    parser.add_argument('--last_x_layers', type=int, default=0,
                        help='Process only the last X layers (0 = all layers)')
    parser.add_argument('--position', type=int, default=-1,
                        help='Which position in the sequence to extract embeddings from')
    parser.add_argument('--dir_type', type=str, default='censorship',
                        help='Type of direction to compute (e.g., refusal, censorship)')

    args = parser.parse_args()
    main(args)