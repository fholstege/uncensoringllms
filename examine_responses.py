import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from tqdm import tqdm
from datetime import datetime
import sys
from accelerate import disk_offload
from utils import set_seed, FileTextStreamer
import random
from config import *


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

    
    print(f"Loading prompts from {args.prompts_file}...")
    # Load prompts
    with open(args.prompts_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

        if args.sample_size is not None:
            # Randomly sample a subset of prompts
            set_seed()
            index = torch.randperm(len(prompts))
            prompts = [prompts[i] for i in index[:args.sample_size]]

    
    print(f"Processing {len(prompts)} prompts...")
    
    # Process each prompt and save to file
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        # Write header information
        output_file.write(f"# Responses from {args.model_id}\n")
        output_file.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"# Prompts file: {args.prompts_file}\n")
        
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
    parser.add_argument('--prompts_file', type=str, default="harmful.txt",
                        help='File containing prompts to test')
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save responses (default: responses_TIMESTAMP.txt)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of prompts to sample from test file (default: all)')


    args = parser.parse_args()
    main(args)
