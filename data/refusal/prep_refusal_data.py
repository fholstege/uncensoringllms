import json
import os
import argparse
from pathlib import Path

def load_json_file(file_path):
    """Load and parse JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_prompts(data):
    """Extract prompts from the JSON data."""
    prompts = []
    for item in data:
        # Extract the prompt from the instruction field
        if "instruction" in item:
            prompt = item["instruction"]
            prompts.append(prompt)
    return prompts

def save_to_txt(prompts, output_file):
    """Save prompts to a text file, one per line."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            # Clean the prompt
            clean_prompt = prompt.strip().replace('\n', ' ')
            f.write(f"{clean_prompt}\n")
    
    print(f"Saved {len(prompts)} prompts to {output_file}")

def main(args):
   
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process harmful data
    harmful_train_prompts = []
    
    # Load and process harmful_train.json
    print("Loading harmful_train.json...")
    harmful_train_data = load_json_file(Path(args.data_dir) / "harmful_train.json")
    harmful_train_prompts.extend(extract_prompts(harmful_train_data))
    
    # Load and process harmful_val.json (to be combined with train)
    print("Loading harmful_val.json...")
    harmful_val_data = load_json_file(Path(args.data_dir) / "harmful_val.json")
    harmful_train_prompts.extend(extract_prompts(harmful_val_data))
    
    # Load harmful_test.json
    print("Loading harmful_test.json...")
    harmful_test_data = load_json_file(Path(args.data_dir) / "harmful_test.json")
    harmful_test_prompts = extract_prompts(harmful_test_data)
    
    # Process harmless data
    harmless_train_prompts = []
    
    # Load and process harmless_train.json
    print("Loading harmless_train.json...")
    harmless_train_data = load_json_file(Path(args.data_dir) / "harmless_train.json")
    harmless_train_prompts.extend(extract_prompts(harmless_train_data))
    
    # Load and process harmless_val.json (to be combined with train)
    print("Loading harmless_val.json...")
    harmless_val_data = load_json_file(Path(args.data_dir) / "harmless_val.json")
    harmless_train_prompts.extend(extract_prompts(harmless_val_data))
    
    # Load harmless_test.json
    print("Loading harmless_test.json...")
    harmless_test_data = load_json_file(Path(args.data_dir) / "harmless_test.json")
    harmless_test_prompts = extract_prompts(harmless_test_data)
    
    # Save to text files
    save_to_txt(harmful_train_prompts, output_dir / "harmful_train.txt")
    save_to_txt(harmful_test_prompts, output_dir / "harmful_test.txt")
    save_to_txt(harmless_train_prompts, output_dir / "harmless_train.txt")
    save_to_txt(harmless_test_prompts, output_dir / "harmless_test.txt")
    
    # Summary
    print("\nData preparation complete!")
    print(f"Harmful: {len(harmful_train_prompts)} training, {len(harmful_test_prompts)} test prompts")
    print(f"Harmless: {len(harmless_train_prompts)} training, {len(harmless_test_prompts)} test prompts")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare refusal data by converting JSON to TXT files")
    parser.add_argument("--data_dir", type=str, 
                        default="/Users/florisholstege/Documents/GitHub/uncensoringllms/data/refusal",
                        help="Directory containing the JSON files")
    parser.add_argument("--output_dir", type=str, 
                        default="/Users/florisholstege/Documents/GitHub/uncensoringllms/data/refusal",
                        help="Directory to save output text files")
   
    
    args = parser.parse_args()
    main(args)
