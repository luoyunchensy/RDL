import os
import json
import random
from collections import defaultdict

# Step 1: Extract all unique relations and deduplicate
def extract_relations(data_file):
    """Extract all unique relations from the original data file"""
    relations = set()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                for triplet in data.get('triplets', []):
                    rel = triplet.get('label', '')
                    if rel:
                        relations.add(rel)
            except json.JSONDecodeError:
                print(f"Warning: Unable to parse line: {line[:50]}...")
                continue
    
    return sorted(list(relations))

# Prompt templates without candidate relations
def get_prompt_templates():
    """Get a list of prompt templates that do NOT use candidate relations"""
    return [
        "Extract the relation triplet (head entity, tail entity, relation) from the given sentence.",
        "Analyze the following sentence and identify the relationship between entities.",
        "In the text \"{sentence}\", find the relation triplet (head, tail, relation).",
        "The following sentence contains a relationship: {sentence}\n\nIdentify the head entity, tail entity, and the type of relation.",
        "This sentence describes a relationship: {sentence}\n\nExtract the head entity, tail entity, and relation type."
    ]

# Step 2: Convert the dataset to Alpaca format without candidate relations
def convert_to_alpaca(data_file):
    """Convert the dataset to Alpaca format without using candidate relations"""
    alpaca_data = []
    prompt_templates = get_prompt_templates()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                for triplet in data.get('triplets', []):
                    tokens = triplet.get('tokens', [])
                    if not tokens:
                        continue
                    
                    sentence = ' '.join(tokens)
                    head_indices = triplet.get('head', [])
                    tail_indices = triplet.get('tail', [])
                    relation = triplet.get('label', '')
                    
                    head_entity = ' '.join([tokens[i] for i in head_indices if 0 <= i < len(tokens)])
                    tail_entity = ' '.join([tokens[i] for i in tail_indices if 0 <= i < len(tokens)])
                    
                    template = random.choice(prompt_templates)
                    instruction = template.format(sentence=sentence)
                    
                    alpaca_sample = {
                        "instruction": instruction,
                        "input": sentence,
                        "output": f"head: {head_entity}\ntail: {tail_entity}\nrelation: {relation}"
                    }
                    
                    alpaca_data.append(alpaca_sample)
            except json.JSONDecodeError:
                print(f"Warning: Unable to parse line: {line[:50]}...")
                continue
    
    return alpaca_data

# Main function
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert FewRL dataset to Alpaca format without candidate relations")
    parser.add_argument("--input", required=True, help="Path to the input FewRL dataset file")
    parser.add_argument("--output", required=True, help="Path to the output Alpaca format dataset file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for template selection")
    parser.add_argument("--extract-only", action="store_true", help="Only extract relations and save them to output file")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Extract all relations
    print("Extracting unique relations...")
    all_relations = extract_relations(args.input)
    print(f"Extracted {len(all_relations)} unique relations")
    
    # Save only relations if requested
    if args.extract_only:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_relations, f, ensure_ascii=False, indent=2)
        print(f"Relations saved to {args.output}")
        return
    
    # Convert to Alpaca format
    print("Converting to Alpaca format (no candidate relations)...")
    alpaca_data = convert_to_alpaca(args.input)
    print(f"Generated {len(alpaca_data)} Alpaca format samples")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Alpaca format data saved to {args.output}")
    
    # Show a sample
    if alpaca_data:
        print("\nSample example:")
        sample = alpaca_data[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input']}")
        print(f"Output: {sample['output']}")

if __name__ == "__main__":
    main()
    
    # python convert_data.py --input path/to/input.jsonl --output path/to/output.json
    
    # python convert_data_withoutcandidate.py --input data/zero_rte/fewrel/unseen_5_seed_0/train.jsonl --output LLaMA-Factory/data/fewrl_train_0.json