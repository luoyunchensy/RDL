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
                # Parse JSON data from each line
                data = json.loads(line.strip())
                
                # Extract relations from triplets
                for triplet in data.get('triplets', []):
                    rel = triplet.get('label', '')
                    if rel:
                        relations.add(rel)
            except json.JSONDecodeError:
                print(f"Warning: Unable to parse line: {line[:50]}...")
                continue
    
    return sorted(list(relations))

# Generate a small candidate relation list for each sample
def generate_candidate_relations(all_relations, true_relation, num_candidates=4):
    """Generate a small candidate relation list including the true relation
    
    Args:
        all_relations: List of all possible relations
        true_relation: The true relation of the sample
        num_candidates: Total number of candidate relations (including the true relation)
    
    Returns:
        A list of candidate relations including the true relation
    """
    # Ensure the true relation is in the candidate list
    candidates = [true_relation]
    
    # Randomly select the remaining candidates from other relations
    other_relations = [r for r in all_relations if r != true_relation]
    num_to_select = min(num_candidates - 1, len(other_relations))
    
    if num_to_select > 0:
        selected = random.sample(other_relations, num_to_select)
        candidates.extend(selected)
    
    # Shuffle the order to avoid positional bias
    random.shuffle(candidates)
    
    return candidates

# List of prompt templates in English
def get_prompt_templates():
    """Get a list of diverse prompt templates in English"""
    return [
        "Extract the relation triplet from the given sentence. Based on the candidate relation list, identify the head entity, tail entity, and the relation between them.",
        "Analyze the following sentence and identify the relationship. Possible relation types include: {relations}. Find the head entity, tail entity, and the type of relation between them.",
        "In the text \"{sentence}\", there is a relation triplet. Please find the correct relation from the candidates {relations}, and identify the head entity and tail entity.",
        "The following sentence contains a relationship: {sentence}\n\nPlease select the correct relation type from the candidates and indicate the head entity and tail entity. Candidate relations: {relations}",
        "This sentence describes a relationship: {sentence}\n\nChoose one from the following relation types: {relations}\n\nExtract the head entity and tail entity that participate in this relationship, along with the relation type."
    ]

# Step 2: Convert the dataset to Alpaca format
def convert_to_alpaca(data_file, all_relations, num_candidates=4):
    """Convert the dataset to Alpaca format
    
    Args:
        data_file: Path to the original data file
        all_relations: List of all possible relations
        num_candidates: Number of candidate relations for each sample
    """
    alpaca_data = []
    prompt_templates = get_prompt_templates()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse JSON data from each line
                data = json.loads(line.strip())
                
                # Process each sample
                for triplet in data.get('triplets', []):
                    tokens = triplet.get('tokens', [])
                    if not tokens:
                        continue
                    
                    # Construct the original sentence
                    sentence = ' '.join(tokens)
                    
                    # Get head, tail and relation
                    head_indices = triplet.get('head', [])
                    tail_indices = triplet.get('tail', [])
                    relation = triplet.get('label', '')
                    
                    # Extract head and tail entity text
                    head_entity = ' '.join([tokens[i] for i in head_indices if 0 <= i < len(tokens)])
                    tail_entity = ' '.join([tokens[i] for i in tail_indices if 0 <= i < len(tokens)])
                    
                    # Generate candidate relation list for the current sample
                    candidate_relations = generate_candidate_relations(
                        all_relations, relation, num_candidates=num_candidates
                    )
                    
                    # Randomly select a prompt template
                    template = random.choice(prompt_templates)
                    
                    # Replace placeholders in the template
                    instruction = template.format(
                        sentence=sentence,
                        relations=', '.join(candidate_relations)
                    ) if '{' in template else template
                    
                    # Construct Alpaca format sample
                    alpaca_sample = {
                        "instruction": instruction,
                        "input": f"Candidate relations: {', '.join(candidate_relations)}\n\nSentence: {sentence}",
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
    
    parser = argparse.ArgumentParser(description="Convert FewRL dataset to Alpaca format")
    parser.add_argument("--input", required=True, help="Path to the input FewRL dataset file")
    parser.add_argument("--output", required=True, help="Path to the output Alpaca format dataset file")
    parser.add_argument("--candidates", type=int, default=4, help="Number of candidate relations for each sample (including the correct answer)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate relation selection")
    parser.add_argument("--extract-only", action="store_true", help="Only extract relations and save them to output file")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
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
    print("Converting to Alpaca format...")
    alpaca_data = convert_to_alpaca(args.input, all_relations, num_candidates=args.candidates)
    print(f"Generated {len(alpaca_data)} Alpaca format samples")
    
    # Save results
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Alpaca format data saved to {args.output}")
    
    # Show a sample example
    if alpaca_data:
        print("\nSample example:")
        sample = alpaca_data[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input']}")
        print(f"Output: {sample['output']}")

if __name__ == "__main__":
    main()
    #python convert_data.py --input data/zero_rte/fewrel/unseen_5_seed_0/train.jsonl --output relations.json --extract-only
    # python convert_data.py --input data/zero_rte/fewrel/unseen_5_seed_0/train.jsonl --output finetune_data/fewrel_train_10.json --candidates 10


# python convert_data.py --input data/zero_rte/fewrel/unseen_5_seed_0/train.jsonl --output LLaMA-Factory/data/fewrl_train_2.json --candidates 2