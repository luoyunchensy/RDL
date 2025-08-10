from vllm import LLM, SamplingParams
import json
import os
import re
import time
from tqdm import tqdm
import numpy as np
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

os.environ['VLLM_USE_MODELSCOPE'] = 'True'

def extract_relations_from_data(data_path):
    """
    Extract all unique relation types from test data
    
    Args:
        data_path: Path to the data file
        
    Returns:
        List of unique relation types
    """
    relations = set()
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            triplet = data["triplets"][0]
            relation = triplet["label"]
            relations.add(relation)
    
    return sorted(list(relations))

def build_prompt(tokens, relation_list, examples=None):
    """
    Build prompts in exactly the same format as during fine-tuning.
    
    Args:
        tokens: List of input text tokens
        relation_list: List of possible relations
        examples: Optional few-shot examples list
        
    Returns:
        Built prompt string, exactly matching the fine-tuning format
    """
    sentence = " ".join(tokens)
    
    # Make sure the format is exactly consistent, including line breaks and spaces
    instruction = (
        f"This sentence describes a relationship: {sentence}\n\n"
        f"Choose one from the following relation types: {', '.join(relation_list)}\n\n"
        f"Extract the head entity and tail entity that participate in this relationship, along with the relation type."
        f"You must respond ONLY with the extracted entities and relation in exactly this format:\n"
        f"head: [head entity]\n"
        f"tail: [tail entity]\n"
        f"relation: [relation type]\n\n"
    )
    
    input_text = (
        f"Candidate relations: {', '.join(relation_list)}\n\n"
        f"Sentence: {sentence}"
    )
    
    # Modify prompt format to match fine-tuning exactly
    if examples and len(examples) > 0:
        few_shot_examples = ""
        for i, ex in enumerate(examples):
            if i > 0:
                few_shot_examples += "\n\n"  # Add extra blank lines between examples
            
            few_shot_examples += f"Instruction: This sentence describes a relationship: {ex['text']}\n\n"
            few_shot_examples += f"Choose one from the following relation types: {', '.join(relation_list)}\n\n"
            few_shot_examples += f"Extract the head entity and tail entity that participate in this relationship, along with the relation type.\n\n"
            few_shot_examples += f"Input: Candidate relations: {', '.join(relation_list)}\n\nSentence: {ex['text']}\n\n"
            few_shot_examples += f"Output: head: {ex['head']}\ntail: {ex['tail']}\nrelation: {ex['relation']}"
        
        # Finally build the complete prompt, ensuring it matches the fine-tuning format exactly
        prompt = f"{few_shot_examples}\n\nInstruction: {instruction}\n\nInput: {input_text}"
    else:
        prompt = f"Instruction: {instruction}\n\nInput: {input_text}"
    
    return prompt

def extract_first_triple(text):
    """
    Extract the first complete triple from output that might contain multiple triples or extra text,
    adapted for fine-tuned model output format.
    
    Args:
        text: Model output text
        
    Returns:
        Cleaned triple text, or original text if extraction fails
    """
    # Try to find head/tail/relation combinations
    head_match = re.search(r"head(?:\s*entity)?(?:\s*[:-])?\s*([^\n]+)", text, re.IGNORECASE)
    tail_match = re.search(r"tail(?:\s*entity)?(?:\s*[:-])?\s*([^\n]+)", text, re.IGNORECASE)
    rel_match = re.search(r"(?:relation|relation\s*type)(?:\s*[:-])?\s*([^\n]+)", text, re.IGNORECASE)
    
    if all([head_match, tail_match, rel_match]):
        head = head_match.group(1).strip()
        tail = tail_match.group(1).strip()
        rel = rel_match.group(1).strip()
        
        # Build clean triple
        clean_triple = f"head: {head}\ntail: {tail}\nrelation: {rel}"
        return clean_triple
    
    # If we can't find a complete triple, return the original text
    return text

def parse_predicted_triple(text):
    """
    Parse relation triples from model-generated text, adapted for fine-tuned model output format.
    
    Args:
        text: Model-generated text
        
    Returns:
        Parsed triple (head, rel, tail) or None (if parsing fails)
    """
    try:
        # Parse for fine-tuned model output format
        head_match = re.search(r"head(?:\s*entity)?(?:\s*[:-])?\s*([^\n]+)", text, re.IGNORECASE)
        tail_match = re.search(r"tail(?:\s*entity)?(?:\s*[:-])?\s*([^\n]+)", text, re.IGNORECASE)
        rel_match = re.search(r"(?:relation|relation\s*type)(?:\s*[:-])?\s*([^\n]+)", text, re.IGNORECASE)
        
        if not all([head_match, tail_match, rel_match]):
            return None
            
        head = head_match.group(1).strip()
        tail = tail_match.group(1).strip()
        rel = rel_match.group(1).strip()
        
        # Validate if the relation is from the provided list (can be a bit flexible since models might output slightly different formats)
        if rel not in relation_list and not any(r.lower() == rel.lower() for r in relation_list):
            # Try to find the most similar relation
            similar_rel = None
            max_similarity = 0
            for r in relation_list:
                similarity = sum(1 for a, b in zip(r.lower(), rel.lower()) if a == b) / max(len(r), len(rel))
                if similarity > max_similarity and similarity > 0.8:  # 80% similarity threshold
                    max_similarity = similarity
                    similar_rel = r
            
            if similar_rel:
                rel = similar_rel
            else:
                return None
            
        return (head, rel, tail)
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.0, top_p=1.0, max_model_len=2048, batch_size=8):
    """
    Get model generation results in batches. Supports local fine-tuned models.
    
    Args:
        prompts: List of prompts
        model: Model name or path
        tokenizer: Tokenizer (optional)
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature parameter
        top_p: Generation top_p parameter
        max_model_len: Maximum model length
        batch_size: Batch size
        
    Returns:
        List of generated texts
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Initialize model (if not already initialized)
    if not hasattr(get_completion, 'llm') or get_completion.llm is None:
        print(f"Initializing model: {model}")
        
        # Check if it's a local fine-tuned model path
        if os.path.exists(model) and not model.startswith("hf://"):
            # Use transformers to load model and get model type
            from transformers import AutoConfig
            try:
                config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                model_type = config.model_type
                print(f"Detected model type: {model_type}")
                
                # Set appropriate tokenizer_path for fine-tuned models
                if tokenizer is None:
                    # For LoRA fine-tuned models, find the original base model
                    adapter_config_path = os.path.join(model, "adapter_config.json")
                    if os.path.exists(adapter_config_path):
                        try:
                            with open(adapter_config_path, 'r') as f:
                                adapter_config = json.load(f)
                            if 'base_model_name_or_path' in adapter_config:
                                tokenizer = adapter_config['base_model_name_or_path']
                                print(f"Using base model tokenizer: {tokenizer}")
                            else:
                                tokenizer = model
                        except Exception as e:
                            print(f"Failed to read adapter_config.json: {e}")
                            tokenizer = model
                    else:
                        tokenizer = model
            except Exception as e:
                print(f"Unable to detect model type: {e}")
                tokenizer = model
                
        # Initialize vLLM engine
        try:
            get_completion.llm = LLM(
                model=model, 
                tokenizer=tokenizer or model,
                max_model_len=max_model_len,
                trust_remote_code=True,
                enforce_eager=True,
                dtype="float16"
            )
        except Exception as e:
            print(f"vLLM initialization failed: {e}")
            print("Trying alternative method to load model...")
            
            # If vLLM fails, try loading with Hugging Face approach
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            try:
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
                tokenizer_obj = AutoTokenizer.from_pretrained(
                    tokenizer or model,
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # Create a simple wrapper class to simulate vLLM interface
                class HFWrapper:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                        
                    def generate(self, prompts, sampling_params):
                        outputs = []
                        for prompt in prompts:
                            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                            with torch.no_grad():
                                output = self.model.generate(
                                    inputs.input_ids,
                                    max_new_tokens=sampling_params.max_tokens,
                                    temperature=sampling_params.temperature,
                                    top_p=sampling_params.top_p,
                                    do_sample=sampling_params.temperature > 0
                                )
                            output_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                            # Create an object similar to vLLM output
                            class Output:
                                def __init__(self, text):
                                    self.outputs = [type('', (), {'text': text})]
                            outputs.append(Output(output_text))
                        return outputs
                
                get_completion.llm = HFWrapper(model_obj, tokenizer_obj)
                print("Successfully loaded model using Hugging Face directly")
            except Exception as e:
                print(f"Hugging Face loading also failed: {e}")
                raise RuntimeError("Unable to load model, please check model path or try different loading method") from e
    
    all_outputs = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(prompts)//batch_size + 1} (size: {len(batch_prompts)})")
        
        try:
            outputs = get_completion.llm.generate(batch_prompts, sampling_params)
            batch_results = [out.outputs[0].text.strip() for out in outputs]
            all_outputs.extend(batch_results)
        except Exception as e:
            print(f"Batch {i//batch_size + 1} generation error: {e}")
            # When error occurs, add empty results for each prompt in this batch
            all_outputs.extend([""] * len(batch_prompts))
    
    return all_outputs

def write_log_from_args_and_metrics(
    model_size: str,
    dataset: str,
    unseen_k: int,
    seed: int,
    metrics: dict,
    output_file: str,
    output_dir: str = "results",
    inference_time=None, 
    inference_time_per_sample=None,
    mode_str: str = "zero_shot"
):
    # Build log file name
    log_filename = f"{model_size}_{dataset}_unseen{unseen_k}_seed{seed}_{mode_str}.log"
    log_path = os.path.join(output_dir, log_filename)
    os.makedirs(output_dir, exist_ok=True)

    # Write log content
    with open(log_path, "w") as f:
        f.write("Evaluation Summary\n")
        f.write(f"Total samples: {metrics['total_gold']}\n")
        f.write(f"Valid predictions: {metrics['total_valid_pred']} ({metrics['total_valid_pred']/metrics['total_gold']:.2%})\n")
        f.write(f"Fully correct predictions: {metrics['correct_all']} ({metrics['correct_all']/metrics['total_gold']:.2%})\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Head+Relation accuracy: {metrics['head_rel_accuracy']:.4f}\n")
        f.write(f"Relation+Tail accuracy: {metrics['rel_tail_accuracy']:.4f}\n")
        f.write(f"Head+Tail accuracy: {metrics['head_tail_accuracy']:.4f}\n")
        f.write(f"Inference time: {inference_time:.2f} seconds (per sample {inference_time_per_sample:.4f}s)\n")
        f.write(f"Raw results saved to: {output_file}\n")

    print(f"Log written to: {log_path}")

def calculate_metrics(gold_triplets, predicted_triplets):
    """
    Calculate performance metrics for relation extraction.
    
    Args:
        gold_triplets: List of gold standard triples
        predicted_triplets: List of predicted triples
        
    Returns:
        Dictionary containing various metrics
    """
    correct_head_rel = 0  # Head entity + relation correct
    correct_rel_tail = 0  # Relation + tail entity correct
    correct_head_tail = 0  # Head entity + tail entity correct
    correct_all = 0  # All triples completely correct
    
    # Process samples without prediction results
    valid_predictions = []
    for pred in predicted_triplets:
        if pred is None:
            valid_predictions.append(None)
        else:
            valid_predictions.append(pred)
    
    for gold, pred in zip(gold_triplets, valid_predictions):
        if pred is None:
            continue
            
        gold_head, gold_rel, gold_tail = gold
        pred_head, pred_rel, pred_tail = pred
        
        if gold_head == pred_head and gold_rel == pred_rel:
            correct_head_rel += 1
            
        if gold_rel == pred_rel and gold_tail == pred_tail:
            correct_rel_tail += 1
            
        if gold_head == pred_head and gold_tail == pred_tail:
            correct_head_tail += 1
            
        if gold == pred:  # Perfect match
            correct_all += 1
    
    # Calculate standard metrics
    total_pred = sum(1 for p in valid_predictions if p is not None)
    total_gold = len(gold_triplets)
    
    precision = correct_all / total_pred if total_pred > 0 else 0
    recall = correct_all / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': correct_all / total_gold if total_gold > 0 else 0,
        # Detailed metrics
        'head_rel_accuracy': correct_head_rel / total_gold if total_gold > 0 else 0,
        'rel_tail_accuracy': correct_rel_tail / total_gold if total_gold > 0 else 0,
        'head_tail_accuracy': correct_head_tail / total_gold if total_gold > 0 else 0,
        # Statistics
        'total_gold': total_gold,
        'total_valid_pred': total_pred,
        'correct_all': correct_all,
        'correct_head_rel': correct_head_rel,
        'correct_rel_tail': correct_rel_tail,
        'correct_head_tail': correct_head_tail
    }

def sample_few_shot_examples(data_path, num_examples=3, seed=42):
    """
    Sample a few examples from the dataset for few-shot prompting.
    
    Args:
        data_path: Path to the data file
        num_examples: Number of examples to sample
        seed: Random seed
        
    Returns:
        List of examples
    """
    np.random.seed(seed)
    
    all_examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            triplet = data["triplets"][0]
            tokens = triplet["tokens"]
            
            head_indices = triplet["head"]
            tail_indices = triplet["tail"]
            
            head = " ".join([tokens[i] for i in head_indices])
            tail = " ".join([tokens[i] for i in tail_indices])
            
            all_examples.append({
                "text": " ".join(tokens),
                "head": head,
                "tail": tail,
                "relation": triplet["label"]
            })
    
    # Randomly select examples
    if len(all_examples) <= num_examples:
        return all_examples
    
    selected_indices = np.random.choice(len(all_examples), num_examples, replace=False)
    return [all_examples[i] for i in selected_indices]

def process_data(data_path, model, tokenizer=None, 
                 max_samples=None, 
                 few_shot=False, 
                 num_examples=3,
                 output_dir="results",
                 temperature=0.0,
                 max_tokens=100,
                 dataset=None,
                 relation_list=None,
                 unseen_k=None,
                 seed=None
                 ):
    """
    Process dataset and evaluate model performance.
    
    Args:
        data_path: Path to the data file
        model: Model name or path
        tokenizer: Tokenizer (optional)
        max_samples: Maximum number of samples to process
        few_shot: Whether to use few-shot learning
        num_examples: Number of few-shot examples
        relation_list: Candidate relation list
        output_dir: Output directory
        temperature: Generation temperature parameter
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary containing results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare examples for few-shot learning
    examples = None
    if few_shot:
        examples = sample_few_shot_examples(data_path, num_examples)
        print(f"Using {len(examples)} few-shot examples")
    
    # Load and prepare data
    prompts = []
    raw_texts = []
    gold_triplets = []
    
    sample_count = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and sample_count >= max_samples:
                break
                
            data = json.loads(line)
            triplet = data["triplets"][0]
            tokens = triplet["tokens"]
            
            # Build prompt
            prompt = build_prompt(tokens, relation_list, examples)
            prompts.append(prompt)
            
            # Save original text
            raw_texts.append(" ".join(tokens))
            
            # Extract gold standard triples
            head_indices = triplet["head"]
            tail_indices = triplet["tail"]
            
            head = " ".join([tokens[i] for i in head_indices])
            tail = " ".join([tokens[i] for i in tail_indices])
            label = triplet["label"]
            
            gold_triplets.append((head, label, tail))
            sample_count += 1
    
    print(f"Running inference on {len(prompts)} samples (temperature={temperature}, max_tokens={max_tokens})...")
    
    # Get model predictions
    start_time = time.time()
    predicted_texts = get_completion(prompts, model, tokenizer, max_tokens=max_tokens, temperature=temperature)
    inference_time = time.time() - start_time
    
    # Parse prediction results
    parsed_triples = []
    for pred_text in tqdm(predicted_texts, desc="Parsing predictions"):
        parsed = parse_predicted_triple(pred_text)
        parsed_triples.append(parsed)
    
    # Calculate metrics
    metrics = calculate_metrics(gold_triplets, parsed_triples)
    
    # Prepare detailed sample results
    samples = []
    for i, (raw_text, gold, pred_text, parsed) in enumerate(zip(raw_texts, gold_triplets, predicted_texts, parsed_triples)):
        sample_result = {
            'id': i,
            'text': raw_text,
            'gold': {
                'head': gold[0],
                'rel': gold[1],
                'tail': gold[2]
            },
            'prediction': pred_text,
            'parsed': None if parsed is None else {
                'head': parsed[0],
                'rel': parsed[1],
                'tail': parsed[2]
            },
            'is_correct': parsed == gold if parsed else False,
            'is_valid': parsed is not None
        }
        samples.append(sample_result)
    
    # Prepare output results
    results = {
        'model': model,
        'data_path': data_path,
        'samples_processed': len(prompts),
        'few_shot': few_shot,
        'num_examples': num_examples if few_shot else 0,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'metrics': metrics,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(prompts),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'samples': samples
    }
    
    # Save results
    mode_str = "few_shot" if few_shot else "zero_shot"
    result_filename = f"{model_name_str}_{dataset}_unseen{unseen_k}_seed{seed}_{mode_str}.json"
    result_file = os.path.join(output_dir, result_filename)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    results["output_file"] = result_file  # Add here

    # Print summary
    print("\nResults Summary:")
    print(f"Samples processed: {len(prompts)}")
    print(f"Valid predictions: {metrics['total_valid_pred']} ({metrics['total_valid_pred']/len(prompts):.2%})")
    print(f"Fully correct: {metrics['correct_all']} ({metrics['correct_all']/len(prompts):.2%})")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Head+Relation accuracy: {metrics['head_rel_accuracy']:.4f}")
    print(f"Relation+Tail accuracy: {metrics['rel_tail_accuracy']:.4f}")
    print(f"Head+Tail accuracy: {metrics['head_tail_accuracy']:.4f}")
    print(f"Inference time: {inference_time:.2f} seconds (per sample {inference_time/len(prompts):.4f}s)")
    print(f"Results saved to: {result_file}")
    
    return results

def preview_dataset(data_path, model, max_samples=10, temperature=0.0, max_tokens=100):
    """Preview the first N samples and prediction results from the dataset"""
    prompts = []
    raw_texts = []
    gold_triplets = []
    
    print(f"Loading first {max_samples} samples from {data_path}...")
    
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
                
            data = json.loads(line)
            triplet = data["triplets"][0]
            tokens = triplet["tokens"]
            
            # Build prompt - use the same format as fine-tuning
            prompt = build_prompt(tokens, relation_list)
            prompts.append(prompt)
            
            # Save original text
            raw_texts.append(" ".join(tokens))
            
            # Extract gold standard triples
            head_indices = triplet["head"]
            tail_indices = triplet["tail"]
            
            head = " ".join([tokens[i] for i in head_indices])
            tail = " ".join([tokens[i] for i in tail_indices])
            label = triplet["label"]
            
            gold_triplets.append((head, label, tail))
    
    # Get model predictions
    print(f"Generating responses... (temperature={temperature}, max_tokens={max_tokens})")
    predicted_texts = get_completion(prompts, model, temperature=temperature, max_tokens=max_tokens)
    
    # Print results
    print(f"\n{Fore.CYAN}Preview and Analysis of First {len(prompts)} Samples{Style.RESET_ALL}")
    print("=" * 80)
    
    correct_count = 0
    
    for i, (raw_text, gold, pred_text) in enumerate(zip(raw_texts, gold_triplets, predicted_texts)):
        parsed = parse_predicted_triple(pred_text)
        
        print(f"{Fore.YELLOW}[Sample {i+1}]{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Text: {raw_text}{Style.RESET_ALL}")
        
        # Print prompt
        print(f"\n{Fore.BLUE}Prompt:{Style.RESET_ALL}")
        print(f"{prompts[i]}")
        
        # Print gold standard triples
        print(f"\n{Fore.GREEN}Gold Standard:{Style.RESET_ALL}")
        print(f"  Head entity: {gold[0]}")
        print(f"  Relation:    {gold[1]}")
        print(f"  Tail entity: {gold[2]}")
        
        # Print model raw output
        print(f"\n{Fore.BLUE}Model Raw Output:{Style.RESET_ALL}")
        print(f"{pred_text}")
        
        # Print cleaned output
        cleaned = extract_first_triple(pred_text)
        if cleaned != pred_text:
            print(f"\n{Fore.CYAN}Cleaned Output:{Style.RESET_ALL}")
            print(f"{cleaned}")
        
        # Print parsing results
        print(f"\n{Fore.MAGENTA}Parsing Results:{Style.RESET_ALL}")
        if parsed:
            # Check if each part matches
            head_match = parsed[0] == gold[0]
            rel_match = parsed[1] == gold[1]
            tail_match = parsed[2] == gold[2]
            
            print(f"  Head entity: {parsed[0]} {Fore.GREEN+'✓'+Style.RESET_ALL if head_match else Fore.RED+'✗'+Style.RESET_ALL}")
            print(f"  Relation:    {parsed[1]} {Fore.GREEN+'✓'+Style.RESET_ALL if rel_match else Fore.RED+'✗'+Style.RESET_ALL}")
            print(f"  Tail entity: {parsed[2]} {Fore.GREEN+'✓'+Style.RESET_ALL if tail_match else Fore.RED+'✗'+Style.RESET_ALL}")
            
            # Check if completely matched
            if head_match and rel_match and tail_match:
                print(f"\n{Fore.GREEN}Result: ✓ Perfect match!{Style.RESET_ALL}")
                correct_count += 1
            else:
                print(f"\n{Fore.YELLOW}Result: ⚠️ Partial match{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Parsing failed!{Style.RESET_ALL}")
        
        print("\n" + "=" * 80)
    
    # Print statistics
    print(f"\n{Fore.CYAN}Statistics:{Style.RESET_ALL}")
    print(f"Total samples: {len(prompts)}")
    print(f"Successfully parsed: {sum(1 for p in [parse_predicted_triple(t) for t in predicted_texts] if p is not None)}")
    print(f"Perfect matches: {correct_count} ({correct_count/len(prompts):.2%})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero-shot/Few-shot Relation Extraction Evaluation")
    parser.add_argument("--dataset", type=str, choices=["wiki", "fewrel"], default="wiki", help="Choose dataset")
    parser.add_argument("--unseen_k", type=int, default=5, help="Number of unseen relations")# 5/10/15
    parser.add_argument("--seed_k", type=int, choices=[0,1,2,3,4],default=0, help="Random seed number")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--few_shot", action="store_true", help="Whether to use few-shot learning")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature parameter")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--preview", action="store_true", help="Only preview first 10 samples")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned model, use this path if specified")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path for fine-tuned model, used for tokenizer")
    
    args = parser.parse_args()
    
    # Construct data path
    args.data = f"./data/zero_rte/{args.dataset}/unseen_{args.unseen_k}_seed_{args.seed_k}/test.jsonl"
    relation_list = extract_relations_from_data(args.data)
    print(f"Relation types extracted from test data: {relation_list}")

    args.model = args.model_path
    model_name_str = os.path.basename(args.model_path)
    # If base_model is specified, use it as tokenizer
    tokenizer_path = args.base_model if args.base_model else args.model_path

    
    mode_str = "few_shot" if args.few_shot else "zero_shot"
    log_file_name = f"{model_name_str}_{args.dataset}_unseen{args.unseen_k}_seed{args.seed_k}_{mode_str}.log"

    print("Starting relation extraction evaluation")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Mode: {'Few-shot' if args.few_shot else 'Zero-shot'}")
    print(f"Temperature: {args.temperature}")
    print(f"Max generation tokens: {args.max_tokens}")
    
    if args.few_shot:
        print(f"Number of examples: {args.num_examples}")
    
    if args.preview:
        preview_dataset(
            data_path=args.data,
            model=args.model,
            tokenizer=tokenizer_path,
            max_samples=10,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    else:
        results = process_data(
            data_path=args.data,
            model=args.model,
            tokenizer=tokenizer_path,
            max_samples=args.max_samples,
            few_shot=args.few_shot,
            num_examples=args.num_examples,
            output_dir=args.output_dir,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dataset=args.dataset,
            relation_list=relation_list,
            unseen_k=args.unseen_k,
            seed=args.seed_k
        )
        
        # 日志写入
        write_log_from_args_and_metrics(
            model_size=model_name_str,
            dataset=args.dataset,
            unseen_k=args.unseen_k,
            seed=args.seed_k,
            metrics=results["metrics"],
            inference_time=results["inference_time"],
            inference_time_per_sample=results["inference_time_per_sample"],
            output_file=os.path.basename(results["output_file"]),
            output_dir=args.output_dir,
            mode_str=mode_str
        )
    
    print("\nEvaluation completed!")

# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-0.5b-base
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-0.5b-base-finetune
# CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-0.5b-instruct 
# CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-0.5b-instruct-finetune    
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-1.5b-base
# CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-1.5b-instruct
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-3b-base
# CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-3b-instruct
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-7b-base
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-7b-base-finetune
# CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-7b-instruct
#note CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-7b-instruct-finetune
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-14b-base
# CUDA_VISIBLE_DEVICES=1 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-14b-instruct

# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-0.5b-base-finetune-full
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-0.5b-instruct-finetune-full

# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-1.5b-base-finetune-full
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-1.5b-base-finetune-full1
# CUDA_VISIBLE_DEVICES=0 python llm_for_rte.py --dataset fewrel --unseen_k 5 --seed_k 0 --model_path models/qwen/qwen2.5-1.5b-instruct-finetune-full