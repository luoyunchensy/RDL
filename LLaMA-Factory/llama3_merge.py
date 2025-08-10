#!/usr/bin/env python3
"""
LLaMA Factory Batch Export Script
Automatically handles export tasks for multiple model configurations
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

# Define all configurations
CONFIGS = [
    # llama3 8b-base
    {
        "model_name_or_path": "../models/llama/llama-3.1-8b-base",
        "adapter_name_or_path": "./output/llama-3.1-8b-base-finetune-lora-wiki-1e-4",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.1-8b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.1-8b-base",
        "adapter_name_or_path": "./output/llama-3.1-8b-base-finetune-lora-wiki-1e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.1-8b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.1-8b-base",
        "adapter_name_or_path": "./output/llama-3.1-8b-base-finetune-lora-wiki-5e-5/checkpoint-2600",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.1-8b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    # llama3 8b-instruct
    {
        "model_name_or_path": "../models/llama/llama-3.1-8b-instruct",
        "adapter_name_or_path": "./output/llama-3.1-8b-instruct-finetune-lora-wiki-1e-4",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.1-8b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.1-8b-instruct",
        "adapter_name_or_path": "./output/llama-3.1-8b-instruct-finetune-lora-wiki-1e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.1-8b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.1-8b-instruct",
        "adapter_name_or_path": "./output/llama-3.1-8b-instruct-finetune-lora-wiki-5e-5/checkpoint-2600",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.1-8b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    # llama3 3b-base
    {
        "model_name_or_path": "../models/llama/llama-3.2-3b-base",
        "adapter_name_or_path": "./output/llama-3.2-3b-base-finetune-lora-wiki-1e-4",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-3b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-3b-base",
        "adapter_name_or_path": "./output/llama-3.2-3b-base-finetune-lora-wiki-1e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-3b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-3b-base",
        "adapter_name_or_path": "./output/llama-3.2-3b-base-finetune-lora-wiki-5e-5/checkpoint-3700",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-3b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    # llama3 3b-instruct
    {
        "model_name_or_path": "../models/llama/llama-3.2-3b-instruct",
        "adapter_name_or_path": "./output/llama-3.2-3b-instruct-finetune-lora-wiki-1e-4",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-3b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-3b-instruct",
        "adapter_name_or_path": "./output/llama-3.2-3b-instruct-finetune-lora-wiki-1e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-3b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-3b-instruct",
        "adapter_name_or_path": "./output/llama-3.2-3b-instruct-finetune-lora-wiki-5e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-3b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    # llama3 1b-base
    {
        "model_name_or_path": "../models/llama/llama-3.2-1b-base",
        "adapter_name_or_path": "./output/llama-3.2-1b-base-finetune-lora-wiki-1e-4",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-1b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-1b-base",
        "adapter_name_or_path": "./output/llama-3.2-1b-base-finetune-lora-wiki-1e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-1b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-1b-base",
        "adapter_name_or_path": "./output/llama-3.2-1b-base-finetune-lora-wiki-5e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-1b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    # llama3 1b-instruct
    {
        "model_name_or_path": "../models/llama/llama-3.2-1b-instruct",
        "adapter_name_or_path": "./output/llama-3.2-1b-instruct-finetune-lora-wiki-1e-4",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-1b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-1b-instruct",
        "adapter_name_or_path": "./output/llama-3.2-1b-instruct-finetune-lora-wiki-1e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-1b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    },
    {
        "model_name_or_path": "../models/llama/llama-3.2-1b-instruct",
        "adapter_name_or_path": "./output/llama-3.2-1b-instruct-finetune-lora-wiki-5e-5",
        "template": "llama3",
        "trust_remote_code": True,
        "export_dir": "../models/llama/llama-3.2-1b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False
    }
]

def create_temp_config(config):
    """Create a temporary configuration file"""
    yaml_content = f"""model_name_or_path: {config['model_name_or_path']}
adapter_name_or_path: {config['adapter_name_or_path']}
template: {config['template']}
trust_remote_code: {str(config['trust_remote_code']).lower()}
export_dir: {config['export_dir']}
export_size: {config['export_size']}
export_device: {config['export_device']}
export_legacy_format: {str(config['export_legacy_format']).lower()}
"""
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_file.write(yaml_content)
    temp_file.close()
    return temp_file.name

def check_adapter_exists(adapter_path):
    """Check if adapter files exist"""
    return os.path.exists(adapter_path)

def check_export_dir_exists(export_dir):
    """Check if export directory already exists"""
    return os.path.exists(export_dir)

def run_export(config_file, model_name):
    """Run the export command"""
    cmd = ["llamafactory-cli", "export", config_file]
    print(f"\n{'='*60}")
    print(f"Exporting: {model_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Export successful!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Only show last 500 characters
        return True
    except subprocess.CalledProcessError as e:
        print(f"Export failed! Error code: {e.returncode}")
        if e.stdout:
            print("Output:", e.stdout[-500:])
        if e.stderr:
            print("Error:", e.stderr[-500:])
        return False
    except FileNotFoundError:
        print("Couldn't find llamafactory-cli command, make sure it's properly installed and in your PATH")
        return False

def main():
    print("LLaMA Factory Batch Export Tool")
    print(f"Total configurations to process: {len(CONFIGS)}")
    
    # User selection
    print("\nChoose operation mode:")
    print("1. Export all")
    print("2. Skip existing export directories")
    print("3. Export only specific model sizes (1b/3b/8b)")
    print("4. Export only specific model types (base/instruct)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    filtered_configs = CONFIGS.copy()
    
    if choice == "3":
        size = input("Enter model size (1b/3b/8b): ").strip()
        filtered_configs = [c for c in CONFIGS if f"-{size}-" in c['adapter_name_or_path']]
        print(f"Filtered to {len(filtered_configs)} {size} model configurations")
    elif choice == "4":
        model_type = input("Enter model type (base/instruct): ").strip()
        filtered_configs = [c for c in CONFIGS if f"-{model_type}" in c['adapter_name_or_path']]
        print(f"Filtered to {len(filtered_configs)} {model_type} model configurations")
    
    # Start processing
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    for i, config in enumerate(filtered_configs, 1):
        model_name = config['adapter_name_or_path'].split('/')[-1]
        
        # Check if adapter files exist
        if not check_adapter_exists(config['adapter_name_or_path']):
            print(f"\n[{i}/{len(filtered_configs)}] Skipping {model_name}: adapter files don't exist")
            skipped_count += 1
            continue
        
        # Check if export directory already exists
        if choice == "2" and check_export_dir_exists(config['export_dir']):
            print(f"\n[{i}/{len(filtered_configs)}] Skipping {model_name}: export directory already exists")
            skipped_count += 1
            continue
        
        # Create temporary config file
        temp_config = create_temp_config(config)
        
        try:
            print(f"\n[{i}/{len(filtered_configs)}] Processing: {model_name}")
            
            # Run export
            if run_export(temp_config, model_name):
                success_count += 1
            else:
                failed_count += 1
                
                # Ask if user wants to continue
                if failed_count == 1:  # Only ask on first failure
                    continue_choice = input("\nExport failed, continue processing remaining configurations? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        print("User chose to stop processing")
                        break
                        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_config)
            except:
                pass
    
    # Show summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("Batch export completed!")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total: {success_count + failed_count + skipped_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()