#!/usr/bin/env python3
"""
Qwen2.5 Factory Batch Export Script
Automatically handles export tasks for multiple Qwen model configurations
"""

import os
import sys
import subprocess
import tempfile
import time
import shutil
from pathlib import Path

# Define all Qwen configurations
CONFIGS = [
    # qwen2.5 0.5b-base
    {
        "model_name_or_path": "../models/qwen/qwen2.5-0.5b-base",
        "adapter_name_or_path": "../output/qwen2.5-0.5b-base-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-0.5b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 1
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-0.5b-base",
        "adapter_name_or_path": "../output/qwen2.5-0.5b-base-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-0.5b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 1
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-0.5b-base",
        "adapter_name_or_path": "../output/qwen2.5-0.5b-base-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-0.5b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 1
    },
    # qwen2.5 0.5b-instruct
    {
        "model_name_or_path": "../models/qwen/qwen2.5-0.5b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-0.5b-instruct-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-0.5b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 1
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-0.5b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-0.5b-instruct-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-0.5b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 1
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-0.5b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-0.5b-instruct-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-0.5b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 1
    },
    # qwen2.5 1.5b-base
    {
        "model_name_or_path": "../models/qwen/qwen2.5-1.5b-base",
        "adapter_name_or_path": "../output/qwen2.5-1.5b-base-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-1.5b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 3
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-1.5b-base",
        "adapter_name_or_path": "../output/qwen2.5-1.5b-base-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-1.5b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 3
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-1.5b-base",
        "adapter_name_or_path": "../output/qwen2.5-1.5b-base-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-1.5b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 3
    },
    # qwen2.5 1.5b-instruct
    {
        "model_name_or_path": "../models/qwen/qwen2.5-1.5b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-1.5b-instruct-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-1.5b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 3
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-1.5b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-1.5b-instruct-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-1.5b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 3
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-1.5b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-1.5b-instruct-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-1.5b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 3
    },
    # qwen2.5 3b-base
    {
        "model_name_or_path": "../models/qwen/qwen2.5-3b-base",
        "adapter_name_or_path": "../output/qwen2.5-3b-base-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-3b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 6
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-3b-base",
        "adapter_name_or_path": "../output/qwen2.5-3b-base-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-3b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 6
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-3b-base",
        "adapter_name_or_path": "../output/qwen2.5-3b-base-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-3b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 6
    },
    # qwen2.5 3b-instruct
    {
        "model_name_or_path": "../models/qwen/qwen2.5-3b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-3b-instruct-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-3b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 6
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-3b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-3b-instruct-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-3b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 6
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-3b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-3b-instruct-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-3b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 6
    },
    # qwen2.5 7b-base
    {
        "model_name_or_path": "../models/qwen/qwen2.5-7b-base",
        "adapter_name_or_path": "../output/qwen2.5-7b-base-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-7b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 14
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-7b-base",
        "adapter_name_or_path": "../output/qwen2.5-7b-base-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-7b-base-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 14
    },
    # qwen2.5 7b-instruct
    {
        "model_name_or_path": "../models/qwen/qwen2.5-7b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-7b-instruct-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-7b-instruct-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 14
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-7b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-7b-instruct-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-7b-instruct-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 14
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-7b-instruct",
        "adapter_name_or_path": "../output/qwen2.5-7b-instruct-finetune-lora-wiki-5e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-7b-instruct-finetune-lora-wiki-5e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 14
    },
    # qwen2.5 14b-base
    {
        "model_name_or_path": "../models/qwen/qwen2.5-14b-base",
        "adapter_name_or_path": "../output/qwen2.5-14b-base-finetune-lora-wiki-1e-4",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-14b-base-finetune-lora-wiki-1e-4",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 28
    },
    {
        "model_name_or_path": "../models/qwen/qwen2.5-14b-base",
        "adapter_name_or_path": "../output/qwen2.5-14b-base-finetune-lora-wiki-1e-5",
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": "../models/qwen/qwen2.5-14b-base-finetune-lora-wiki-1e-5",
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "estimated_size_gb": 28
    }
]

def get_disk_usage(path="/"):
    """Get disk usage information"""
    usage = shutil.disk_usage(path)
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)
    free_gb = usage.free / (1024**3)
    used_percent = (used_gb / total_gb) * 100
    return total_gb, used_gb, free_gb, used_percent

def check_disk_space(required_gb, path="/"):
    """Check if there's enough disk space"""
    _, _, free_gb, used_percent = get_disk_usage(path)
    
    if used_percent > 95:
        print(f"Warning: Disk usage is at {used_percent:.1f}%, you should clean up some space first")
        return False
    
    if free_gb < required_gb:
        print(f"Not enough disk space: need {required_gb}GB, but only {free_gb:.1f}GB available")
        return False
    
    return True

def print_disk_status():
    """Print disk status"""
    total, used, free, used_percent = get_disk_usage()
    print(f"\nDisk Status:")
    print(f"   Total capacity: {total:.1f}GB")
    print(f"   Used: {used:.1f}GB ({used_percent:.1f}%)")
    print(f"   Available: {free:.1f}GB")
    
    if used_percent > 90:
        print("   Critical: Disk space is severely low!")
    elif used_percent > 80:
        print("   Warning: Disk space is getting tight")
    else:
        print("   Good: Plenty of disk space available")

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
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    return os.path.exists(adapter_config_path)

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
    print("Qwen2.5 Factory Batch Export Tool")
    print(f"Total configurations to process: {len(CONFIGS)}")
    
    # Show disk status
    print_disk_status()
    
    # Check basic disk space
    _, _, free_gb, used_percent = get_disk_usage()
    if used_percent > 98:
        print("\nCritical: Disk space is severely low, cannot continue! Please clean up some space first.")
        print("Suggested actions:")
        print("1. sudo rm -rf /tmp/*")
        print("2. conda clean --all")
        print("3. pip cache purge")
        print("4. Delete unnecessary large files")
        return
    
    # User selection
    print("\nChoose operation mode:")
    print("1. Export all")
    print("2. Skip existing export directories")
    print("3. Export only specific model sizes (0.5b/1.5b/3b/7b/14b)")
    print("4. Export only specific model types (base/instruct)")
    print("5. Smart selection based on disk space (recommended)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    filtered_configs = CONFIGS.copy()
    
    if choice == "3":
        size = input("Enter model size (0.5b/1.5b/3b/7b/14b): ").strip()
        filtered_configs = [c for c in CONFIGS if f"-{size}-" in c['adapter_name_or_path']]
        print(f"Filtered to {len(filtered_configs)} {size} model configurations")
    elif choice == "4":
        model_type = input("Enter model type (base/instruct): ").strip()
        filtered_configs = [c for c in CONFIGS if f"-{model_type}" in c['adapter_name_or_path']]
        print(f"Filtered to {len(filtered_configs)} {model_type} model configurations")
    elif choice == "5":
        # Smart selection: choose models based on available space
        if free_gb > 30:
            print("Plenty of space, can handle all models")
        elif free_gb > 20:
            print("Limited space, suggest skipping 14B models")
            filtered_configs = [c for c in CONFIGS if c['estimated_size_gb'] <= 14]
        elif free_gb > 10:
            print("Space is tight, suggest only processing small models (<=3B)")
            filtered_configs = [c for c in CONFIGS if c['estimated_size_gb'] <= 6]
        else:
            print("Very low space, suggest only processing 0.5B and 1.5B models")
            filtered_configs = [c for c in CONFIGS if c['estimated_size_gb'] <= 3]
        print(f"Smart filtered to {len(filtered_configs)} configurations")
    
    # Start processing
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    start_time = time.time()
    
    for i, config in enumerate(filtered_configs, 1):
        model_name = config['adapter_name_or_path'].split('/')[-1]
        
        # Check if adapter files exist
        if not check_adapter_exists(config['adapter_name_or_path']):
            print(f"\n[{i}/{len(filtered_configs)}] Skipping {model_name}: adapter config file doesn't exist")
            skipped_count += 1
            continue
        
        # Check if export directory already exists
        if choice == "2" and check_export_dir_exists(config['export_dir']):
            print(f"\n[{i}/{len(filtered_configs)}] Skipping {model_name}: export directory already exists")
            skipped_count += 1
            continue
        
        # Check disk space
        required_space = config.get('estimated_size_gb', 10)
        if not check_disk_space(required_space):
            print(f"\n[{i}/{len(filtered_configs)}] Skipping {model_name}: not enough disk space")
            skipped_count += 1
            continue
        
        # Create temporary config file
        temp_config = create_temp_config(config)
        
        try:
            print(f"\n[{i}/{len(filtered_configs)}] Processing: {model_name}")
            print(f"Estimated space needed: {required_space}GB")
            
            # Run export
            if run_export(temp_config, model_name):
                success_count += 1
                print_disk_status()  # Show disk status after each success
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
    print_disk_status()
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total: {success_count + failed_count + skipped_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()