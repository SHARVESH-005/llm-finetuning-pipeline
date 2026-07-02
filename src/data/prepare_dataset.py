import os
import yaml
from datasets import load_dataset

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_dpo_prompt(example):
    # Ultrafeedback binarized has columns: prompt, chosen, rejected
    # The chosen and rejected are typically lists of messages (e.g. role: user/assistant)
    # TRL DPOTrainer expects dicts with 'prompt', 'chosen', and 'rejected' string values
    
    # Simple extraction for demo purposes
    prompt_msgs = example['prompt']
    chosen_msgs = example['chosen']
    rejected_msgs = example['rejected']
    
    # We will format this into a basic string prompt for standard LLMs
    # (In production, you'd use a specific chat template)
    return {
        "prompt": prompt_msgs,
        "chosen": chosen_msgs,
        "rejected": rejected_msgs
    }

def main():
    config = load_config()
    dataset_name = config['data']['dataset_name']
    processed_dir = config['data']['processed_dir']
    
    print(f"Downloading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    # Optional formatting if needed for TRL compatibility
    # TRL's DPOTrainer requires standard keys: prompt, chosen, rejected
    print("Formatting dataset for DPO and SFT...")
    
    # For SFT, we only train on the 'chosen' responses.
    # For DPO, we train on chosen vs rejected.
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # We save a local subset for faster iterations during dev/Colab
    subset_size = 1000 # Just for demo limits
    
    train_subset = dataset[config['data']['train_split']].select(range(subset_size))
    test_subset = dataset[config['data']['test_split']].select(range(100))
    
    train_path = os.path.join(processed_dir, 'train')
    test_path = os.path.join(processed_dir, 'test')
    
    train_subset.save_to_disk(train_path)
    test_subset.save_to_disk(test_path)
    
    print(f"Successfully processed and saved subsets to {processed_dir}")

if __name__ == "__main__":
    main()
