import os
import yaml
import subprocess
import json

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # We will evaluate the base model and then evaluate our fine-tuned DPO model
    # To prove improvement mathematically using lm-evaluation-harness
    
    tasks = "gsm8k,truthfulqa_mc1" # Standard benchmarks for reasoning and truthfulness
    
    # 1. Base Model Eval (Optional to run here, usually done once)
    base_model = config['model']['base_model_name_or_path']
    print(f"--- Running Eval on Base Model: {base_model} ---")
    
    # Example command using lm-eval
    # lm_eval --model hf --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tasks gsm8k --limit 10
    
    # We use a limit here for fast testing. Remove limit for full benchmark.
    base_cmd = [
        "lm_eval", 
        "--model", "hf", 
        "--model_args", f"pretrained={base_model}", 
        "--tasks", "gsm8k", 
        "--limit", "10",
        "--output_path", "./eval_results/base_model"
    ]
    
    print("Executing: " + " ".join(base_cmd))
    # subprocess.run(base_cmd) # Commented out so it doesn't auto-run and crash without GPU
    
    # 2. DPO Model Eval
    # To evaluate a PEFT model, lm-eval supports peft arg
    dpo_adapter = config['model']['dpo_adapter_path']
    print(f"\n--- Running Eval on DPO Aligned Model ---")
    
    dpo_cmd = [
        "lm_eval", 
        "--model", "hf", 
        "--model_args", f"pretrained={base_model},peft={dpo_adapter}", 
        "--tasks", "gsm8k", 
        "--limit", "10",
        "--output_path", "./eval_results/dpo_model"
    ]
    
    print("Executing: " + " ".join(dpo_cmd))
    # subprocess.run(dpo_cmd)
    
    print("\nEvaluation scripts generated. Run them in an environment with GPU to see TruthfulQA and GSM8K scores.")

if __name__ == "__main__":
    main()
