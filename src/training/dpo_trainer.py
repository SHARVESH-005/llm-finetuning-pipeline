import os
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_dpo_prompt(example):
    # Ultrafeedback returns prompt, chosen, rejected as lists of dicts
    # DPOTrainer requires them to be strings or prompt-formatted strings.
    # We create simple string representations.
    def to_string(messages):
        text = ""
        for msg in messages:
            text += f"<|{msg['role']}|>\n{msg['content']}</s>\n"
        return text

    prompt = to_string(example['prompt'])
    chosen = to_string(example['chosen'])
    rejected = to_string(example['rejected'])
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def main():
    config = load_config()
    base_model_id = config['model']['base_model_name_or_path']
    sft_adapter_path = config['model']['sft_adapter_path']
    out_dir = config['model']['dpo_adapter_path']
    
    print(f"Loading tokenizer from {sft_adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (4-bit)...")
    # For DPO we need the model and a reference model. 
    # Usually we load the SFT model as 'model', and base model as 'ref_model'.
    # DPOTrainer can automatically create a ref model by disabling adapters if we use PEFT.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Load the SFT adapter onto the base model
    print(f"Loading SFT adapter from {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
    
    # We can use the same Lora config or a new one for DPO. We will train the current adapter.
    
    print("Loading dataset...")
    train_dataset = load_from_disk(os.path.join(config['data']['processed_dir'], 'train'))
    train_dataset = train_dataset.map(format_dpo_prompt, remove_columns=train_dataset.column_names)

    from trl import DPOConfig
    training_args = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['dpo']['learning_rate']),
        logging_steps=config['training']['logging_steps'],
        max_steps=config['training']['save_steps'],
        save_strategy="steps",
        save_steps=config['training']['save_steps'],
        optim="paged_adamw_32bit",
        fp16=True,
        remove_unused_columns=False, # Required for DPOTrainer
        beta=config['dpo']['beta'],
        max_length=config['training']['max_seq_length'],
        max_prompt_length=config['training']['max_seq_length'] // 2,
    )
    
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # DPOTrainer handles this automatically when using PEFT
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    print("Starting DPO Training (Alignment)...")
    dpo_trainer.train()
    
    print(f"Saving DPO aligned adapter to {out_dir}...")
    dpo_trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
