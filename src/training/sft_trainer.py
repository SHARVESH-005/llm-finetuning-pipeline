import os
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_sft_prompt(example):
    # DPO dataset has chosen/rejected. For SFT we only train on the 'chosen' good behavior.
    messages = example['chosen']
    # This is a very simplified formatting. We usually apply the tokenizer's chat_template.
    text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        text += f"<|{role}|>\n{content}</s>\n"
    return {"text": text}

def main():
    config = load_config()
    model_id = config['model']['base_model_name_or_path']
    out_dir = config['model']['sft_adapter_path']
    
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model with 4-bit quantization (QLoRA) using BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config['lora']['target_modules']
    )
    
    model = get_peft_model(model, peft_config)
    
    print("Loading dataset...")
    train_dataset = load_from_disk(os.path.join(config['data']['processed_dir'], 'train'))
    
    # Format dataset for causal language modeling
    train_dataset = train_dataset.map(format_sft_prompt, remove_columns=train_dataset.column_names)

    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        logging_steps=config['training']['logging_steps'],
        max_steps=config['training']['save_steps'], # Just run a few steps for demo
        save_strategy="steps",
        save_steps=config['training']['save_steps'],
        optim="paged_adamw_32bit",
        fp16=True,
        seed=config['training']['seed'],
        dataset_text_field="text",
        max_length=config['training']['max_seq_length']
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("Starting SFT Training...")
    trainer.train()
    
    print(f"Saving SFT adapter to {out_dir}...")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
