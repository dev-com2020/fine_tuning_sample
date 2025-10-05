#!/usr/bin/env python3
"""
Główny skrypt do fine-tuningu modeli z LM Studio
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd

# Dodaj katalog główny do ścieżki
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging(log_level: str = "INFO") -> None:
    """Konfiguruje logowanie"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training/fine_tune.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Wczytuje konfigurację z pliku YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def load_dataset(data_path: str, max_samples: Optional[int] = None) -> Dataset:
    """Wczytuje dane treningowe"""
    logging.info(f"Wczytywanie danych z: {data_path}")
    
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line.strip()))
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if max_samples:
            data = data[:max_samples]
    else:
        raise ValueError("Nieobsługiwany format pliku danych")
    
    return Dataset.from_list(data)

def prepare_model_and_tokenizer(config: Dict[str, Any]) -> tuple:
    """Przygotowuje model i tokenizer"""
    model_config = config['model']
    training_config = config['training']
    
    logging.info(f"Ładowanie modelu: {model_config['name']}")
    
    # Wczytaj tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['path'] + model_config['name'],
        padding_side=model_config.get('padding', 'right')
    )
    
    # Dodaj pad token jeśli nie istnieje
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Wczytaj model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['path'] + model_config['name'],
        torch_dtype=torch.bfloat16 if training_config.get('bf16', False) else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Konfiguruj LoRA jeśli włączone
    if config.get('lora', {}).get('enabled', False):
        logging.info("Konfigurowanie LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            lora_dropout=config['lora']['lora_dropout'],
            target_modules=config['lora']['target_modules']
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length: int):
    """Tokenizuje przykłady"""
    # Formatuj teksty zgodnie z szablonem
    texts = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        response = examples['response'][i]
        
        # Użyj szablonu z konfiguracji LM Studio
        text = f"<s>[INST] {instruction} [/INST] {response}</s>"
        texts.append(text)
    
    # Tokenizuj
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # Ustaw labels na to samo co input_ids dla language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning modeli z LM Studio")
    parser.add_argument("--config", default="configs/training_config.yaml", 
                       help="Ścieżka do pliku konfiguracyjnego")
    parser.add_argument("--data", help="Ścieżka do danych treningowych (nadpisuje config)")
    parser.add_argument("--output", help="Katalog wyjściowy (nadpisuje config)")
    parser.add_argument("--epochs", type=int, help="Liczba epok (nadpisuje config)")
    parser.add_argument("--lr", type=float, help="Learning rate (nadpisuje config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (nadpisuje config)")
    
    args = parser.parse_args()
    
    # Wczytaj konfigurację
    config = load_config(args.config)
    
    # Nadpisz parametry z argumentów
    if args.data:
        config['data']['train_file'] = args.data
    if args.output:
        config['training']['output_dir'] = args.output
    if args.epochs:
        config['training']['num_train_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['per_device_train_batch_size'] = args.batch_size
        config['training']['per_device_eval_batch_size'] = args.batch_size
    
    # Konfiguruj logowanie
    setup_logging(config.get('logging', {}).get('log_level', 'INFO'))
    
    logging.info("Rozpoczynanie fine-tuningu")
    logging.info(f"Konfiguracja: {config}")
    
    # Wczytaj dane
    train_dataset = load_dataset(
        config['data']['train_file'],
        config['data'].get('max_train_samples')
    )
    
    eval_dataset = None
    if config['data'].get('eval_file'):
        eval_dataset = load_dataset(
            config['data']['eval_file'],
            config['data'].get('max_eval_samples')
        )
    
    # Przygotuj model i tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)
    
    # Tokenizuj dane
    logging.info("Tokenizowanie danych")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, config['model']['max_length']
        ),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(
                x, tokenizer, config['model']['max_length']
            ),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Argumenty treningu
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        gradient_checkpointing=config.get('memory_optimization', {}).get('gradient_checkpointing', False),
        dataloader_pin_memory=config.get('memory_optimization', {}).get('dataloader_pin_memory', False),
        fp16=config.get('memory_optimization', {}).get('fp16', False),
        bf16=config.get('memory_optimization', {}).get('bf16', True),
        seed=config.get('seed', 42),
        report_to="tensorboard" if config.get('logging', {}).get('use_tensorboard', True) else None
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Rozpocznij trening
    logging.info("Rozpoczynanie treningu")
    trainer.train()
    
    # Zapisz model
    logging.info("Zapisywanie modelu")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    logging.info("Fine-tuning zakończony pomyślnie")

if __name__ == "__main__":
    main()
