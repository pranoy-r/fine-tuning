import logging
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, AutoTokenizer

class QwenInstructionDataset(Dataset):
    def __init__(self, dataset: Any, tokenizer: Any, max_length: int = 1024) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        messages = item["messages"]
      
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        input_ids = self.tokenizer.encode(
            full_text, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_length
        )
        
        labels = list(input_ids)

        context_messages = [msg for msg in messages if msg["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        
        # Mask labels
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

def get_qwen_dataloaders(
    dataset_name_or_path: str,
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 1024,
    num_workers: int = 0,
    val_split_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    
    raw_dataset = load_dataset("json", data_files=dataset_name_or_path, split="train")
    split_dataset = raw_dataset.train_test_split(test_size=val_split_ratio, seed=seed)
    
    train_dataset = QwenInstructionDataset(split_dataset["train"], tokenizer, max_length)
    val_dataset = QwenInstructionDataset(split_dataset["test"], tokenizer, max_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=False        
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,        
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader

if __name__ == "__main__":
    import json
    import os

    test_file = "test_data.jsonl"
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I am doing great, thank you!"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 is 4."}
            ]
        }
    ]

    with open(test_file, "w") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")

    model_id = "Qwen/Qwen2.5-7B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dl, val_dl = get_qwen_dataloaders(
        dataset_name_or_path=test_file,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        num_workers=0
    )

  
    batch = next(iter(train_dl))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    for i in range(len(input_ids)):
        print(f"\n--- Sample {i+1} ---")
        print("Input IDs:", input_ids[i])
        print("Labels:", labels[i])
        
        # print the decoded input and labels for verification
        decoded_input = tokenizer.decode(input_ids[i], skip_special_tokens=False)
        decoded_labels = tokenizer.decode([id for id in labels[i] if id != -100], skip_special_tokens=False)
        print("Decoded Input:", decoded_input)
        print("Decoded Labels:", decoded_labels)