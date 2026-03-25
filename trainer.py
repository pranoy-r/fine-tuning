from __future__ import annotations

import os
import math
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import get_qwen_dataloaders
from peft import set_peft_model_state_dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Union, Dict, Any
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from accelerate import Accelerator
import wandb
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
@torch.no_grad()
def sample_generations(
    model: nn.Module, 
    val_loader: DataLoader, 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
    accelerator: Accelerator, 
    device: torch.device, 
    global_step: int, 
    num_samples: int = 2
) -> None:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    logging.info(f"[Step {global_step}] Sampling text generations...")

    try:
        batch = next(iter(val_loader))
    except StopIteration:
        val_loader_iter = iter(val_loader)
        batch = next(val_loader_iter)

    input_ids = batch["input_ids"].to(device)[:num_samples]
    attention_mask = batch["attention_mask"].to(device)[:num_samples]
    
    current_batch_size = input_ids.size(0)
    if current_batch_size == 0:
        return

    # Generate text
    generated_ids = unwrapped_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )

    # Decode and log
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    samples_table = wandb.Table(columns=["Step", "Input Prompt", "Generated Output"])
    
    for i in range(current_batch_size):
        samples_table.add_data(global_step, decoded_inputs[i], decoded_preds[i])

    if accelerator.is_main_process:
        accelerator.log({"val/text_samples": samples_table}, step=global_step)

    model.train()


@torch.no_grad()
def validate(
    model: nn.Module, 
    val_loader: DataLoader, 
    accelerator: Accelerator, 
    device: torch.device, 
    global_step: int
) -> float:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    
    if accelerator.is_main_process:
        print(f"\n[Step {global_step}] Starting Validation...")
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = unwrapped_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        total_loss += outputs.loss.item()
        num_batches += 1
        
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    
    if accelerator.is_main_process:
        print(f"Validation Results - Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
        
        accelerator.log({
            "val/loss": avg_loss,
            "val/perplexity": perplexity
        }, step=global_step)
        
    model.train()
    return avg_loss


def resume_from_checkpoint(
    device: torch.device, 
    path: str,  
    model: nn.Module, 
    optim: Optimizer, 
    scheduler: Optional[_LRScheduler]
    ) -> int:

   
    if os.path.isdir(path):
        filename = os.path.join(path, "training_state.pth")
    else:
        filename = path

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find checkpoint file: {filename}")

    
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    global_step = checkpoint['step']
    
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

 
    # For LORA
    checkpoint_dir = os.path.dirname(filename)
    

    weights_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

    if os.path.exists(weights_path):
        # Load weights and inject them into the model
        adapters_weights = torch.load(weights_path, map_location=device)
        set_peft_model_state_dict(model, adapters_weights)
        logging.info(f"Successfully resumed LoRA weights from {weights_path}")
    else:
        logging.warning(f"No adapter weights found in {checkpoint_dir}. Model remains in base state.")
    
    logging.info(f"Resumed optimizer/scheduler state from: {filename} at step {global_step}")
    return global_step


def save_ckpt(
    accelerator: Accelerator, 
    model: nn.Module, 
    optim: Optimizer, 
    scheduler: Optional[_LRScheduler], 
    global_step: int, 
    ckpt_saved_dir: str, 
    save_intermediate_models: bool = False
    ) -> None:
    
    unwrapped_model = accelerator.unwrap_model(model)
    
    if not save_intermediate_models:
        step_dir = os.path.join(ckpt_saved_dir, 'final-model')
    else:
        step_dir = os.path.join(ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}')
        
    os.makedirs(step_dir, exist_ok=True)

    unwrapped_model.save_pretrained(
        step_dir,
        is_main_process=accelerator.is_main_process, 
        save_function=accelerator.save
    )

    state_file = os.path.join(step_dir, 'training_state.pth')
    checkpoint = {
        'step': global_step,
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    accelerator.save(checkpoint, state_file)
    logging.info("Saving LoRA checkpoint and states to: %s ...", step_dir)



class QwenTrainer:
    def __init__(
        self, 
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        train_dl: DataLoader,
        val_dl: DataLoader,
        optim: Optimizer,
        scheduler: Optional[_LRScheduler],
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        ckpt_every: int = 500,
        eval_every: int = 500,
        sample_every: int = 250,    
        save_intermediate_models: bool = False,
        ckpt_saved_dir: str = 'ckpt',
        resume: Optional[str] = None,
        accelerator_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.global_step = 0

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="wandb"
        )

        self.accelerator.init_trackers(**accelerator_kwargs)
        device = self.accelerator.device

        self.model, self.optim, self.scheduler, self.train_dl, self.val_dl = self.accelerator.prepare(
            model, optim, scheduler, train_dl, val_dl
        )

        self.tokenizer = tokenizer
        
        if resume:
            self.global_step = resume_from_checkpoint(device,
                                                    resume,
                                                    self.accelerator.unwrap_model(model),
                                                    self.optim,
                                                    self.scheduler)

        effective_steps_per_epoch = math.ceil(len(self.train_dl) / gradient_accumulation_steps)
        effective_training_steps = num_epochs * effective_steps_per_epoch

        logging.info(f"Effective batch size per device: {batch_size * gradient_accumulation_steps}")
        logging.info(f"Effective steps per epoch: {effective_steps_per_epoch}")
        logging.info(f"Effective Total training steps: {effective_training_steps}")

        self.start_epoch = self.global_step // effective_steps_per_epoch
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.ckpt_every = ckpt_every
        self.eval_every = eval_every
        self.sample_every = sample_every
        self.save_intermediate_models = save_intermediate_models
        self.ckpt_saved_dir = ckpt_saved_dir

    @property
    def device(self):
        return self.accelerator.device

    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.num_epochs):
            with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
                for batch in train_dl:
                    
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    with self.accelerator.accumulate(self.model):
                        self.optim.zero_grad(set_to_none=True)
                        
                        # Forward Pass
                        outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels
                        )
                        loss = outputs.loss
                            
                        self.accelerator.backward(loss)
                        
                        if self.accelerator.sync_gradients and self.max_grad_norm:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        self.optim.step()    
                        if self.scheduler is not None:
                            self.scheduler.step()

            
                    if self.accelerator.sync_gradients:
                        
                        # Checkpointing
                        if not (self.global_step % self.ckpt_every) and self.global_step != 0:
                            if self.accelerator.is_main_process:
                                save_ckpt(
                                    self.accelerator, 
                                    self.model, 
                                    self.optim,
                                    self.scheduler,
                                    self.global_step,
                                    self.ckpt_saved_dir,
                                    self.save_intermediate_models
                                )
                            self.accelerator.wait_for_everyone()
                        
                        # Sampling
                        if not (self.global_step % self.sample_every) and self.global_step != 0:
                            if self.accelerator.is_main_process:
                                sample_generations(
                                    self.model,
                                    self.val_dl,
                                    self.tokenizer,
                                    self.accelerator,
                                    self.device,
                                    self.global_step,
                                    num_samples=2
                                )
                            self.accelerator.wait_for_everyone()

                        # Validation
                        if not (self.global_step % self.eval_every) and self.global_step != 0:
                            validate(
                                self.model,
                                self.val_dl,
                                self.accelerator,
                                self.device,
                                self.global_step
                            )
                            self.accelerator.wait_for_everyone() 
                        
                        # Logging
                        if self.accelerator.is_main_process:
                            log_dict = {
                                "train/loss": loss.item(),
                                "train/lr": self.optim.param_groups[0]['lr']
                            }
                            self.accelerator.log(log_dict, step=self.global_step)
                        
                        self.global_step += 1

        # Save the final model
        if self.accelerator.is_main_process:
            save_ckpt(
                self.accelerator,
                self.model,
                self.optim,
                self.scheduler,
                self.global_step,
                self.ckpt_saved_dir,
                save_intermediate_models=False
            )

        self.accelerator.end_training()        
        print("Train finished!")




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()

    # Project / Dataset
    parser.add_argument('--project_name', type=str, default='Qwen-Finetune', help="WandB project name")
    parser.add_argument('--dataset_path', type=str, default='my_data.jsonl', help="Path to training dataset (JSONL format)")
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-7B-Instruct', help="Hugging Face model ID")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint state to resume from")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=100, help="LR warmup steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Adam weight decay")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
    
    # LoRA Hyperparameters
    parser.add_argument('--lora_rank', type=int, default=16, help="LoRA Rank (r)")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA Alpha")

    # Logging / Checkpointing
    parser.add_argument('--ckpt_every', type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument('--save_intermediate_models', default=True, action='store_true', help="Whether to save intermediate models during training")
    parser.add_argument('--eval_every', type=int, default=500, help="Evaluate every N steps")
    parser.add_argument('--sample_every', type=int, default=250, help="Sample and log text every N steps")
    parser.add_argument('--ckpt_saved_dir', type=str, default='qwen_checkpoints', help="Directory to save outputs")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Model in 4-bit (QLoRA setup)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" 
    )
    
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    

    train_dl, val_dl = get_qwen_dataloaders(
        dataset_name_or_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    steps_per_epoch = max(1, math.ceil(len(train_dl) / args.gradient_accumulation_steps))
    num_training_steps = args.num_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    training_params = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm
    }

    logging_params = {
        "ckpt_every": args.ckpt_every,
        "eval_every": args.eval_every,
        "sample_every": args.sample_every,
        "save_intermediate_models": args.save_intermediate_models,
        "ckpt_saved_dir": args.ckpt_saved_dir,
        "resume": args.resume,
    }

    accelerator_kwargs = {
        "project_name": args.project_name,
        "init_kwargs": {"wandb": {"config": vars(args)}}
    }

    # Initialize and run Trainer
    trainer = QwenTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dl=train_dl,
        val_dl=val_dl,
        optim=optim,
        scheduler=scheduler,
        accelerator_kwargs=accelerator_kwargs,
        **training_params,
        **logging_params,
    )

    trainer.train()