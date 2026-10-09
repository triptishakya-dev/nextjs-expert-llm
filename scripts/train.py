
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = ROOT / "models" / "nextjs-lora"

MAX_LENGTH = 256

# True = short test; False = full training
SMOKE_TEST = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

print("GPU:", torch.cuda.get_device_name(0))

# ====================================
# 1. Load tokenizer
# ====================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====================================
# 2. Load dataset
# ====================================

print("Loading Next.js dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": str(ROOT / "data/train.jsonl"),
        "validation": str(
            ROOT / "data/validation.jsonl"
        ),
    },
)

if SMOKE_TEST:
    dataset["train"] = dataset["train"].select(
        range(min(64, len(dataset["train"])))
    )

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print("Training samples:", len(tokenized["train"]))

# ====================================
# 3. Load base model
# ====================================

print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ====================================
# 4. Configure LoRA
# ====================================

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

model.enable_input_require_grads()
model.print_trainable_parameters()

# ====================================
# 5. Training configuration
# ====================================

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=5e-5,
    num_train_epochs=1,

    max_steps=10 if SMOKE_TEST else -1,

    fp16=True,
    bf16=False,

    gradient_checkpointing=True,

    optim="adamw_torch",
    logging_steps=1 if SMOKE_TEST else 10,

    save_strategy="no",
    eval_strategy="no",

    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    processing_class=tokenizer,
    data_collator=collator,
)

# ====================================
# 6. Train
# ====================================

print("Starting training...")

trainer.train()

# ====================================
# 7. Save adapter
# ====================================

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("Training completed!")
print("Model adapter saved:", OUTPUT_DIR)
