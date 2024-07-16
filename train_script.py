# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load the Vishwamai model and tokenizer
model_name = "VishwamAI/vishwamai"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and preprocess the MARIO-Math-Reasoning/Gaokao2023-Math-En dataset
dataset = load_dataset("MARIO-Math-Reasoning/Gaokao2023-Math-En")

def preprocess_function(examples):
    inputs = [f"Question: {q}\nAnswer:" for q in examples["question"]]
    targets = [f" {a}" for a in examples["answer"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Split the dataset into train and evaluation sets
train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(int(len(tokenized_dataset["train"]) * 0.8)))
eval_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(int(len(tokenized_dataset["train"]) * 0.8), len(tokenized_dataset["train"])))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.push_to_hub("VishwamAI/vishwamai-finetuned")
tokenizer.push_to_hub("VishwamAI/vishwamai-finetuned")