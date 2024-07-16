# Vishwamai

This repository contains JAX example code for loading and running the Vishwamai open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` - see [Downloading the weights](#downloading-the-weights)

Then, run

```shell
pip install -r requirements.txt
python run.py
```

to test the code.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Model Specifications

Vishwamai is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/VishwamAI/vishwamai):
```
git clone https://github.com/VishwamAI/chat-agent.git && cd chat-agent
pip install huggingface_hub[hf_transfer]
huggingface-cli download VishwamAI/vishwamai --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# Training the Model on Hugging Face

To train the Vishwamai model on Hugging Face, follow these steps:

1. Clone the repository and install the required dependencies:
   ```
   git clone https://github.com/VishwamAI/chat-agent.git
   cd chat-agent
   pip install -r requirements.txt
   pip install transformers datasets accelerate
   ```

2. Prepare your dataset in a format compatible with Hugging Face's datasets library.

3. Create a training script (e.g., `train_vishwamai.py`) using the Hugging Face Transformers library. Here's a basic example:

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
   from datasets import load_dataset

   # Load the Vishwamai model and tokenizer
   model = AutoModelForCausalLM.from_pretrained("VishwamAI/vishwamai")
   tokenizer = AutoTokenizer.from_pretrained("VishwamAI/vishwamai")

   # Load your dataset
   dataset = load_dataset("your_dataset_name")

   # Define training arguments
   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       save_steps=10_000,
       save_total_limit=2,
   )

   # Create Trainer instance
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset["train"],
       tokenizer=tokenizer,
   )

   # Start training
   trainer.train()
   ```

4. Run the training script:
   ```
   python train_vishwamai.py
   ```

5. After training, you can push your model to the Hugging Face Model Hub:
   ```
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model.push_to_hub("VishwamAI/vishwamai-finetuned")
   tokenizer.push_to_hub("VishwamAI/vishwamai-finetuned")
   ```

Note: Make sure you have sufficient computational resources, as training large language models like Vishwamai requires significant GPU capacity and memory.

# License

The code and associated Vishwamai weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Vishwamai.