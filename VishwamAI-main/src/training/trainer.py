import jax
import jax.numpy as jnp
import optax
from typing import Dict, Iterable
from functools import partial
from src.features.image_to_text import image_to_text
from src.features.pdf_to_text import pdf_to_text
from src.features.summarization import summarize_text
from src.features.audio_to_text import audio_to_text
import os
import numpy as np
import time

class VishwamAITrainer:
    def __init__(self, model, config: Dict, optimizer, opt_state):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.opt_state = opt_state

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, opt_state, batch):
        def loss_fn(params):
            logits = self.model.apply(params, batch['input_ids'], is_training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['labels']).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, logits

    def preprocess_input(self, input_data):
        if input_data.endswith('.pdf'):
            return pdf_to_text(input_data)
        elif input_data.endswith(('.png', '.jpg', '.jpeg')):
            return image_to_text(input_data)
        elif input_data.endswith(('.wav', '.mp3', '.flac')):
            return audio_to_text(input_data)
        else:
            return input_data

    def preprocess_math_input(self, input_data):
        # Tokenize mathematical expressions
        tokens = self.config['tokenizer'].encode(input_data)
        return tokens

    def train(self, train_dataset: Iterable, eval_dataset: Iterable, num_epochs: int):
        params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.config['max_seq_length']), dtype=jnp.int32))

        checkpoint_dir = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            start_time = time.time()
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            train_loss = 0
            train_steps = 0
            for batch in train_dataset:
                print(f"Processing batch {train_steps + 1}")
                batch['input_ids'] = self.preprocess_input(batch['input_ids'])
                batch['input_ids'] = self.preprocess_math_input(batch['input_ids'])
                params, self.opt_state, loss, _ = self.train_step(params, self.opt_state, batch)
                train_loss += loss
                train_steps += 1

                if train_steps % 100 == 0:
                    print(f"Step {train_steps}: Current Train Loss: {loss:.4f}")

            eval_metrics = self.evaluate(params, eval_dataset)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds")
            print(f"Train Loss: {train_loss / train_steps:.4f}")
            print(f"Eval Metrics: {eval_metrics}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.npy')
            print(f"Checkpoint path: {checkpoint_path}")
            print(f"Saving checkpoint to {checkpoint_path}")
            np.save(checkpoint_path, params)
            print(f"Checkpoint saved successfully at {checkpoint_path}")

            if self._should_stop_early(eval_metrics):
                print("Early stopping criteria met. Ending training.")
                break

        return params

    def evaluate(self, params, eval_dataset: Iterable) -> Dict[str, float]:
        total_loss = 0
        total_steps = 0
        for batch in eval_dataset:
            batch['input_ids'] = self.preprocess_input(batch['input_ids'])
            batch['input_ids'] = self.preprocess_math_input(batch['input_ids'])
            logits = self.model.apply(params, batch['input_ids'], is_training=False)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['labels']).mean()
            total_loss += loss
            total_steps += 1

        avg_loss = total_loss / total_steps
        perplexity = jnp.exp(avg_loss)
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }

    def _should_stop_early(self, eval_metrics: Dict[str, float]) -> bool:
        # Implement early stopping logic here
        # For example, stop if perplexity is below a certain threshold
        return eval_metrics['perplexity'] < self.config.get('early_stopping_perplexity_threshold', float('inf'))
