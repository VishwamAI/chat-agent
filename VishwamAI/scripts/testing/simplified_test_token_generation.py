import time
import sentencepiece as spm
import jax
import jax.numpy as jnp
import haiku as hk

class VishwamAIModel(hk.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = self._create_tokenizer()
        self.transformer = self._create_transformer()

    def _create_tokenizer(self):
        sp = spm.SentencePieceProcessor()
        sp.Load("/home/ubuntu/chat-agent/VishwamAI/data/vishwamai.spm")
        return sp

    def _create_transformer(self):
        def transformer_fn(x):
            return hk.nets.ResNet50(1000)(x)
        return hk.transform(transformer_fn)

    def __call__(self, input_ids, rng):
        embeddings = self.tokenizer.EncodeAsIds(input_ids)
        logits = self.transformer.apply(self.params, rng, embeddings)
        return logits

    def generate_text(self, prompt, max_length=100):
        input_ids = self.tokenizer.EncodeAsIds(prompt)
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)
        for _ in range(max_length):
            predictions = self(input_ids, rng)
            next_token = jnp.argmax(predictions[:, -1, :], axis=-1)
            input_ids = jnp.concatenate([input_ids, next_token], axis=-1)
            if next_token == self.tokenizer.PieceToId("[EOS]"):
                break
        return self.tokenizer.DecodeIds(input_ids.tolist())

def main():
    model = VishwamAIModel()
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    model.params = model.transformer.init(rng, dummy_input)

    prompt = "Once upon a time"
    start_time = time.time()
    generated_text = model.generate_text(prompt)
    end_time = time.time()
    print(f"Generated text: {generated_text}")
    print(f"Time taken for token generation: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
