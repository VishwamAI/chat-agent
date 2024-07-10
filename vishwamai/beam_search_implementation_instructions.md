# Beam Search Implementation Instructions for Vishwamai Model

This document provides detailed instructions for implementing a beam search strategy in the `generate` method of the `VishwamaiForCausalLM` class within the `model.py` file. Beam search is a popular decoding algorithm that keeps track of multiple hypotheses (beams) at each step and selects the most likely sequences.

## Steps to Implement Beam Search

1. **Add `num_beams` Parameter to `generate` Method:**
   - Open the `model.py` file.
   - Locate the `generate` method within the `VishwamaiForCausalLM` class.
   - Add a new parameter `num_beams` to the method signature with a default value of 5.

   ```python
   def generate(
       self,
       prompts: Union[str, Sequence[str]],
       device: Any,
       output_len: int = 100,
       temperature: float = 0.95,
       top_p: float = 1.0,
       top_k: int = 100,
       num_beams: int = 5,
   ) -> Union[str, Sequence[str]]:
   ```

2. **Initialize Beams:**
   - Inside the `generate` method, initialize the beams with the input token IDs and a score of 0.0.

   ```python
   beams = [(token_ids_tensor.clone(), 0.0)] * num_beams
   ```

3. **Update Token Generation Logic:**
   - Modify the token generation loop to maintain multiple candidate sequences (beams).
   - For each beam, generate the next token probabilities and apply top-p sampling.
   - Validate the generated tokens using the Earley parser and update the beams with valid tokens and their scores.

   ```python
   for i in range(max_seq_len - min_prompt_len):
       new_beams = []
       for beam_token_ids, beam_score in beams:
           logits = self(
               input_token_ids=beam_token_ids,
               input_positions=input_positions_tensor,
               kv_write_indices=None,
               kv_caches=kv_caches,
               mask=curr_mask_tensor,
               output_positions=output_positions_tensor,
               temperatures=temperatures_tensor,
               top_ps=top_ps_tensor,
               top_ks=top_ks_tensor,
           )[1]
           probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
           next_token_ids = self.sample_top_p(probs, top_p)

           # Grammar masking: validate sequences using Earley parser
           valid_tokens = []
           for token in next_token_ids:
               sequence = beam_token_ids[0, :output_index.item()].tolist() + [token]
               sequence_str = ' '.join([self.tokenizer.decode([t]) for t in sequence])
               try:
                   self.earley_parser.parse(sequence_str)
                   valid_tokens.append(token)
               except:
                   continue
           if not valid_tokens:
               valid_tokens = [self.tokenizer.pad_id] * batch_size

           for token in valid_tokens:
               new_beam_token_ids = beam_token_ids.clone()
               new_beam_token_ids.index_copy_(1, output_index, torch.tensor([token]).to(device))
               new_beam_score = beam_score + torch.log(probs[0, token]).item()
               new_beams.append((new_beam_token_ids, new_beam_score))

       # Select top beams
       beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]

       # Update input tensors for the next iteration
       input_token_ids_tensor = beams[0][0]
       input_positions_tensor = output_index.unsqueeze(dim=-1)
       curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
       output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
       output_index = output_index + 1
   ```

4. **Update Final Token Selection:**
   - Ensure that the final token selection process uses the best beam.

   ```python
   token_ids = beams[0][0].tolist()
   results = []
   for i, tokens in enumerate(token_ids):
       trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i]) + output_len]
       if self.tokenizer.eos_id in trimmed_output:
           eos_index = trimmed_output.index(self.tokenizer.eos_id)
           trimmed_output = trimmed_output[:eos_index]
       results.append(self.tokenizer.decode(trimmed_output))
   ```

5. **Test the Implementation:**
   - After making the changes, test the `generate` method to ensure that the beam search strategy is working as expected.
   - Verify that the generated text is of higher quality and that the grammar masking logic correctly validates sequences across all beams.

By following these steps, you will implement a beam search strategy in the `generate` method of the `VishwamaiForCausalLM` class, improving the quality of text generation in the Vishwamai model.
