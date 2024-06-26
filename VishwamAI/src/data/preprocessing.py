def preprocess(data):
    # 1. Tokenize the text
    tokens = tokenizer.encode(data)

    # 2. Add special tokens
    tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]

    # 3. Pad or truncate the sequence
    pad_len = max_len - len(tokens)
    tokens = tokens + [PAD_TOKEN] * pad_len

    # 4. Create a tensor of token IDs
    tokens_tensor = torch.tensor(tokens)

    # 5. Create a tensor of segment IDs
    segment_ids = [0] * len(tokens)
    segment_ids_tensor = torch.tensor(segment_ids)

    # 6. Create a tensor of attention mask
    attention_mask = [1] * len(tokens)
    attention_mask_tensor = torch.tensor(attention_mask)

    return tokens_tensor, segment_ids_tensor, attention_mask_tensor
