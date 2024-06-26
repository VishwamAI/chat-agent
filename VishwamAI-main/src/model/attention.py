# Suggested code may be subject to a license. Learn more: ~LicenseLog:2743477399.
# Suggested code may be subject to a license. Learn more: ~LicenseLog:953013427.
def build_attention(config, is_training):
  """Builds an attention layer.

  Args:
    config: A `BertConfig` object.
    is_training: Whether this layer is in training mode.

  Returns:
    An `Attention` object.
  """
  if config.attention_probs_dropout_prob is None:
    config.attention_probs_dropout_prob = 0.0
  if config.hidden_size % config.num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (config.hidden_size, config.num_attention_heads))
  attention_head_size = int(config.hidden_size / config.num_attention_heads)
  all_head_size = attention_head_size * config.num_attention_heads

  attention_layer = Attention(
      config.num_attention_heads, config.size_per_head,
      attention_probs_dropout_prob=config.attention_probs_dropout_prob,
      initializer_range=config.initializer_range, key_value_proj_dim=config.key_value_proj_dim,
      use_bias=config.use_bias, output_attention_probs=config.output_attention_probs,
      query_act=config.query_act, key_act=config.key_act,
      value_act=config.value_act, attention_type=config.attention_type)
  return attention_layer
