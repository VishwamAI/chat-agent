import tensorflow as tf
import keras_nlp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_sentencepiece_tokenizer(data, vocabulary_size, model_type="unigram", proto_output_file="vishwamai.spm", lowercase=False):
    """
    Train a SentencePiece tokenizer and save the model to a file.

    Args:
        data: A `tf.data.Dataset` or a list of filenames.
        vocabulary_size: int. The maximum size of the vocabulary to be trained.
        model_type: str. The model algorithm must be one of `"unigram"`, `"bpe"`, `"word"` or `"char"`. Defaults to `"unigram"`.
        proto_output_file: str. The file path to save the trained SentencePiece model. Defaults to "vishwamai.spm".
        lowercase: bool. If True, the input text will be lowercased before tokenization. Defaults to `False`.

    Returns:
        None
    """
    try:
        proto = keras_nlp.tokenizers.compute_sentence_piece_proto(
            data,
            vocabulary_size=vocabulary_size,
            model_type=model_type,
            proto_output_file=proto_output_file,
            lowercase=lowercase
        )
        logging.info(f"SentencePiece model trained and saved to {proto_output_file}")
    except Exception as e:
        logging.error(f"Error during SentencePiece model training: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    # Replace 'example_data' with the actual dataset
    example_data = tf.data.TextLineDataset(["/home/ubuntu/chat-agent/VishwamAI/scripts/text_data.txt"])
    train_sentencepiece_tokenizer(example_data, vocabulary_size=1000)
