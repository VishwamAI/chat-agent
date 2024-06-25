import sentencepiece as spm
import tensorflow as tf
import logging
import tempfile
import os

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
        # Convert the dataset to a list of strings
        data_list = [str(line.numpy(), 'utf-8') for line in data]

        # Write the data list to a temporary text file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write('\n'.join(data_list))
            temp_file_path = temp_file.name

        # Train the SentencePiece model
        spm.SentencePieceTrainer.Train(
            input=temp_file_path,
            model_prefix=proto_output_file.split('.')[0],
            vocab_size=vocabulary_size,
            model_type=model_type,
            character_coverage=1.0,
            input_sentence_size=1000000,
            shuffle_input_sentence=True
        )
        logging.info(f"SentencePiece model trained and saved to {proto_output_file}")

        # Clean up the temporary file
        os.remove(temp_file_path)
    except Exception as e:
        logging.error(f"Error during SentencePiece model training: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    # Replace 'example_data' with the actual dataset
    example_data = tf.data.TextLineDataset(["/home/ubuntu/chat-agent/VishwamAI/scripts/text_data.txt"])
    train_sentencepiece_tokenizer(example_data, vocabulary_size=64)
