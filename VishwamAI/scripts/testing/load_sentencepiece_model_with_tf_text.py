import tensorflow as tf
import tensorflow_text as tf_text

def load_sentencepiece_model(model_path):
    try:
        with tf.io.gfile.GFile(model_path, "rb") as f:
            model_proto = f.read()
        if not model_proto:
            raise ValueError("Model file is empty or could not be read.")
        print("Model file read successfully.")
        print(f"Model file size: {len(model_proto)} bytes")
        print(f"Model file snippet: {model_proto[:100]}")

        tokenizer = tf_text.SentencepieceTokenizer(
            model=model_proto,
            out_type=tf.int32,
            nbest_size=-1,
            alpha=1.0,
            add_bos=False,
            add_eos=False,
            reverse=False
        )
        print("Tokenizer initialized successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading SentencePiece model: {e}")

if __name__ == "__main__":
    model_path = "/home/ubuntu/chat-agent/VishwamAI/scripts/vishwamai.model"
    load_sentencepiece_model(model_path)
