import sentencepiece as spm
import tensorflow_text as tf_text
import tensorflow as tf

def test_tokenizer_initialization():
    model_path = "/home/ubuntu/chat-agent/VishwamAI/data/vishwamai.spm"
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    model_proto = sp.serialized_model_proto()
    print(f"Model proto type: {type(model_proto)}")
    print(f"Model proto content: {model_proto[:100]}")
    try:
        tokenizer = tf_text.SentencepieceTokenizer(model=model_proto, out_type=tf.int32)
        print("Tokenizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")

if __name__ == "__main__":
    test_tokenizer_initialization()
