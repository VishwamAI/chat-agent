import sentencepiece as spm

def verify_serialized_model():
    model_path = '/home/ubuntu/chat-agent/VishwamAI/scripts/vishwamai.serialized'
    sp = spm.SentencePieceProcessor()
    with open(model_path, 'rb') as f:
        model_content = f.read()
    try:
        sp.LoadFromSerializedProto(model_content)
        print('SentencePiece model loaded successfully.')
    except Exception as e:
        print(f'Failed to load SentencePiece model: {e}')

if __name__ == "__main__":
    verify_serialized_model()
