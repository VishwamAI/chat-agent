import sentencepiece as spm

def verify_sentencepiece_model(model_path):
    sp = spm.SentencePieceProcessor()
    try:
        sp.Load(model_path)
        print("SentencePiece model loaded successfully.")
        print(f"Model size: {sp.GetPieceSize()} pieces")
        print(f"First 10 pieces: {sp.IdToPiece(list(range(10)))}")
    except Exception as e:
        print(f"Error loading SentencePiece model: {e}")

if __name__ == "__main__":
    model_path = "/home/ubuntu/chat-agent/VishwamAI/scripts/vishwamai.model"
    verify_sentencepiece_model(model_path)
