import sentencepiece as spm
sp = spm.SentencePieceProcessor()
model_path = 'vishwamai.serialized'
with open(model_path, 'rb') as f:
    model_content = f.read()
try:
    sp.LoadFromSerializedProto(model_content)
    print('SentencePiece model loaded successfully.')
except Exception as e:
    print(f'Failed to load SentencePiece model: {e}')

