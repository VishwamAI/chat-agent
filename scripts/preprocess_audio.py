import os
import sys
import time
import random
from vosk import Model, KaldiRecognizer
import wave
import json

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_audio(input_dir, output_file):
    # List all audio files in the input directory
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    # Initialize a list to store the extracted text
    extracted_texts = []

    # Load the Vosk model
    model = Model("/home/ubuntu/chat-agent/VishwamAI-main/models/vosk-model-small-en-us-0.15")

    # Process each audio file
    for audio_file in audio_files:
        # Open the audio file
        audio_path = os.path.join(input_dir, audio_file)
        wf = wave.open(audio_path, "rb")

        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print(f"Audio file {audio_file} must be WAV format mono PCM.")
            extracted_texts.append("[ERROR: Invalid audio format]")
            continue

        # Initialize the recognizer
        recognizer = KaldiRecognizer(model, wf.getframerate())

        # Process the audio file
        extracted_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_json = json.loads(result)
                extracted_text += result_json.get("text", "")

        # Append the final result
        final_result = recognizer.FinalResult()
        final_result_json = json.loads(final_result)
        extracted_text += final_result_json.get("text", "")
        extracted_texts.append(extracted_text)

        # Close the audio file
        wf.close()

    # Write the extracted texts to the output file
    with open(output_file, 'w') as f:
        for text in extracted_texts:
            f.write(text + '\n')

if __name__ == "__main__":
    input_dir = '/home/ubuntu/chat-agent/VishwamAI-main/data/raw/audio'
    output_file = '/home/ubuntu/chat-agent/VishwamAI-main/data/processed/audio_preprocessed.txt'
    preprocess_audio(input_dir, output_file)
