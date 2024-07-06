import os
import sys
import speech_recognition as sr
from pydub import AudioSegment
import moviepy.editor as mp

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_video(input_dir, output_file):
    # List all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]

    # Initialize a list to store the extracted text
    extracted_texts = []

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Process each video file
    for video_file in video_files:
        # Open the video file
        video_path = os.path.join(input_dir, video_file)
        video = mp.VideoFileClip(video_path)

        # Extract audio from the video
        audio_path = os.path.join(input_dir, 'temp_audio.wav')
        video.audio.write_audiofile(audio_path)

        # Use SpeechRecognition to extract text from the audio
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            extracted_text = recognizer.recognize_google(audio_data)
            extracted_texts.append(extracted_text)

        # Remove the temporary audio file
        os.remove(audio_path)

    # Write the extracted texts to the output file
    with open(output_file, 'w') as f:
        for text in extracted_texts:
            f.write(text + '\n')

if __name__ == "__main__":
    input_dir = 'data/raw/videos'
    output_file = 'data/processed/videos_preprocessed.txt'
    preprocess_video(input_dir, output_file)
