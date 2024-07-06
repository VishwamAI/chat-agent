from pydub import AudioSegment
import speech_recognition as sr

def audio_to_text(audio_path: str) -> str:
    """
    Converts audio file to text using SpeechRecognition.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        str: The extracted text.
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        audio.export("temp.wav", format="wav")

        # Initialize recognizer
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return text
    except Exception as e:
        return f"Error: {e}"
