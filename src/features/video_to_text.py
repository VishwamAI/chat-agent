import cv2
import pytesseract

def video_to_text(video_path: str) -> str:
    """
    Extracts text from a video file using Tesseract OCR.

    Args:
        video_path (str): The path to the video file.

    Returns:
        str: The extracted text.
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        text = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use Tesseract to extract text from the frame
            frame_text = pytesseract.image_to_string(gray)
            text += frame_text

        cap.release()
        return text
    except Exception as e:
        return f"Error: {e}"
