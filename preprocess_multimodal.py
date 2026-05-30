import os
from faster_whisper import WhisperModel

# Initialize the local Whisper model
# 'tiny' is fast and lightweight. For better multilingual accuracy later, you can swap to 'base' or 'small'.
print("Loading local Whisper model (this may take a moment on first run)...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe_audio(file_path):
    """Transcribes local audio/video files to text using offline Whisper."""
    if not os.path.exists(file_path):
        # For this tutorial/mock data, if the file doesn't actually exist on your disk yet,
        # we will return a placeholder so your script doesn't crash.
        return f"[Mock Transcription for {os.path.basename(file_path)}: Pothole or utility issue reported.]"
    
    print(f"Processing local file: {file_path}")
    # Whisper automatically handles both audio (.wav/.mp3) and video (.mp4) files directly!
    segments, info = model.transcribe(file_path, beam_size=5)
    
    # Combine the transcribed text segments
    transcribed_text = " ".join([segment.text for segment in segments])
    return transcribed_text.strip()

# Test the function logic
if __name__ == "__main__":
    test_file = "mock_uploads/complaint_02.wav"
    text = transcribe_audio(test_file)
    print("\n--- Test Result ---")
    print(f"Resulting Text: {text}")