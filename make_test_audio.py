import pyttsx3
import os

# Ensure our upload directory exists
os.makedirs("mock_uploads", exist_ok=True)

# The message we want to turn into an audio file
complaint_message = "The main water pipeline burst on 5th avenue. Water is flooding the street and entering basements!"

print("Generating local audio file...")
engine = pyttsx3.init()
# Save the speech directly into our expected file path
engine.save_to_file(complaint_message, 'mock_uploads/complaint_02.wav')
engine.runAndWait()
print("Success! 'mock_uploads/complaint_02.wav' has been created.")