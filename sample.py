import os

# Add ffmpeg path manually
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

import whisper
import sounddevice as sd
import soundfile as sf
import tempfile

def record_audio(duration=5, samplerate=16000):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio, samplerate)
    return temp_file.name

# Load model
model = whisper.load_model("base")

# Record & Transcribe
audio_path = record_audio()
result = model.transcribe(audio_path)

print("\n📄 Transcription:\n", result["text"])
