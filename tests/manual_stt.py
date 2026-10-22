import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio.audio_service import AudioService
from app.services.speech_service import SpeechService

def main():
    print("Initializing services...")
    audio = AudioService()
    speech = SpeechService()
    
    print("\n🎙  Recording started automatically! Speak into your microphone.")
    print("   Press ENTER when you are finished speaking to stop.")
    
    audio_bytes = audio.record_until_keypress()
    
    print(f"\n✓ Recorded {len(audio_bytes)} bytes. Sending to Deepgram...")
    
    try:
        transcript = speech.transcribe(audio_bytes)
        print(f"\n📝 Final Transcript: '{transcript}'")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
