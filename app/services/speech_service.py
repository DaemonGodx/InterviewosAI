import logging
import io
from deepgram import DeepgramClient
from app.config import DEEPGRAM_API_KEY

logger = logging.getLogger(__name__)

class SpeechServiceError(Exception):
    pass

class SpeechService:
    def __init__(self):
        if not DEEPGRAM_API_KEY or DEEPGRAM_API_KEY == "dummy" or DEEPGRAM_API_KEY == "your_deepgram_api_key_here":
            logger.warning("Deepgram API Key is missing or invalid. Speech services may fail.")
        self.client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio bytes to text using Deepgram STT.
        """
        try:
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-2",
                smart_format=True
            )
            
            # Navigate Deepgram response dict
            transcript = response.results.channels[0].alternatives[0].transcript
            if not transcript or not transcript.strip():
                return ""
            return transcript.strip()
        except Exception as e:
            logger.error("Deepgram STT failed.")
            raise SpeechServiceError("Transcription failed. Check voice configuration and try again.") from e

    def speak(self, text: str) -> bytes:
        """
        Synthesizes text to audio bytes using Deepgram TTS.
        """
        try:
            options = {
                "model": "aura-asteria-en",
                "encoding": "linear16",
                "sample_rate": 16000
            }
            # Use the audio generate method returning an iterator of bytes
            generator = self.client.speak.v1.audio.generate(
                text=text,
                model="aura-asteria-en",
                encoding="linear16",
                sample_rate=16000
            )
            audio_bytes = b"".join(list(generator))
            return audio_bytes
        except Exception as e:
            logger.error("Deepgram TTS failed.")
            raise SpeechServiceError("TTS Synthesis failed. Check voice configuration and try again.") from e
