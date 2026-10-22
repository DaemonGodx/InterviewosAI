import sys
import os
import threading
import queue
import logging
import io
import sounddevice as sd
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

class AudioServiceError(Exception):
    pass

class AudioService:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_until_keypress(self) -> bytes:
        """
        Records audio from the microphone until the user presses Enter.
        Returns the recorded audio as WAV bytes.
        """
        q = queue.Queue()
        is_recording = True

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio recording status: {status}")
            if is_recording:
                q.put(indata.copy())

        try:
            stream = sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=callback)
            with stream:
                import time
                start_time = time.time()
                while True:
                    # input() will block until a newline is read.
                    # If it returns instantly, it means there was a stray newline in the buffer.
                    input()
                    if time.time() - start_time > 1.0:
                        break
                    
                is_recording = False
                
            # Drain queue
            audio_data = []
            while not q.empty():
                audio_data.append(q.get())
                
            if not audio_data:
                raise AudioServiceError("No audio data was recorded.")
                
            audio_data = np.concatenate(audio_data, axis=0)
            
            # Save to bytes buffer as WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.sample_rate, format='WAV', subtype='PCM_16')
            return buffer.getvalue()
            
        except Exception as e:
            logger.error("Failed to record audio.")
            raise AudioServiceError("Failed to record audio.") from e

    def play_audio(self, audio_bytes: bytes):
        """
        Plays Deepgram linear16 little-endian PCM at the configured format.
        Blocks until playback is completely finished.
        """
        try:
            frame_width = 2 * self.channels
            if not audio_bytes or len(audio_bytes) % frame_width:
                raise AudioServiceError("Audio data is not valid 16-bit PCM for the configured channel count.")
            data = np.frombuffer(audio_bytes, dtype="<i2")
            if self.channels > 1:
                data = data.reshape((-1, self.channels))
            sd.play(data, self.sample_rate)
            sd.wait() # Wait until file is done playing
        except AudioServiceError:
            raise
        except Exception as e:
            logger.error("Failed to play audio.")
            raise AudioServiceError("Failed to play audio.") from e
