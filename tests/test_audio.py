
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from app.services.speech_service import SpeechService, SpeechServiceError
from app.audio.audio_service import AudioService, AudioServiceError


def test_audio_service_record(mocker):
    # Mock audio recording dependencies.
    mocker.patch("app.audio.audio_service.sd.InputStream")
    mocker.patch("app.audio.audio_service.sf.write")
    mocker.patch("builtins.input", return_value="")

    service = AudioService()

    # Prevent the recording queue from blocking indefinitely.
    with patch("app.audio.audio_service.queue.Queue") as mock_queue:
        instance = mock_queue.return_value
        instance.empty.side_effect = [False, True]
        instance.get.return_value = np.zeros((10, 1))

        result = service.record_until_keypress()

        assert isinstance(result, bytes)


def test_audio_service_play(mocker):
    mocker_play = mocker.patch("app.audio.audio_service.sd.play")
    mocker_wait = mocker.patch("app.audio.audio_service.sd.wait")

    service = AudioService()
    service.play_audio(np.array([0, 1024, -1024], dtype="<i2").tobytes())

    args, _ = mocker_play.call_args
    np.testing.assert_array_equal(args[0], np.array([0, 1024, -1024], dtype="<i2"))
    assert args[1] == 16000
    mocker_wait.assert_called_once()


def test_deepgram_pcm_output_is_played_as_pcm(mocker):
    mocker.patch("app.services.speech_service.DEEPGRAM_API_KEY", "test-key")
    mock_client = mocker.patch("app.services.speech_service.DeepgramClient").return_value
    pcm_bytes = np.array([10, -10, 32767], dtype="<i2").tobytes()
    mock_client.speak.v1.audio.generate.return_value = [pcm_bytes[:2], pcm_bytes[2:]]
    play = mocker.patch("app.audio.audio_service.sd.play")
    mocker.patch("app.audio.audio_service.sd.wait")

    audio = SpeechService().speak("Hello")
    AudioService().play_audio(audio)

    np.testing.assert_array_equal(play.call_args.args[0], np.array([10, -10, 32767], dtype="<i2"))
    assert play.call_args.args[1] == 16000


def test_audio_service_rejects_malformed_pcm(mocker):
    mocker.patch("app.audio.audio_service.sd.play")
    with pytest.raises(AudioServiceError, match="16-bit PCM"):
        AudioService().play_audio(b"\x00")


def test_speech_service_transcribe(mocker):
    mocker.patch("app.services.speech_service.DEEPGRAM_API_KEY", "dummy")
    mocker_client = mocker.patch(
        "app.services.speech_service.DeepgramClient"
    )

    # Mock the Deepgram transcription response.
    mock_response = MagicMock()
    mock_response.results.channels[0].alternatives[0].transcript = (
        "Test transcript."
    )

    mock_client_instance = mocker_client.return_value
    mock_client_instance.listen.v1.media.transcribe_file.return_value = (
        mock_response
    )

    service = SpeechService()
    transcript = service.transcribe(b"dummy_audio")

    assert transcript == "Test transcript."

    mock_client_instance.listen.v1.media.transcribe_file.assert_called_once()


def test_speech_service_speak(mocker):
    mocker.patch("app.services.speech_service.DEEPGRAM_API_KEY", "dummy")
    mocker_client = mocker.patch(
        "app.services.speech_service.DeepgramClient"
    )

    # Mock the audio chunks returned by Deepgram TTS.
    mock_client_instance = mocker_client.return_value
    mock_client_instance.speak.v1.audio.generate.return_value = [
        b"dummy_",
        b"audio",
    ]

    service = SpeechService()
    audio = service.speak("Hello world")

    assert audio == b"dummy_audio"

    mock_client_instance.speak.v1.audio.generate.assert_called_once()


def test_speech_service_empty_transcript(mocker):
    mocker.patch("app.services.speech_service.DEEPGRAM_API_KEY", "dummy")
    mocker_client = mocker.patch(
        "app.services.speech_service.DeepgramClient"
    )

    # Mock a response containing only whitespace.
    mock_response = MagicMock()
    mock_response.results.channels[0].alternatives[0].transcript = "   "

    mock_client_instance = mocker_client.return_value
    mock_client_instance.listen.v1.media.transcribe_file.return_value = (
        mock_response
    )

    service = SpeechService()
    transcript = service.transcribe(b"dummy_audio")

    assert transcript == ""


def test_speech_service_transcribe_failure(mocker):
    mocker.patch("app.services.speech_service.DEEPGRAM_API_KEY", "dummy")
    mocker_client = mocker.patch(
        "app.services.speech_service.DeepgramClient"
    )

    mock_client_instance = mocker_client.return_value
    mock_client_instance.listen.v1.media.transcribe_file.side_effect = (
        RuntimeError("Deepgram connection failed")
    )

    service = SpeechService()

    with pytest.raises(SpeechServiceError, match="Transcription failed"):
        service.transcribe(b"dummy_audio")


def test_speech_service_speak_failure(mocker):
    mocker.patch("app.services.speech_service.DEEPGRAM_API_KEY", "dummy")
    mocker_client = mocker.patch(
        "app.services.speech_service.DeepgramClient"
    )

    mock_client_instance = mocker_client.return_value
    mock_client_instance.speak.v1.audio.generate.side_effect = (
        RuntimeError("Deepgram connection failed")
    )

    service = SpeechService()

    with pytest.raises(SpeechServiceError, match="TTS Synthesis failed"):
        service.speak("Hello world")
