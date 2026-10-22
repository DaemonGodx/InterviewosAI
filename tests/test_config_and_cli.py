import subprocess
import sys

import pytest
from pydantic import ValidationError

from app import config
from app.models.schemas import InterviewConfig


def test_text_mode_does_not_require_deepgram(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq")
    monkeypatch.setattr(config, "DEEPGRAM_API_KEY", None)
    config.validate_config("Text")


def test_voice_mode_requires_deepgram(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq")
    monkeypatch.setattr(config, "DEEPGRAM_API_KEY", None)
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY") as error:
        config.validate_config("Voice")
    assert "test-groq" not in str(error.value)


def test_question_count_must_be_positive():
    with pytest.raises(ValidationError):
        InterviewConfig(target_role="Engineer", experience_level="Fresher", question_count=0)


def test_module_help_exits_without_starting_interview():
    result = subprocess.run(
        [sys.executable, "-m", "app", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "Prepare and run" in result.stdout
    assert "Configuration Error" not in result.stdout
