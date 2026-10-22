import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


def _configured(value):
    if not value or not value.strip():
        return False
    return value.strip().lower() not in {
        "dummy", "your_groq_api_key_here", "your_deepgram_api_key_here",
    }

def validate_config(interview_mode="Text"):
    if interview_mode not in {"Text", "Voice"}:
        raise ValueError("Interview mode must be Text or Voice.")
    missing = []
    if not _configured(GROQ_API_KEY):
        missing.append("GROQ_API_KEY")
    if interview_mode == "Voice" and not _configured(DEEPGRAM_API_KEY):
        missing.append("DEEPGRAM_API_KEY")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
