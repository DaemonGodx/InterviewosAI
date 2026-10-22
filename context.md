# InterviewOS — Current Project Context

## Purpose and scope

InterviewOS is a local command-line practice interviewer. It uses a candidate resume and selected role/experience settings to prepare a reviewed interview plan, conduct text or optional voice interviews, analyze responses, adapt within bounded policies, generate an evidence-based report, and save sessions locally in SQLite.

This is a preparation tool, not a hiring system. There is no web application, account system, cloud synchronization, proctoring, face analysis, or behavioral monitoring.

## Current application flow

1. `python -m app` starts the Rich CLI. `python -m app --help` displays command help without starting an interview.
2. `ResumeService` extracts PDF or DOCX text and builds a structured candidate profile through the configured Groq client.
3. `PlannerService` prepares a complete technical, behavioral, or mixed question plan. Questions are semantically checked and repaired with bounded retries and safe fallbacks. The candidate can review, edit, or regenerate plan questions before starting.
4. `InterviewEngine` runs the LangGraph state machine. Each new utterance is classified before it can enter answer analysis.
5. Substantive answers use the existing structured assessment and adaptive-decision flow. Accepted follow-ups remain tied to their source competency; the plan cursor and follow-up caps bound progression.
6. The evaluator checks aligned question, answer, analysis, and metadata history, then summarizes usable evidence, scores, confidence, and readiness.
7. `Database` saves the session in the existing SQLite JSON columns. Older session records load with compatibility defaults.

## Interview behavior

- **Technical** interviews focus on engineering knowledge, implementation, debugging, architecture, and trade-offs.
- **Behavioral** interviews focus on work behavior, communication, collaboration, ownership, judgment, learning, and motivation.
- **Mixed** plans are deterministically allocated at approximately 60% Technical and 40% Behavioral, with both categories included when the plan has at least two questions. A one-question Mixed plan is Technical.
- Conversational intents include answers, admission of not knowing, repeat, clarify, hint, skip, correction, off-topic, and ambiguous turns. Non-scoring conversational requests do not enter the aligned answer/analysis history. Skips use blank/no-evidence semantics.
- Adaptive follow-ups and conversational retries are bounded. The configured per-question follow-up cap is 0–3 (default 1); additional safeguards prevent repeated unknown or ambiguous turns from looping indefinitely.
- Evaluation uses successfully assessed evidence. Failed analysis is unavailable evidence, not a candidate score. Sparse evidence limits confidence and readiness claims.

## Voice and persistence

Voice mode uses Deepgram for speech-to-text and text-to-speech, with local `sounddevice` capture/playback. Recording stops when the candidate presses Enter. TTS linear16 output is played as raw 16-bit PCM at 16 kHz. Text mode does not require a Deepgram API key; Voice mode does.

The application stores session records in `interviewos.db` in the repository directory by default. The database is local and is not synchronized remotely.

## Repository map

- `app/__main__.py` — CLI and text/voice interview loop.
- `app/engine/graph.py`, `state.py`, `nodes.py` — LangGraph routing, typed state, conversational handling, analysis, adaptation, and evaluation.
- `app/models/schemas.py`, `db.py` — Pydantic models and SQLite persistence.
- `app/services/planner_service.py` — deterministic plan allocation, question generation, semantic validation, review/edit/regeneration support.
- `app/services/resume_service.py` — PDF/DOCX extraction and candidate profiling.
- `app/services/speech_service.py`, `app/audio/audio_service.py` — Deepgram and local audio boundaries.
- `app/evaluation_harness.py` and `tests/evaluations/` — offline fixture-driven checks of production conversational paths.
- `tests/` — automated tests plus manually invoked smoke scripts whose filenames start with `manual_` to avoid accidental pytest collection.

## Setup and commands

Use Python 3.10 or newer. Install dependencies from `requirements.txt`, create `.env` with `GROQ_API_KEY`, and add `DEEPGRAM_API_KEY` only for Voice mode.

```powershell
python -m pip install -r requirements.txt
python -m app --help
python -m app
python -m pytest -q
python -m app.evaluation_harness
```

Optional manual smoke scripts require the corresponding live service and environment:

```powershell
python tests/manual_question_generation.py
python tests/manual_stt.py
```

The question-generation smoke script uses Groq. The STT smoke script requires Deepgram credentials and a working microphone. They are not part of the deterministic unit suite.

## Current limitations

- No web UI, account/authentication layer, cloud data store, proctoring, or computer-vision analysis.
- No OCR for scanned PDFs and no support for legacy binary `.doc` resumes.
- Voice capture is Enter-to-stop and depends on local audio devices.
- Automated evaluation is practice feedback based on recorded answers; it is not a hiring recommendation.
- Live provider and physical audio-device behavior require manual verification. The automated tests mock external services.
