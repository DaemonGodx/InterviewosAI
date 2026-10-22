# InterviewOS

InterviewOS is a local, command-line interview practice application. It parses a candidate resume, builds a reviewed interview plan, collects text or voice answers, optionally adapts the next planned question, evaluates the recorded evidence, and saves sessions to SQLite.

## What is implemented

- Resume text extraction from PDF and DOCX, followed by candidate profile generation through Groq structured output.
- Technical, Behavioral, and Mixed interview plans. Technical questions primarily assess engineering knowledge and practice. Behavioral questions assess work behavior such as communication, collaboration, ownership, judgment, and learning.
- A plan review step before the interview. The candidate can confirm the plan, edit one question, regenerate one question, or regenerate the plan. Edited and regenerated questions keep their intended category and are revalidated.
- A LangGraph interview flow with bounded adaptive actions. The confirmed plan is the default sequence; follow-ups consume planned slots, keep the same category, and are limited by `max_follow_ups_per_question` (0–3, default 1). Difficulty adaptations apply to the next unasked slot. The reviewed question wording is retained in the stored plan.
- Candidate utterances are classified before analysis. Repeat, clarification, hint, correction, skip, off-topic, and ambiguous turns are handled separately from scored answers. Repeated ambiguity and admissions of not knowing have bounded fallback behavior.
- Text mode and Deepgram voice mode. Voice uses Deepgram STT/TTS and local `sounddevice`; TTS linear16 PCM is decoded as 16-bit little-endian audio at 16 kHz before playback.
- Deterministic evaluation of successfully analyzed answers, including category-specific scores, evidence summaries, bounded scores, and confidence based on usable answers and competency breadth. Failed analysis is unavailable evidence.
- SQLite persistence for plans, answers, analyses, decisions, adaptation history, and evaluations, with compatibility defaults for older saved sessions.

## Mixed interview allocation

Mixed plans use approximately 60% Technical and 40% Behavioral questions. The Technical count is rounded to the nearest whole question, with at least one question of each category whenever the plan contains at least two questions. Categories are interleaved. A one-question Mixed interview deterministically selects Technical and the CLI warns that Behavioral coverage is omitted.

## Evaluation confidence and scores

Only nonblank answers with a successful, usable analysis contribute to scores. Confidence also reflects the number of distinct primary competencies assessed: fewer than three usable answers is **Very Low**; fewer than four usable answers or fewer than two competencies is **Low**; fewer than six usable answers or fewer than three competencies is **Moderate**; otherwise it is **High**. Sparse evidence results in “Insufficient evidence” for role readiness.

Per-answer aggregation is explicit: Technical overall evidence is `0.50 × correctness + 0.30 × depth + 0.20 × communication`; Behavioral uses the same weights, with correctness interpreted as relevance/specificity. Overall score is the mean of usable per-answer scores multiplied by 10 (0–100). Technical, behavioral, problem-solving, and communication dimensions are left unavailable when their category/evidence was not assessed. Readiness is limited to observed interview evidence and does not infer personality or general suitability.

## Setup

Use Python 3.10 or newer. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create `.env` in the repository root. Groq is required for resume profiling and interview planning. Deepgram is required only for Voice mode.

```env
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

Keep real credentials private. `.env` is local configuration and should not be committed.

## Run

```powershell
python -m app --help
python -m app
```

Choose Text mode to run without a Deepgram key. Voice mode checks for Deepgram credentials after mode selection. The supported resume formats are `.pdf` and `.docx`; legacy binary `.doc` files are not supported.

## Tests

Run the automated test suite and the offline conversational behavior harness:

```powershell
python -m pytest -q
python -m app.evaluation_harness
```

Automated tests mock Groq, Deepgram, and audio-device calls; they do not require live credentials or hardware. The manual smoke scripts are kept under `tests/` but are not named for pytest collection:

```powershell
python tests/manual_question_generation.py
python tests/manual_stt.py
```

The question-generation smoke script requires a valid Groq key. The STT smoke script requires a valid Deepgram key and a working microphone. Neither script is part of the deterministic automated suite.

## Known limitations

- The application is a local CLI prototype. There is no web UI, authentication, proctoring, or behavioral monitoring.
- Voice recording stops when the candidate presses Enter; it does not use voice activity detection.
- Audio playback and recording depend on the machine's available `sounddevice` input/output devices.
- PDF extraction does not perform OCR on scanned documents.
- Readiness estimates are constrained summaries of interview evidence, not hiring recommendations. One or two usable answers are insufficient for a confident readiness judgment.
- SQLite stores sessions locally in `interviewos.db`; no account sync or remote database is implemented.
