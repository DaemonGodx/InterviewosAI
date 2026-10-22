# InterviewOS — Development Track

## Project state

InterviewOS is an implemented local CLI prototype. The active architecture is Python + Rich CLI, Pydantic schemas, LangChain/Groq structured model calls, a LangGraph interview flow, Deepgram speech services for optional Voice mode, and SQLite persistence. The current working tree may also contain local artifacts and legacy-file changes; this document describes the active `app/` architecture, not a clean-release claim.

## Architecture and lifecycle

1. The CLI validates the required Groq configuration and accepts a PDF or DOCX resume.
2. `ResumeService` extracts text and requests a typed candidate profile.
3. `InterviewConfig` collects role, experience, interview type, question count, and mode. Deepgram is required only for Voice mode.
4. `PlannerService` creates and semantically validates a complete question plan. The CLI allows confirm, edit, one-question regeneration, or full regeneration before starting.
5. The LangGraph engine delivers planned questions, collects answers, performs rubric-aware analysis, and applies bounded adaptive decisions to the next eligible plan slot.
6. The evaluation node validates parallel histories and summarizes usable evidence, category scores, confidence, and constrained readiness.
7. The CLI displays the report and SQLite saves session state for previous-session history.

## Interview policies

- **Technical:** primarily assesses technical knowledge, implementation, debugging, architecture, problem-solving, and engineering trade-offs.
- **Behavioral:** primarily assesses communication, teamwork, conflict resolution, leadership behavior, ownership, adaptability, motivation, judgment, and learning.
- **Mixed:** schedules approximately 60% Technical and 40% Behavioral, interleaved; both categories appear when there are at least two questions. One question deterministically selects Technical and has limited coverage.
- The candidate confirms the plan before questions are asked. Adaptive actions preserve the confirmed question wording. Follow-ups reference the prior answer, remain in-category, and consume a configured unasked slot.
- Evaluation excludes failed analyses as unavailable evidence. Confidence depends on usable analyses and competency breadth. Sparse evidence does not produce a confident role-readiness conclusion.

## Implemented work

| Area | Current status |
|---|---|
| Resume/profile, planning, and review CLI | Implemented; provider calls require Groq |
| Technical, Behavioral, and Mixed planning | Implemented with semantic validation and bounded repair |
| LangGraph answer analysis and adaptive flow | Implemented with bounded actions and confirmed-plan defaults |
| Evidence-based evaluation | Implemented with state alignment checks and bounded category scores |
| Text mode | Implemented; Deepgram credentials are not required |
| Voice mode | Implemented through existing Deepgram and local audio abstractions; linear16 PCM playback boundary corrected |
| SQLite persistence/history | Implemented with normalized graph dictionaries and backward-compatible loading |
| Web UI / proctoring / monitoring | Not implemented |

## Verification status

Phase 4–6 regression run: **92 tests passed** with no warnings. The automated suite mocks external provider calls. `python -m app --help` displayed help and exited; `python -m app` reached the menu and exited in an isolated temporary working directory. Real microphone/speaker operation and live provider credentials still require manual verification in the target environment.

## Setup and test commands

```powershell
python -m pip install -r requirements.txt
python -m app --help
python -m app
python -m pytest -q
```

Groq is required for profile generation and planning. Deepgram is required only when the candidate selects Voice mode. Supported resume inputs are PDF and DOCX; scanned PDF OCR and legacy `.doc` parsing are not implemented.

## Remaining limitations

- Local CLI prototype only; no web interface, user accounts, proctoring, or behavioral monitoring.
- Voice capture is Enter-to-stop and depends on local audio hardware/device configuration.
- PDF text extraction has no OCR.
- Interview readiness is an evidence-limited practice summary and should not be treated as a hiring decision.
- Previous sessions are stored in local SQLite without cloud synchronization.
