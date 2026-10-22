# InterviewOS — Next Steps

## Current status

The local CLI interview flow is implemented: resume profiling, deterministic mixed-plan allocation, plan review, conversational turn handling, structured answer analysis, bounded adaptive follow-ups, evidence-based evaluation, SQLite persistence, and optional Deepgram voice support. The current architecture and limitations are summarized in `context.md` and `README.md`.

`AUDIT.md` was a point-in-time implementation audit and has been removed. Re-run a fresh code audit if a future engineering task requires one.

## Remaining verification

- [ ] Run `python -m pytest -q` on the current checkout before a release or commit and record its actual result.
- [ ] Run `python -m app.evaluation_harness` when validating conversational routing changes.
- [ ] Manually run `python tests/manual_question_generation.py` with a valid Groq key when checking live question generation.
- [ ] Manually run `python tests/manual_stt.py` with a valid Deepgram key and working microphone when checking live transcription.
- [ ] Perform a complete Voice-mode interview on the target machine to verify microphone capture, transcription, speech synthesis, playback, and session completion together.

The manual smoke scripts are intentionally separate from automated pytest collection because they require live provider credentials or hardware. Automated tests should continue to mock Groq, Deepgram, and audio-device calls.

## Deferred / out of scope

No implementation work is currently approved for a web UI, user accounts, cloud sync, OCR, legacy `.doc` support, proctoring, computer vision, behavioral monitoring, or voice activity detection. Add these only with a separate scoped request.
