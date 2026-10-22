# InterviewOS behavior evaluation

The versioned offline suite exercises the compiled production interview graph and its nodes with deterministic fixture model outputs. It makes no model-provider, network, Deepgram, or audio-device calls.

From the repository root:

```powershell
python -m app.evaluation_harness
python -m app.evaluation_harness --case repeat-08 --case skip-11
python -m app.evaluation_harness --json reports\conversation-evaluation.json
python -m pytest
```

The first three commands run the offline behavior harness; `--json` writes the same run as a machine-readable report and still prints the terminal summary. The last command runs the normal test suite separately. The commands were verified with Python 3.13 in this checkout.

Case pass/fail counts summarize deterministic production-contract checks. Each individual semantic check is also counted: `pass` means its deterministic constraint succeeded, `fail` means a required deterministic contract failed, and `inconclusive` means the offline interface cannot judge semantic quality reliably. Inconclusive checks are never reported as passed. Reports include check counts by dimension, case counts by category, and list every non-passing check with a reason and a safely shortened observation.

## Limitations

Intent labels and model-generated response/analysis outputs are fixtures. The suite verifies how production code routes and handles those outputs, not live natural-language classification accuracy or the semantic quality of live model prose. Hint leakage is screened against case-specific forbidden terms and is not a semantic proof. Adaptive follow-up semantic relevance is therefore reported as inconclusive. Live evaluation is not included: the production answer-analysis call combines scoring and adaptive recommendations into one result, and the repository has no independent judge interface. Scoring those live outputs would let the model under evaluation determine its own result or require a new judge architecture, which is outside this evaluation-only milestone.

The dataset and rubric are versioned at `datasets/conversation_cases_v1.json` and `rubric_v1.json`. Add synthetic cases with unique stable IDs; malformed fields, unsupported intents, and duplicate IDs fail during dataset loading with a case-specific error.
