# Cognitivo Hackathon

Build and fine-tune an evidence-grounded market signal agent over RBA, ASX, and AFR data.

## Challenge Scope

This hackathon evaluates how effectively an agent answers financial-market questions using the
approved local datasets.

| Area | Scope |
|---|---|
| Question coverage | Easy, medium, and hard questions using one or more approved datasets. |
| Approved data | RBA cash-rate decisions, ASX company prices, and the AFR news corpus. |
| Evaluation split | 15 public practice questions with answers; the remaining official questions stay with the organizers. |
| Supplied reasoning brain | **Qwen3.6-35B-A3B-FP8** through the LiteLLM `agent-brain` alias. Qwen3.6-35B-A3B-FP8 performs planning, tool selection, tool-call generation, and iterative reasoning. |
| Fine-tuning target | `Llama-3.1-Nemotron-Nano-8B-v1`. Teams fine-tune Nemotron for grounded financial-domain answer synthesis. |
| Official score | 30% fine-tuned model quality, 30% architecture and repository quality, and 40% hidden-question performance. |

## Objective

Build and fine-tune a domain model, integrate it into a well-engineered agent, and demonstrate that
the complete system can answer unseen financial-market questions. The benchmark includes easy,
medium, and hard questions as well as single-dataset and cross-dataset questions.

The solution is expected to do more than produce a plausible sentence. It must show measurable
fine-tuning quality, a reproducible repository and system architecture, grounded use of the supplied
data, and clear answers on the hidden evaluation set.

## Required Model Roles

The submitted solution uses two model roles. Do not train Nemotron to replace the supplied Qwen3.6-35B-A3B-FP8
reasoning brain or use Nemotron as the primary tool-calling model.

| Component | Required responsibility |
|---|---|
| Qwen3.6-35B-A3B-FP8 through `agent-brain` | Receives the question, plans the approach, selects `query_data` or retrieval tools, emits tool calls and arguments, reviews tool results, and decides whether another tool call is required. Participants do not fine-tune Qwen3.6-35B-A3B-FP8. |
| Agent runtime | Validates and executes Qwen3.6-35B-A3B-FP8's tool calls against the approved local datasets, records the trace, and returns structured results to Qwen3.6-35B-A3B-FP8. The model requests calls; the application code executes them. |
| Fine-tuned Nemotron through `DOMAIN_FT_MODEL` | Receives the question and accumulated verified tool results after the Qwen3.6-35B-A3B-FP8 reasoning loop, then synthesizes the final concise financial-domain answer. This is the model participants fine-tune and assess against the supplied base Nemotron. |

```text
question
  -> Qwen3.6-35B-A3B-FP8 agent-brain plans and emits tool calls
  -> agent runtime executes query_data / retrieve
  -> tool results return to Qwen3.6-35B-A3B-FP8 until reasoning is complete
  -> fine-tuned Nemotron synthesizes the final answer
  -> POST /query returns {"answer": "..."}
```

The organizer may also use Qwen3.6-35B-A3B-FP8 as the independent LLM judge. That evaluation call is
separate from the Qwen3.6-35B-A3B-FP8 brain inside the submitted agent and does not replace the required
fine-tuned Nemotron synthesis step.

The cluster bootstrap begins with `DOMAIN_PREDICT_MODE=mock` for pre-training integration tests.
Teams must switch to `DOMAIN_PREDICT_MODE=llm` after serving their adapter and before official
evaluation so the submitted solution actually uses the fine-tuned Nemotron model.

## Task Format

For each question, the evaluator sends one JSON object in the format shown by
`questions_template.json`: a single `question` field containing the question text. The agent must
return one JSON object in the format shown by `answer_template.json`. The machine-readable rules in
`validate.json` are used to confirm that the response can be parsed.

Questions may require direct retrieval, filtering, counting, financial calculations, chronological
comparison, ranking, or reasoning across multiple datasets.

`public_questions.jsonl` contains 15 calibration cases in the same format the evaluation harness
uses. Each line is a JSON object with `id`, `prompt`, `difficulty`, `datasets`, and a `grading`
object listing the expected facts and their point values. Pass only the `prompt` field as the
`question` to your agent. The `grading.components[].expected_fact` values show what the judge checks
for.

> **Public questions are calibration cases.** Use them to test retrieval, calculations, formatting,
> and the complete agent pipeline. Do not implement question-ID-specific hard-coded answers.

## Required Response

Every response must include:

- `answer`: a direct response containing every requested component. This is the only response field scored in the hidden-question benchmark.
- `steps`: the number of reasoning steps the agent took. This optional integer is recorded for private organizer diagnostics.
- `tool_trace`: an ordered list of tool calls made during reasoning. This optional list is recorded for private organizer diagnostics.

Only `answer` is required. `steps` and `tool_trace` are optional but strongly recommended because
they help organizers diagnose agent behavior. The response is validated against `validate.json`.

```json
{
  "answer": "Direct answer with all requested values.",
  "steps": 3,
  "tool_trace": [
    {
      "tool": "tool_name",
      "args": {"param": "value"},
      "result": "tool output summary"
    }
  ]
}
```

## Scoring

The final hackathon score is calculated from three independently assessed categories. Each category
is normalized to a score out of 100 before its official weighting is applied.

| Category | Weight | What is assessed |
|---|---:|---|
| Fine-tuned model quality | 30% | Quality and relevance of the fine-tuned model; measurable improvement over the supplied base model; training-data preparation; evaluation evidence; model behavior, robustness, and successful use of the fine-tuned model during inference. |
| Architecture and repository quality | 30% | Agent and model architecture; appropriate use of deterministic tools, retrieval, and data processing; code quality and reliability; API-contract compliance; reproducibility; repository structure; README and run instructions; training artifacts; logs; pinned commit; security and documented limitations. |
| Hidden-question evaluation | 40% | Performance of the submitted agent on unseen easy, medium, hard, single-dataset, and cross-dataset questions. Answers are graded using component-based correctness and partial credit. |

```text
final_score =
    (fine_tuned_model_score * 0.30)
  + (architecture_repository_score * 0.30)
  + (hidden_question_score * 0.40)
```

### Fine-Tuned Model Quality - 30%

Judges review the submitted training evidence and may test the declared fine-tuned model endpoint.
Teams must demonstrate that the fine-tuned model is genuinely used by the submitted solution. The
assessment considers:

- The relevance, quality, and documented preparation of the fine-tuning data.
- The training method, configuration, hyperparameters, checkpoints, and model-selection rationale.
- Quantitative and qualitative comparison with the supplied base model on held-out or validation examples.
- Robustness, consistency, domain understanding, and avoidance of unsupported claims.
- Evidence that the final agent uses Qwen for planning and tool-call generation, then routes the verified tool results through the fine-tuned Nemotron model for final synthesis.

Training evidence must be reproducible and must not contain hidden evaluation data.

### Architecture and Repository Quality - 30%

Judges inspect the public GitHub repository at the exact commit SHA declared in `submission.json`.
The assessment considers:

- A clear end-to-end architecture covering Qwen planning/tool-call generation, runtime tool execution, fine-tuned Nemotron synthesis, retrieval, and data flow.
- Correct separation of responsibilities: Qwen selects and requests tools, application code executes them, and fine-tuned Nemotron writes the grounded final answer.
- Correct use of structured parsing and deterministic calculations for dataset-derived facts.
- Maintainable source code, sensible module boundaries, error handling, timeouts, and safe fallbacks.
- Compliance with `GET /health` and `POST /query`, including valid JSON responses.
- Complete README, run instructions, architecture explanation, training summary, and known limitations.
- Useful training artifacts, configurations, metrics, and non-sensitive logs in the required folders.
- A clean, accessible repository with no credentials, hidden evaluation material, or machine-specific secrets.

### Hidden-Question Evaluation - 40%

Each hidden response is worth a maximum of 10 points. Points are allocated across the factual or
analytical components explicitly requested by that question.

| Scoring rule | What is assessed |
|---|---|
| Component-based correctness | Correct facts, dates, counts, calculations, rankings, sentiment, market direction, and every other output explicitly requested in the question. |
| Partial credit | Multi-part questions award the points attached to each independently correct requested component; one incorrect component does not erase correct components. |
| Equivalent expression | Equivalent date formats, harmless numeric formatting differences, and sentiment synonyms that preserve the reference meaning are accepted. Each case's `grading.tolerance_note` defines any permitted numeric tolerance or rounding. |
| Response validity | Only `answer` is required for automated hidden-question scoring. `steps` and `tool_trace` are retained for private diagnostics. Malformed or missing `answer` fields score zero. |

The grader requires only information requested by the prompt. Extra dates, prices, quotations, or
article drivers appearing in supporting material are not mandatory unless the question asks for
them.

#### Response-Time Rules

| Response time | Hidden-question scoring effect |
|---:|---|
| 60 seconds or less | Full earned points. |
| More than 60 seconds and no more than 300 seconds | 20% is deducted from the points earned for that question. |
| More than 300 seconds | Timeout and zero points for that question. |

`GET /health` is a hard gate for the hidden-question evaluation. If the registered agent does not
return HTTP 200 during the pre-evaluation health check, the team is skipped and receives no
hidden-question points for that run.

The harness sends up to three hidden questions concurrently to each team by default. The submitted
agent, tool runtime, and model servers must safely handle at least three simultaneous `/query`
requests without mixing state or responses.

### Ranking and Feedback

The final ranking uses the weighted score across all three categories. The public leaderboard shows
rank, team, and final score. Organizers retain private diagnostics, and each team may receive a
sanitized report for its own submission. Hidden expected facts and other teams' private data are not
shared.

## Required Deliverables

- A working agent that accepts the specified question object and returns the specified response object.
- Source code and any data-query or retrieval tools created by the team.
- Fine-tuning or training evidence: scripts, preparation notes, configuration, logs, and a short model summary.
- Evidence comparing the supplied base model with the team's fine-tuned model.
- A reachable fine-tuned model or a documented organizer-approved method for technical model assessment.
- A final `submission.json` registering the team, repository commit, and agent endpoint.
- A valid sample `Participant_Package/answer_template.json` demonstrating the agent's per-question response contract.
- A README explaining the architecture, how to run the agent, and known limitations.
- Useful logs or traces that allow the organizers to diagnose failed requests.

### Submission Structure

Submit a GitHub-style project using this structure:

```text
TeamSubmission/
  README.md
  submission.json
  src/
    .gitkeep
  training/
    .gitkeep
  logs/
    .gitkeep
  Participant_Package/
    answer_template.json
    Challenge_Brief.md
    public_questions.jsonl
    questions_template.json
    Setup_Instructions.md
    submission-guide.md
    submission_template.json
    validate.json
    handout/
      01_training_guide.md
      02_execution_guide.md
      03_scoring_and_examples.md
```

| Path | Purpose |
|---|---|
| `README.md` | Team name, architecture, exact run command, model integration, and known limitations. |
| `submission.json` | Final team metadata, repository commit, agent endpoint, and fine-tuned model information described in `submission-guide.md`. Include a reachable model endpoint when it is used for technical model assessment. |
| `src/` | The team's submitted agent implementation, including its data-query or retrieval tools. |
| `training/` | Data-preparation notes, fine-tuning configuration, scripts, metrics, and model summary. |
| `logs/` | Useful non-sensitive training or agent-run logs. |
| `training/.gitkeep`, `logs/.gitkeep` | Placeholders that keep initially empty required folders in Git. Replace or supplement them with the team's evidence and logs. |
| `Participant_Package/` | Challenge materials, examples, validation rules, and participant handouts. |
| `Participant_Package/answer_template.json` | Example response body for `POST /query`. |
| `Participant_Package/questions_template.json` | Example request body for `POST /query`. |
| `Participant_Package/submission_template.json` | Reference template for the real root `submission.json`. |
| `Participant_Package/validate.json` | Machine-readable validation schema for agent responses. |
| `Participant_Package/public_questions.jsonl` | Public calibration questions used to test the complete agent pipeline. |
| `Participant_Package/Challenge_Brief.md` | Official challenge scope, scoring, deliverables, rules, and technical reference. |
| `Participant_Package/Setup_Instructions.md` | Participant environment, dataset, model, and setup information. |
| `Participant_Package/submission-guide.md` | Detailed repository, API, scoring, and submission requirements. |
| `Participant_Package/handout/` | Training, execution, scoring, and worked-example guides. |

## Rules and Constraints

- Use only approved local datasets and services during official scoring.
- Do not use unrestricted external browsing during scoring.
- Do not alter the source datasets.
- All official responses must be valid JSON and follow the required contract.
- Return a response for every question, even when evidence is insufficient. State the limitation clearly in the `answer` field instead of returning an empty response or inventing a figure.
- Malformed, crashing, or timed-out responses may receive no credit for that case.
- Keep secrets and credentials out of submitted files and logs.

## Technical Reference

The points below are non-negotiable for reproducibility. Scores are computed by running the same
tool calls against the same data. If your implementation uses a different search scope or field
set, your counts will not match.

### Dataset Field Schemas

| Dataset | Fields |
|---|---|
| AFR | `HEADLINE, SUBHEAD, INTRO, TEXT, NEWSPAPER, PUBLICATIONDATE` |
| ASX | `ticker, date, open, high, low, close, volume` |
| RBA | `Effective Date, Change % points, Cash rate target%` (UTF-8 BOM encoding) |

### AFR Text Search

> **All AFR pattern counts must search across `HEADLINE`, `SUBHEAD`, `INTRO`, and `TEXT`
> combined.** Searching only the headline or only the body will produce different counts that will
> not match the reference answers. Use case-insensitive, once-per-record matching: a record counts
> once even if the pattern appears in multiple fields.

Whole-word searches must use word-boundary anchors, such as `\bNAB\b` rather than just `NAB`.
Short acronyms without boundaries will match substrings in unrelated words and significantly
inflate counts.

### Fine-Tuning Reference Baseline

Participants receive **Llama-3.1-Nemotron-Nano-8B-v1**. The configuration below is a confirmed
working starting point.

> **Note:** These values are a reference baseline, not a required configuration. Teams are
> encouraged to experiment with the tunable parameters and justify their choices while staying
> within the available hardware, event time, and model-context constraints.

| Parameter | Reference starting value |
|---|---|
| NeMo container | `nvcr.io/nvidia/nemo:25.09` |
| LoRA rank | 32 |
| Sequence length | 512 (longer sequences may run out of memory on a single node) |
| Learning rate | `5e-5` recommended (`1e-4` causes a loss spike after warmup) |
| Training steps | 100 for a full run; the step 20 checkpoint already shows meaningful improvement |

### Model Serving Endpoints

| Service | Default endpoint | Notes |
|---|---|---|
| LiteLLM proxy | `http://localhost:4000` | Configured by organizers; use `LITELLM_BASE_URL`, `BRAIN_MODEL=agent-brain`, `DOMAIN_FT_MODEL=domain-ft`, and switch `DOMAIN_PREDICT_MODE` from `mock` to `llm` after the adapter is live. |
| Qwen3.6-35B-A3B-FP8 reasoning brain (vLLM) | Port `8000` on the assigned brain/agent node | Served by organizers behind the `agent-brain` alias for planning and tool-call generation. |
| Fine-tuned Nemotron (vLLM) | Port `8001` on the assigned fine-tuning/model node | Team deploys after training behind the `domain-ft` alias for final synthesis. |

Each team receives a two-node GIGABYTE Atom cluster with one NVIDIA GB10 per node. The organizers
provide the actual hostnames and IP addresses. Any hostname or IP shown in a command must be replaced
with the value assigned to your cluster.

> **Keep all credentials and endpoint URLs in environment variables.** Do not hard-code them in
> source files. Source the organizer-provided `~/team.env` before starting your services. The
> evaluation harness calls the registered agent endpoint; it does not inject variables into the
> participant's running process.
