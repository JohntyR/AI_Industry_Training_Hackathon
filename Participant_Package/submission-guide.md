# Team Submission Guide

This guide defines what each team must submit and how the organizers will call the submitted agent during evaluation.

## Required Repository Structure

Each team repository must be public and reachable by the organizers before the submission deadline.

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

| Path | Required | Purpose |
|---|---:|---|
| `submission.json` | Yes | Final team metadata, pinned repository commit, and live agent endpoint registration. |
| `README.md` | Yes | Project summary, architecture, run instructions, endpoint notes, and known limitations. |
| `src/` | Yes | Agent source code and data-query or retrieval tools. |
| `training/` | Yes | Fine-tuning scripts, preparation notes, configuration, logs, metrics, and model summary. |
| `logs/` | Yes | Useful non-sensitive training or agent-run logs. |
| `Participant_Package/` | Yes | Challenge references, sample request and response files, validation rules, and handouts. |

The supplied environment contains the common dependencies used during the event. Official scoring calls the registered agent endpoint rather than rebuilding every project live.

## Final `submission.json`

`submission.json` is the single source of truth for your team — it registers your team identity, your GitHub commit, and tells the evaluation harness where to reach your agent. The harness reads this file directly; there is no separate teams config to edit.

The file `Participant_Package/submission_template.json` is a reference example. Update the real
`submission.json` at the repository root; the evaluator does not use the template file as your registration.

```json
{
  "schema_version": "1.0",
  "team_id": "team-example",
  "team_name": "Example Team",
  "github_url": "https://github.com/example/team-agent",
  "commit_sha": "0123456789abcdef0123456789abcdef01234567",
  "agent": {
    "endpoint": "http://172.20.x.x:5000",
    "health_path": "/health",
    "query_path": "/query",
    "timeout_seconds": 300
  },
  "model": {
    "endpoint": "http://172.20.x.x:8001/v1",
    "model_name": "nemotron-8b-finance-lora"
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `team_id` | Yes | Short identifier used in report filenames — no spaces |
| `team_name` | Yes | Display name on the leaderboard |
| `github_url` | Yes | Public repo URL |
| `commit_sha` | Yes | Exact 40-char commit hash to be judged |
| `agent.endpoint` | Yes | Full URL where your server is reachable — use IP, not hostname |
| `agent.health_path` | Yes | Path the harness calls for the pre-eval health check (usually `/health`) |
| `agent.query_path` | Yes | Path the harness POSTs questions to (usually `/query`) |
| `agent.timeout_seconds` | Yes | Set to 300; used per request and capped by the organizer's `--timeout` value |
| `model.model_name` | Yes | Name or alias of the fine-tuned model used by the submitted solution |
| `model.endpoint` | Conditional | Reachable OpenAI-compatible endpoint when direct model testing is the agreed assessment method |

Fine-tuned model quality is an official scoring category. If you cannot expose the model endpoint,
agree on another technical assessment method with the organizers before the deadline and document
it in your README.

These fields have different owners during evaluation:

- `agent.endpoint`, `health_path`, `query_path`, and `timeout_seconds` are used directly by the
  hidden-question harness.
- `github_url` and `commit_sha` are used by organizers when cloning the public repository for the
  architecture and repository assessment. The hidden-question harness records them as metadata but
  does not clone the repository itself.
- `model.model_name` and `model.endpoint` support the separate fine-tuned-model assessment. They are
  recorded in organizer reports but are not used to answer hidden questions.

**Use your machine's IP address** (`ip addr` to find it), not `localhost`. The organizer harness runs on a different machine. Do not put credentials or API keys in this file.

## Agent API Contract

Your submitted agent server must live in `src/` and expose the two endpoints below. Teams may choose
their own internal module structure while following the required API and architecture contracts.

The required internal model flow is Qwen3.6-35B-A3B-FP8 through `agent-brain` for planning and tool-call generation,
application code for executing those tool calls, and fine-tuned Nemotron for final answer synthesis
from the verified results. Participants fine-tune Nemotron, not Qwen3.6-35B-A3B-FP8.

The cluster bootstrap initially sets `DOMAIN_PREDICT_MODE=mock`. Change it to
`DOMAIN_PREDICT_MODE=llm` after the fine-tuned adapter is available and before official evaluation.

### `GET /health`

Returns HTTP 200 when your agent is ready. The harness checks this before starting the evaluation. If the health check fails, your team is skipped.

```json
{"status": "ok"}
```

### `POST /query`

Each request contains one question and your agent must return one JSON response. The harness uses
up to **three concurrent requests per team** by default, so `/query`, shared state, and model serving
must safely handle at least three simultaneous calls.

**Request body template** (`questions_template.json`):
```json
{
  "question": "From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?"
}
```

**Response template** (`answer_template.json`):
```json
{
  "answer": "41 of the 175 decision records changed the rate: 20 increases and 21 decreases.",
  "steps": 3,
  "tool_trace": [
    {
      "tool": "query_data",
      "args": {"dataset": "rba", "metric": "count_changes"},
      "result": "41 changes: 20 increases, 21 decreases"
    }
  ]
}
```

| Field | Required | Graded | Notes |
|---|---|---|---|
| `answer` | Yes | **Yes** | The only response field scored by the automated hidden-question judge |
| `steps` | No | No | Total tool calls plus synthesis steps; retained for private diagnostics |
| `tool_trace` | No | No | List of `{tool, args, result}`; retained for private diagnostics |

**Slow-response penalty:** If your agent takes longer than **60 seconds** to return a response, **20% of the earned points for that question are deducted**. This is applied before the hidden-question category score is calculated and is visible in your private per-question report. Design your agent to return within 60 seconds.

**Malformed or timed-out responses** score zero for that question.

## Official Scoring

The final hackathon score combines three category scores, each normalized to 100:

| Category | Weight | What judges assess |
|---|---:|---|
| Fine-tuned model quality | 30% | Training preparation and method, improvement over the supplied base model, evaluation evidence, robustness, and actual use of the fine-tuned model. |
| Architecture and repository quality | 30% | End-to-end design, tools and retrieval, code quality, API compliance, reliability, reproducibility, documentation, artifacts, logs, and repository hygiene. |
| Hidden-question evaluation | 40% | Component-based correctness on unseen questions after any response-time deductions. |

```text
final_score =
    (fine_tuned_model_score * 0.30)
  + (architecture_repository_score * 0.30)
  + (hidden_question_score * 0.40)
```

The complete rubric is in `Challenge_Brief.md`.

## Hidden-Question Scoring - 40%

Each question has one or more grading components. Points are awarded per component — **partial credit is possible**.

```
question max score = 10 points
  ├── component C01 (5 pts): "BHP.AX was best at +22.17%"   → YES/NO
  └── component C02 (5 pts): "AMP.AX was worst at -50.04%"  → YES/NO
```

The LLM judge checks each component independently against your answer. Satisfying one component in
this example earns 5/10; satisfying both earns 10/10.

**What earns partial credit:**
- Stating the right number but missing a secondary fact (e.g. date of first occurrence)
- Getting some tickers right in a multi-ticker comparison but missing others

**What earns zero for a component or question:**
- An unsupported or guessed answer that does not satisfy the expected fact; tool use itself is not directly scored
- Correct number buried in wrong context ("there are 41 records in total, of which 20 are holds" — judge reads "20 holds" not "20 increases")
- Hedging that changes meaning ("approximately 41", "likely around 20")
- Empty or error response

**Equivalent formatting is accepted:** `"1,234"` = `"1234"`, `"Jan 2024"` = `"2024-01"`, minor rephrasing that preserves meaning.
Each evaluation case may also declare a `grading.tolerance_note`. The judge receives and applies that
case-specific rule, such as `+/-0.02` percentage points for calculated returns. Public calibration
questions expose their tolerance notes; hidden questions use the same schema without revealing the
expected facts.

## What the leaderboard shows

The public leaderboard shows only **Rank**, **Team**, and **Score** — nothing else. Latency, tool usage, step counts, and availability are recorded per team but not published on the public leaderboard. Teams do not see other teams' internals.

Only the weighted final Score determines rank. The hidden-question category is calculated as
`sum(earned_points) / sum(max_points) x 100%` after slow-response penalties, then contributes 40%
to the final score.

**Health check is a hard gate.** If `GET /health` does not return 200 at the start of the run, the team is skipped entirely — no questions are graded. Test your endpoint from a different machine before submitting.

## Your own detailed report

After the eval run, each team receives a **detailed private report** covering only their own agent:

- Hidden-question score, tool rate, avg steps, avg latency, P95 latency, availability, slow penalty
- Per-question breakdown: earned/max, latency, tool usage, per-component YES/NO verdicts

The report **excludes hidden grading facts** — you see whether each component passed but not the expected fact the judge checked. Other teams' data is never included.

## README Requirements

The `README.md` must include:

- Team name and short project summary.
- Exact command used to run the agent.
- Agent endpoint paths and expected response shape.
- High-level architecture showing Qwen planning and tool-call generation, runtime tool execution, retrieval, and fine-tuned Nemotron answer synthesis.
- Training summary explaining what was fine-tuned, which preparation method was used, and where supporting evidence is stored.
- Base-versus-fine-tuned evaluation results and the method organizers should use to assess the final model.
- Known limitations and failure cases.

## Training Evidence

The `training/` folder must contain enough evidence for judges to understand and reproduce the team's fine-tuning work. Suitable contents include:

- Training or fine-tuning scripts.
- Data-preparation scripts or notebook exports.
- Configuration and hyperparameters.
- Training logs, metrics, or screenshots.
- A short model card or model summary.
- Held-out results or representative comparisons showing improvement over the supplied base model.

## Submission Checklist

Before submitting, confirm that:

- The repository is public and the organizers can clone it without credentials.
- `submission.json` is at the repository root with your final IP, port, and commit SHA filled in.
- `commit_sha` is the exact 40-character commit hash to be judged.
- `Participant_Package/answer_template.json` is present and follows the required response shape.
- `GET /health` returns 200 from the IP in `submission.json`.
- `POST /query` accepts `{"question": "..."}` and returns a JSON object with a non-empty `answer`; `steps` and `tool_trace` are optional.
- `/query` and the model-serving stack handle at least three concurrent requests safely.
- Most responses return within 60 seconds (responses over 60s incur a 20% point deduction).
- `README.md`, `src/`, `training/`, and `logs/` contain the required material.
- The fine-tuned Nemotron model is used during inference.
- The supplied Qwen3.6-35B-A3B-FP8 `agent-brain` alias performs planning, tool selection, and tool-call generation.
- Application code executes Qwen's tool requests and returns structured results to the reasoning loop.
- `DOMAIN_PREDICT_MODE=llm` is enabled after the adapter is served; the bootstrap `mock` mode is disabled.
- Fine-tuned Nemotron synthesizes the final answer from the question and verified tool results.
- `model.model_name` identifies the fine-tuned model and `model.endpoint` is reachable when direct testing is the agreed assessment method.
- The repository documents base-versus-fine-tuned results and the complete Qwen/runtime/Nemotron architecture.
- No credentials or organizer-only evaluation material are committed.
