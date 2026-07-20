# Participant Setup Information

The Atom environment is already prepared for participants. The datasets, Qwen3-35B reasoning brain,
base Nemotron model, Python environment, and local model-serving services are supplied by the
organizers.

Each team receives a two-node GIGABYTE Atom cluster with one NVIDIA GB10 and 128 GB unified memory
per node. The organizers provide the hostnames and IP addresses for each cluster; do not assume
fixed machine names or copy addresses from another team.

Participants should not download replacement datasets or switch to unrestricted external services during scoring.

## Supplied Datasets

The hackathon uses exactly three datasets:

| Dataset | Supplied folder | Intended use |
|---|---|---|
| RBA cash-rate decisions | `RBA-Rates-2010-2026` | Rate changes, hikes/cuts, dates, targets, and period comparisons |
| ASX company prices | `ASX-18-companies-2015-2021-Jasonl` | Returns, volume, rankings, drawdowns, baskets, and event windows |
| AFR news corpus | `AFR Jasonl` | Article retrieval, pattern counts, date aggregation, and news evidence |

Use structured parsing and deterministic calculations for RBA and ASX data. Use the supplied local full-text search, indexed search, or RAG service for AFR records. Cross-dataset answers must respect the overlapping date coverage and clearly identify missing coverage.

## Supplied Models and Required Roles

The supplied **Qwen3-35B** model is available through the LiteLLM `agent-brain` alias. Qwen3-35B is the agent's
reasoning and orchestration model: it plans the approach, selects tools, emits tool calls and
arguments, examines results, and decides whether the tool loop should continue. Participants do
not fine-tune Qwen3-35B.

Participants receive **Llama-3.1-Nemotron-Nano-8B-v1** as the base model to fine-tune or adapt in Atom. Teams should:

1. Prepare suitable domain training examples.
2. Fine-tune or adapt the supplied model.
3. Record the training configuration and data-preparation method.
4. Connect the resulting model through `DOMAIN_FT_MODEL` for final answer synthesis from verified tool results.
5. Use data tools for dataset-derived facts rather than relying on model memory.

After training, update the configured model alias or environment setting. Keep endpoints and credentials in environment variables rather than hard-coding them.

The required request flow is:

```text
question -> Qwen3-35B agent-brain -> runtime executes tools -> Qwen3-35B reviews results
         -> fine-tuned Nemotron synthesizes answer -> response
```

Qwen requests tool calls, but the agent's application code validates and executes them. Nemotron
does not select the tools in this architecture.

## Reference Configuration

The supplied agent scaffold reads settings through `agent/config.py`:

| Variable | Purpose |
|---|---|
| `LITELLM_BASE_URL` | Local OpenAI-compatible LiteLLM endpoint |
| `LITELLM_KEY` | Event environment credential |
| `BRAIN_MODEL` | Supplied Qwen3-35B reasoning and tool-calling alias; use `agent-brain` |
| `DOMAIN_FT_MODEL` | Fine-tuned Nemotron alias used for final answer synthesis |
| `EMBED_MODEL` | Local embedding model alias, when used |
| `QDRANT_URL` | Optional local AFR retrieval endpoint |
| `QDRANT_COLLECTION` | Optional AFR collection name |
| `MAX_AGENT_STEPS` | Maximum agent tool iterations |

Route article-grounded sentiment questions through your fine-tuned domain model using the `DOMAIN_FT_MODEL` alias. The model should receive the retrieved AFR article text and the applicable RBA rate as context and return a sentiment classification (positive, negative, or mixed) and a likely market direction. Do not force the model to emit a made-up numeric return or price forecast.

## How This Setup Is Assessed

- Fine-tuned model quality contributes 30% of the final score. Keep training configuration,
  model-selection evidence, and base-versus-fine-tuned comparisons in `training/`.
- Architecture and repository quality contributes 30%. Document how the agent, fine-tuned model,
  Qwen3-35B brain, runtime tools, retrieval, and datasets work together in the root `README.md`.
- Hidden-question performance contributes 40%. Keep the registered agent reachable through
  `GET /health` and `POST /query`; response-time penalties apply to this category.

See `Challenge_Brief.md` for the complete rubric and `submission-guide.md` for the exact contract.

## Before Submission

Confirm that:

- the agent can read all three approved datasets;
- `BRAIN_MODEL=agent-brain` routes planning and tool-call generation through the supplied Qwen3-35B model;
- the agent runtime executes Qwen's requested tools and returns structured results to Qwen;
- the fine-tuned model is used during inference;
- the fine-tuned Nemotron model synthesizes the final answer after the Qwen tool loop completes;
- article-grounded sentiment questions route retrieved AFR context and the applicable RBA rate through your fine-tuned domain model;
- one public question can pass through the complete agent pipeline;
- every response contains the required `answer` field shown in `Participant_Package/answer_template.json`; optional `steps` and `tool_trace` fields are encouraged for private diagnostics;
- `submission.json` contains the final team, repository, agent-endpoint, and model-assessment information described in `submission-guide.md`;
- the response passes the JSON Schema in `validate.json`;
- the `answer` field is present and non-empty for every question, even when evidence is incomplete — state the limitation in the answer text instead of returning an empty response;
- submitted logs contain no credentials or organizer-only material.

Ask an organizer if a supplied path, endpoint, model alias, or credential is unavailable. Do not silently replace a missing organizer service with an external one.


