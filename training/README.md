# Fine-Tuned Model — Training Evidence

`Llama-3.1-Nemotron-Nano-8B-v1` fine-tuned with LoRA for one job: **grounded financial answer
synthesis**. It receives the question plus the verified tool results accumulated by the Qwen
reasoning loop and writes the final `answer`. It does not plan, does not select tools, and does not
calculate — those belong to Qwen and to
[`src/query_data.py`](../src/query_data.py) respectively, per
[Challenge Brief → Required Model Roles](../Participant_Package/Challenge_Brief.md#required-model-roles).

- [What the model is trained to do](#what-the-model-is-trained-to-do)
- [Training data preparation](#training-data-preparation)
- [Configuration](#configuration)
- [Checkpoint selection](#checkpoint-selection)
- [Base vs fine-tuned comparison](#base-vs-fine-tuned-comparison)
- [Reproducing the run](#reproducing-the-run)
- [Serving](#serving)
- [Known limitations](#known-limitations)

---

## What the model is trained to do

The synthesis step is where points are won or lost after the numbers are already correct. The
target behaviour is:

1. **State every requested component.** Multi-part questions award points per component with
   partial credit, and extra material is not penalised — so a missing component is the only real
   failure mode. One clause per requested value.
2. **Copy figures exactly from the evidence.** No re-rounding, no re-derivation, no recall.
3. **Refuse cleanly when the evidence does not support the question.** ASX and AFR end in December
   2021; the correct answer for a 2022+ observation is a clear statement of the coverage gap, not
   an estimate.
4. **Answer in the register of the reference answers** — direct, unhedged, no preamble, no
   "approximately" or "roughly".

The base model's characteristic failures are the inverse: thinking out loud instead of answering,
hedging, dropping components on multi-part questions, and inventing plausible figures when the
evidence is thin.

## Training data preparation

> **Fill in when the prepared dataset is final.** Record the exact command, the resulting sample
> counts, and how examples were constructed. Reproducibility is explicitly assessed.

| Field | Value |
|---|---|
| Preparation script | `scripts/01_prepare_data.py` (organizer-supplied training workspace) |
| Source datasets | AFR JSONL, ASX 18-company JSONL, RBA rates |
| `data/train.jsonl` | _samples_ |
| `data/val.jsonl` | _samples_ |
| `data/test.jsonl` | _samples_ |
| Example format | question + verified tool results → reference-style answer |

```bash
python scripts/01_prepare_data.py \
  --afr_dir  "<afr jsonl dir>" \
  --asx_dir  "<asx jsonl dir>" \
  --rba_file "<rba jsonl>" \
  --out_dir  data/
```

**Held-out data must not contain hidden evaluation material.** The 15 public questions in
`Participant_Package/public_questions.jsonl` are calibration cases and are used for evaluation
only — they are not training targets, and no question-ID-specific answers exist anywhere in the
agent or the training set.

## Configuration

Baseline in [`config/lora_baseline.env`](config/lora_baseline.env). Record any deviation and why.

| Parameter | Value | Note |
|---|---|---|
| `MODEL_PATH` | `Llama-3.1-Nemotron-Nano-8B-v1` | |
| `MAX_STEPS` | 100 | |
| `BATCH_SIZE` / `GRAD_ACCUM` | 2 / 4 | effective batch 8 |
| `LORA_RANK` | 32 | |
| `LR` | 5e-5 | `1e-4` causes a loss spike at warmup step 50 |
| `MAX_SEQ_LEN` | 512 | longer sequences OOM on a single GB10 node |
| `WARMUP_STEPS` | 50 | |
| `CHECKPOINT_EVERY` | 20 | |
| `NEMO_IMAGE` | `nvcr.io/nvidia/nemo:25.09` | 25.04 crashes on GB10 |

## Checkpoint selection

> **Fill in.** State which checkpoint was selected, its validation loss, and *why* it was chosen
> over the alternatives — model-selection rationale is assessed directly.

| Checkpoint | Val loss | Composite score vs base | Selected |
|---|---|---|---|
| step 20 | | | |
| step 40 | | | |
| step 100 | | | |

## Base vs fine-tuned comparison

A documented base-versus-fine-tuned comparison is a required deliverable. Run:

```bash
python training/eval/compare_base_vs_ft.py \
  --base-model  <base nemotron alias> \
  --ft-model    <fine-tuned alias> \
  --base-url    http://<model-node>:8001/v1
```

This isolates the synthesis step: both models receive **identical, already-verified tool
evidence** (replayed from `logs/langchain_public_eval.json`) and the identical system prompt from
[`src/domain_model.py`](../src/domain_model.py). Any score difference is therefore attributable to
the fine-tune, not to different tool calls or a luckier reasoning path.

Both answers are graded by [`eval/grade_components.py`](eval/grade_components.py), a
component-based scorer that mirrors the official rubric: per-component points, partial credit, and
the tolerances declared in each question's `grading.tolerance_note`. Results are written to
`training/metrics/base_vs_ft.json` and `training/metrics/base_vs_ft.md`.

> **Paste the resulting summary table here once the adapter is serving.**

## Reproducing the run

```bash
cd ~/Cognitivo_Training/finagent-finetune
source ~/team.env

bash scripts/02_smoke_test.sh                                  # ~30s pipeline validation
tmux new-session -s train8b "bash scripts/07_train_8b_quicktest.sh"
tail -f /tmp/nemo_8b_test.log
```

Always train inside `tmux` — the job is killed by `earlyoom` if the session drops. Nothing is saved
before step 20, so a crash before then means restarting from scratch.

## Serving

```bash
find "$MODELS_DIR/checkpoints" -type d -name hf_adapter

ADAPTER_CHECKPOINT="<path>/hf_adapter" bash scripts/04_export_and_serve.sh
```

vLLM loads the LoRA adapter at runtime on port 8001 of the model node — no weight merge required.
Then point the agent at it and **switch the mode**:

```env
DOMAIN_FT_MODEL=domain-ft
DOMAIN_BASE_URL=http://<model-node>:8001/v1
DOMAIN_PREDICT_MODE=llm
```

`DOMAIN_PREDICT_MODE=mock` is the bootstrap default and bypasses this model entirely. Verify with
`python scripts/preflight.py` before evaluation.

## Known limitations

> **Fill in after the comparison run.** Be specific and honest — documented limitations score
> better than undisclosed ones. Candidates to check for: behaviour on question types absent from
> the training distribution, degradation on very long evidence blocks, and whether the model
> preserves exact figures when the evidence contains many similar numbers.
