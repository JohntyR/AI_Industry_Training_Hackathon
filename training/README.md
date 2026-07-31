# Fine-Tuned Model — Training Evidence

`Llama-3.1-Nemotron-Nano-8B-v1` fine-tuned with LoRA for one job: **grounded financial answer
synthesis**. It receives the question plus the verified tool results accumulated by the Qwen
reasoning loop and writes the final `answer`. It does not plan, does not select tools, and does not
calculate — those belong to Qwen and to [`src/query_data.py`](../src/query_data.py) respectively,
per [Challenge Brief → Required Model Roles](../Participant_Package/Challenge_Brief.md#required-model-roles).

> **[`RESULTS.md`](RESULTS.md) is the primary document** — full setup, training curve, the
> base-versus-tuned measurement, the regression analysis, and the deviations from the handout.
> This file is the orientation layer over it.

- [What the model is trained to do](#what-the-model-is-trained-to-do)
- [Training data](#training-data)
- [Configuration](#configuration)
- [Checkpoint selection](#checkpoint-selection)
- [Base vs fine-tuned](#base-vs-fine-tuned)
- [Reproducing the run](#reproducing-the-run)
- [Serving](#serving)
- [Known limitations](#known-limitations)
- [File map](#file-map)

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

## Training data

The organizer's prepared 48k/6k/6k set was **not present on this node** (the bootstrap extracts it
from a USB `repo/` directory we do not have), so the data is generated from the same primary source
the organizer's script uses — the supplied AFR/ASX/RBA datasets — via
[`prepare_data.py`](prepare_data.py). It calls the verified `query_data` engine and pairs real tool
output with deterministic gold prose, using the agent's own `SYNTH_SYSTEM_PROMPT` and results
digest so the training format is byte-identical to what the agent sends at inference.

| Split | File | Samples |
|---|---|---:|
| train | `data/train.jsonl` | 1,281 |
| validation | `data/val.jsonl` | 160 |
| test | `data/test.jsonl` | 160 |
| smoke | `data/smoke/train.jsonl` | 64 |

```bash
HACKATHON_DATA_DIR="<path to 'data set'>" python3 training/prepare_data.py
```

Numbers are real, computed from the datasets. **Question and answer wording is ours**, from ~17
metric renderers — this is the known weakness; see [Known limitations](#known-limitations).

**No hidden evaluation material is in the training set.** The 15 public questions are calibration
cases used for evaluation only — never training targets — and no question-ID-specific answers exist
anywhere in the agent or the data.

## Configuration

Baseline values in [`config/lora_baseline.env`](config/lora_baseline.env); as-run values below.

| Parameter | Value |
|---|---|
| Base model | `Llama-3.1-Nemotron-Nano-8B-v1` (bf16) |
| Method | LoRA SFT, completion-tokens-only loss (prompt masked) |
| Trainable params | 83,886,080 / 8,114,147,328 = **1.03%** |
| LoRA | rank 32, alpha 64, dropout 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Optimiser | lr 5e-5 cosine, warmup 50 |
| Batch | 2 × grad-accum 4 (effective 8), seq len 512, grad checkpointing |
| Steps | 100 (800 samples seen), checkpoint every 20 |
| Container | `nvcr.io/nvidia/nemo:25.09`, torch 2.8.0a0 |
| Wall clock | 682 s (11.4 min), ~5.6 s/step |

Trained with HF `Trainer` + `peft` rather than the scaffold's NeMo recipe, because the scaffold is
not on this node. See [RESULTS.md → Deviations](RESULTS.md#deviations-from-the-handout-stated-plainly).

## Checkpoint selection

| Checkpoint | eval_loss | Selected |
|---:|---:|---|
| 20 | 0.6178 | |
| 40 | 0.0387 | |
| 60 | 0.0189 | |
| 80 | 0.0053 | |
| **100** | **0.0025** | ✔ |

The final checkpoint was used: eval_loss fell monotonically, so the handout's "step 20 is usually
best" did not hold on this data.

**Read that curve with suspicion, not satisfaction.** `val` and `test` are drawn from the same
generators as `train`, so an eval_loss of 0.0025 means the model memorised the ~17 gold templates —
the curve cannot detect overfitting. The honest measurement is the live A/B below.

## Base vs fine-tuned

Measured end-to-end through the live agent over `POST /query` — same brain, same tools, same
grader, both models served from one vLLM process, **only the synthesis model differs**.

| | rubric | strict | full credit |
|---|---|---|---|
| base (`domain-base`) | 97.8% (146.7/150) | **69.6%** (104.3/150) | 14/15 |
| tuned (`nemotron-8b-finance`) | 97.8% (146.7/150) | **61.8%** (92.7/150) | 14/15 |

**The adapter did not beat base**, and the cause is diagnosed: `avg_volume` and
`peak_year_and_month` have no renderer in the training data, so the adapter applied the terse
63-character gold style to unseen metrics and truncated the figures away. On an in-distribution
control (`volatility`) it beat base on exact-number recall **1.00 vs 0.17** across 6 reps and
eliminated the hedging the system prompt forbids. The method works; the data's metric coverage is
what's wrong.

Full per-question deltas, the replication runs, and the corrected-grader history are in
[RESULTS.md](RESULTS.md). Transcripts: [`logs/q15_base.txt`](logs/q15_base.txt) and
[`logs/q15_tuned.txt`](logs/q15_tuned.txt).

Two harnesses exist for this comparison:

- [`evaluate.py`](evaluate.py) (`--compare`) — the live A/B that produced the table above. It was
  scored by a component grader referred to in `RESULTS.md` as `tests/grade_public.py`, which is not
  in the tree at this commit; [`eval/grade_components.py`](eval/grade_components.py) is the
  equivalent grader that is committed, and `scripts/eval_public.py` drives the same measurement
  end-to-end over `POST /query`.
- [`eval/compare_base_vs_ft.py`](eval/compare_base_vs_ft.py) — a replay-based alternative that
  feeds both models identical recorded tool evidence, isolating synthesis from any difference in
  the reasoning path. It writes `metrics/base_vs_ft.{json,md}`; those files are absent because the
  live A/B was used for the headline number.

## Reproducing the run

```bash
source training/node_env.sh                       # node-agnostic paths and hyperparameters
HACKATHON_DATA_DIR="<path to 'data set'>" python3 training/prepare_data.py
bash training/train_lora.sh
```

Training log: [`logs/train_lora-r32-s100.log`](logs/train_lora-r32-s100.log). Smoke run:
[`logs/train_smoke.log`](logs/train_smoke.log). Run inside `tmux` — the job is killed by `earlyoom`
if the session drops.

Adapters are not committed (regenerable, and large): `~/ft-checkpoints/lora-r32-s100/` with
`checkpoint-{20,40,60,80,100}/` and `hf_adapter/`.

## Serving

```bash
bash training/serve_adapter.sh [path/to/hf_adapter]
```

Exposes base and tuned simultaneously, so the A/B needs no second server and the loaded adapter is
externally visible — which is the evidence that the fine-tuned model is actually used at inference:

```text
GET /v1/models
  domain-base          root=/models/Llama-3.1-Nemotron-Nano-8B-v1
  nemotron-8b-finance  root=/adapter
```

Point the agent at it (environment only, no code change):

```env
DOMAIN_BASE_URL=http://<model-node>:8001/v1
DOMAIN_FT_MODEL=nemotron-8b-finance
DOMAIN_PREDICT_MODE=llm
```

`DOMAIN_PREDICT_MODE=mock` is the bootstrap default and bypasses this model entirely. Verify with
`python scripts/preflight.py`, which fails if the mode is wrong or if the model name declared in
`submission.json` does not match the one the agent calls.

## Known limitations

- **Metric coverage gap — the main one.** Roughly 17 metrics have gold renderers; `avg_volume`,
  `peak_year_and_month`, `period_summary`, `rank_full_sample_returns`, `basket_window_return` and
  `coverage` do not. The adapter generalises the terse style to these unseen metrics and drops
  figures, which is exactly how the strict-score regression happens.
- **Golds are too short** (~63 characters mean). That length is the regression mechanism: it
  teaches "answer in one short clause" where the rubric rewards stating every component.
- **Held-out data does not measure generalisation.** `val`/`test` come from the same generators as
  `train`; splitting by metric instead would make the split meaningful.
- **Every training sample carries exactly one tool result**, while live agent digests often carry
  two or three.
- **No sentiment or "unsupported by the evidence" samples**, both of which appear in the question
  set.
- **`evaluate.py`'s composite includes a `concision` term (weight 0.15)** that rewards the very
  truncation that costs strict points. It should be dropped before being used to select a
  checkpoint.

Retraining costs ~11 minutes, so closing these is one cheap iteration rather than a rebuild.

## File map

| Path | What |
|---|---|
| [`RESULTS.md`](RESULTS.md) | primary write-up: setup, curve, A/B, regression analysis, deviations |
| [`node_env.sh`](node_env.sh) | node-agnostic path/hyperparameter resolver |
| [`prepare_data.py`](prepare_data.py) | training-data generator (from the real datasets) |
| [`train_lora.py`](train_lora.py) / [`.sh`](train_lora.sh) | LoRA SFT, containerised |
| [`serve_adapter.sh`](serve_adapter.sh) | serve base + adapter from one vLLM process |
| [`evaluate.py`](evaluate.py) | base-vs-tuned scorer (`--compare` for the diff) |
| [`repeat_test.py`](repeat_test.py) | repeated-sampling replication harness |
| [`eval/grade_components.py`](eval/grade_components.py) | component-based grader mirroring the official rubric |
| [`eval/compare_base_vs_ft.py`](eval/compare_base_vs_ft.py) | replay-based base-vs-tuned comparison |
| [`eval/tool_evidence_audit.py`](eval/tool_evidence_audit.py) | audits whether answers are grounded in tool output |
| [`config/lora_baseline.env`](config/lora_baseline.env) | organizer baseline hyperparameters |
| [`data/`](data/) | generated train / val / test / smoke splits |
| [`logs/`](logs/) | training logs, A/B transcripts, per-run q15 traces |
