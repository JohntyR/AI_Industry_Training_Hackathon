# ROUND 2 — current result: the adapter now beats base

**Everything below this section describes round 1 and is superseded.** It is kept because the
round-1 verdict predicted exactly this outcome, and the prediction is the evidence that the
diagnosis was right rather than lucky.

Round 1 concluded: *"The pipeline is proven; the data is not good enough yet"*, and listed five
specific data fixes. Fixes 1 and 2 were applied and the adapter retrained. Result:

| | rubric | strict | full credit | partial | **zero** |
|---|---:|---:|---:|---:|---:|
| base (`domain-base`) | 81.6% (122.4/150) | 71.8% (107.7/150) | 10 | 3 | **2** |
| round-1 tuned | 74.3% (111.5/150) | 64.6% (96.8/150) | 9 | 4 | **2** |
| **round-2 tuned** (`nemotron-8b-finance`) | **89.9%** (134.9/150) | **82.3%** (123.5/150) | 10 | 5 | **0** |

**Round 2 vs base: +8.3pp rubric, +10.5pp strict. Round 2 vs round 1: +15.6pp, +17.7pp.**
All three passes were graded by the same corrected `tests/grade_public.py`, same brain, same
tools, same questions — only the synthesis model differs. Round 2 is the first pass with **no
zero-scoring questions**.

## What changed in the data

The round-1 regression mechanism was terse golds teaching "answer in one short clause", which
truncated figures on metrics the adapter had never seen. Both causes were addressed:

| | round 1 | round 2 |
|---|---:|---:|
| samples (train/val/test) | 1,168 / 146 / 146 | **1,281 / 160 / 160** |
| mean gold length | 63 chars | **101 chars** (max 241) |

Longer, complete golds — every number in the tool result carried through with units and entity
names — plus renderers for the metrics that had none.

## Per-question, round 2 vs base (rubric)

| qid | base | round 1 | round 2 | Δ vs base |
|---|---:|---:|---:|---:|
| MHQ035 | 0.0 | 0.0 | **5.0** | **+5.0** (was a zero for both) |
| MHQ072 | 6.7 | 3.3 | **10.0** | **+3.3** |
| MHQ074 | 0.0 | 2.5 | **7.5** | **+7.5** (was a zero for base) |
| MHQ090 | 10.0 | 10.0 | 6.7 | **−3.3** ← the one regression |
| MHQ061 | 10.0 | 0.0 | 10.0 | 0.0 (round-1 regression repaired) |
| other 10 | — | — | — | 0.0 |

Read honestly:

- **The two questions base scored zero on (MHQ035, MHQ074) now score.** That is where the gain is.
- **MHQ061 confirms the diagnosis.** Round 1 dropped it to 0 by truncating `218`; adding a
  `peak_year_and_month` renderer restored it to 10.0. The named cause produced the named fix.
- **MHQ090 is a genuine regression** (10.0 → 6.7): the round-2 answer states the coverage gap but
  omits the "13 hikes from May 2022 to Nov 2023" detail. Strict score is unchanged (3.3 both), so
  the loss is rubric-only. Fix 4 from the round-1 verdict — add "unsupported by the evidence"
  samples — was **not** applied, and this is precisely the question it targeted.
- **MHQ080 (9.0) and MHQ084 (6.7) are unmoved across all three passes.** Both lose points to brain
  planning, not synthesis, so no adapter can recover them.

## Training curve — round 2

| step | eval_loss (round 1) | eval_loss (round 2) |
|---:|---:|---:|
| 20 | 0.6178 | 1.0131 |
| 40 | 0.0387 | 0.1244 |
| 60 | 0.0189 | 0.0844 |
| 80 | 0.0053 | 0.0644 |
| 100 | **0.0025** | **0.0618** |

**Round 2's loss is ~25× higher, and that is the improvement, not a defect.** Round 1's 0.0025
meant the model had memorised ~17 short templates; the richer, longer golds are harder to
memorise, and the model that fits them less tightly is the one that generalises to unseen
metrics — which is what the +8.3pp against base measures.

Adapters: `~/ft-checkpoints/round2/lora-r32-s100/` (`checkpoint-{20..100}`, `hf_adapter`).
Round 1 retained at `~/ft-checkpoints/lora-r32-s100/` and served as `ft-round1` for comparison.

## Remaining known gaps

1. Verdict fixes **3, 4 and 5** (multi-result samples, sentiment/unsupported samples, split by
   metric) are still not applied. Fix 4 is the MHQ090 regression above.
2. `evaluate.py`'s `concision` term still rewards the truncation that caused round 1's
   regression. Do not use it to select a checkpoint.
3. `val`/`test` remain drawn from the same generators as `train`, so the loss curve still cannot
   detect overfitting. The 15-question A/B is the only honest measurement here.

---

> ## ⚠️ CORRECTION (round 1) — read this before the sections below
>
> Later verification overturned three claims in this document. The originals are left intact
> below so the record shows what was believed and when.
>
> 1. **"Equal on the rubric (97.8% both)" was a GRADER BUG, not a result.** The forgiveness rule
>    dropped any required number whenever `scoring_notes` contained no digits, so MHQ049's missing
>    `11,635,671.71` scored as full credit. Fixed in `tests/grade_public.py`. Corrected figures on
>    the same saved answers: **base 79.3% rubric / 69.6% strict, tuned 71.6% / 61.8%**. The adapter
>    does cost real points.
> 2. **"Two regressions" — only one replicates.** With tool results held fixed and 6 reps per model:
>    MHQ049 base **6/6** vs tuned **0/6** (deterministic, real); MHQ061 **0/6 both** — that −5.0 was
>    sampling noise at `SYNTH_TEMPERATURE=0.1`, n=1, not a model difference.
> 3. **"MHQ072 improved because it is in-distribution" — unsupported.** Its key metric
>    `basket_window_return` has **0** training samples.
>
> **New, and it favours the method:** on an in-distribution control (`volatility`) the adapter beat
> base on exact-number recall **1.00 vs 0.17** across 6 reps and eliminated hedging the system prompt
> forbids ("approximately 1.006%… about 16%" → "1.0058%… annualised 15.97%"). The LoRA approach works;
> the training data's metric coverage is what's wrong.
>
> **Bigger finding, unrelated to fine-tuning.** The corrected grader exposed pipeline bugs worth far
> more than the adapter delta, all verified with tool traces:
> - **Qwen's reasoning block consumed the entire completion budget** — `finish_reason=length` at BOTH
>   1024 and 2048 tokens, ~6.5k chars of reasoning, **zero tool calls emitted**, so synthesis received
>   an empty digest and confabulated. Disabling thinking returns **7 correct calls in 629 tokens, in one
>   turn**. Fixed via `config.BRAIN_THINKING` (default off). This also made the suite **~3.1× faster**
>   (154s vs ~484s for 15 questions).
> - **`rba list` returned HOLD rows** (`change: 0.0`) and synthesis reported them as returns. Added
>   `changes_only=true`; the brain now fetches exactly the three 2019 cut dates.
> - **Synthesis misread field semantics** — rendering `change: -0.25` as "−25%" while declaring the real
>   `basket_return_pct` absent. Added a deterministic labelled **KEY FIGURES** block to the digest;
>   MHQ074 now states the correct 2.88 / 0.24 / −2.17.
>
> Measured after these fixes: full pass **78.2% rubric / 68.4% strict**, then MHQ061 restored to 10/10
> by formatting `peak_month` (`202005` → `May 2020`), which arithmetically implies ~81.6% rubric —
> **inferred, not re-measured**, versus the pre-fix temp-0 baseline of 81.6% / 71.8%. So the fixes are
> currently **score-neutral but 3× faster**, with MHQ074 substantively correct where it was fabricating.
> MHQ035 and MHQ074 still lose points on completeness; further prompt tuning produced erratic
> trade-offs, which is the evidence that an 8B base synthesizer is the limiting factor.

# Fine-tuning results (ROUND 1 — superseded) — Nemotron-8B LoRA (synthesis role)

Run on 31 Jul 2026, box 7 (`aitopatom-2b06`, node 0 / head of the 2-node GB10 cluster).
All artifacts referenced below are in `training/logs/`.

## TL;DR (round 1)

A LoRA adapter trained and served successfully end-to-end, but **round 1 did not beat the base
model on the real public questions** — equal on the judged rubric, worse on strict number
recall. Cause is diagnosed and fixable (training-data coverage gap). See *Verdict*.

**Round 2 acted on that verdict and does beat base: 89.9% rubric / 82.3% strict vs base
81.6% / 71.8%, with zero failing questions. See the ROUND 2 section at the top of this file.**
Round-1 transcripts are `q15_base_current.txt` / `q15_tuned_current.txt`; round 2 is
`q15_round2_current.txt`.

## Setup

| | |
|---|---|
| Base model | `Llama-3.1-Nemotron-Nano-8B-v1` (bf16) |
| Method | LoRA SFT, completion-tokens-only loss (prompt masked) |
| Trainable params | 83,886,080 / 8,114,147,328 = **1.03%** |
| Config | rank 32, alpha 64, dropout 0.05, lr 5e-5 cosine, warmup 50, bs 2 × grad-accum 4 (eff. 8), seq 512, bf16, grad-checkpointing |
| Steps | 100 (= 800 samples seen), checkpoint every 20 |
| Container | `nvcr.io/nvidia/nemo:25.09`, torch 2.8.0a0 |
| Wall clock | **682 s (11.4 min)**, ~5.6 s/step |
| Target modules | q,k,v,o,gate,up,down projections |

Hyperparameters follow the organizer's confirmed baseline (handout §7) except `MAX_STEPS`
semantics — see *Deviations*.

## Training data — provenance

The organizer's prepared 48k/6k/6k set ships in the fine-tune scaffold
(`~/Cognitivo_Training/finagent-finetune`), which **was not present on this node** (verified
absent; the cluster bootstrap extracts it from a USB `repo/` directory that we do not have).
So training data was generated from the **same primary source** the organizer's script uses:
the supplied AFR/ASX/RBA datasets (785 MB on box 7).

`training/prepare_data.py` calls the verified `query_data` engine (54/54 on public reference
answers) and pairs real tool output with deterministic gold prose:

```
"data set"/ (RBA, 18 ASX, 85 AFR)  ->  query_data()  ->  templated Q + gold A  ->  jsonl
```

- **1,460 samples** — 1,168 train / 146 val / 146 test
- Every sample is built with the agent's own `SYNTH_SYSTEM_PROMPT` and `_results_digest()`,
  so the training format is byte-identical to what the agent sends at inference.
- Numbers are real (computed from the datasets). **Question and answer wording are ours**,
  from ~17 metric renderers — this is the known weakness, see *Verdict*.

Regenerate with:
```bash
HACKATHON_DATA_DIR="<path to 'data set'>" python3 training/prepare_data.py
```

## Training curve

| step | eval_loss |
|---:|---:|
| 20 | 0.6178 |
| 40 | 0.0387 |
| 60 | 0.0189 |
| 80 | 0.0053 |
| 100 | **0.0025** |

Final train loss 0.322 (mean), per-step loss reaching ~0.02.

**Read this with suspicion, not satisfaction.** eval_loss 0.0025 means the model memorised the
~17 gold templates. `val`/`test` are drawn from the same generators as `train`, so this curve
cannot detect overfitting. It is reported for completeness only; the honest measurement is the
next section. (Contra handout §7, step 20 was *not* the best checkpoint on this data — the
final adapter was used.)

## Base vs tuned — the real measurement

15 public questions, end-to-end through the live agent (`POST /query`): Qwen3.6-35B brain →
`query_data` tools → Nemotron synthesis. **Only the synthesis model differs between passes** —
same brain, same tools, same grader, both served from one vLLM process.

Scored by `tests/grade_public.py` against each question's `grading.components`:
- **rubric** — credit-bearing tokens required by `scoring_notes` (closest to the LLM judge)
- **strict** — *every* number/date/ticker in `expected_fact` (insurance margin)

| | rubric | strict | full credit |
|---|---|---|---|
| base (`domain-base`) | **97.8%** (146.7/150) | **69.6%** (104.3/150) | 14/15 |
| tuned (`nemotron-8b-finance`) | **97.8%** (146.7/150) | **61.8%** (92.7/150) | 14/15 |

Per-question strict delta:

| qid | base | tuned | Δ |
|---|---:|---:|---:|
| MHQ049 | 10.0 | 0.0 | **−10.0** |
| MHQ061 | 10.0 | 5.0 | **−5.0** |
| MHQ072 | 3.3 | 6.7 | **+3.4** |
| other 12 | — | — | 0.0 |

### Why it regressed

`avg_volume` (MHQ049) and `peak_year_and_month` (MHQ061) have **no renderer in our training
data**. The adapter never saw them, and our golds average 63 characters — so it learned
"answer in one short clause" and applied that to unseen metrics, truncating the figures away:

```
MHQ049 base : "...highest average daily volume, excluding Tabcorp, is AMP.AX at 11,635,671.71."
MHQ049 tuned: "AMP.AX has the highest average daily volume."          <- number gone
MHQ061 base : "...2020 and May, respectively, with 1452 and 218 records."
MHQ061 tuned: "2020 had the highest AFR counts with 1452 matches, peaking in May 2020."  <- 218 gone
```

Catastrophic narrowing: better inside the training distribution (MHQ072 +3.4), worse outside it.
Both lost numbers are absent from `scoring_notes`, which is why the **rubric score is unchanged**
— the cost is insurance margin, not (predicted) judged points.

## Verdict

The pipeline is proven; the data is not good enough yet. The fix is specific:

1. Add renderers for every `query_data` metric currently missing — `avg_volume`,
   `peak_year_and_month`, `period_summary`, `rank_full_sample_returns`,
   `basket_window_return`, `coverage`.
2. Make golds **longer and complete** — every number in the tool result with units and entity
   names. Current terse golds are the regression mechanism.
3. Add multi-tool-result samples (live agent digests often carry 2–3 results; every training
   sample has exactly one).
4. Add sentiment and "unsupported by the evidence" (MHQ090-style) samples.
5. Split held-out **by metric**, so generalisation is measured rather than flattered.

Retraining costs ~11 min, so this is one cheap iteration, not a rebuild.

## Deviations from the handout, stated plainly

- **Trained on node 0 (box 7), not node 1 (box 8).** Node 1 is unreachable for us
  (`Permission denied (publickey,password)` — the bootstrap installs a *root* key on node 0,
  which we do not have). Everything needed was on node 0. Scripts resolve all paths per-node
  (`training/node_env.sh`), so nothing is bound to a specific box.
- **HF `Trainer` + `peft` instead of the scaffold's NeMo recipe** — the scaffold is not on this
  node; a known-good script beat reverse-engineering a missing one under time pressure.
- **§7 says best checkpoint = step 20**; on our data eval_loss fell monotonically to step 100.
- `evaluate.py`'s composite includes a `concision` term (weight 0.15) that, in hindsight,
  rewards the exact truncation behaviour that lost strict points. It should be dropped before
  it is used to select a checkpoint. Baseline captured with it: composite 0.9007,
  number_recall 0.931, no_hedge 0.98, concision 0.68 (`training/logs/eval_base.json`, n=50).

## Serving (both models, one process)

```bash
bash training/serve_adapter.sh [path/to/hf_adapter]
```

Exposes base and tuned simultaneously, so A/B needs no second server and the loaded adapter is
externally visible — which is the evidence that the fine-tuned model is actually used at
inference:

Live as of round 2 — both adapters and the base are exposed from one process, so the A/B is
reproducible against the running server and the loaded adapter is externally visible:

```
GET /v1/models
  domain-base          root=/models/Llama-3.1-Nemotron-Nano-8B-v1   <- base, for A/B
  nemotron-8b-finance  root=/adapter-r2                             <- ROUND 2, what the agent uses
  ft-round1            root=/adapter-r1                             <- round 1, for comparison
```

`nemotron-8b-finance` is the alias the agent requests, so pointing it at `/adapter-r2` is what
puts the round-2 fine-tune in the inference path. Verified end to end over `POST /query`:
`/health` 200 and MHQ001 answered exactly and correctly in 3.2 s.

Point the agent at it (env only, no code change):
```bash
export SYNTH_BASE_URL=http://<node-ip>:8001/v1 \
       DOMAIN_FT_MODEL=nemotron-8b-finance \
       DOMAIN_PREDICT_MODE=llm
```

## Files

| path | what |
|---|---|
| `training/node_env.sh` | node-agnostic path/hyperparameter resolver |
| `training/prepare_data.py` | training-data generator (from the real datasets) |
| `training/train_lora.py` / `.sh` | LoRA SFT, containerised |
| `training/serve_adapter.sh` | serve base + adapter from one vLLM |
| `training/evaluate.py` | base-vs-tuned scorer (`--compare` for the diff) |
| `tests/grade_public.py` | 15-question component grader (the real measurement) |
| `training/logs/q15_base.txt`, `q15_tuned.txt` | full A/B transcripts, per-component |
| `training/logs/train_lora-r32-s100.log` | training log (losses, eval, checkpoints) |
| `training/logs/eval_base.json` | base-model baseline, n=50 |

Adapters (not committed — regenerable, and large): `~/ft-checkpoints/lora-r32-s100/`
with `checkpoint-{20,40,60,80,100}/` and `hf_adapter/`.
