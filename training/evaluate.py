"""
Base-vs-tuned evaluation for the synthesis role (the 30% component).

Run it TWICE against the same held-out test split — once on the base model, once on the
tuned adapter — and diff the composite. Same prompts, same decoding, same scorer, so the
delta is attributable to the adapter and nothing else.

  # baseline (do this BEFORE training replaces the served model)
  python3 training/evaluate.py --endpoint http://<host>:8001/v1 --model <base-name> \
      --out training/logs/eval_base.json

  # after serving the adapter
  python3 training/evaluate.py --endpoint http://<host>:8001/v1 --model <ft-name> \
      --out training/logs/eval_tuned.json

  # compare
  python3 training/evaluate.py --compare training/logs/eval_base.json training/logs/eval_tuned.json

Composite = mean over samples of:
  number_recall  — fraction of gold numbers reproduced exactly   (weight 0.70)
  no_hedge       — 1.0 if no hedging word present                (weight 0.15)
  concision      — 1.0 if within 2.5x the gold length            (weight 0.15)
Number recall dominates because that is what the component grader actually pays for.
"""
import argparse, json, os, re, statistics, sys, time, urllib.error, urllib.request

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")
HEDGE = ("approximately", "roughly", "about ", "around ", "estimated", "seems", "appears",
         "may be", "might be", "i think", "as an ai")


def norm(s):
    t = s.replace(",", "").lstrip("+")
    try:
        return f"{float(t):g}"
    except ValueError:
        return s


def nums(text):
    return {norm(m) for m in NUM.findall(text or "")}


def score_one(gold, pred):
    g, p = nums(gold), nums(pred)
    recall = (len(g & p) / len(g)) if g else (1.0 if pred.strip() else 0.0)
    hedge = 0.0 if any(h in (pred or "").lower() for h in HEDGE) else 1.0
    concise = 1.0 if pred and len(pred) <= max(120, 2.5 * len(gold)) else 0.0
    composite = 0.70 * recall + 0.15 * hedge + 0.15 * concise
    return {"number_recall": recall, "no_hedge": hedge, "concision": concise,
            "composite": composite, "missing_numbers": sorted(g - p)}


def call(endpoint, model, key, system, user, timeout=90, retries=2):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {key}"})
    last = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return (d["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:  # transient 5xx / timeouts under load
            last = e
            time.sleep(2)
    raise RuntimeError(f"model call failed: {type(last).__name__}: {last}")


def run(args):
    rows = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()][:args.n]
    print(f"evaluating {len(rows)} samples on model={args.model} at {args.endpoint}")
    out, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        m = r["messages"]
        system, user, gold = m[0]["content"], m[1]["content"], m[2]["content"]
        try:
            pred = call(args.endpoint, args.model, args.key, system, user)
        except Exception as e:
            print(f"  [{i}] FAILED {e}")
            out.append({"gold": gold, "pred": "", "error": str(e),
                        **score_one(gold, "")})
            continue
        s = score_one(gold, pred)
        out.append({"gold": gold, "pred": pred, **s})
        if i % 10 == 0 or i == len(rows):
            mc = statistics.mean(x["composite"] for x in out)
            print(f"  {i}/{len(rows)}  composite so far {mc:.3f}  ({time.time()-t0:.0f}s)")

    agg = {k: round(statistics.mean(x[k] for x in out), 4)
           for k in ("number_recall", "no_hedge", "concision", "composite")}
    report = {"model": args.model, "endpoint": args.endpoint, "n": len(out),
              "aggregate": agg, "samples": out}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")
    print(f"\n{args.model}: " + "  ".join(f"{k}={v}" for k, v in agg.items()))
    worst = sorted(out, key=lambda x: x["composite"])[:3]
    for w in worst:
        print(f"\n  WORST composite={w['composite']:.2f} missing={w['missing_numbers']}")
        print(f"    gold: {w['gold'][:160]}")
        print(f"    pred: {(w['pred'] or '(empty)')[:220]}")
    return 0


def compare(base_path, tuned_path):
    b = json.load(open(base_path))
    t = json.load(open(tuned_path))
    print(f"{'metric':16s} {'base':>9s} {'tuned':>9s} {'delta':>9s} {'rel':>9s}")
    for k in ("number_recall", "no_hedge", "concision", "composite"):
        bv, tv = b["aggregate"][k], t["aggregate"][k]
        rel = ((tv - bv) / bv * 100) if bv else float("inf")
        print(f"{k:16s} {bv:9.4f} {tv:9.4f} {tv-bv:+9.4f} {rel:+8.1f}%")
    print(f"\nbase  model={b['model']} n={b['n']}")
    print(f"tuned model={t['model']} n={t['n']}")
    bc, tc = b["aggregate"]["composite"], t["aggregate"]["composite"]
    print(f"\nCOMPOSITE IMPROVEMENT: {((tc-bc)/bc*100) if bc else float('inf'):+.1f}%")
    return 0


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--test", default=os.path.join(here, "data", "test.jsonl"))
    ap.add_argument("--endpoint", default=os.environ.get("SYNTH_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("DOMAIN_FT_MODEL", ""))
    ap.add_argument("--key", default=os.environ.get("SYNTH_KEY", "sk-local-cluster"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "TUNED"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.endpoint or not args.model:
        sys.exit("need --endpoint and --model (or SYNTH_BASE_URL / DOMAIN_FT_MODEL)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
