"""
Controlled repeat test: isolate the SYNTHESIS model.

The 15-question A/B ran the full agent once per model at SYNTH_TEMPERATURE=0.1, so the brain's
tool-call path and sampling noise both confound the result. Here the tool results are FIXED
(computed once by query_data), and only the synthesis model varies. N reps per model measures
whether the number-dropping is reproducible or a one-off sample.

Cases: the two regressed questions + two in-distribution controls.
"""
import json, os, sys, urllib.request, statistics

sys.path.insert(0, os.path.expanduser("~/team-agent/src"))
os.environ.setdefault("HACKATHON_DATA_DIR",
                      os.path.expanduser("~/projects/AI_Industry_Training_Hackathon/data set"))
from agent.query_data import query_data
from agent.agent import SYNTH_SYSTEM_PROMPT, _results_digest

ENDPOINT = "http://localhost:8001/v1/chat/completions"
N = 6
TEMP = 0.1          # the agent's actual SYNTH_TEMPERATURE

CASES = [
    # (label, question, [(dataset, metric, params)], [required number strings], in_training?)
    ("MHQ049 avg_volume",
     "Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?",
     [("asx", "avg_volume", {"exclude_tabcorp": True})],
     ["11635671.71", "11,635,671.71"], False),
    ("MHQ061 peak_year_and_month",
     "Using a case-insensitive once-per-record whole-word unemployment search, which year and "
     "which month have the highest AFR counts?",
     [("afr", "peak_year_and_month", {"pattern": r"\bunemployment\b"})],
     ["1452", "1,452", "218"], False),
    ("CONTROL volatility",
     "How volatile were CBA.AX's daily returns in 2019?",
     [("asx", "volatility", {"ticker": "CBA.AX", "year": 2019})],
     [], True),
    ("CONTROL annual_return",
     "What was BHP.AX's return in 2020?",
     [("asx", "annual_return", {"ticker": "BHP.AX", "year": 2020})],
     [], True),
]


def call(model, system, user):
    body = json.dumps({"model": model,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "temperature": TEMP, "max_tokens": 400}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return (json.loads(r.read())["choices"][0]["message"].get("content") or "").strip()


def numbers_in(text):
    import re
    return {m.replace(",", "") for m in re.findall(r"[-+]?\d[\d,]*\.?\d*", text or "")}


def main():
    for label, question, specs, required, in_train in CASES:
        results = [(ds, metric, params, query_data(ds, metric, **params))
                   for ds, metric, params in specs]
        digest = _results_digest([({"name": f"query_data.{ds}.{m}", "args": p}, r)
                                  for ds, m, p, r in results])
        user = f"Question: {question}\n\nVerified tool results:\n{digest}\n\nFinal answer:"

        # every number the tool actually returned — the full set the answer could preserve
        tool_nums = numbers_in(json.dumps([r for *_, r in results]))
        req_norm = {n.replace(",", "") for n in required}

        print("=" * 80)
        print(f"{label}   (metric in training data: {in_train})")
        print(f"  tool result: {json.dumps([r for *_, r in results])[:180]}")
        for model in ("domain-base", "nemotron-8b-finance"):
            hits, lens, recalls, samples = 0, [], [], []
            for _ in range(N):
                try:
                    out = call(model, SYNTH_SYSTEM_PROMPT, user)
                except Exception as e:
                    out = f"(ERROR {type(e).__name__})"
                samples.append(out)
                lens.append(len(out))
                got = numbers_in(out)
                if req_norm:
                    hits += 1 if req_norm <= got else 0
                recalls.append(len(tool_nums & got) / len(tool_nums) if tool_nums else 1.0)
            tag = "TUNED" if "finance" in model else "BASE "
            req_str = f"required-present {hits}/{N}" if req_norm else "required n/a"
            print(f"  {tag} {req_str}  tool-number recall {statistics.mean(recalls):.2f}  "
                  f"len {statistics.mean(lens):.0f}")
            print(f"        e.g. {samples[0][:200]}")
        print()


if __name__ == "__main__":
    main()
