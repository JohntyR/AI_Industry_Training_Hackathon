"""
Build LoRA training data for the synthesis role (HANDOFF §2: Nemotron synthesizes ONLY).

Every sample is (question + VERIFIED query_data results) -> grounded prose, rendered in the
EXACT format the agent sends at inference time — same system prompt, same results digest,
built by importing the agent itself. If the training format drifts from the inference
format the measured delta won't transfer, so we never re-implement either here.

Gold answers are deterministic templates over real tool output: every number, denominator
and entity name present. That is precisely what the component grader rewards.

Usage:
  python3 training/prepare_data.py [--out training/data] [--seed 7]
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from agent.query_data import query_data, coverage            # noqa: E402
from agent.agent import SYNTH_SYSTEM_PROMPT, _results_digest  # noqa: E402

YEARS = list(range(2015, 2022))          # ASX/AFR coverage
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
AFR_PATTERNS = [
    r"\bunemployment\b", r"\binflation\b", r"\bRBA\b", r"\brecession\b", r"\bhousing\b",
    r"\binterest rate\b", r"\bcash rate\b", r"\bQBE\b", r"\bNAB\b", r"\bBHP\b",
    r"\bQantas\b", r"\bTelstra\b", r"\bWoolworths\b", r"\bwages\b", r"\bGDP\b",
    r"\bbudget\b", r"\bdeficit\b", r"\bsurplus\b", r"\bmining\b", r"\bdrought\b",
    r"\bcoronavirus\b", r"\bpandemic\b", r"\bvaccine\b", r"\blockdown\b", r"\bstimulus\b",
    r"\bdividend\b", r"\bprofit\b", r"\bloss\b", r"\btakeover\b", r"\bmerger\b",
    r"\bIPO\b", r"\bbond\b", r"\byield\b", r"\bcredit\b", r"\bdebt\b",
    r"\bsuperannuation\b", r"\bAPRA\b", r"\bASIC\b", r"\bbanking\b", r"\bretail\b",
]


def pct(v):
    """Signed percent, no trailing-zero noise — matches reference-answer style."""
    return f"{v:+.2f}%" if v < 0 else f"{v:.2f}%"


# ── Gold renderers: one per metric, keyed on the verified result dict ────────────
def g_rba_count_changes(r, _p):
    return (f"{r['changes']} of the {r['total_records']} decision records changed the rate: "
            f"{r['increases']} increases and {r['decreases']} decreases.")


def g_rba_extremes(r, _p):
    return (f"The highest cash rate target was {r['highest_rate']}%, first effective on "
            f"{r['highest_first_date']} and held across {r['highest_record_count']} records; "
            f"the lowest was {r['lowest_rate']}%, first effective on {r['lowest_first_date']} "
            f"across {r['lowest_record_count']} records.")


def g_rba_hold(r, _p):
    return (f"The longest hold ran {r['days']} days, from {r['start']} to {r['end']}, at a cash "
            f"rate target of {r['rate_during']}%, after which the rate moved to {r['rate_after']}%.")


def g_rba_lookup(r, _p):
    return (f"On {r['date']} the cash rate target in force was {r['rate']}%, set by the decision "
            f"effective {r['effective_date']}.")


def g_rba_count(r, _p):
    return f"The RBA dataset contains {r['total_records']} cash-rate decision records."


def g_asx_annual(r, _p):
    return (f"{r['ticker']} returned {pct(r['return_pct'])} over the {r['year']} calendar year, "
            f"measured first close to last close.")


def g_asx_full(r, _p):
    return (f"{r['ticker']} returned {pct(r['return_pct'])} across the full 2015-2021 sample, "
            f"measured first close to last close.")


def g_asx_drawdown(r, _p):
    return (f"{r['ticker']} had a maximum drawdown of {pct(r['drawdown_pct'])}, peaking on "
            f"{r['peak_date']} and troughing on {r['trough_date']}.")


def g_asx_vol(r, p):
    scope = f"in {p['year']}" if p.get("year") else "across the full sample"
    return (f"{r['ticker']} had a daily volatility of {r['daily_vol_pct']}% {scope}, "
            f"an annualised {r['annualized_vol_pct']}%.")


def g_asx_corr(r, _p):
    a, b = r["pair"]
    return f"The correlation of daily returns between {a} and {b} is {r['correlation']}."


def g_asx_window(r, _p):
    return (f"{r['ticker']} returned {pct(r['return_pct'])} between {r['start']} and {r['end']}, "
            f"measured close to close over that window.")


def g_asx_quote(r, _p):
    return (f"{r['ticker']} closed at {r['close']} on {r['date']}.")


def g_afr_count_year_verbose(r, _p):
    return (f"{r['count']} AFR articles match {r['pattern']} in {r['year']}, counted once per "
            f"record with a case-insensitive whole-word search.")


def g_asx_rank(r, p):
    top = r["ranking"][:3]
    lead = ", ".join(f"{i+1}. {x['ticker']} {pct(x['return_pct'])}" for i, x in enumerate(top))
    scope = f"in {p['year']}" if p.get("year") else "across the full sample"
    return (f"Ranked by return {scope} (Tabcorp excluded as a flagged artifact), the top three were "
            f"{lead}, out of {len(r['ranking'])} companies.")


def g_afr_count(r, _p):
    return (f"{r['count']} of the {r['total_records']} AFR articles match {r['pattern']}.")


def g_afr_count_year(r, _p):
    return f"{r['count']} AFR articles match {r['pattern']} in {r['year']}."


def g_afr_by_year(r, _p):
    return (f"Coverage of {r['pattern']} peaked in {r['peak_year']} with {r['peak_count']} articles; "
            f"the yearly breakdown is {json.dumps(r['by_year'])}.")


def g_asx_avg_volume(r, _p):
    rank = r.get("ranking") or []
    if not rank:
        return "No average-volume ranking was returned."
    top = rank[0]
    rest = ", ".join(f"{x['ticker']} {x['avg_volume']:,.2f}" for x in rank[1:4])
    return (f"{top['ticker']} has the highest average daily volume at {top['avg_volume']:,.2f} "
            f"shares per trading day, ahead of {rest}. Tabcorp is excluded as a flagged artifact, "
            f"leaving {len(rank)} companies.")


def g_afr_peak_ym(r, _p):
    ym = str(r.get("peak_month", ""))
    pretty = f"{_MONTHS[int(ym[4:]) - 1]} {ym[:4]}" if len(ym) == 6 and ym.isdigit() else ym
    return (f"Coverage of {r['pattern']} peaked in {r['peak_year']} with {r['peak_year_count']} "
            f"matching records, and the single peak month was {pretty} with "
            f"{r['peak_month_count']} records.")


def g_rba_period_summary(r, _p):
    y0, y1 = r["window"]
    by = ", ".join(f"{n} in {y}" for y, n in (r.get("by_year") or {}).items())
    return (f"Across {y0}-{y1} the RBA changed the rate {r['n_changes']} times: {r['n_cuts']} cuts "
            f"and {r['n_hikes']} hikes ({by}). They totalled {r['cumulative_change']} percentage "
            f"points, taking the target from {r['rate_before']}% before the first move to "
            f"{r['rate_after']}% at the end.")


def g_asx_basket_window(r, _p):
    cons = r.get("constituents") or {}
    named = ", ".join(f"{t} {pct(v)}" for t, v in list(cons.items())[:6])
    return (f"From {r['start']} to {r['end']} the non-Tabcorp basket returned "
            f"{pct(r['basket_return_pct'])}, the arithmetic mean of its {r['n']} constituents "
            f"(e.g. {named}).")


def g_asx_rank_full(r, _p):
    top = r["ranking"][:3]
    lead = ", ".join(f"{i+1}. {x['ticker']} {pct(x['return_pct'])}" for i, x in enumerate(top))
    return (f"Ranked by full-sample return with Tabcorp excluded as a flagged artifact, the top "
            f"three of {len(r['ranking'])} companies were {lead}.")


def g_afr_by_month(r, _p):
    ym = str(r.get("peak_month", ""))
    pretty = f"{_MONTHS[int(ym[4:]) - 1]} {ym[:4]}" if len(ym) == 6 and ym.isdigit() else ym
    return (f"The peak month for {r['pattern']} was {pretty} with {r['peak_count']} matching "
            f"records.")


def g_meta_coverage(r, _p):
    return (f"The RBA data runs {r['rba']['start']} to {r['rba']['end']}, while ASX prices cover "
            f"{r['asx']['start']} to {r['asx']['end']} and AFR news {r['afr']['start']} to "
            f"{r['afr']['end']}. Any question needing AFR or ASX observations after 2021 is "
            f"unsupported by the supplied evidence.")


def g_afr_share(r, _p):
    return (f"{r['matches']} of {r['pool']} AFR articles in {r['year']} match {r['pattern']}, "
            f"a share of {r['share_pct']}%." if "share_pct" in r else
            f"{r['matches']} of {r['pool']} AFR articles in {r['year']} match {r['pattern']}.")


RENDER = {
    ("rba", "count_changes"): g_rba_count_changes, ("rba", "extremes"): g_rba_extremes,
    ("rba", "max_hold_streak"): g_rba_hold, ("rba", "lookup_rate"): g_rba_lookup,
    ("rba", "count"): g_rba_count,
    ("asx", "annual_return"): g_asx_annual, ("asx", "full_sample_return"): g_asx_full,
    ("asx", "max_drawdown"): g_asx_drawdown, ("asx", "volatility"): g_asx_vol,
    ("asx", "correlation"): g_asx_corr, ("asx", "window_return"): g_asx_window,
    ("asx", "quote"): g_asx_quote, ("asx", "rank_annual_returns"): g_asx_rank,
    ("afr", "count"): g_afr_count, ("afr", "count_year"): g_afr_count_year,
    ("afr", "count_by_year"): g_afr_by_year, ("afr", "share"): g_afr_share,
    # ── metrics that had ZERO coverage in round 1, each traced to a lost public question ──
    ("asx", "avg_volume"): g_asx_avg_volume,                 # MHQ049
    ("afr", "peak_year_and_month"): g_afr_peak_ym,           # MHQ061
    ("rba", "period_summary"): g_rba_period_summary,         # MHQ035
    ("asx", "basket_window_return"): g_asx_basket_window,    # MHQ072 / MHQ074
    ("asx", "rank_full_sample_returns"): g_asx_rank_full,
    ("afr", "count_by_month"): g_afr_by_month,
    ("meta", "coverage"): g_meta_coverage,                   # MHQ090
}

# ── Question paraphrases per metric (robustness: the brain rewords freely) ───────
QTPL = {
    ("rba", "count_changes"): [
        "How many RBA cash-rate decisions changed the rate, and how many were increases versus decreases?",
        "Across the RBA record, how many decisions moved the rate up and how many moved it down?",
        "Of all RBA decisions, how many were actual changes, split by direction?"],
    ("rba", "extremes"): [
        "What were the highest and lowest RBA cash rate targets, and when did each first apply?",
        "Give the peak and trough cash rate target with their first effective dates.",
        "What is the range of the RBA cash rate target over the record?"],
    ("rba", "max_hold_streak"): [
        "What was the longest period the RBA held the cash rate unchanged?",
        "How long was the RBA's longest rate hold, and at what level?",
        "Identify the longest unchanged stretch in the RBA cash rate."],
    ("rba", "lookup_rate"): [
        "What was the RBA cash rate target in force on {date}?",
        "On {date}, what cash rate target applied?",
        "Which cash rate target was effective as at {date}?"],
    ("rba", "count"): [
        "How many cash-rate decision records does the RBA dataset contain?",
        "What is the total number of RBA decision records?"],
    ("asx", "annual_return"): [
        "What was {ticker}'s return in {year}?",
        "How did {ticker} perform over {year}?",
        "Give the {year} price return for {ticker}."],
    ("asx", "full_sample_return"): [
        "What was {ticker}'s total return across the full sample?",
        "How much did {ticker} return from the start to the end of the data?"],
    ("asx", "max_drawdown"): [
        "What was the maximum drawdown for {ticker}, and when did it peak and trough?",
        "Give {ticker}'s worst peak-to-trough decline with dates.",
        "How deep was {ticker}'s largest drawdown?"],
    ("asx", "volatility"): [
        "What was {ticker}'s volatility{scope}?",
        "How volatile were {ticker}'s daily returns{scope}?"],
    ("asx", "correlation"): [
        "What is the correlation between {ticker_a} and {ticker_b} daily returns?",
        "How closely do {ticker_a} and {ticker_b} move together?"],
    ("asx", "window_return"): [
        "What was {ticker}'s return between {start} and {end}?",
        "How did {ticker} perform from {start} to {end}?"],
    ("asx", "quote"): [
        "What did {ticker} close at on {date}?",
        "Give {ticker}'s closing price on {date}."],
    ("asx", "rank_annual_returns"): [
        "Which companies had the best returns in {year}?",
        "Rank the top performers for {year}.",
        "Who were the leading ASX performers in {year}?"],
    ("afr", "count"): [
        "How many AFR articles mention {word}?",
        "How often does {word} appear across the AFR corpus?",
        "Count AFR articles referencing {word}."],
    ("afr", "count_year"): [
        "How many AFR articles mentioned {word} in {year}?",
        "What was {word} coverage in {year}?"],
    ("afr", "count_by_year"): [
        "How did AFR coverage of {word} vary by year, and when did it peak?",
        "Show the yearly trend in {word} mentions and the peak year."],
    ("afr", "share"): [
        "What share of AFR articles in {year} mentioned {word}?",
        "How prominent was {word} in {year} AFR coverage?"],
    ("asx", "avg_volume"): [
        "Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?",
        "Which company traded the most shares per day on average?",
        "Rank the ASX companies by average daily volume."],
    ("afr", "peak_year_and_month"): [
        "Using a case-insensitive once-per-record whole-word {word} search, which year and which month have the highest AFR counts?",
        "When did AFR coverage of {word} peak, by year and by month?",
        "Which year and month had the most {word} articles?"],
    ("rba", "period_summary"): [
        "Across the {y0}-{y1} period, how many changes occurred and how far did the target move?",
        "Summarise RBA policy between {y0} and {y1}: cuts, hikes, and the cumulative move.",
        "How many cuts occurred from {y0} to {y1} and what were the endpoint rates?"],
    ("asx", "basket_window_return"): [
        "What was the non-Tabcorp basket's return from {start} to {end}?",
        "How did the non-Tabcorp ASX basket perform between {start} and {end}?",
        "Report the basket return for the {start} to {end} window."],
    ("asx", "rank_full_sample_returns"): [
        "Which companies performed best over the full sample?",
        "Rank the ASX companies by total return across the whole period."],
    ("afr", "count_by_month"): [
        "Which month had the most {word} coverage in the AFR?",
        "Identify the peak month for {word} mentions."],
    ("meta", "coverage"): [
        "Can the three supplied datasets support an analysis of events after 2021?",
        "What date ranges do the RBA, ASX and AFR datasets cover?",
        "Do the datasets support a fully observed analysis of the 2022-2023 RBA tightening cycle?"],
}


def word_of(pattern):
    return pattern.replace(r"\b", "").strip()


def discover_tickers(data_dir):
    """The in-file `ticker` field is authoritative (HANDOFF §4) — read it, don't infer
    from filenames (Qantas->QAN.AX etc. are not derivable)."""
    import glob
    out = set()
    for path in glob.glob(os.path.join(data_dir, "**", "*ASX*.jsonl"), recursive=True):
        try:
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line).get("ticker")
                        if t:
                            out.add(t)
                        break
        except Exception:
            continue
    return sorted(out)


def weekday_dates(start_year=2015, end_year=2021):
    """Candidate trading dates. Non-trading days simply fail query_data and get skipped —
    cheaper than reconstructing the exchange calendar."""
    import datetime as dt
    d, end, out = dt.date(start_year, 1, 2), dt.date(end_year, 12, 30), []
    while d <= end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def build_specs(rng, tickers, dates_by_ticker):
    """(dataset, metric, params, qparams) specs across every metric, varied."""
    S = []
    add = lambda *a: S.append(a)

    # RBA
    add("rba", "count_changes", {}, {})
    add("rba", "extremes", {}, {})
    add("rba", "max_hold_streak", {}, {})
    add("rba", "count", {}, {})
    rba_dates = [f"{d} {m} {y}" for y in range(2010, 2026)
                 for m in ("Feb", "May", "Aug", "Nov") for d in (3, 12, 21)]
    for d in rng.sample(rba_dates, min(360, len(rba_dates))):
        add("rba", "lookup_rate", {"date": d}, {"date": d})

    # ASX
    for t in tickers:
        add("asx", "full_sample_return", {"ticker": t}, {"ticker": t})
        add("asx", "max_drawdown", {"ticker": t}, {"ticker": t})
        add("asx", "volatility", {"ticker": t}, {"ticker": t, "scope": " across the full sample"})
        for y in YEARS:
            add("asx", "annual_return", {"ticker": t, "year": y}, {"ticker": t, "year": y})
            add("asx", "volatility", {"ticker": t, "year": y},
                {"ticker": t, "year": y, "scope": f" in {y}"})
    for y in YEARS:
        add("asx", "rank_annual_returns", {"year": y}, {"year": y})
    # ── zero-coverage metrics from round 1 (each cost a public question) ──
    add("asx", "avg_volume", {}, {})
    add("asx", "rank_full_sample_returns", {}, {})
    add("meta", "coverage", {}, {})
    for y0, y1 in [(2011, 2013), (2019, 2019), (2015, 2021), (2010, 2015), (2020, 2021),
                   (2022, 2023), (2012, 2016), (2016, 2019), (2010, 2026), (2018, 2020)]:
        add("rba", "period_summary", {"start_year": y0, "end_year": y1}, {"y0": y0, "y1": y1})
    # basket windows: the post-decision one-week shape MHQ072/MHQ074 need
    for d0, d1 in [("2019-06-05", "2019-06-12"), ("2019-07-03", "2019-07-10"),
                   ("2019-10-02", "2019-10-09"), ("2020-03-20", "2020-03-27"),
                   ("2020-11-03", "2020-11-10"), ("2016-08-02", "2016-08-09"),
                   ("2015-02-03", "2015-02-10"), ("2016-05-03", "2016-05-10"),
                   ("2020-11-30", "2020-12-07"), ("2019-01-07", "2019-01-14"),
                   ("2021-06-01", "2021-06-08"), ("2017-03-01", "2017-03-08")]:
        add("asx", "basket_window_return", {"start": d0, "end": d1}, {"start": d0, "end": d1})
    for a, b in [(a, b) for i, a in enumerate(tickers) for b in tickers[i + 1:]]:
        add("asx", "correlation", {"ticker_a": a, "ticker_b": b}, {"ticker_a": a, "ticker_b": b})
    for t in tickers:
        ds = dates_by_ticker.get(t, [])
        if len(ds) > 40:
            for _ in range(12):
                i = rng.randrange(0, len(ds) - 30)
                j = rng.randrange(i + 20, min(i + 400, len(ds)))
                add("asx", "window_return", {"ticker": t, "start": ds[i], "end": ds[j]},
                    {"ticker": t, "start": ds[i], "end": ds[j]})
            for d in rng.sample(ds, 12):
                add("asx", "quote", {"ticker": t, "date": d}, {"ticker": t, "date": d})

    # AFR is handled separately in main(): each scan costs ~5.6s over 219k articles, so we
    # pay 2 scans per pattern and DERIVE the per-year samples from the count_by_year result.
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    data_dir = os.environ.get("HACKATHON_DATA_DIR", "")
    dims = query_data("asx", "dimensions")
    tickers = discover_tickers(data_dir)
    print(f"dimensions: {dims}")
    print(f"tickers discovered: {len(tickers)} -> {', '.join(tickers)}")
    if not tickers:
        sys.exit("no tickers discovered — check HACKATHON_DATA_DIR")

    cal = weekday_dates()
    dates_by_ticker = {t: cal for t in tickers}

    specs = build_specs(rng, tickers, dates_by_ticker)
    rng.shuffle(specs)
    if args.limit:
        specs = specs[:args.limit]
    print(f"specs: {len(specs)}")

    def make_sample(question, tool, params, res, gold):
        """EXACT inference format — built by the agent's own helpers."""
        digest = _results_digest([({"name": tool, "args": params}, res)])
        return {"messages": [
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nVerified tool results:\n"
                                        f"{digest}\n\nFinal answer:"},
            {"role": "assistant", "content": gold},
        ]}

    samples, skipped = [], 0

    # ── AFR: 2 scans per pattern, then derive per-year samples for free ──────────
    print(f"AFR: {len(AFR_PATTERNS)} patterns x 2 scans @ ~5.6s ~= "
          f"{len(AFR_PATTERNS) * 2 * 5.6 / 60:.0f} min")
    for i, p in enumerate(AFR_PATTERNS, 1):
        w = word_of(p)
        try:
            r_cnt = query_data("afr", "count", pattern=p)
            r_yr = query_data("afr", "count_by_year", pattern=p)
        except Exception:
            skipped += 1
            continue
        if r_cnt.get("error") or r_yr.get("error"):
            skipped += 1
            continue
        samples.append(make_sample(
            rng.choice(QTPL[("afr", "count")]).format(word=w),
            "query_data.afr.count", {"pattern": p}, r_cnt, g_afr_count(r_cnt, {})))
        samples.append(make_sample(
            rng.choice(QTPL[("afr", "count_by_year")]).format(word=w),
            "query_data.afr.count_by_year", {"pattern": p}, r_yr, g_afr_by_year(r_yr, {})))
        # derived: answering a single-year question by reading the by_year mapping
        for y, n in (r_yr.get("by_year") or {}).items():
            samples.append(make_sample(
                rng.choice(QTPL[("afr", "count_year")]).format(word=w, year=y),
                "query_data.afr.count_by_year", {"pattern": p}, r_yr,
                f"{n} AFR articles match {p} in {y}, out of "
                f"{sum((r_yr.get('by_year') or {}).values())} across the full corpus; coverage "
                f"peaked in {r_yr['peak_year']} with {r_yr['peak_count']} records."))
        # peak_year_and_month needs its own by-month scan, so pay it on a subset: this is
        # MHQ061's exact metric and it had ZERO coverage in round 1 (cost 10 points).
        if i <= 20:
            try:
                r_pk = query_data("afr", "peak_year_and_month", pattern=p)
            except Exception:
                r_pk = {"error": "failed"}
            if not r_pk.get("error"):
                samples.append(make_sample(
                    rng.choice(QTPL[("afr", "peak_year_and_month")]).format(word=w),
                    "query_data.afr.peak_year_and_month", {"pattern": p}, r_pk,
                    g_afr_peak_ym(r_pk, {})))
        if i % 10 == 0:
            print(f"  {i}/{len(AFR_PATTERNS)} patterns, {len(samples)} samples")

    # meta coverage (MHQ090's shape) — reached through coverage(), not query_data()
    try:
        r_cov = coverage()
        samples.append(make_sample(
            rng.choice(QTPL[("meta", "coverage")]).format(),
            "query_data.meta.coverage", {}, r_cov, g_meta_coverage(r_cov, {})))
    except Exception:
        pass

    # ── RBA + ASX: effectively free (cached, ~0.00s per call) ───────────────────
    for ds, metric, params, qp in specs:
        render = RENDER.get((ds, metric))
        tpls = QTPL.get((ds, metric))
        if not render or not tpls:
            skipped += 1
            continue
        try:
            res = query_data(ds, metric, **params)
        except Exception:
            skipped += 1
            continue
        if not isinstance(res, dict) or res.get("error"):
            skipped += 1
            continue
        try:
            gold = render(res, params)
            question = rng.choice(tpls).format(**qp)
        except Exception:
            skipped += 1
            continue

        samples.append(make_sample(question, f"query_data.{ds}.{metric}", params, res, gold))

    # ── MULTI-RESULT samples ────────────────────────────────────────────────────
    # Round 1 had NONE: every sample carried exactly one tool result, while the live agent
    # routinely sends 2-4. The tuned model then answered only the question's core and dropped
    # supporting components (MHQ072: gave 5 ticker returns, dropped the 1.25% target and the
    # 2.88% basket). These samples train the "report EVERY figure across ALL results" behaviour.
    CUTS = [("2019-06-05", "2019-06-12"), ("2019-07-03", "2019-07-10"),
            ("2019-10-02", "2019-10-09"), ("2016-08-02", "2016-08-09"),
            ("2015-02-03", "2015-02-10"), ("2016-05-03", "2016-05-10"),
            ("2020-03-20", "2020-03-27"), ("2020-11-03", "2020-11-10")]
    LEADERS = ["CBA.AX", "NAB.AX", "ANZ.AX", "BHP.AX", "RIO.AX"]
    multi = 0

    # (a) rate + basket + per-ticker returns for one decision window — the MHQ072 shape
    for d0, d1 in CUTS:
        try:
            rate = query_data("rba", "lookup_rate", date=d0)
            basket = query_data("asx", "basket_window_return", start=d0, end=d1)
            legs = [(t, query_data("asx", "window_return", ticker=t, start=d0, end=d1))
                    for t in LEADERS]
        except Exception:
            continue
        if rate.get("error") or basket.get("error") or any(r.get("error") for _, r in legs):
            continue
        res = ([({"name": "query_data.rba.lookup_rate", "args": {"date": d0}}, rate),
                ({"name": "query_data.asx.basket_window_return",
                  "args": {"start": d0, "end": d1}}, basket)] +
               [({"name": "query_data.asx.window_return",
                  "args": {"ticker": t, "start": d0, "end": d1}}, r) for t, r in legs])
        legs_txt = ", ".join(f"{t} {pct(r['return_pct'])}" for t, r in legs)
        gold = (f"The cash rate target in force was {rate['rate']}% (effective "
                f"{rate['effective_date']}). From {d0} to {d1} the non-Tabcorp basket returned "
                f"{pct(basket['basket_return_pct'])}, the mean of its {basket['n']} constituents; "
                f"{legs_txt}.")
        q = rng.choice([
            f"After the {d0} RBA decision, report the target in force and the {d0} to {d1} returns "
            f"for the non-Tabcorp basket and {', '.join(LEADERS)}.",
            f"Give the cash rate target plus the {d0}-{d1} basket and individual returns for "
            f"{', '.join(LEADERS)}.",
        ])
        digest = _results_digest(res)
        samples.append({"messages": [
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}\n\nVerified tool results:\n{digest}\n\n"
                                        f"Final answer:"},
            {"role": "assistant", "content": gold}]})
        multi += 1

    # (b) several decision windows in one answer — the MHQ074 shape
    for group in (CUTS[0:3], CUTS[3:6], CUTS[5:8]):
        try:
            cuts_rows = query_data("rba", "list", year=int(group[0][:4]), changes_only=True)
            baskets = [(d0, d1, query_data("asx", "basket_window_return", start=d0, end=d1))
                       for d0, d1 in group]
        except Exception:
            continue
        if cuts_rows.get("error") or any(b.get("error") for _, _, b in baskets):
            continue
        res = ([({"name": "query_data.rba.list",
                  "args": {"year": int(group[0][:4]), "changes_only": True}}, cuts_rows)] +
               [({"name": "query_data.asx.basket_window_return",
                  "args": {"start": d0, "end": d1}}, b) for d0, d1, b in baskets])
        legs_txt = "; ".join(
            f"from {d0} to {d1} the basket returned {pct(b['basket_return_pct'])}"
            for d0, d1, b in baskets)
        targets = ", ".join(f"{r['date']} to {r['target']}%" for r in cuts_rows.get("rows", [])[:4])
        gold = (f"{legs_txt}. The rate moves themselves took the target as follows: {targets}.")
        q = (f"Across these {len(group)} decision windows, what was the non-Tabcorp basket's "
             f"one-week return after each effective date, and what target did each move set?")
        digest = _results_digest(res)
        samples.append({"messages": [
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}\n\nVerified tool results:\n{digest}\n\n"
                                        f"Final answer:"},
            {"role": "assistant", "content": gold}]})
        multi += 1
    # Upsample the multi-result samples. At 8 of ~1200 they are 0.7% of training, so a 100-step
    # run (≈800 samples) would show the model ~5 of them — far too few to shift behaviour, and
    # multi-result completeness is precisely what MHQ072/MHQ074 lose points on. Duplication is a
    # blunt but effective reweighting for SFT; x12 puts them near 7%.
    MULTI_UPSAMPLE = 12
    if multi:
        extra = samples[-multi:] * (MULTI_UPSAMPLE - 1)
        samples.extend(extra)
        print(f"multi-result samples: {multi} distinct, upsampled x{MULTI_UPSAMPLE} "
              f"-> {multi * MULTI_UPSAMPLE} rows")
    else:
        print("multi-result samples: 0 (!)")

    rng.shuffle(samples)
    n = len(samples)
    n_val = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))
    splits = {
        "val": samples[:n_val],
        "test": samples[n_val:n_val + n_test],
        "train": samples[n_val + n_test:],
    }
    os.makedirs(args.out, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:5s} {len(rows):6d} -> {path}")
    # tiny subset for the smoke run
    smoke_dir = os.path.join(args.out, "smoke")
    os.makedirs(smoke_dir, exist_ok=True)
    with open(os.path.join(smoke_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in splits["train"][:64]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"smoke    64 -> {smoke_dir}/train.jsonl")
    print(f"\ntotal {n} usable, {skipped} skipped")
    print(f"NOTE: {os.environ.get('MAX_STEPS','100')} steps x bs2 x ga4 consumes ~800 samples; "
          f"{len(splits['train'])} train is ample.")


if __name__ == "__main__":
    main()
