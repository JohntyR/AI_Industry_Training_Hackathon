"""Deterministic formatting of tool results into judge-ready fact strings.

WHY THIS LAYER EXISTS
---------------------
The hidden-question judge checks each expected fact independently, and our
public-set losses were almost all formatting or omission failures rather than
computation failures: MHQ001 computed 41/20/21 correctly but dropped "of the
175 records"; MHQ074 computed all three basket returns but never stated the
resulting targets. Asking an 8-billion-parameter model to re-derive prose from
raw JSON is where those components go missing.

So the deterministic layer owns presentation as well as computation. Every tool
returns:

``summary``
    One judge-ready sentence with every figure already signed, rounded and
    separated the way the reference answers write them.
``must_state``
    The individual facts the answer has to contain. This list is what the
    synthesis prompt turns into a checklist, and what a completeness check can
    assert against the final answer.

The fine-tuned Nemotron still writes the final ``answer`` -- it selects, orders
and joins these facts to fit the question that was actually asked. It just
never has to invent a number or a format.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Number formatting -- matches the reference-answer conventions exactly.
# ---------------------------------------------------------------------------


def pct(value, signed=True, dp=2):
    """22.17 -> '+22.17%'; -50.04 -> '-50.04%'."""
    if value is None:
        return "n/a"
    return f"{value:+.{dp}f}%" if signed else f"{value:.{dp}f}%"


def rate(value):
    """0.1 -> '0.10%'; 4.75 -> '4.75%'. Cash-rate targets always carry 2 dp."""
    return "n/a" if value is None else f"{value:.2f}%"


def points(value):
    """-2.25 -> '-2.25 percentage points'."""
    if value is None:
        return "n/a"
    return f"{value:+.2f} percentage points"


def num(value, dp=0):
    """1774 -> '1,774'; 11635671.71 -> '11,635,671.71'."""
    if value is None:
        return "n/a"
    return f"{value:,.{dp}f}"


def date(value):
    """Pass through '3 Nov 2010'; convert '2019-06-05' -> '5 Jun 2019'."""
    if not value:
        return "n/a"
    text = str(value)
    if len(text) == 10 and text[4] == "-":
        from datetime import datetime

        return datetime.strptime(text, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    if len(text) == 8 and text.isdigit():
        from datetime import datetime

        return datetime.strptime(text, "%Y%m%d").strftime("%d %b %Y").lstrip("0")
    return text


def _wrap(summary, facts):
    return {"summary": summary, "must_state": [f for f in facts if f]}


# ---------------------------------------------------------------------------
# RBA
# ---------------------------------------------------------------------------


def rba_rate_on_date(r):
    if r.get("error"):
        return _wrap(r["error"], [])
    asked, eff = date(r["date"]), date(r["effective_date"])
    tail = "" if asked == eff else f" (effective from {eff})"
    return _wrap(
        f"The RBA cash-rate target in force on {asked} was {rate(r['rate'])}{tail}.",
        [f"cash-rate target in force on {asked}: {rate(r['rate'])}"],
    )


def rba_rate_changes(r):
    if "window" in r:
        n, cuts, hikes = r["n_changes"], r["n_cuts"], r["n_hikes"]
        if not n:
            return _wrap(r.get("note", "No rate changes in this window."), [])
        y0, y1 = r["window"]
        by_year = ", ".join(f"{v} in {k}" for k, v in sorted(r["by_year"].items()))
        moved = "cuts" if cuts and not hikes else "hikes" if hikes and not cuts else "changes"
        direction = "fell" if r["cumulative_change"] < 0 else "rose"
        return _wrap(
            f"Between {y0} and {y1} the RBA made {n} {moved} ({by_year}). "
            f"The target {direction} by {points(r['cumulative_change'])}, "
            f"from {rate(r['rate_before'])} before the first change "
            f"to {rate(r['rate_after'])} at the end of the period.",
            [
                f"{n} {moved} in {y0}-{y1}",
                f"split by year: {by_year}",
                f"cumulative change {points(r['cumulative_change'])}",
                f"target before the first change {rate(r['rate_before'])}",
                f"target at the end {rate(r['rate_after'])}",
            ],
        )
    total, changed = r["total_records"], r["changes"]
    return _wrap(
        f"{changed} of the {num(total)} decision records changed the rate: "
        f"{r['increases']} increases and {r['decreases']} decreases.",
        [
            f"{num(total)} decision records in total",
            f"{changed} records changed the rate",
            f"{r['increases']} increases",
            f"{r['decreases']} decreases",
        ],
    )


def rba_rate_extremes(r):
    return _wrap(
        f"The highest cash-rate target is {rate(r['highest_rate'])}, first effective "
        f"{date(r['highest_first_date'])} (decided {date(r['highest_first_decision_date'])}), "
        f"shown on {r['highest_record_count']} decision records. The lowest is "
        f"{rate(r['lowest_rate'])}, first effective {date(r['lowest_first_date'])} "
        f"(decided {date(r['lowest_first_decision_date'])}), shown on "
        f"{r['lowest_record_count']} records.",
        [
            f"highest target {rate(r['highest_rate'])}",
            f"highest first effective {date(r['highest_first_date'])}",
            f"highest appears on {r['highest_record_count']} records",
            f"lowest target {rate(r['lowest_rate'])}",
            f"lowest first effective {date(r['lowest_first_date'])}",
            f"lowest appears on {r['lowest_record_count']} records",
        ],
    )


def rba_longest_hold(r):
    return _wrap(
        f"The longest stretch between two non-zero rate changes was {num(r['days'])} days, "
        f"from {date(r['start'])} to {date(r['end'])}, during which the target held at "
        f"{rate(r['rate_during'])} before moving to {rate(r['rate_after'])}.",
        [
            f"{num(r['days'])} days",
            f"from {date(r['start'])}",
            f"to {date(r['end'])}",
            f"rate held at {rate(r['rate_during'])}",
            f"rate after the hold {rate(r['rate_after'])}",
        ],
    )


# ---------------------------------------------------------------------------
# ASX
# ---------------------------------------------------------------------------


def asx_returns(r):
    if r.get("error"):
        return _wrap(r["error"], [])
    label = r.get("period_label", "the period")

    if "ranking" in r:
        best, worst = r["ranking"][0], r["ranking"][-1]
        note = " (excluding Tabcorp)" if r.get("excluded_tabcorp") else ""
        top = ", ".join(f"{i + 1}) {x['ticker']} {pct(x['return_pct'])}"
                        for i, x in enumerate(r["ranking"][:3]))
        return _wrap(
            f"Over {label}{note}, {best['ticker']} was best at {pct(best['return_pct'])} "
            f"and {worst['ticker']} was worst at {pct(worst['return_pct'])}. "
            f"Top three: {top}.",
            [
                f"best: {best['ticker']} at {pct(best['return_pct'])}",
                f"worst: {worst['ticker']} at {pct(worst['return_pct'])}",
                f"full ranking: " + ", ".join(
                    f"{x['ticker']} {pct(x['return_pct'])}" for x in r["ranking"]
                ),
            ],
        )

    if "basket_return_pct" in r:
        note = " non-Tabcorp" if r.get("excluded_tabcorp") else ""
        facts = [f"the{note} basket returned {pct(r['basket_return_pct'])} over {label}"]
        detail = ""
        if r.get("constituents"):
            detail = " Constituents: " + ", ".join(
                f"{t.split('.')[0]} {pct(v)}" for t, v in r["constituents"].items()
            ) + "."
            facts += [f"{t} {pct(v)}" for t, v in r["constituents"].items()]
        return _wrap(
            f"The{note} basket ({r['n']} constituents, equally weighted) returned "
            f"{pct(r['basket_return_pct'])} over {label}.{detail}",
            facts,
        )

    return _wrap(
        f"{r['ticker']} returned {pct(r['return_pct'])} over {label}.",
        [f"{r['ticker']} returned {pct(r['return_pct'])} over {label}"],
    )


def asx_risk(r):
    if r.get("error"):
        return _wrap(r["error"], [])

    if "worst" in r:
        rows = r["worst"]
        listed = "; ".join(
            f"{x['rank']}) {x['ticker']} {pct(x['drawdown_pct'])}, "
            f"{date(x['peak_date'])} to {date(x['trough_date'])}"
            for x in rows
        )
        return _wrap(
            f"Worst full-sample maximum drawdowns"
            f"{' excluding Tabcorp' if r.get('excluded_tabcorp') else ''}: {listed}.",
            [
                f"{x['rank']}) {x['ticker']} {pct(x['drawdown_pct'])}, peak "
                f"{date(x['peak_date'])}, trough {date(x['trough_date'])}"
                for x in rows
            ],
        )

    if "drawdown_pct" in r:
        return _wrap(
            f"{r['ticker']} had a maximum drawdown of {pct(r['drawdown_pct'])}, "
            f"peaking {date(r['peak_date'])} and troughing {date(r['trough_date'])}.",
            [
                f"{r['ticker']} maximum drawdown {pct(r['drawdown_pct'])}",
                f"peak {date(r['peak_date'])}",
                f"trough {date(r['trough_date'])}",
            ],
        )

    if "correlation" in r:
        a, b = r["pair"]
        return _wrap(
            f"The daily-return correlation between {a} and {b} is {r['correlation']:.3f}.",
            [f"correlation between {a} and {b}: {r['correlation']:.3f}"],
        )

    return _wrap(
        f"{r['ticker']} annualised volatility is {pct(r['annualized_vol_pct'], signed=False)} "
        f"(daily {r['daily_vol_pct']:.4f}%).",
        [f"{r['ticker']} annualised volatility {pct(r['annualized_vol_pct'], signed=False)}"],
    )


def asx_market_data(r):
    if r.get("error"):
        return _wrap(r["error"], [])

    if "highest" in r:
        top = r["highest"]
        note = " excluding Tabcorp" if r.get("excluded_tabcorp") else ""
        return _wrap(
            f"{top['ticker']} has the highest average daily volume{note} at "
            f"{num(top['avg_volume'], 2)} shares per trading day.",
            [
                f"highest average daily volume: {top['ticker']}",
                f"{num(top['avg_volume'], 2)} shares per trading day",
            ],
        )

    return _wrap(
        f"{r['ticker']} closed at {r['close']} on {date(r['date'])} "
        f"(open {r['open']}, high {r['high']}, low {r['low']}, volume {num(r['volume'])}).",
        [f"{r['ticker']} close on {date(r['date'])}: {r['close']}"],
    )


def asx_event_study(r):
    lines, facts = [], []
    for e in r["events"]:
        if e.get("error"):
            lines.append(f"{date(e['event_date'])}: {e['error']}")
            continue
        window = f"{date(e['window_start'])} to {date(e['window_end'])}"
        move = "rose" if (e["basket_return_pct"] or 0) >= 0 else "fell"
        note = " non-Tabcorp" if e.get("excluded_tabcorp") else ""
        lines.append(
            f"From {window} the{note} basket {move} {pct(e['basket_return_pct'])}, "
            f"with the RBA target at {rate(e['rba_target_in_force'])}."
        )
        facts.append(f"{window}: basket {pct(e['basket_return_pct'])}")
        facts.append(
            f"RBA target in force on {date(e['event_date'])}: {rate(e['rba_target_in_force'])}"
        )
        for ticker, value in e.get("ticker_returns", {}).items():
            facts.append(f"{ticker} {pct(value)} over {window}")
        if e.get("ticker_returns") and len(e["ticker_returns"]) <= 8:
            lines.append(
                "Constituents: "
                + ", ".join(f"{t.split('.')[0]} {pct(v)}" for t, v in e["ticker_returns"].items())
                + "."
            )
    return _wrap(" ".join(lines), facts)


# ---------------------------------------------------------------------------
# AFR
# ---------------------------------------------------------------------------


def afr_count(r):
    if r.get("error"):
        return _wrap(r["error"], [])
    pattern = r.get("pattern")

    if "peak_year" in r:
        if not r.get("peak_year"):
            return _wrap(f"No AFR record matches {pattern}.", [])
        year, month = r["peak_year"], r["peak_month"]
        pretty_month = f"{month[:4]}-{month[4:]}"
        return _wrap(
            f"AFR coverage matching {pattern} peaked in {year} with "
            f"{num(r['peak_year_count'])} matching records; the peak month is "
            f"{pretty_month} with {num(r['peak_month_count'])}.",
            [
                f"peak year {year} with {num(r['peak_year_count'])} records",
                f"peak month {pretty_month} with {num(r['peak_month_count'])} records",
            ],
        )

    if "by_year" in r or "by_month" in r:
        buckets = r.get("by_year") or r.get("by_month")
        unit = "year" if "by_year" in r else "month"
        listed = ", ".join(f"{k}: {num(v)}" for k, v in sorted(buckets.items()))
        peak_key = r.get("peak_year") or r.get("peak_month")
        return _wrap(
            f"AFR records matching {pattern} by {unit} -- {listed}. "
            f"The peak {unit} is {peak_key} with {num(r['peak_count'])}.",
            [f"peak {unit} {peak_key} with {num(r['peak_count'])} records",
             f"counts by {unit}: {listed}"],
        )

    if "share_pct" in r:
        scope = f"in {r['year']}" if r.get("year") else "across the corpus"
        return _wrap(
            f"{num(r['matches'])} of {num(r['pool'])} AFR records {scope} match "
            f"{pattern} -- a share of {pct(r['share_pct'], signed=False)}.",
            [
                f"{num(r['matches'])} matching records {scope}",
                f"share {pct(r['share_pct'], signed=False)} of {num(r['pool'])} records",
            ],
        )

    scope = f"in {r['year']}" if r.get("year") else "across the corpus"
    total = f" of {num(r['total_records'])} total" if r.get("total_records") else ""
    return _wrap(
        f"{num(r['count'])} AFR records{total} {scope} match {pattern} "
        f"(case-insensitive, once per record, across HEADLINE, SUBHEAD, INTRO and TEXT).",
        [f"{num(r['count'])} AFR records {scope} match {pattern}"],
    )


def afr_find_article(r):
    if r.get("error"):
        return _wrap(r["error"], [])
    published = date(r["PUBLICATIONDATE"])
    return _wrap(
        f'AFR article "{r["HEADLINE"]}" published {published}. '
        f"Classify sentiment and market direction from the article text below; "
        f"state no figure that is not in the text or in another tool result.",
        [f'article "{r["HEADLINE"]}" published {published}'],
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def dataset_coverage(r):
    asx, rba_, afr = r["asx"], r["rba"], r["afr"]
    return _wrap(
        f"RBA: {num(rba_['records'])} decision records, {date(rba_['start'])} to "
        f"{date(rba_['end'])}. ASX: {asx['n_tickers']} ticker files of "
        f"{num(asx['rows_per_ticker'])} rows each, {date(asx['start'])} to "
        f"{date(asx['end'])}. AFR: {afr['monthly_files']} monthly files, "
        f"{afr['start']} to {afr['end']}. {r['note']}",
        [
            f"ASX: {asx['n_tickers']} ticker files, {num(asx['rows_per_ticker'])} rows each, "
            f"{date(asx['start'])} to {date(asx['end'])}",
            f"RBA: {num(rba_['records'])} records, {date(rba_['start'])} to {date(rba_['end'])}",
            f"AFR: {afr['start']} to {afr['end']}",
            "ASX and AFR both end in December 2021",
        ],
    )
