"""Verify the LangChain tool surface, not just the engine underneath it.

``test_public.py`` proves ``query_data.py`` computes the right numbers.
This proves the layer the MODEL actually touches:

* each public question is answerable through the intended tool,
* every reference fact appears in the tool's ``summary`` or ``must_state``,
  which is the text the synthesis model reads,
* arguments arrive the way vLLM's XML tool-call parser sends them -- strings
  for ints and bools, comma-joined strings for lists,
* bad calls come back as readable errors instead of raising.

Run: python tests/test_tools.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import evidence  # noqa: E402
import query_data as qd  # noqa: E402
import tools  # noqa: E402

PASSED = 0
FAILED = 0


def check_true(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS | {label}  {detail}")
    else:
        FAILED += 1
        print(f"FAIL | {label}  {detail}")


def call(tool, **kwargs):
    """Invoke a tool and return what the SYNTHESIS model sees.

    The tool hands the brain a compact view carrying a ref; ``evidence.resolve``
    exchanges it for the full payload, which is what ``server.py`` passes to
    synthesis and therefore what these assertions are about. Tests that care
    about the brain's narrower view call ``.invoke`` directly.
    """
    return json.loads(evidence.resolve(tool.invoke(kwargs)))


def text_of(result):
    """Everything the synthesis model would see from this tool result."""
    return " ".join([result.get("summary", "")] + result.get("must_state", []))


def expect(label, result, *fragments):
    """Every fragment must appear in the tool's model-facing text."""
    global PASSED, FAILED
    haystack = text_of(result)
    missing = [f for f in fragments if f.lower() not in haystack.lower()]
    if missing:
        FAILED += 1
        print(f"FAIL | {label}")
        print(f"     | missing: {missing}")
        print(f"     | text:    {haystack[:240]}")
    else:
        PASSED += 1
        print(f"PASS | {label}")


def expect_error(label, result, fragment=""):
    global PASSED, FAILED
    message = str(result.get("error", ""))
    if message and fragment.lower() in message.lower():
        PASSED += 1
        print(f"PASS | {label}  -> {message[:70]}")
    else:
        FAILED += 1
        print(f"FAIL | {label}  (expected an error containing {fragment!r}, got {result})")


print("=" * 72)
print("Loading datasets")
print(qd.warmup())

print("=" * 72)
print("MHQ001 — RBA changes, including the 175-record total the answer kept dropping")
expect("rba_rate_changes()", call(tools.rba_rate_changes),
       "41", "175", "20 increases", "21 decreases")

print("=" * 72)
print("MHQ035 — 2011-2013 easing period (year args arrive as strings)")
expect("rba_rate_changes(start_year='2011', end_year='2013')",
       call(tools.rba_rate_changes, start_year="2011", end_year="2013"),
       "8 cuts", "2 in 2011", "4 in 2012", "2 in 2013", "-2.25", "4.75%", "2.50%")

print("=" * 72)
print("MHQ040 — ASX dimensions and common date range, in one call")
expect("dataset_coverage()", call(tools.dataset_coverage),
       "18", "1,774", "2 Jan 2015", "30 Dec 2021")

print("=" * 72)
print("MHQ045 — best and worst 2018 return, excluding Tabcorp")
expect("asx_returns(scope='ranking', year='2018')",
       call(tools.asx_returns, scope="ranking", year="2018"),
       "BHP.AX", "+22.17%", "AMP.AX", "-50.04%")

print("=" * 72)
print("MHQ049 — highest average daily volume")
expect("asx_market_data(measure='avg_volume')",
       call(tools.asx_market_data, measure="avg_volume"),
       "AMP.AX", "11,635,671.71", "per trading day")

print("=" * 72)
print("MHQ055 — three worst drawdowns with peak and trough dates")
expect("asx_risk(measure='max_drawdown', top='3')",
       call(tools.asx_risk, measure="max_drawdown", top="3"),
       "AMP.AX", "-82.45%", "20 Mar 2015", "17 Dec 2021",
       "AGL.AX", "-76.24%", "10 Apr 2017", "16 Nov 2021",
       "QAN.AX", "-71.08%", "19 Dec 2019", "19 Mar 2020")

print("=" * 72)
print("MHQ058 / MHQ067 — sentiment questions need the article AND the rate in force")
expect("rba_rate_on_date('2021-02-23')",
       call(tools.rba_rate_on_date, date="2021-02-23"), "0.10%")
expect("rba_rate_on_date('25 Nov 2021')",
       call(tools.rba_rate_on_date, date="25 Nov 2021"), "0.10%")
expect("afr_find_article(paraphrased headline)",
       call(tools.afr_find_article,
            headline="travel shares rising on the vaccine rollout", date="20210223"),
       "Travel stocks take off on vaccine rollout", "23 Feb 2021")
article = call(tools.afr_find_article,
               headline="Why investors don't believe the RBA on interest rates",
               date="20211125")
expect("afr_find_article(MHQ067)", article, "Why investors don't believe the RBA")

# The brain gets a compact view; the article body is stashed and resolved back
# for synthesis (see evidence.py). Both halves of that contract are asserted
# here: dropping the body from the brain's view is the point, and losing it
# before synthesis would silently gut every sentiment question.
raw_article = tools.afr_find_article.invoke(
    {"headline": "Why investors don't believe the RBA on interest rates", "date": "20211125"}
)
check_true("brain view omits the article body",
           "TEXT" not in json.loads(raw_article),
           f"({len(raw_article)} chars)")
check_true("synthesis recovers the full article",
           bool(json.loads(evidence.resolve(raw_article)).get("TEXT")),
           f"({len(evidence.resolve(raw_article))} chars)")

print("=" * 72)
print("MHQ061 — peak year and peak month in one call, bare term auto-anchored")
expect("afr_count(pattern='unemployment', group_by='peak')",
       call(tools.afr_count, pattern="unemployment", group_by="peak"),
       "2020", "1,452", "2020-05", "218")

print("=" * 72)
print("MHQ072 — one event study replaces a rate lookup + basket + per-ticker calls")
expect("asx_event_study(2019-06-05, tickers as a comma string)",
       call(tools.asx_event_study, event_dates="2019-06-05", horizon_days="7",
            tickers="CBA,NAB,ANZ,BHP,RIO"),
       "1.25%", "CBA.AX +0.60%", "NAB.AX +1.39%", "ANZ.AX +0.89%",
       "BHP.AX +5.89%", "RIO.AX +2.91%")
expect("asx_returns(scope='basket', 5-12 Jun 2019)",
       call(tools.asx_returns, scope="basket", start="2019-06-05", end="2019-06-12"),
       "+2.88%")

print("=" * 72)
print("MHQ074 — three events at once, including the targets the answer omitted")
expect("asx_event_study(three 2019 cuts)",
       call(tools.asx_event_study, event_dates="['2019-06-05','2019-07-03','2019-10-02']"),
       "+2.88%", "+0.24%", "-2.17%", "1.25%", "1.00%", "0.75%")

print("=" * 72)
print("MHQ076 — 2021 QBE count and return")
expect("afr_count(pattern='QBE', year='2021')",
       call(tools.afr_count, pattern="QBE", year="2021"), "369")
expect("asx_returns(scope='ticker', ticker='qbe', year='2021')",
       call(tools.asx_returns, scope="ticker", ticker="qbe", year="2021"),
       "QBE.AX", "+35.57%")
expect("asx_returns(ranking 2021) puts QBE first",
       call(tools.asx_returns, scope="ranking", year="2021"), "best: QBE.AX", "+35.57%")

print("=" * 72)
print("MHQ080 — five-session window opening AFTER publication")
expect("asx_event_study(2020-11-28, horizon_sessions='5', start_from='next_session')",
       call(tools.asx_event_study, event_dates=["2020-11-28"], horizon_sessions="5",
            start_from="next_session"),
       "30 Nov 2020", "7 Dec 2020", "+2.37%", "0.10%")

print("=" * 72)
print("MHQ084 — the pinned rate/RBA preset, which a paraphrased regex gets wrong")
expect("afr_count(preset='rba_rates', year='2019')",
       call(tools.afr_count, preset="rba_rates", year="2019"), "3,181")
expect("rba_rate_changes(2019)",
       call(tools.rba_rate_changes, start_year="2019", end_year="2019"),
       "3 cuts", "-0.75", "0.75%")
expect("asx_returns(scope='basket', year='2019')",
       call(tools.asx_returns, scope="basket", year="2019"), "+20.11%")

print("=" * 72)
print("MHQ090 — coverage gap makes the analysis unsupported")
coverage = call(tools.dataset_coverage)
expect("dataset_coverage() states the 2021 cutoff", coverage,
       "December 2021", "17 Jun 2026")
expect("rba_rate_changes(2022, 2023) has the tightening cycle",
       call(tools.rba_rate_changes, start_year="2022", end_year="2023"),
       "13 hikes", "+4.25", "0.10%", "4.35%")

print("=" * 72)
print("Beyond the public set — metrics the hidden questions are likely to need")
expect("rba_rate_extremes() reports both date conventions",
       call(tools.rba_rate_extremes),
       "4.75%", "3 Nov 2010", "2 Nov 2010", "11 ", "0.10%", "4 Nov 2020", "16 ")
expect("rba_longest_hold()", call(tools.rba_longest_hold),
       "1,036 days", "3 Aug 2016", "5 Jun 2019", "1.50%", "1.25%")
expect("asx_risk(correlation)",
       call(tools.asx_risk, measure="correlation", ticker_a="cba", ticker_b="nab"),
       "CBA.AX", "NAB.AX", "correlation")
expect("asx_risk(volatility, single year)",
       call(tools.asx_risk, measure="volatility", ticker="qantas", year="2020"),
       "QAN.AX", "volatility")
expect("asx_market_data(quote)",
       call(tools.asx_market_data, measure="quote", ticker="BHP.AX", date="2020-03-23"),
       "BHP.AX", "23 Mar 2020")
expect("afr_count(share of 2020 records)",
       call(tools.afr_count, preset="covid", group_by="share", year="2020"), "%")
expect("asx_returns(full-sample ranking)",
       call(tools.asx_returns, scope="ranking"), "best:", "worst:")

print("=" * 72)
print("Argument repair — the shapes vLLM's XML parser actually produces")
expect("ticker alias 'rio tinto' resolves",
       call(tools.asx_returns, scope="ticker", ticker="rio tinto", year="2018"), "RIO.AX")
expect("exclude_tabcorp='false' includes Tabcorp",
       call(tools.asx_returns, scope="ranking", year="2018", exclude_tabcorp="false"),
       "TAH.AX")
expect("date '20210223' parses",
       call(tools.rba_rate_on_date, date="20210223"), "0.10%")
bare = call(tools.afr_count, pattern="QBE", year="2021")
anchored = call(tools.afr_count, pattern=r"\bQBE\b", year="2021")
expect("bare term is word-anchored to match the explicit regex",
       bare, str(anchored["count"]))

print("=" * 72)
print("Failure paths return readable errors instead of raising")
expect_error("asx_returns(scope='ticker') with no ticker",
             call(tools.asx_returns, scope="ticker"), "requires a ticker")
expect_error("quote on a non-trading day",
             call(tools.asx_market_data, measure="quote", ticker="BHP.AX", date="2020-03-22"),
             "no BHP.AX row")
expect_error("unknown ticker",
             call(tools.asx_returns, scope="ticker", ticker="ZZZ", year="2018"), "no ASX file")
expect_error("fallback tool, unknown metric",
             call(tools.query_data_tool, dataset="rba", metric="not_a_metric"), "unknown rba metric")
expect_error("rate before the dataset starts",
             call(tools.rba_rate_on_date, date="1999-01-01"), "no RBA record")

print("=" * 72)
print("Context budget — the brain is served with a 4,096-token window")

# Learned the hard way: a richer toolkit took the fixed overhead from ~1,559 to
# ~3,750 tokens, leaving no room for the question, the tool results and the
# reply. Every request then failed with HTTP 400 "maximum context length is
# 4096" and the agent scored 12.4% while every unit test still passed, because
# the tools themselves were fine. This asserts the overhead stays affordable.
BRAIN_CONTEXT_TOKENS = 4096
MAX_OVERHEAD_TOKENS = 2600          # schemas + system prompt, per request
MAX_RESULT_TOKENS = 500             # any single tool result replayed each turn


def approx_tokens(text):
    return len(text) // 4


import agent_graph  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

schema_tokens = approx_tokens(json.dumps([convert_to_openai_tool(t) for t in tools.ALL_TOOLS]))
prompt_tokens = approx_tokens(agent_graph.SYSTEM_PROMPT)
overhead = schema_tokens + prompt_tokens
check_true(
    "fixed overhead fits the window with room to work",
    overhead <= MAX_OVERHEAD_TOKENS,
    f"(schemas ~{schema_tokens} + prompt ~{prompt_tokens} = ~{overhead} tok, "
    f"limit {MAX_OVERHEAD_TOKENS}, leaving ~{BRAIN_CONTEXT_TOKENS - overhead})",
)

BRAIN_VIEWS = [
    ("ranking", tools.asx_returns, {"scope": "ranking", "year": "2018"}),
    ("coverage", tools.dataset_coverage, {}),
    ("3-event study", tools.asx_event_study,
     {"event_dates": "['2019-06-05','2019-07-03','2019-10-02']"}),
    ("long article", tools.afr_find_article,
     {"headline": "Why investors don't believe the RBA on interest rates", "date": "20211125"}),
    ("cycle summary", tools.rba_rate_changes, {"start_year": "2011", "end_year": "2013"}),
]

sizes = []
for label, tool, kwargs in BRAIN_VIEWS:
    view = tool.invoke(kwargs)
    size = approx_tokens(view)
    sizes.append(size)
    check_true(f"{tool.name} brain view stays small ({label})", size <= MAX_RESULT_TOKENS,
               f"(~{size} tok, limit {MAX_RESULT_TOKENS})")
    # Whatever the brain sees, synthesis must still get the fact checklist.
    resolved = evidence.resolve(view)
    facts = json.loads(resolved).get("must_state") or json.loads(resolved).get("error")
    check_true(f"{tool.name} checklist survives to synthesis ({label})", bool(facts))

# The failure that keeps recurring: a question needing four tool calls overflows
# the window and every request 400s. Budget the whole conversation, not just one
# result -- each turn costs the assistant's tool-call message plus the result,
# and all of it is re-sent on the next turn.
worst = max(sizes)
conversation = overhead + 60 + 4 * (60 + worst)
check_true(
    "a four-tool-call conversation fits the 4,096-token window",
    conversation < BRAIN_CONTEXT_TOKENS,
    f"(~{conversation} tok with the largest result ~{worst})",
)

print("=" * 72)
print(f"RESULT: {PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
