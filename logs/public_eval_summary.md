# Public-Question Evaluation

Run against `http://127.0.0.1:5000` over `POST /query` -- the full pipeline: Qwen planning,
runtime tool execution, fine-tuned Nemotron synthesis.

| Metric | Value |
|---|---:|
| Component score | 77.34/150.0 (51.6%) |
| After time penalties | 77.34/150.0 (51.6%) |
| Slowest response | 137.99s |
| Over 60s | 1 of 15 |
| Tool calls made | 29 |
| Answers with no tool evidence | 1 |
| Questions hitting a degraded path | 15 |
| Failed requests | 0 |

> **This run is not valid.** 1 of 15 answers were produced from zero tool results, so any points shown are coincidence rather than evidence — a generic refusal can satisfy a label-only component.

| Question | Difficulty | Time | Tools | Score | Penalty | Missing components |
|---|---|---:|---:|---:|---|---|
| MHQ001 | easy | 10.09s | 1 | 10.0/10.0 | - | - |
| MHQ035 | medium | 16.04s | 1 | 10.0/10.0 | - | - |
| MHQ040 | easy | 9.63s | 1 | 10.0/10.0 | - | - |
| MHQ045 | medium | 16.56s | 1 | 10.0/10.0 | - | - |
| MHQ049 | medium | 13.17s | 1 | 10.0/10.0 | - | - |
| MHQ055 | hard | 17.23s | 1 | 10.0/10.0 | - | - |
| MHQ058 | easy | 38.96s | 2 | 2.0/10.0 | - | C02, C03 |
| MHQ061 | medium | 20.86s | 1 | 10.0/10.0 | - | - |
| MHQ067 | hard | 42.55s | 2 | 2.0/10.0 | - | C02, C03 |
| MHQ072 | medium | 137.99s | 0 | 0.0/10.0 | -20% | C01, C02, C03 |
| MHQ074 | hard | 43.77s | 4 | 0.0/10.0 | - | C01, C02, C03, C04 |
| MHQ076 | easy | 22.82s | 3 | 0.0/10.0 | - | C01 |
| MHQ080 | medium | 45.78s | 3 | 0.0/10.0 | - | C01, C02, C03, C06, C07 |
| MHQ084 | medium | 50.3s | 7 | 3.34/10.0 | - | C01, C02 |
| MHQ090 | hard | 9.97s | 1 | 0.0/10.0 | - | C01, C02, C03 |

## Answers

### MHQ001 — 10.0/10.0

> From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?

Verified results: {"total_records": 175, "changes": 41, "increases": 20, "decreases": 21}

### MHQ035 — 10.0/10.0

> Across the 2011-2013 easing period, how many cuts occurred and how far did the target fall?

Verified results: {"window": [2011, 2013], "n_changes": 8, "n_cuts": 8, "n_hikes": 0, "by_year": {"2011": 2, "2012": 4, "2013": 2}, "cumulative_change": -2.25, "rate_before": 4.75, "rate_after": 2.5}

### MHQ040 — 10.0/10.0

> What are the dimensions and common date range of the ASX dataset?

Verified results: {"n_tickers": 18, "rows_per_ticker": 1774, "start_date": "2015-01-02", "end_date": "2021-12-30"}

### MHQ045 — 10.0/10.0

> Excluding Tabcorp, which ticker had the best and worst 2018 return?

Verified results: {"year": 2018, "excluded_tabcorp": true, "ranking": [{"ticker": "BHP.AX", "return_pct": 22.17}, {"ticker": "QAN.AX", "return_pct": 17.88}, {"ticker": "GPT.AX", "return_pct": 9.46}, {"ticker": "RIO.AX", "return_pct": 7.94}, {"ticker": "CMW.AX", "return_pct": 6.13}, {"ticker": "IAG.AX", "return_pct": 1.96}, {"ticker": "TCL.AX", "return_pct": -0.03}, {"ticker": "TPG.AX", "return_pct": -0.74}, {"ticker": "SUN.AX", "return_pct": -3.53}, {"ticker": "QBE.AX", "return_pct": -3.62}, {"ticker": "CBA.AX", "return_pct": -4.42}, {"ticker": "AZJ.AX", "return_pct": -7.94}, {"ticker": "ANZ.AX", "return_pct": -9.24}, {"ticker": "AGL.AX", "return_pct": -11.38}, {"ticker": "NAB.AX", "return_pct": -12.31}, {"ticker": "SGP.AX", "return_pct": -15.3}, {"ticker": "AMP.AX", "return_pct": -50.04}], "best": {"ticker": "BHP.AX", "return_pct": 22.17}, "worst": {"ticker": "AMP.AX", "return_pct": -50.04}}

### MHQ049 — 10.0/10.0

> Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?

Verified results: {"excluded_tabcorp": true, "ranking": [{"ticker": "AMP.AX", "avg_volume": 11635671.71}, {"ticker": "QAN.AX", "avg_volume": 9585960.83}, {"ticker": "BHP.AX", "avg_volume": 8986996.14}, {"ticker": "SGP.AX", "avg_volume": 7732021.03}, {"ticker": "AZJ.AX", "avg_volume": 7226183.35}, {"ticker": "ANZ.AX", "avg_volume": 6394244.77}, {"ticker": "NAB.AX", "avg_volume": 6314939.81}, {"ticker": "IAG.AX", "avg_volume": 6213509.96}, {"ticker": "GPT.AX", "avg_volume": 5886983.29}, {"ticker": "TCL.AX", "avg_volume": 5197565.66}, {"ticker": "QBE.AX", "avg_volume": 4846211.1}, {"ticker": "CMW.AX", "avg_volume": 4536898.82}, {"ticker": "CBA.AX", "avg_volume": 3123979.3}, {"ticker": "SUN.AX", "avg_volume": 2889976.91}, {"ticker": "AGL.AX", "avg_volume": 2429828.0}, {"ticker": "RIO.AX", "avg_volume": 1978049.58}, {"ticker": "TPG.AX", "avg_volume": 1762158.26}], "highest": {"ticker": "AMP.AX", "avg_volume": 11635671.71}}

### MHQ055 — 10.0/10.0

> Rank the three worst non-Tabcorp full-sample maximum drawdowns and identify each peak and trough date.

Verified results: {"excluded_tabcorp": true, "worst": [{"rank": 1, "ticker": "AMP.AX", "drawdown_pct": -82.45, "peak_date": "20 Mar 2015", "trough_date": "17 Dec 2021"}, {"rank": 2, "ticker": "AGL.AX", "drawdown_pct": -76.24, "peak_date": "10 Apr 2017", "trough_date": "16 Nov 2021"}, {"rank": 3, "ticker": "QAN.AX", "drawdown_pct": -71.08, "peak_date": "19 Dec 2019", "trough_date": "19 Mar 2020"}]}

### MHQ058 — 2.0/10.0

> Retrieve the AFR article "Travel stocks take off on vaccine rollout" published 23 Feb 2021 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for ASX travel shares.

Verified results: {"date": "2021-02-23", "rate": 0.1, "effective_date": "3 Feb 2021"}

Missing:
- `C02` The article's sentiment is positive.
- `C03` The likely direction for ASX travel shares is upward.

### MHQ061 — 10.0/10.0

> Using a case-insensitive once-per-record whole-word unemployment search, which year and which month have the highest AFR counts?

Verified results: {"pattern": "\\bunemployment\\b", "peak_year": "2020", "peak_year_count": 1452, "peak_month": "202005", "peak_month_count": 218}

### MHQ067 — 2.0/10.0

> Retrieve the AFR article "Why investors don't believe the RBA on interest rates" published 25 Nov 2021 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for the broad ASX, especially rate-sensitive shares.

Verified results: {"date": "2021-11-25", "rate": 0.1, "effective_date": "3 Nov 2021"}

Missing:
- `C02` The article's sentiment is mixed with a negative bias.
- `C03` The likely broad-ASX impact is mixed-to-down, with rate-sensitive shares under pressure.

### MHQ072 — 0.0/10.0

> After the 5 Jun 2019 RBA cut, report the new target and the 5-12 Jun returns for the non-Tabcorp basket, CBA.AX, NAB.AX, ANZ.AX, BHP.AX, and RIO.AX.

The supplied datasets do not contain the information required to answer this question.

Missing:
- `C01` The RBA cut to 1.25%.
- `C02` From 5 to 12 Jun, the non-Tabcorp basket rose +2.88%
- `C03` CBA +0.60%, NAB +1.39%, ANZ +0.89%, BHP +5.89%, and RIO +2.91%.

### MHQ074 — 0.0/10.0

> Across the three 2019 RBA cuts, what was the non-Tabcorp basket's one-week return after each effective date?

Verified results: {"start": "2019-10-08", "end": "2019-10-15", "n": 17, "basket_return_pct": 0.37, "constituents": {"AGL.AX": 0.16, "AMP.AX": 0.3, "ANZ.AX": 1.5, "AZJ.AX": 1.19, "BHP.AX": 1.32, "CBA.AX": 0.9, "CMW.AX": 0.0, "GPT.AX": -1.76, "IAG.AX": 1.29, "NAB.AX": 1.28, "QAN.AX": 4.53, "QBE.AX": 0.98, "RIO.AX": 2.28, "SGP.AX": 0.66, "SUN.AX": -4.97, "TCL.AX": -2.01, "TPG.AX": -1.29}}

Missing:
- `C01` After 5 Jun the basket rose +2.88% by 12 Jun
- `C02` after 3 Jul it rose +0.24% by 10 Jul
- `C03` after 2 Oct it fell -2.17% by 9 Oct.
- `C04` The cuts took the target to 1.25%, 1.00%, and 0.75%, respectively.

### MHQ076 — 0.0/10.0

> For 2021, report the once-per-record whole-word QBE AFR count and QBE.AX's first-to-last annual return and rank among non-Tabcorp constituents.

Verified results: {"year": 2021, "excluded_tabcorp": true, "ranking": [{"ticker": "QBE.AX", "return_pct": 35.57}, {"ticker": "NAB.AX", "return_pct": 33.83}, {"ticker": "GPT.AX", "return_pct": 27.95}, {"ticker": "ANZ.AX", "return_pct": 27.05}, {"ticker": "CBA.AX", "return_pct": 26.74}, {"ticker": "SUN.AX", "return_pct": 21.79}, {"ticker": "SGP.AX", "return_pct": 7.97}, {"ticker": "CMW.AX", "return_pct": 7.9}, {"ticker": "BHP.AX", "return_pct": 6.92}, {"ticker": "TCL.AX", "return_pct": 4.08}, {"ticker": "QAN.AX", "return_pct": 1.43}, {"ticker": "AZJ.AX", "return_pct": -2.4}, {"ticker": "RIO.AX", "return_pct": -4.51}, {"ticker": "IAG.AX", "return_pct": -4.97}, {"ticker": "TPG.AX", "return_pct": -15.09}, {"ticker": "AMP.AX", "return_pct": -35.9}, {"ticker": "AGL.AX", "return_pct": -44.38}], "best": {"ticker": "QBE.AX", "return_pct": 35.57}, "worst": {"ticker": "AGL.AX", "return_pct": -44.38}}

Missing:
- `C01` There are 369 AFR records matching whole-word QBE in 2021, and QBE.AX had the best non-Tabcorp 2021 return at +35.57%.

### MHQ080 — 0.0/10.0

> Retrieve the AFR article "Energy stocks shine as vaccines fuel oil rally" published 28 Nov 2020 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for ASX energy shares. Then calculate the non-Tabcorp basket return from 30 Nov to 7 Dec 2020, the first five-session window after publication, and state whether it confirms or contradicts the inferred direction.

Verified results: {"start": "2020-11-30", "end": "2020-12-07", "n": 17, "basket_return_pct": 2.37, "constituents": {"AGL.AX": -0.96, "AMP.AX": 1.75, "ANZ.AX": 3.09, "AZJ.AX": 0.47, "BHP.AX": 11.35, "CBA.AX": 2.11, "CMW.AX": 5.26, "GPT.AX": 0.64, "IAG.AX": 0.97, "NAB.AX": 1.57, "QAN.AX": 2.04, "QBE.AX": -0.4, "RIO.AX": 14.38, "SGP.AX": -1.54, "SUN.AX": 1.09, "TCL.AX": -1.07, "TPG.AX": -0.39}}

Missing:
- `C01` The RBA cash-rate target in force was 0.10%.
- `C02` The article's sentiment is positive.
- `C03` The likely direction for ASX energy shares is upward.
- `C06` The non-Tabcorp basket rose 2.37% from 30 Nov to 7 Dec 2020.
- `C07` The observed five-session move confirmed the inferred upward direction.

### MHQ084 — 3.34/10.0

> For 2019, report the RBA cut count, cumulative change and year-end target; the AFR rate/RBA-pattern count; and the non-Tabcorp ASX basket's simple average annual return.

Verified results: {"start": "2019-01-01", "end": "2019-12-31", "n": 17, "basket_return_pct": 20.11, "constituents": {"AGL.AX": 7.29, "AMP.AX": -20.16, "ANZ.AX": 9.66, "AZJ.AX": 29.12, "BHP.AX": 28.58, "CBA.AX": 19.2, "CMW.AX": 26.88, "GPT.AX": 10.2, "IAG.AX": 15.58, "NAB.AX": 10.95, "QAN.AX": 28.85, "QBE.AX": 36.49, "RIO.AX": 44.46, "SGP.AX": 42.29, "SUN.AX": 10.91, "TCL.AX": 34.34, "TPG.AX": 7.31}}

Missing:
- `C01` The RBA cut three times for -0.75 percentage points and ended at 0.75%.
- `C02` AFR contains 3,181 records matching interest rates?|cash rate|rate cut|rate hike|\\bRBA\\b.

### MHQ090 — 0.0/10.0

> Can the three supplied datasets support a fully observed analysis of how AFR news and ASX prices reacted to the 2022-2023 RBA tightening cycle?

Verified results: {"rba": {"start": "3 Feb 2010", "end": "17 Jun 2026"}, "asx": {"start": "2015-01-02", "end": "2021-12-30"}, "afr": {"start": "2015-01", "end": "2021-12"}}

Missing:
- `C01` No.
- `C02` The RBA data covers the 13 hikes from May 2022 to Nov 2023, but both AFR and ASX datasets end in 2021.
- `C03` A three-dataset reaction analysis is therefore unsupported by the supplied evidence.

