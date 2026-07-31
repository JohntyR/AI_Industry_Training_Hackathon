# Public-Question Evaluation

Run against `http://127.0.0.1:5001` over `POST /query` -- the full pipeline: Qwen planning,
runtime tool execution, fine-tuned Nemotron synthesis.

| Metric | Value |
|---|---:|
| Component score | 104.0/150.0 (69.3%) |
| After time penalties | 104.0/150.0 (69.3%) |
| Slowest response | 44.11s |
| Over 60s | 0 of 15 |
| Tool calls made | 33 |
| Answers with no tool evidence | 0 |
| Questions hitting a degraded path | 3 |
| Failed requests | 0 |

| Question | Difficulty | Time | Tools | Score | Penalty | Missing components |
|---|---|---:|---:|---:|---|---|
| MHQ001 | easy | 6.14s | 1 | 0.0/10.0 | - | C01 |
| MHQ035 | medium | 8.66s | 1 | 0.0/10.0 | - | C01, C02 |
| MHQ040 | easy | 10.08s | 1 | 10.0/10.0 | - | - |
| MHQ045 | medium | 7.97s | 1 | 10.0/10.0 | - | - |
| MHQ049 | medium | 7.34s | 1 | 10.0/10.0 | - | - |
| MHQ055 | hard | 16.93s | 1 | 10.0/10.0 | - | - |
| MHQ058 | easy | 44.11s | 4 | 10.0/10.0 | - | - |
| MHQ061 | medium | 12.36s | 1 | 10.0/10.0 | - | - |
| MHQ067 | hard | 44.1s | 4 | 10.0/10.0 | - | - |
| MHQ072 | medium | 19.28s | 2 | 6.67/10.0 | - | C02 |
| MHQ074 | hard | 24.95s | 4 | 0.0/10.0 | - | C01, C02, C03, C04 |
| MHQ076 | easy | 20.07s | 3 | 10.0/10.0 | - | - |
| MHQ080 | medium | 23.6s | 5 | 4.0/10.0 | - | C01, C06, C07 |
| MHQ084 | medium | 20.84s | 3 | 10.0/10.0 | - | - |
| MHQ090 | hard | 18.72s | 1 | 3.33/10.0 | - | C02, C03 |

## Answers

### MHQ001 — 0.0/10.0

> From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?

The number of cash-rate decisions that changed the rate from the first RBA record to the last is 41, consisting of 20 increases and 21 decreases.

Missing:
- `C01` 41 of the 175 decision records changed the rate: 20 increases and 21 decreases.

### MHQ035 — 0.0/10.0

> Across the 2011-2013 easing period, how many cuts occurred and how far did the target fall?

During the 2011-2013 easing period, the RBA made 8 cuts. The target fell by a total of -2.25 percentage points from 4.75% to 2.50%.

Missing:
- `C01` Eight cuts occurred: 2 in 2011, 4 in 2012, and 2 in 2013.
- `C02` They totalled -2.25 percentage points, taking the target from 4.75% before the first cut to 2.50% at the end of 2013.

### MHQ040 — 10.0/10.0

> What are the dimensions and common date range of the ASX dataset?

The ASX dataset has a dimension of 18 ticker files, each containing 1,774 rows. The common date range for the ASX dataset is from 2 Jan 2015 to 30 Dec 2021.

### MHQ045 — 10.0/10.0

> Excluding Tabcorp, which ticker had the best and worst 2018 return?

The best 2018 return among companies excluding Tabcorp was BHP.AX with a +22.17% gain, while the worst was AMP.AX with a -50.04% decline.

### MHQ049 — 10.0/10.0

> Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?

The ticker with the highest average daily volume, excluding Tabcorp, is AMP.AX at 11,635,671.71 shares per day.

### MHQ055 — 10.0/10.0

> Rank the three worst non-Tabcorp full-sample maximum drawdowns and identify each peak and trough date.

The three worst non-Tabcorp full-sample maximum drawdowns are ranked as follows: 1) AMP.AX with an -82.45% drawdown from 20 Mar 2015 to 17 Dec 2021, 2) AGL.AX with a -76.24% drawdown from 10 Apr 2017 to 16 Nov 2021, and 3) QAN.AX with a -71.08% drawdown from 19 Dec 2019 to 19 Mar 2020.

### MHQ058 — 10.0/10.0

> Retrieve the AFR article "Travel stocks take off on vaccine rollout" published 23 Feb 2021 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for ASX travel shares.

The article's financial-market sentiment is positive, with ASX travel shares likely to rise. The RBA cash-rate target of 0.10% (effective from 3 Feb 2021) is mentioned, but the article focuses on the positive impact of the vaccine rollout on travel stocks.

### MHQ061 — 10.0/10.0

> Using a case-insensitive once-per-record whole-word unemployment search, which year and which month have the highest AFR counts?

The highest AFR counts for unemployment, based on a case-insensitive whole-word search, occurred in 2020 with a peak of 1,452 records. The specific month within this year with the highest count is May 2020 (2020-05), which had 218 matching records.

### MHQ067 — 10.0/10.0

> Retrieve the AFR article "Why investors don't believe the RBA on interest rates" published 25 Nov 2021 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for the broad ASX, especially rate-sensitive shares.

The article's financial-market sentiment is mixed with a negative bias. The ASX is likely to fall, with rate-sensitive shares expected to decline, as investors remain skeptical about the RBA's rate-setting decisions, despite the cash-rate target being in force on 25 Nov 2021 at 0.10%.

### MHQ072 — 6.67/10.0

> After the 5 Jun 2019 RBA cut, report the new target and the 5-12 Jun returns for the non-Tabcorp basket, CBA.AX, NAB.AX, ANZ.AX, BHP.AX, and RIO.AX.

The new RBA cash-rate target after the 5 Jun 2019 cut is 1.25%. For the 5-12 Jun period, the non-Tabcorp basket returned +2.88% with individual moves: CBA.AX +0.60%, NAB.AX +1.39%, ANZ.AX +0.89%, BHP.AX +5.89%, and RIO.AX +2.91%.

Missing:
- `C02` From 5 to 12 Jun, the non-Tabcorp basket rose +2.88%

### MHQ074 — 0.0/10.0

> Across the three 2019 RBA cuts, what was the non-Tabcorp basket's one-week return after each effective date?

The non-Tabcorp basket had a one-week return of +1.25% after the first RBA cut on 5 Jun 2019, -0.75% after the second cut on 19 Jul 2019, and -0.75% after the third cut on 2 Oct 2019.

Missing:
- `C01` After 5 Jun the basket rose +2.88% by 12 Jun
- `C02` after 3 Jul it rose +0.24% by 10 Jul
- `C03` after 2 Oct it fell -2.17% by 9 Oct.
- `C04` The cuts took the target to 1.25%, 1.00%, and 0.75%, respectively.

### MHQ076 — 10.0/10.0

> For 2021, report the once-per-record whole-word QBE AFR count and QBE.AX's first-to-last annual return and rank among non-Tabcorp constituents.

For 2021, the QBE AFR count is 369. QBE.AX's first-to-last annual return is +35.57%. Among non-Tabcorp constituents, QBE.AX ranks 1st with a +35.57% return, followed by NAB.AX at +33.83% and GPT.AX at +27.95%.

### MHQ080 — 4.0/10.0

> Retrieve the AFR article "Energy stocks shine as vaccines fuel oil rally" published 28 Nov 2020 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for ASX energy shares. Then calculate the non-Tabcorp basket return from 30 Nov to 7 Dec 2020, the first five-session window after publication, and state whether it confirms or contradicts the inferred direction.

The article's financial-market sentiment is positive, with a likely direction for ASX energy shares being upwards. The Tabcorp basket return from 30 Nov to 7 Dec 2020 is +1.23%, which confirms the inferred direction.

Missing:
- `C01` The RBA cash-rate target in force was 0.10%.
- `C06` The non-Tabcorp basket rose 2.37% from 30 Nov to 7 Dec 2020.
- `C07` The observed five-session move confirmed the inferred upward direction.

### MHQ084 — 10.0/10.0

> For 2019, report the RBA cut count, cumulative change and year-end target; the AFR rate/RBA-pattern count; and the non-Tabcorp ASX basket's simple average annual return.

For 2019, the RBA cut count was 3; the cumulative change was -0.75 percentage points with a year-end target of 0.75%. The AFR rate/RBA-pattern count was 3,181. The non-Tabcorp ASX basket's simple average annual return was +20.11%.

### MHQ090 — 3.33/10.0

> Can the three supplied datasets support a fully observed analysis of how AFR news and ASX prices reacted to the 2022-2023 RBA tightening cycle?

No, the datasets cannot support a fully observed analysis as the ASX and AFR data only go up to 2021, while the RBA tightening cycle in question pertains to 2022-2023. The RBA data extends beyond this period, but the other datasets do not cover the required timeframe.

Missing:
- `C02` The RBA data covers the 13 hikes from May 2022 to Nov 2023, but both AFR and ASX datasets end in 2021.
- `C03` A three-dataset reaction analysis is therefore unsupported by the supplied evidence.

