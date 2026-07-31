# Public-Question Evaluation

Run against `http://127.0.0.1:5001` over `POST /query` -- the full pipeline: Qwen planning,
runtime tool execution, fine-tuned Nemotron synthesis.

| Metric | Value |
|---|---:|
| Component score | 123.67/150.0 (82.4%) |
| After time penalties | 123.67/150.0 (82.4%) |
| Slowest response | 22.69s |
| Over 60s | 0 of 15 |
| Tool calls made | 30 |
| Answers with no tool evidence | 0 |
| Questions hitting a degraded path | 3 |
| Failed requests | 0 |

| Question | Difficulty | Time | Tools | Score | Penalty | Missing components |
|---|---|---:|---:|---:|---|---|
| MHQ001 | easy | 6.61s | 1 | 10.0/10.0 | - | - |
| MHQ035 | medium | 12.4s | 1 | 10.0/10.0 | - | - |
| MHQ040 | easy | 13.68s | 1 | 10.0/10.0 | - | - |
| MHQ045 | medium | 15.46s | 1 | 10.0/10.0 | - | - |
| MHQ049 | medium | 5.34s | 1 | 10.0/10.0 | - | - |
| MHQ055 | hard | 14.19s | 1 | 10.0/10.0 | - | - |
| MHQ058 | easy | 16.07s | 3 | 10.0/10.0 | - | - |
| MHQ061 | medium | 7.47s | 1 | 10.0/10.0 | - | - |
| MHQ067 | hard | 22.69s | 3 | 0.0/10.0 | - | C01, C02, C03 |
| MHQ072 | medium | 22.17s | 2 | 10.0/10.0 | - | - |
| MHQ074 | hard | 16.16s | 3 | 0.0/10.0 | - | C01, C02, C03, C04 |
| MHQ076 | easy | 14.38s | 3 | 10.0/10.0 | - | - |
| MHQ080 | medium | 16.49s | 5 | 7.0/10.0 | - | C01, C07 |
| MHQ084 | medium | 16.35s | 3 | 10.0/10.0 | - | - |
| MHQ090 | hard | 19.04s | 1 | 6.67/10.0 | - | C02 |

## Answers

### MHQ001 — 10.0/10.0

> From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?

From the first to the last RBA cash-rate decision, there were 41 rate changes, comprising 20 increases and 21 decreases, out of a total of 175 decision records.

### MHQ035 — 10.0/10.0

> Across the 2011-2013 easing period, how many cuts occurred and how far did the target fall?

During the 2011-2013 easing period, 8 cuts were made, with 2 in 2011, 4 in 2012, and 2 in 2013. The target fell by a total of -2.25 percentage points, starting from a target of 4.75% before the first change and ending at a target of 2.50% at the end of the period, with the cycle running from 2 November 2011 to 7 August 2013.

### MHQ040 — 10.0/10.0

> What are the dimensions and common date range of the ASX dataset?

The ASX dataset has a dimension of 18 ticker files, each containing 1,774 rows, and it spans from 2 January 2015 to 30 December 2021. The RBA dataset, in comparison, has 175 decision records from 3 February 2010 to 17 June 2026. The AFR dataset covers the time range from 2015-01 to 2021-12. Both ASX and AFR datasets end on 30 December 2021 and 2021, respectively, while RBA runs until 2026.

### MHQ045 — 10.0/10.0

> Excluding Tabcorp, which ticker had the best and worst 2018 return?

The best 2018 return among the specified tickers was BHP.AX at +22.17%, while the worst was AMP.AX at -50.04%. The full ranking, excluding Tabcorp, is BHP.AX +22.17%, QAN.AX +17.88%, GPT.AX +9.46%, RIO.AX +7.94%, CMW.AX +6.13%, IAG.AX +1.96%, TCL.AX -0.03%, TPG.AX -0.74%, SUN.AX -3.53%, QBE.AX -3.62%, CBA.AX -4.42%, AZJ.AX -7.94%, ANZ.AX -9.24%, AGL.AX -11.38%, NAB.AX -12.31%, SGP.AX -15.30%, and AMP.AX -50.04%.

### MHQ049 — 10.0/10.0

> Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?

The ticker with the highest average daily volume, excluding Tabcorp, is AMP.AX at 11,635,671.71 shares per trading day.

### MHQ055 — 10.0/10.0

> Rank the three worst non-Tabcorp full-sample maximum drawdowns and identify each peak and trough date.

The three worst non-Tabcorp full-sample maximum drawdowns are 1) AMP.AX at -82.45% from 20 Mar 2015 to 17 Dec 2021, with a peak on 20 Mar 2015 and a trough on 17 Dec 2021; 2) AGL.AX at -76.24% from 10 Apr 2017 to 16 Nov 2021, with a peak on 10 Apr 2017 and a trough on 16 Nov 2021; and 3) QAN.AX at -71.08% from 19 Dec 2019 to 19 Mar 2020, with a peak on 19 Dec 2019 and a trough on 19 Mar 2020.

### MHQ058 — 10.0/10.0

> Retrieve the AFR article "Travel stocks take off on vaccine rollout" published 23 Feb 2021 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for ASX travel shares.

The article "Travel stocks take off on vaccine rollout" published on 23 Feb 2021 expresses a positive financial-market sentiment, with ASX travel shares likely to continue their upward trajectory. The RBA cash-rate target of 0.10% in force on that date is mentioned, but it does not directly influence the article's sentiment.

### MHQ061 — 10.0/10.0

> Using a case-insensitive once-per-record whole-word unemployment search, which year and which month have the highest AFR counts?

The year with the highest AFR counts is 2020, with 1,452 matching records. The month with the highest AFR counts in that year is May 2020 (2020-05), with 218 records.

### MHQ067 — 0.0/10.0

> Retrieve the AFR article "Why investors don't believe the RBA on interest rates" published 25 Nov 2021 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for the broad ASX, especially rate-sensitive shares.

The article "Why investors don't believe the RBA on interest rates" published on 25 Nov 2021 expresses a negative sentiment towards the RBA's interest rate forecasts, with a -0.10% cash-rate target in force on that date. The likely direction for the broad ASX, especially rate-sensitive shares, is a decrease in value, reflecting investors' skepticism about the RBA's rate predictions.

Missing:
- `C01` The RBA cash-rate target in force was 0.10%.
- `C02` The article's sentiment is mixed with a negative bias.
- `C03` The likely broad-ASX impact is mixed-to-down, with rate-sensitive shares under pressure.

### MHQ072 — 10.0/10.0

> After the 5 Jun 2019 RBA cut, report the new target and the 5-12 Jun returns for the non-Tabcorp basket, CBA.AX, NAB.AX, ANZ.AX, BHP.AX, and RIO.AX.

The new RBA cash-rate target after the 5 Jun 2019 cut was 1.25%. The non-Tabcorp basket, excluding Tabcorp, returned +2.88% from 5 Jun 2019 to 12 Jun 2019, with individual constituent returns of +0.60% for CBA.AX, +1.39% for NAB.AX, +0.89% for ANZ.AX, +5.89% for BHP.AX, and +2.91% for RIO.AX, all with the RBA target at 1.25%.

### MHQ074 — 0.0/10.0

> Across the three 2019 RBA cuts, what was the non-Tabcorp basket's one-week return after each effective date?

The non-Tabcorp basket had a one-week return of +0.25% after the first 2019 RBA cut on 5 Jun 2019, +0.50% after the second cut on 19 Jul 2019, and +0.75% after the third cut on 2 Oct 2019. The three cuts in 2019 ran from 5 Jun 2019 to 2 Oct 2019, with the target falling by -0.75 percentage points from 1.50% to 0.75%.

Missing:
- `C01` After 5 Jun the basket rose +2.88% by 12 Jun
- `C02` after 3 Jul it rose +0.24% by 10 Jul
- `C03` after 2 Oct it fell -2.17% by 9 Oct.
- `C04` The cuts took the target to 1.25%, 1.00%, and 0.75%, respectively.

### MHQ076 — 10.0/10.0

> For 2021, report the once-per-record whole-word QBE AFR count and QBE.AX's first-to-last annual return and rank among non-Tabcorp constituents.

For 2021, there were 369 AFR records matching QBE. QBE.AX had a +35.57% annual return, ranking first among non-Tabcorp constituents with a +35.57% return, followed by NAB.AX at +33.83% and GPT.AX at +27.95%.

### MHQ080 — 7.0/10.0

> Retrieve the AFR article "Energy stocks shine as vaccines fuel oil rally" published 28 Nov 2020 and use the RBA cash-rate target in force on that date. Classify the article's financial-market sentiment as positive, negative, or mixed; state the likely direction for ASX energy shares. Then calculate the non-Tabcorp basket return from 30 Nov to 7 Dec 2020, the first five-session window after publication, and state whether it confirms or contradicts the inferred direction.

The article "Energy stocks shine as vaccines fuel oil rally" published 28 Nov 2020 expresses a positive financial-market sentiment, with ASX energy shares likely to rise. The non-Tabcorp basket rose by +2.37% from 30 Nov to 7 Dec 2020, confirming the inferred positive direction.

Missing:
- `C01` The RBA cash-rate target in force was 0.10%.
- `C07` The observed five-session move confirmed the inferred upward direction.

### MHQ084 — 10.0/10.0

> For 2019, report the RBA cut count, cumulative change and year-end target; the AFR rate/RBA-pattern count; and the non-Tabcorp ASX basket's simple average annual return.

For 2019, the RBA cut count was 3, the cumulative change was -0.75 percentage points, and the year-end target was 0.75%. The AFR rate/RBA-pattern count was 3,181. The non-Tabcorp ASX basket's simple average annual return was +20.11%, with constituent returns ranging from -20.16% for AMP.AX to +44.46% for RIO.AX.

### MHQ090 — 6.67/10.0

> Can the three supplied datasets support a fully observed analysis of how AFR news and ASX prices reacted to the 2022-2023 RBA tightening cycle?

The three supplied datasets cannot support a fully observed analysis of how AFR news and ASX prices reacted to the 2022-2023 RBA tightening cycle as the AFR and ASX datasets only cover data up to December 2021, which is before the RBA tightening cycle of 2022-2023. The RBA dataset includes data beyond 2021 but the analysis requires data from both RBA and AFR/ASX, which is not available for the relevant time period. The ASX and AFR datasets end on 30 Dec 2021 / Dec 2021, and the RBA runs to 2026. The question implies a breakdown of AFR news and ASX prices, but the AFR and ASX datasets do not have data after 2021, which is necessary for the analysis. The AFR has 85 monthly files from 2015-01 to 2021-12, and the ASX has 18 ticker files of 1,774 rows each from 2 Jan 2015 to 30 Dec 2021. The RBA has 175 decision records from 3 Feb 2010 to 17 Jun 2026.

Missing:
- `C02` The RBA data covers the 13 hikes from May 2022 to Nov 2023, but both AFR and ASX datasets end in 2021.

