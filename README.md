[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zul456/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/zul456/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# Forecast GTA VI's launch-day Twitch viewers hour by hour

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
on [make.fun](https://make.fun).

`main.py` (Python 3.13, standard library only, no network) forecasts Twitch
concurrent viewers in the "Grand Theft Auto VI" category for each hour 2–24 of
release day 2026-11-19 UTC, given everything recorded up to the end of hour 1
(hour 1's print included). It reads every `*.in` file from `$MAKEFUN_INPUT_DIR`
(or `/input/`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR`
(or `/output/`): 23 rows, hours 2–24, five quantiles (0.05 / 0.25 / 0.5 / 0.75 /
0.95) per hour. Fully deterministic — run it twice and the outputs are
byte-identical.

## Model

For each hour h = 2..24, a log-linear regression is fitted at run time from the
case's own comparable launches (past big launches' release-day hourly curves,
including GTA VI's own premiere history):

    log(viewers_h / viewers_1) = a_h + b_h * log(p1) + c_h * I(rolling)

- **Anchoring:** every hour is a ratio to hour 1's print, so the forecast scales
  with whatever hour 1 actually is.
- **Shape:** the slope on log(p1) is negative — across 40+ launch curves,
  log(hour-1 print) correlates about −0.94 with log(peak ratio), so small
  hour-1 launches rise much more over the day than big ones. The regression is
  fitted locally in log(p1) (an adaptive window, widened until at least 12
  points), so the shape adapts to this launch's hour-1 level.
- **Unlock type:** the indicator I(rolling) is 1 for rolling midnight-local
  console unlocks and 0 for simultaneous global (PC) ones. Rolling launches
  rise more than simultaneous ones at the same hour-1 level, and GTA VI
  unlocks at midnight local, country by country — so the indicator corrects the
  level for exactly this kind of launch. At a hypothetical hour-1 print of
  1,000,000 the model still predicts a modest rise (peak median ≈ 1.19×), not
  the flat/declining curve a plain regression would give.
- **Spread:** the five quantiles come from the empirical residuals of the fit
  (an honest ~90% band), and the log-quantiles are lightly smoothed over
  neighbouring hours. Quantiles are never equal and the forecast is never flat.

If a case has no usable comparables, or any step fails, the forecast falls back
to the house baseline shape (flat at hour 1 with a widening spread), so a case
is never left invalid.

## Practice results

Scored locally on the four practice cases (pinball loss on the log scale;
lower is better) with a local scorer against the SullyGnome release-day hourly
averages:

| case | game | loss | typical miss | 90% coverage |
|------|------|------|--------------|--------------|
| p00 | Marvel's Wolverine | 0.2447 | 55.7% | 1.000 |
| p01 | Red Dead Redemption 2 | 0.2214 | 48.0% | 0.913 |
| p02 | God of War Ragnarök | 0.1244 | 19.8% | 1.000 |
| p03 | Marvel's Spider-Man 2 | 0.1720 | 28.2% | 0.870 |
| **mean** | | **0.1906** | **37.9%** | **0.946** |

House baseline on the same practice: mean loss **0.3871** (typical miss
~82.8%, 90% coverage ~0.783). With the final's full 42-comparable info set
(all four practice launches included as comparables), the same program scores
mean **0.1576** on practice.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.2 s for all four practice cases (limit: 8 s per case).

## License

MIT — see [LICENSE](LICENSE).
