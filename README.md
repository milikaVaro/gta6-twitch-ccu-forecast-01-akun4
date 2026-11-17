# GTA VI launch-day Twitch CCU forecast — akun 4 rebuild

Competition: `gta6-twitch-ccu-forecast-01` (make.fun). Forecast five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of Twitch concurrent viewers
for the "Grand Theft Auto VI" category, hours 2–24 of release day
(2026-11-19 UTC), given the hour-1 print.

## Model

Per-hour **three-way ensemble (OLS + Theil-Sen + recency-weighted OLS)**
on `log(viewers_h)` vs `log(viewers_1)` across the rolling-midnight
comparables, with a **sharp spread (×0.75)**:

- OLS: the plain fit.
- Theil-Sen: median of pairwise slopes — robust to outlying launches
  (upper median for even counts, exactly as the backtest computes it).
- WOLS: weights `1.0 + 0.1*i` in list order — newer launches count more.
- Center: average of the three fits. Spread: ensemble residual sd × 0.75,
  clamped to [0.20, 1.00].

Leave-one-out backtest over the 12 rolling comparables: **mean pinball
loss 0.2425** — better than WOLS alone (0.2434) and plain OLS (0.2473);
the third estimator breaks the tie between the other two.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Stdlib only, deterministic, no network. Reads every `<id>.in`, writes
`<id>.out` with 23 rows (hours 2–24), five quantiles each.

## Rebuild note

This replaces the account's first entry (a slope-shrunk regression,
practice loss 0.194, rank 1.326). Same repo, new history: `makefun
init` root (freshness anchor) + LICENSE + this work.
