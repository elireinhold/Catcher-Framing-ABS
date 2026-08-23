# Catcher Framing Under an Automated Strike Zone (ABS)

A Bayesian analysis of MLB catcher framing skill (2025–2026 Statcast data),
quantifying how much defensive value each catcher provides and how the introduction
of ABS has affected the value of catcher framing.

## Motivation

This season (2026), the MLB introduced the Automated Ball-Strike (ABS) system in a limited capacity.
Players can use ABS to challenge umpire calls, but umpires are still making the vast majority of calls.
If it is eventually fully adopted league-wide, umpires would no longer make ball/strike judgment calls, 
and the call would be determined entirely by whether the pitch crossed the strike zone, with no human error from umpires. 
The skill of catcher framing and manipulating umpire calls would basically become worthless. However, for the near future, it
seems like there will be a hybrid between ABS and human umpires making strike/ball decisions.

This project compares the value of catcher framing in 2025 and 2026 to investigate: **How much of catcher framing value under traditional umpiring is eliminated by the current introduction of ABS?** 

## Data

- Statcast pitch-level data, 2025 full season and 2026 season through
  2026-08-16 (~625,000 pitches after filtering).
- Batter-specific strike zone bounds (`sz_top`, `sz_bot`) used to construct
  a ground-truth automated-zone indicator for every pitch.
- Catchers with fewer than 100 total pitches were excluded (141 qualified
  catchers).

## Methodology

Four hierarchical Bayesian logistic models were fit with PyMC (NUTS via
`nutpie`), each refining the research question:

| Model | Outcome | What it isolates |
|---|---|---|
| **1 — Baseline framing** | `called_strike` | Catcher effect on strike probability, controlling for pitch location (quadratic in plate x/z), velocity, movement, count, pitch type, and handedness. |
| **2/3 — ABS disagreement** | `umpire_disagreement` | Catcher effect on the probability an umpire's call differs from ABS|
| **4 — Directional framing value** | `called_strike` | Catcher effect on strike probability, controlling directly for the *true* automated-zone outcome (`automated_strike`, built from real `sz_top`/`sz_bot`). This is the model used for the final results below|

Framing value is converted from log-odds to runs using a standard
sabermetric constant (~0.125 runs per called strike), and each catcher's
resulting `framing_runs` is equivalent to the number of runs
that channel would eliminate under a fully automated zone.

## Key Results (2026 season, through 2026-08-16)

- **League-wide net total: ~6.2 framing runs.** This is close to 0, indicating
  that gains for catchers who are good at framing are largely offset by the loss for
  those who are poor. ABS is close to value-neutral for the sport in aggregate.
- **Total value in motion (sum of absolute values): ~52 runs** across 141
  catchers (~61 runs in the fuller 2025 season). If ABS fully took over human umpires,
  this is the framing value at risk of being eliminated or redistributed.
- Framing value is **highly unevenly distributed**: a handful of elite
  and poor framers account for a disproportionate share of the total,
  while the median catcher's framing value is close to zero. Most catchers hover around zero.
- Findings were corroborated across three independent model
  specifications (location-based, ABS-disagreement, and ABS-directional).

See `bayesian_model_4_framing_value_2025_2026.csv` for the full per-catcher breakdown.
See `figures/bayesian_model_4_framing_value.png` for the plot showing the change in catcher effect of all eligible catchers from 2025 to 2026.
You can search the catcher ID on Baseball-Savants to find a catcher's name.

## Conclusion
Overall, I can conclude that the introduction of ABS in 2026 has not eliminated or significantly affected the overall value of catcher framing.
Some players who might have struggled under the old system prosper under the new ABS, while some players are the opposite or
just remain the same. This phenomenon occurs in many settings as people have to adapt to new changes in environments. I also did not factor in things like injuries, age, and league experience that could affect catchers and their framing abilities between seasons. The total value
in motion is only ~52 across an entire season full of 2430 games, indicating that catcher framing is not very valuable in general.
If ABS were to fully eliminate human umpiring, then of course the art of catcher framing would be obsolete.

## AI-Assisted Development
I used AI to help build, debug, and refine the Bayesian models, while independently evaluating modeling decisions and interpreting results. This process strengthened my skills in Bayesian modeling, statistical inference, and applied machine learning.
