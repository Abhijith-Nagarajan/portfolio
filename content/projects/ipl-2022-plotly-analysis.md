---
title: "IPL 2022 Analysis with Plotly"
description: "A storytelling EDA of the 2022 Indian Premier League season — visualizing toss-decision impact, powerplay performance per team, and batsmen / bowler effectiveness across the three phases of a T20 innings. Built entirely in Plotly (subplots, box plots, scatter quadrants) and published as a Medium article."
date: 2022-06-15
featured: false
weight: 60
domains: ["Data Visualization"]
externalUrl: "https://medium.com/p/11d755c91300"
tags:
  - "Python"
  - "pandas"
  - "numpy"
  - "Plotly"
  - "Data Visualization"
  - "EDA"
  - "Storytelling"
  - "Sports Analytics"
  - "Cricket"
  - "Jupyter"
  - "Data Science"
---

A self-directed exploration of the IPL 2022 season — partly to dig into a sport I follow closely, partly to get hands-on with **Plotly** specifically (subplots, hover-aware box plots, scatter-quadrant layouts) rather than my usual Matplotlib / Seaborn defaults. Published as a [Medium article](https://medium.com/p/11d755c91300) so the interactive charts actually render for readers.

## Analytical lens

The notebook works through seven dimensions, organized so the per-tournament patterns set the context before drilling into per-team specifics:

1. **Toss decision overall** — what did teams choose, and did it matter?
2. **Time of game start** — does the difference in evening vs day-night games change toss behavior?
3. **Matches batted and bowled per team** — split of how each franchise approached innings.
4. **Games won per decision per team** — does the toss decision predict wins differently for different sides?
5. **Powerplay analysis** — runs scored and wickets lost in overs 1–6, separately for batting-first and chasing.
6. **Batsmen analysis** — performance broken into three phases of a T20 innings (powerplay / middle overs / slog overs).
7. **Bowlers analysis** — same three-phase split, using a wickets-vs-economy scatter quadrant to surface effectiveness tiers.

## A few findings worth surfacing

- **Toss matters less than the narrative suggests.** In ~80% of games (59 of 74), the toss-winning team chose to field first — yet those teams won *fewer* than half of the matches they chose to chase. The toss is a behavioral signal more than a structural advantage.
- **Powerplay aggression separated the field.** DC and PBKS came out swinging — averaging ~9 runs/over in the first 6 overs across most of their innings. GT and RR were the steady accumulators. CSK / MI / RCB / SRH were heavily top-order dependent — when openers failed, the middle order struggled to recover.
- **Different teams had different chase profiles.** GT chased significantly better than the average; RR, LSG, and RCB had better records when batting first. The "always bowl first" heuristic doesn't survive a per-team breakdown.

## Plotly techniques used

- **`make_subplots`** for the per-team grids (2×5) so all ten franchises sit on a comparable scale in one figure
- **Box plots** for distribution-aware comparisons (powerplay scores per team, wickets lost in PP)
- **Bar charts with consistent per-team color encoding** — each franchise gets its actual brand color (`team_colors` dict) so visual continuity is preserved across the seven sections
- **Scatter with quadrant-style marker sizing** for the bowlers analysis — wickets vs. economy on the axes, marker size encoding the wickets × economy threshold combination (so the visual surface area highlights effective + economical bowlers)
- **`chart-studio` integration** so the interactive figures embed cleanly in the Medium write-up

## Where to read / fork

- **[Medium article](https://medium.com/p/11d755c91300)** — best place to read, since the Plotly figures are interactive
- **[Kaggle notebook](https://www.kaggle.com/code/itsabhijith/ipl-analysis-plotly-eda)** — source + charts, runnable in-browser
- **[GitHub repo](https://github.com/Abhijith-Nagarajan/IPL_2022_Analysis)** — the notebook in version control

## What I'd revisit

- **Multi-season comparison** — the same analytical lens applied across the last 5 seasons would surface whether DC/PBKS-style aggressive powerplay starts are stable team identities or one-season effects.
- **Player consistency indices** — current batsmen / bowler analysis is per-tournament. A consistency score (e.g., coefficient of variation on per-innings strike rate / economy) would distinguish "form-dependent" from "reliable" players.
- **Venue effects** — bake in toss × venue interactions. The 2022 season was played at a small number of grounds; some of the "toss decision didn't matter" effect may be venue confounded.

[Read on Medium →](https://medium.com/p/11d755c91300)
