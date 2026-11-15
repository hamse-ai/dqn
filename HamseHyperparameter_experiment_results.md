# Hamse — Hyperparameter Experiment Results

## Setup

All ten runs used Air Raid (`ALE/AirRaid-v5`) and a 300,000-step training target. The run loop finishes the current episode before stopping, so recorded totals can be slightly above 300,000 steps. Learning rate, gamma, and batch size were held fixed; these experiments compare epsilon settings.

The run script selects the best model by the highest 20-episode moving-average reward. The table includes that peak moving average, the final 20-episode average, and the best individual episode reward to show both peak and end-of-run behavior.

## Results

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 20,000 | Highest peak 20-episode average: 2,626.25; final average: 1,271.25; best episode: 4,225. Performance declined after its peak. |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Peak average: 1,753.75; final average: 1,571.25; best episode: 3,600. End average was closer to its peak than Experiment 1. |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 200,000 | Peak average: 1,893.75; final average: 1,802.50; best episode: 4,650. Highest final average among the ten runs. |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 300,000 | Peak and final averages were both 1,202.50; best episode: 2,425. Lowest best episode reward among the ten runs. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Peak average: 1,427.50; final average: 1,217.50; best episode: 4,000. |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Peak average: 1,435.00; final average: 716.25; best episode: 3,625. Final average was substantially below its peak. |
| 7 | 0.0001 | 0.99 | 32 | 1.0 | 0.2 | 100,000 | Lowest peak average: 973.75; final average: 717.50; best episode: 2,575. |
| 8 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | Peak average: 1,841.25; final average: 1,552.50; best individual episode: 4,875, the highest single-episode score. |
| 9 | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | 20,000 | Peak average: 1,243.75; final average: 867.50; best episode: 3,275. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 50,000 | Peak average: 1,580.00; final average: 1,220.00; best episode: 2,850. |

## Findings

- By peak 20-episode average, Experiment 1 performed best (2,626.25). Its final average fell to 1,271.25, so this was a strong peak but not a sustained result.
- Experiment 3 had the strongest final average (1,802.50) and a peak average of 1,893.75, making it the steadier candidate based on these recorded summaries.
- Experiment 7 had the lowest peak average (973.75). Experiment 8 achieved the highest single-episode score (4,875), but its peak moving average was 1,841.25.
- These runs do not establish the effect of learning rate, gamma, or batch size because those values were identical in every experiment. They isolate changes to the epsilon schedule.

## Run artifacts

Each folder contains its episode metrics, reward plot, and summary:

| Experiment | Summary | Metrics | Plot |
|---:|---|---|---|
| 1 | [summary](Hamse/exp01/summary.json) | [metrics](Hamse/exp01/metrics.csv) | [plot](Hamse/exp01/plot.png) |
| 2 | [summary](Hamse/exp02/summary.json) | [metrics](Hamse/exp02/metrics.csv) | [plot](Hamse/exp02/plot.png) |
| 3 | [summary](Hamse/exp03/summary.json) | [metrics](Hamse/exp03/metrics.csv) | [plot](Hamse/exp03/plot.png) |
| 4 | [summary](Hamse/exp04/summary.json) | [metrics](Hamse/exp04/metrics.csv) | [plot](Hamse/exp04/plot.png) |
| 5 | [summary](Hamse/exp05/summary.json) | [metrics](Hamse/exp05/metrics.csv) | [plot](Hamse/exp05/plot.png) |
| 6 | [summary](Hamse/exp06/summary.json) | [metrics](Hamse/exp06/metrics.csv) | [plot](Hamse/exp06/plot.png) |
| 7 | [summary](Hamse/exp07/summary.json) | [metrics](Hamse/exp07/metrics.csv) | [plot](Hamse/exp07/plot.png) |
| 8 | [summary](Hamse/exp08/summary.json) | [metrics](Hamse/exp08/metrics.csv) | [plot](Hamse/exp08/plot.png) |
| 9 | [summary](Hamse/exp09/summary.json) | [metrics](Hamse/exp09/metrics.csv) | [plot](Hamse/exp09/plot.png) |
| 10 | [summary](Hamse/exp10/summary.json) | [metrics](Hamse/exp10/metrics.csv) | [plot](Hamse/exp10/plot.png) |
