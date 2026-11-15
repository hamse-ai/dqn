# Hyperparameter Experiment Results

**Member Name:** Hamse

**Environment:** `ALE/AirRaid-v5`

## Experimental procedure

Ten experiments were conducted using the group's `train.py`. The commands are listed in `run_hamse.sh`. Each run's model, episode metrics, plot, and summary are stored in `Hamse/expXX/`.

The experiments focus on the **epsilon schedule**. Learning rate, gamma, and batch size were held fixed; therefore, the results describe the observed performance of the tested epsilon configurations and do not measure the effects of changing the other three hyperparameters.

The following settings were held constant across all runs:

| Fixed setting | Value |
|---------------|-------|
| Training target | 300,000 environment steps |
| Replay buffer size | 50,000 transitions |
| Learning starts after | 5,000 steps |
| Network update | Every 4 steps |
| Target network update | Every 2,000 steps |
| Random seed | 0 |
| Reward used for learning | Clipped to its sign; reported scores use the game's actual reward |

The recorded duration was approximately 53–56 minutes per run. The summaries do not identify the hardware used. Runs may exceed 300,000 steps slightly because training stops after the current episode ends.

**Metric definitions:**

| Column | Meaning |
|--------|---------|
| Start avg | Average score of the first 20 episodes. |
| Best avg20 | Highest rolling average over 20 episodes during training; the model was saved when this value improved. |
| End avg | Average score of the final 20 episodes. |
| Best episode | Highest score in a single episode. |

## Results

Scores in the average columns are rounded to the nearest point. The three scores in each behavior description are averages from the first, middle, and final thirds of training.

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Start avg | Best avg20 | End avg | Best episode | Observed behaviour |
|-----|---------------|-------|------------|---------------|-------------|-----------------------|-----------|------------|---------|--------------|--------------------|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 20,000 | 425 | **2,626** | 1,271 | 4,225 | Mean rewards by training third were 404 → 285 → 1,281. The best avg20 occurred near 272k steps, followed by a decline. This configuration produced the highest peak average. |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | 602 | 1,754 | 1,571 | 3,600 | Mean rewards by third were 435 → 374 → 1,031, indicating a late recovery. The final average remained relatively close to the peak. |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 200,000 | 584 | 1,894 | **1,803** | 4,650 | Mean rewards by third were 545 → 502 → 1,057. The final average was the highest; the best avg20 occurred near 296k steps. |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 300,000 | 501 | 1,203 | 1,203 | 2,425 | Mean rewards increased modestly across thirds (576 → 607 → 740). Peak and final averages were equal. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | 741 | 1,428 | 1,218 | 4,000 | Mean rewards by third were 508 → 441 → 992, showing a mid-training decline and late recovery. The best single-episode score exceeded the final average substantially. |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | 572 | 1,435 | 716 | 3,625 | Mean rewards by third increased (524 → 585 → 841), while the final average was substantially below the peak, indicating an unstable ending. |
| 7 | 0.0001 | 0.99 | 32 | 1.0 | 0.2 | 100,000 | 691 | 974 | 718 | 2,575 | This configuration had the lowest peak average. Mean rewards changed little across thirds (566 → 541 → 635). |
| 8 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | 616 | 1,841 | 1,553 | **4,875** | Mean rewards by third were 449 → 413 → 999, showing late improvement. This configuration produced the highest single-episode score, although its final average was below its peak. |
| 9 | 0.0001 | 0.99 | 32 | 0.5 | 0.01 | 20,000 | 398 | 1,244 | 868 | 3,275 | This configuration had the lowest start average, followed by late improvement (310 → 345 → 844). The final average remained below the peak. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 50,000 | 602 | 1,580 | 1,220 | 2,850 | Mean rewards by third were 455 → 439 → 1,005, indicating a late improvement. The final average remained below the peak. |

## Best- and lowest-performing configurations

| | Exp | Learning rate | Gamma | Batch | Epsilon start → end | Epsilon decay | Best avg20 | End avg | Best episode |
|---|-----|---------------|-------|-------|---------------------|---------------|------------|---------|--------------|
| **Best peak** | 1 | 0.0001 | 0.99 | 32 | 1.0 → 0.05 | 20,000 | **2,626** | 1,271 | 4,225 |
| **Lowest peak** | 7 | 0.0001 | 0.99 | 32 | 1.0 → 0.2 | 100,000 | 974 | 718 | 2,575 |

- **Highest peak average: Exp 1.** It reached a 20-episode average of 2,626 and a best episode score of 4,225. Its final average fell to 1,271, so the peak was not sustained.
- **Strongest final performance: Exp 3.** Its final average (1,803) was the highest and remained close to its peak (1,894). This is the preferred configuration when prioritizing performance at the end of training.
- **Lowest peak average: Exp 7.** Its peak average was 974. Exp 6 had the lowest final average (716); Exp 7 finished at 718.

## Observed effects and limitations

**Epsilon decay showed no monotonic relationship with performance.** With epsilon start 1.0 and end 0.05, the 20,000-step decay in Exp 1 had the highest peak, while the 200,000-step decay in Exp 3 had the highest final average. The 300,000-step decay in Exp 4 had lower peak and final averages. Faster decay coincided with the strongest peak, while slower decay coincided with the stronger finish; these results do not establish that decay rate alone caused the differences.

**The tested epsilon-end values did not have a single clear winner.** At a 100,000-step decay, epsilon ends of 0.01, 0.1, and 0.2 produced peak averages of 1,428, 1,435, and 974. The 0.2 setting had the lowest peak in this group. There was no run with epsilon end 0.05 at the same 100,000-step decay, so the effect relative to that baseline cannot be isolated.

**The epsilon-start comparisons are limited.** Exp 8 and Exp 9 both started at 0.5 but used different epsilon-end and decay values. Their results compare complete schedules rather than epsilon start alone.

All ten runs used the same learning rate (0.0001), gamma (0.99), and batch size (32). No conclusions about the effects of changing those values can be drawn from these experiments.

## Selected configuration

**Learning rate 0.0001, gamma 0.99, batch size 32, epsilon 1.0 → 0.05 with decay 200,000 (Exp 3).** This configuration is selected because it had the highest final 20-episode average (1,803) and finished close to its best average (1,894). Exp 1 had the strongest peak, but its average declined considerably by the end.

The results are based on one seed per configuration, and Air Raid scores varied across episodes. Additional seeds and experiments varying learning rate, gamma, and batch size would be required to draw firm conclusions about those hyperparameters.

The model, metrics, plot, and summary for each run are in `Hamse/expXX/`:

| Exp | Run artifacts |
|-----|---------------|
| 1 | [summary](Hamse/exp01/summary.json), [metrics](Hamse/exp01/metrics.csv), [plot](Hamse/exp01/plot.png) |
| 2 | [summary](Hamse/exp02/summary.json), [metrics](Hamse/exp02/metrics.csv), [plot](Hamse/exp02/plot.png) |
| 3 | [summary](Hamse/exp03/summary.json), [metrics](Hamse/exp03/metrics.csv), [plot](Hamse/exp03/plot.png) |
| 4 | [summary](Hamse/exp04/summary.json), [metrics](Hamse/exp04/metrics.csv), [plot](Hamse/exp04/plot.png) |
| 5 | [summary](Hamse/exp05/summary.json), [metrics](Hamse/exp05/metrics.csv), [plot](Hamse/exp05/plot.png) |
| 6 | [summary](Hamse/exp06/summary.json), [metrics](Hamse/exp06/metrics.csv), [plot](Hamse/exp06/plot.png) |
| 7 | [summary](Hamse/exp07/summary.json), [metrics](Hamse/exp07/metrics.csv), [plot](Hamse/exp07/plot.png) |
| 8 | [summary](Hamse/exp08/summary.json), [metrics](Hamse/exp08/metrics.csv), [plot](Hamse/exp08/plot.png) |
| 9 | [summary](Hamse/exp09/summary.json), [metrics](Hamse/exp09/metrics.csv), [plot](Hamse/exp09/plot.png) |
| 10 | [summary](Hamse/exp10/summary.json), [metrics](Hamse/exp10/metrics.csv), [plot](Hamse/exp10/plot.png) |
