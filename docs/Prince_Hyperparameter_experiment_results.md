# Hyperparameter Experiment Results

**Member Name:** Prince

**Environment:** `ALE/AirRaid-v5`

## How I ran my experiments

I ran 10 experiments, each with a different combination of hyperparameters, using our group's `train.py`. I defined all 10 runs in `run_prince.sh`, so anyone can repeat them with one command. My models, metrics and plots are in `Prince/expXX/` (`dqn_model.pth`, `metrics.csv`, `plot.png`, `summary.json`).

My focus was **gamma** and **batch size**. Amaliza's experiments covered the learning rate and epsilon settings, so I kept those at our baseline values (lr 0.0001, epsilon 1.0 -> 0.05, decay 100,000). Between us, the group tested all six hyperparameters.

I kept these settings the same in every run, so that only the hyperparameters I was testing changed the result:

| Fixed setting | Value |
|---------------|-------|
| Training length | 300,000 environment steps |
| Replay buffer size | 50,000 transitions |
| Learning starts after | 5,000 steps |
| Network update | every 4 steps |
| Target network update | every 2,000 steps |
| Random seed | 0 |
| Reward used for learning | clipped to its sign (scores I report are the real game scores) |

I trained on my MacBook Pro (Apple M2):

| Experiments | Where I ran them | Time per run |
|-------------|------------------|--------------|
| 1–5 | M2 CPU | about 85–92 minutes (Exp 3 and 4 took longer because the laptop was in use or asleep during those runs) |
| 6–10 | M2 GPU (Apple MPS) | about 11–20 minutes |

After Exp 5, I changed `train.py` to use the Mac GPU. This only changes the speed: the code, settings and seed are the same.

**How I read the numbers:**

| Column | Meaning |
|--------|---------|
| Start avg | Average score of the first 20 episodes. Epsilon is close to 1.0 there, so this is roughly what **random play** scores (about 520–720). |
| Best avg20 | Best 20-episode average during training. This is when `dqn_model.pth` was saved. |
| End avg | Average score of the last 20 episodes, when the agent mostly plays greedily (epsilon around 0.05–0.10). |
| Best episode | Highest score in a single game. |

I say an agent **learned** if its scores at the end are clearly above random play.

## Results

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Start avg | Best avg20 | End avg | Best episode | Observed behaviour |
|-----|---------------|-------|------------|---------------|-------------|-----------------------|-----------|------------|---------|--------------|--------------------|
| 1 | 0.0001 | 0.90 | 32 | 1.0 | 0.05 | 100000 | 621 | 695 | **300** | 1975 | **My worst run.** Did not learn. Its best average came at step ~17k, while it was still playing randomly. After that, scores dropped (thirds: 499 -> 426 -> 448) and ended at 300, well below random play. Episodes got shorter (613 -> 256 steps). With gamma 0.90 the agent only looks about 10 steps ahead, which is too short-sighted for Air Raid. |
| 2 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 100000 | 550 | 759 | 529 | 2150 | Barely learned. Scores dipped in the middle of training (thirds: 477 -> 363 -> 441) and ended around random play (529). Its best average came late (step ~287k), so it was slowly starting to improve, but gamma 0.95 was still too low to learn much in 300k steps. Episodes got shorter (566 -> 389 steps). |
| 3 | 0.0001 | 0.97 | 32 | 1.0 | 0.05 | 100000 | 606 | 885 | 708 | 2425 | Learned a little. This is the first gamma where scores rose in every third of training (461 -> 574 -> 636), and it ended above random play (708). The improvement was small but steady. |
| 4 | 0.0001 | 0.995 | 32 | 1.0 | 0.05 | 100000 | 665 | 1552 | 1052 | 3475 | Learned well, after a slow start. Scores dropped in the middle of training (thirds: 537 -> 353 -> 1000), then rose strongly in the last third to a best average of 1,552 at step ~273k. Episodes got much longer (652 -> 1001 steps), so the agent learned to survive. My best gamma with batch 32. |
| 5 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100000 | 599 | 1276 | **1276** | 4450 | Learned, and was still improving when training stopped: its best average came at the very end (step ~301k). Scores rose through training (thirds: 484 -> 512 -> 896) and episodes more than doubled (573 -> 1336 steps). Slower to improve than Exp 4, because with gamma this high the targets depend on rewards far in the future, which take longer to learn. |
| 6 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 100000 | 535 | 719 | 565 | 2700 | Did not learn much. Scores dipped in the middle (thirds: 482 -> 348 -> 495) and ended around random play (565). With only 16 experiences per update, each gradient step is very noisy, so learning was unstable. |
| 7 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 100000 | 602 | **1975** | **1378** | 3925 | **My best run.** Scores rose strongly through training (thirds: 485 -> 634 -> 1139), reaching a best 20-episode average of **1,975** at step ~267k, about 3x random play. It also had my highest final average (1,378). Episodes nearly doubled in length (604 -> 1126 steps). Batch 64 gave smoother gradients than 16 or 32. |
| 8 | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 100000 | 721 | 1751 | 1019 | 3350 | Learned well. It had the highest last-third average of all my runs (thirds: 556 -> 826 -> 1236) and a best average of 1,751 at step ~236k. It dropped to 1,019 by the end, and each update took longer to compute than with batch 64, so overall batch 64 was the better choice. |
| 9 | 0.0001 | 0.95 | 64 | 1.0 | 0.05 | 100000 | 622 | 851 | 780 | 2550 | Learned a little. Scores rose slowly (thirds: 530 -> 597 -> 650) and ended at 780. Compared with Exp 2 (same gamma, batch 32), batch 64 made it better (end avg 529 -> 780), but the low gamma still held it back: it was far behind Exp 7 (same batch, gamma 0.99). |
| 10 | 0.0001 | 0.995 | 128 | 1.0 | 0.05 | 100000 | 518 | 1326 | 1272 | **5200** | Learned steadily, and was one of my most stable runs: scores rose in every third (634 -> 862 -> 998) and the final average (1,272) stayed close to its best (1,326). It had my **best single game (5,200)** and the longest episode (3,434 steps). Its best average was lower than Exp 4 (same gamma, batch 32), but it ended higher (1,052 -> 1,272). |

The training curves (reward and episode length) for each run are in `Prince/expXX/plot.png`.

## My best and worst configuration

| | Exp | Learning rate | Gamma | Batch | Epsilon decay | Best avg20 | End avg | Best episode |
|---|-----|---------------|-------|-------|---------------|------------|---------|--------------|
| **Best** | 7 | 0.0001 | 0.99 | 64 | 100,000 | **1,975** | **1,378** | 3,925 |
| **Worst** | 1 | 0.0001 | 0.90 | 32 | 100,000 | 695 | **300** | 1,975 |

- **Best: Exp 7** (lr 0.0001, gamma 0.99, batch 64, epsilon 1.0 -> 0.05 with decay 100,000). It reached my highest 20-episode average (**1,975**, about 3x random play) and my highest final average (**1,378**). Gamma 0.99 makes the agent value future rewards enough to learn to survive, and batch 64 averages the loss over enough experiences to keep updates stable, while still being fast to train.
- **Worst: Exp 1** (gamma 0.90). It never did better than random play after the early random phase and ended at **300**. With gamma 0.90, a reward 20 steps away is worth only about 12% of its value, so the agent barely cares about the future. In Air Raid, points come from surviving and defending buildings over a long time, so a short-sighted agent does badly.

## What helped and what hurt

**Gamma was the most important setting in my experiments.** With lr 0.0001, batch 32 and the default epsilon:

| Gamma | Exp | Best avg20 | End avg | What I saw |
|-------|-----|------------|---------|------------|
| 0.90 | 1 | 695 | 300 | Too low: fell below random play |
| 0.95 | 2 | 759 | 529 | Too low: stayed around random play |
| 0.97 | 3 | 885 | 708 | Small, steady improvement |
| 0.99 | Amaliza's Exp 1 (baseline) | 1,295 | 1,295 | Steady improvement |
| 0.995 | 4 | **1,552** | 1,052 | Best with batch 32, after a slow start |
| 0.999 | 5 | 1,276 | **1,276** | Good, but slower to improve; still rising at the end |

Raising gamma from 0.90 to 0.995 improved results almost every step of the way. Above that (0.999), learning got slower, because the agent has to estimate rewards very far in the future.

**Batch size** (with lr 0.0001, gamma 0.99 and the default epsilon):

| Batch size | Exp | Best avg20 | End avg | What I saw |
|------------|-----|------------|---------|------------|
| 16 | 6 | 719 | 565 | Too noisy: barely learned |
| 32 | Amaliza's Exp 1 (baseline) | 1,295 | 1,295 | Steady improvement |
| 64 | 7 | **1,975** | **1,378** | Best overall |
| 128 | 8 | 1,751 | 1,019 | Good, but slower per update and dropped more at the end |

**Changing one setting at a time** (each pair differs in only one hyperparameter):

| Hyperparameter | Compared | Result | Effect | Why I think it happened |
|----------------|----------|--------|--------|-------------------------|
| Gamma | Exp 1 (0.90) vs Exp 4 (0.995) | Best avg 695 -> **1,552** | Helped | A far-sighted agent values surviving longer, which is what earns points. |
| Batch size | Exp 6 (16) vs Exp 7 (64) | Best avg 719 -> **1,975** | Helped | Averaging the loss over more experiences makes each update less noisy. |
| Batch size | Exp 2 (32) vs Exp 9 (64), both gamma 0.95 | End avg 529 -> **780** | Helped | Same reason, but the low gamma still limited how much it could learn. |
| Gamma | Exp 9 (0.95) vs Exp 7 (0.99), both batch 64 | Best avg 851 -> **1,975** | Helped | Even with a good batch size, a short-sighted agent learns much less. |
| Batch size | Exp 4 (32) vs Exp 10 (128), both gamma 0.995 | Best avg 1,552 -> 1,326, end avg 1,052 -> **1,272** | Mixed | The bigger batch gave a lower peak but more stable learning. |

**Episode length followed the reward:** in the runs that learned (4, 5, 7, 8, 10), episodes got much longer because the agent survived longer. In the runs that failed (1, 2), episodes got shorter.

## My final configuration and why I chose it

**lr 0.0001, gamma 0.99, batch 64, epsilon 1.0 -> 0.05 with decay 100,000 (Exp 7).** It had the best 20-episode average and the best final average of my 10 runs, and it improved strongly in the last part of training. Its saved `dqn_model.pth` (taken at its best average) is my strongest model for `play.py`.

Exp 10 (gamma 0.995, batch 128) is a close alternative: it was the most stable of my good runs and had my best single game (5,200). Exp 5 (gamma 0.999) was still improving when training stopped, so with more training steps it might catch up.

**What I would keep in mind:** 300,000 steps is short for Atari, and Air Raid scores vary a lot from one game to the next, so I would not over-interpret small differences between runs. The large gaps (for example Exp 7 vs Exp 1) are the conclusions I trust most. My next step would be to combine my findings with Amaliza's: gamma 0.99–0.995 and batch 64, with her best learning rate and faster epsilon decay, and more training steps.
