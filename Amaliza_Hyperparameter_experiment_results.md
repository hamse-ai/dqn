# Hyperparameter Experiment Results

**Member Name:** Amaliza

**Environment:** `ALE/AirRaid-v5`
**Fixed settings for all 10 runs (from `train.py`):** 300,000 environment steps, replay buffer 50,000, learning starts after 5,000 steps, one update every 4 steps, target network update every 2,000 steps, seed 0, rewards clipped to their sign for learning (the logged reward is the real game score).

All runs were started with `run_amaliza.sh`. Models, metrics and plots are in `Amaliza/expXX/` (`dqn_model.pth`, `metrics.csv`, `plot.png`, `summary.json`). My laptop has no GPU and too little RAM for the 50,000-transition buffer, so the runs were trained on free cloud T4 GPUs: Exp 1 on Google Colab, Exp 2–10 on Kaggle (same code, same settings, same seed, ~12–16 minutes per run).

**How to read the numbers:**
- *Start avg* = average score of the first 20 episodes. Epsilon is close to 1.0 there, so this is roughly what **random play** scores (about 450–620).
- *Best avg20* = the best 20-episode average during training (this is when `dqn_model.pth` was saved).
- *End avg* = average score of the last 20 episodes, when the agent mostly plays greedily (epsilon ≈ 0.05–0.15).
- An agent has **learned** if its scores at the end are clearly above random play.

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Start avg | Best avg20 | End avg | Best episode | Observed behaviour |
|-----|---------------|-------|------------|---------------|-------------|-----------------------|-----------|------------|---------|--------------|--------------------|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 578 | 1295 | **1295** | 3325 | Learned steadily (baseline). Scores improved in every third of training (454 -> 612 -> 754) and its best 20-episode average came at the very end of training, so it was still improving when training stopped. Episodes more than doubled in length (552 -> 1194 steps): the agent learned to survive much longer. Highest final average of all 10 runs. |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 476 | 655 | 203 | 1825 | Did not learn. Its best average came early (step ~33k) while it was still mostly random, then scores fell to about 200, which is worse than random play. Episodes also got shorter (484 -> 349 steps). The learning rate is too high, so the Q-values become unstable. |
| 3 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 591 | 773 | 326 | 2150 | Did not learn. Same pattern as Exp 2: peak at step ~17k (random phase), then a steady decline to about 330. Still too high a learning rate. |
| 4 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 595 | 819 | 324 | 2950 | Unstable. Scores dropped in the middle of training, recovered in the last third (average 529), reaching a best average of 819 at step ~233k, but ended low again (324). It learns something, but not reliably at batch 32. |
| 5 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 462 | 839 | 790 | 2375 | Learned slowly but steadily. The last third of training averaged 660 (vs 476 in the first third) and the final average (790) stayed close to its best. Episodes got longer (493 -> 736 steps): the agent survives longer. One of the most stable runs. |
| 6 | 0.00001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 542 | 796 | 290 | 1925 | Did not learn. Best average came at step ~25k (random phase). The learning rate is so small that the network barely changes in 300k steps, so when epsilon drops and the agent follows its own Q-values, it plays worse than random. |
| 7 | 0.00025 | 0.99 | 64 | 1.0 | 0.05 | 100000 | 619 | 1321 | 986 | 3800 | Learned well. Scores rose through training (thirds: 517 -> 858 -> 991) and episodes got longer (585 -> 802 steps). Same learning rate as Exp 4 but batch 64, and it is much better and more stable: a bigger batch gives a less noisy gradient. |
| 8 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 50000 | 606 | **1803** | **1009** | **4350** | **Best run.** Flat for the first ~230k steps, then a strong rise: the 20-episode average climbed to 1,803 at step ~276k, with a best single game of 4,350 and episodes up to ~2,900 steps. Same learning rate as Exp 5, but epsilon decays twice as fast, so the agent spends more of training using (and correcting) its own policy. |
| 9 | 0.0005 | 0.95 | 32 | 1.0 | 0.05 | 100000 | 566 | 575 | 171 | 1800 | **Worst run.** Never beat random play; the average fell steadily to 171 and episodes got shorter (555 -> 245 steps). A high learning rate plus a lower gamma (shorter-sighted agent) was even worse than Exp 3 alone. |
| 10 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 100000 | 569 | 1434 | 680 | 3300 | Learned. Best average 1,434 at step ~198k and scores improved through training (thirds: 527 -> 570 -> 755). Compared with Exp 4 (same settings, epsilon end 0.05), keeping 10% exploration made learning clearly better and more stable. |

## Best and worst configuration

- **Best: Exp 8** (lr 0.00005, gamma 0.99, batch 32, epsilon 1.0 -> 0.05 with decay 50,000). It reached the highest 20-episode average (**1,803**), and the best single game (**4,350**). Its peak average is about 3x what random play scores, and it still ended at **1,009**. A small learning rate keeps updates stable, and the faster epsilon decay gives the agent more time to learn from its own (greedy) play.
- **Worst: Exp 9** (lr 0.0005, gamma 0.95). It never did better than random play and ended at **171**. Large updates made the Q-values unstable, and the lower gamma made the agent value future rewards less, which does not help in a game where surviving longer is what earns points.

## What helped and what hurt

- **Learning rate was the most important setting.** At batch 32 with the default epsilon:
  - Too high (0.001, 0.0005 in Exp 2, 3) -> performance collapsed below random play.
  - 0.00025 (Exp 4) -> unstable, ups and downs.
  - 0.0001 (Exp 1) -> steady improvement, still rising at the end (final average 1,295).
  - 0.00005 (Exp 5) -> slow but steady improvement.
  - Too low (0.00001, Exp 6) -> the network barely learned in 300k steps.
- **Bigger batch helped:** at lr 0.00025, batch 64 (Exp 7, end avg 986) was far better than batch 32 (Exp 4, end avg 324). Averaging the loss over more experiences makes each update less noisy.
- **Faster epsilon decay helped:** at lr 0.00005, decay 50,000 (Exp 8, best avg 1,803) beat decay 100,000 (Exp 5, best avg 839). With the slower decay, the agent was still acting randomly for a large part of training.
- **Keeping a bit more exploration helped:** at lr 0.00025, epsilon end 0.1 (Exp 10, best avg 1,434) beat epsilon end 0.05 (Exp 4, best avg 819).
- **Lower gamma hurt:** gamma 0.95 with lr 0.0005 (Exp 9) was the worst run, even worse than the same learning rate with gamma 0.99 (Exp 3).
- **Episode length follows reward:** in the runs that learned (1, 5, 7, 8, 10), episodes got longer because the agent survived longer; in the runs that failed (2, 3, 6, 9), episodes got shorter.

## Repeat run: how reliable are these results?

Before the Kaggle runs, I had already run all 10 experiments once on Google Colab with exactly the same settings and seed. GPU training is not perfectly deterministic, so the two runs give different numbers. Exp 1 in the table above is the Colab run; this table compares Exp 2–10.

| Exp | Change tested | Kaggle best avg20 / end avg (table above) | Colab repeat best avg20 / end avg | Same conclusion? |
|-----|---------------|---------------------------|-----------------------------|------------------|
| 2 | lr 0.001 | 655 / 203 | 668 / 230 | Yes: did not learn |
| 3 | lr 0.0005 | 773 / 326 | 616 / 231 | Yes: did not learn |
| 4 | lr 0.00025 | 819 / 324 | 823 / 818 | Partly: same peak, but the Colab run ended well |
| 5 | lr 0.00005 | 839 / 790 | 1155 / 920 | Yes: slow, steady learning |
| 6 | lr 0.00001 | 796 / 290 | 635 / 248 | Yes: did not learn |
| 7 | lr 0.00025, batch 64 | 1321 / 986 | **1671** / 780 | Yes: one of the two best runs |
| 8 | lr 0.00005, decay 50,000 | **1803** / **1009** | 1619 / 783 | Yes: one of the two best runs |
| 9 | lr 0.0005, gamma 0.95 | 575 / 171 | 729 / 201 | Yes: worst / did not learn |
| 10 | lr 0.00025, eps end 0.1 | 1434 / 680 | 1186 / 813 | Yes: learned |

**What the repeat shows:**
- The **big effects repeat**: a too-high learning rate (Exp 2, 3, 9) or too-low one (Exp 6) failed both times, and Exp 7 and Exp 8 were the two best runs both times.
- The **small differences are noise**: Exp 7 beat Exp 8 on Colab, and Exp 8 beat Exp 7 on Kaggle. Exp 4 ended at 324 on Kaggle but 818 on Colab, which shows how unstable lr 0.00025 with batch 32 is. In the Colab repeat, batch 64 (Exp 7) and epsilon end 0.1 (Exp 10) did not improve Exp 4's *final* average, only its peak (1,671 and 1,186 vs 823), so those two effects are less certain than the learning-rate effect.
- So the reliable lessons are: **keep the learning rate low (0.00005–0.0001)**, and **a bigger batch or a faster epsilon decay helps the agent reach higher scores**.

## Final configuration and justification

**lr 0.00005, gamma 0.99, batch 32, epsilon 1.0 -> 0.05 with decay 50,000 (Exp 8).** It had the best 20-episode average and the best single game of my 10 runs, and it improved strongly in the last part of training. Its saved `dqn_model.pth` (taken at its best average) is my strongest model for `play.py`.

Exp 1 (lr 0.0001) is a close alternative: it had the highest *final* average (1,295) and was still improving when training stopped, so with more training steps it might catch up.

**Caveat:** 300,000 steps is short for Atari, and Air Raid scores are very noisy from one game to the next. The repeat run above shows that small differences between runs should not be over-interpreted; the large gaps (for example Exp 8 vs Exp 9) are the reliable conclusions. A natural next step would be combining the best ideas: a low learning rate (0.00005–0.0001) with batch 64, the faster epsilon decay, and more training steps.
