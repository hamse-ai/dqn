# Hyperparameter Experiment Results

**Member Name:** Amaliza

**Environment:** `ALE/AirRaid-v5`

## How I ran my experiments

I ran 10 experiments, each with a different combination of hyperparameters, using our group's `train.py`. I defined all 10 runs in `run_amaliza.sh`, so anyone can repeat them with one command. My models, metrics and plots are in `Amaliza/expXX/` (`dqn_model.pth`, `metrics.csv`, `plot.png`, `summary.json`).

I kept these settings the same in every run, so that only the six hyperparameters I was testing changed the result: 300,000 environment steps, a replay buffer of 50,000 transitions, learning starting after 5,000 steps, one update every 4 steps, a target network update every 2,000 steps, and seed 0. For learning, rewards are clipped to their sign; the scores I report are the real game scores.

My laptop has no GPU and not enough RAM for the 50,000-transition replay buffer, so I trained on free cloud GPUs (NVIDIA T4). I ran experiment 1 on **Google Colab**. Colab's free GPU then ran out, so I moved to **Kaggle** and ran the other nine experiments (2–10) there. Both use the same code, settings and seed, and each run took about 12–15 minutes.

**How I read the numbers:**
- *Start avg* is the average score of the first 20 episodes. Epsilon is close to 1.0 there, so this is roughly what **random play** scores (about 450–620).
- *Best avg20* is the best 20-episode average during training. This is when `dqn_model.pth` was saved.
- *End avg* is the average score of the last 20 episodes, when the agent mostly plays greedily (epsilon around 0.05–0.15).
- I say an agent **learned** if its scores at the end are clearly above random play.

## Results

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Start avg | Best avg20 | End avg | Best episode | Observed behaviour |
|-----|---------------|-------|------------|---------------|-------------|-----------------------|-----------|------------|---------|--------------|--------------------|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 578 | 1295 | **1295** | 3325 | My baseline, and it learned steadily. Its scores improved in every third of training (454 -> 612 -> 754), and its best 20-episode average came at the very end, so it was still improving when training stopped. Episodes more than doubled in length (552 -> 1194 steps), meaning the agent learned to survive much longer. It had the highest final average of all my runs. |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 476 | 655 | 203 | 1825 | Did not learn. Its best average came early (step ~33k), while it was still mostly playing randomly. After that, scores fell to about 200, worse than random play, and episodes got shorter (484 -> 349 steps). I think the learning rate was too high, so the Q-values became unstable. |
| 3 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 591 | 773 | 326 | 2150 | Did not learn. I saw the same pattern as Exp 2: a peak at step ~17k (still the random phase), then a steady decline to about 330. The learning rate was still too high. |
| 4 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 595 | 819 | 324 | 2950 | Unstable. Scores dropped in the middle of training, recovered in the last third (average 529) and reached a best average of 819 at step ~233k, but ended low again (324). It learned something, but not reliably with batch 32. |
| 5 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 462 | 839 | 790 | 2375 | Learned slowly but steadily. The last third of training averaged 660 (vs 476 in the first third), and the final average (790) stayed close to its best. Episodes got longer (493 -> 736 steps), so the agent survived longer. One of my most stable runs. |
| 6 | 0.00001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | 542 | 796 | 290 | 1925 | Did not learn. Its best average came at step ~25k, during the random phase. The learning rate was so small that the network barely changed in 300k steps, so once epsilon dropped and the agent followed its own Q-values, it played worse than random. |
| 7 | 0.00025 | 0.99 | 64 | 1.0 | 0.05 | 100000 | 619 | 1321 | 986 | 3800 | Learned well. Scores rose through training (thirds: 517 -> 858 -> 991) and episodes got longer (585 -> 802 steps). It has the same learning rate as Exp 4 but batch 64, and it was much better and more stable: a bigger batch gives a less noisy gradient. |
| 8 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 50000 | 606 | **1803** | 1009 | **4350** | **My best run.** It was flat for the first ~230k steps, then rose strongly: the 20-episode average climbed to 1,803 at step ~276k, with a best single game of 4,350 and episodes of up to ~2,900 steps. It has the same learning rate as Exp 5, but epsilon decays twice as fast, so the agent spent more of training using (and correcting) its own policy. |
| 9 | 0.0005 | 0.95 | 32 | 1.0 | 0.05 | 100000 | 566 | 575 | 171 | 1800 | **My worst run.** It never beat random play: the average fell steadily to 171 and episodes got shorter (555 -> 245 steps). A high learning rate together with a lower gamma (a shorter-sighted agent) did even worse than Exp 3. |
| 10 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 100000 | 569 | 1434 | 680 | 3300 | Learned. Its best average was 1,434 at step ~198k, and scores improved through training (thirds: 527 -> 570 -> 755). Compared with Exp 4 (same settings but epsilon end 0.05), keeping 10% exploration made learning clearly better and more stable. |

## My best and worst configuration

- **Best: Exp 8** (lr 0.00005, gamma 0.99, batch 32, epsilon 1.0 -> 0.05 with decay 50,000). It reached my highest 20-episode average (**1,803**) and my best single game (**4,350**). Its peak average is about 3x what random play scores, and it still ended at **1,009**. I believe the small learning rate kept the updates stable, while the faster epsilon decay gave the agent more time to learn from its own (greedy) play.
- **Worst: Exp 9** (lr 0.0005, gamma 0.95). It never did better than random play and ended at **171**. The large updates made the Q-values unstable, and the lower gamma made the agent care less about future rewards, which does not help in a game where surviving longer is what earns points.

## What helped and what hurt

- **The learning rate was the most important setting.** With batch 32 and the default epsilon, I found:
  - Too high (0.001 and 0.0005, Exp 2 and 3): performance collapsed below random play.
  - 0.00025 (Exp 4): unstable, with ups and downs.
  - 0.0001 (Exp 1): steady improvement, still rising at the end (final average 1,295).
  - 0.00005 (Exp 5): slow but steady improvement.
  - Too low (0.00001, Exp 6): the network barely learned in 300k steps.
- **A bigger batch helped:** at lr 0.00025, batch 64 (Exp 7, end avg 986) did far better than batch 32 (Exp 4, end avg 324). Averaging the loss over more experiences makes each update less noisy.
- **A faster epsilon decay helped:** at lr 0.00005, decay 50,000 (Exp 8, best avg 1,803) beat decay 100,000 (Exp 5, best avg 839). With the slower decay, the agent was still acting randomly for a large part of training.
- **Keeping a bit more exploration helped:** at lr 0.00025, epsilon end 0.1 (Exp 10, best avg 1,434) beat epsilon end 0.05 (Exp 4, best avg 819).
- **A lower gamma hurt:** gamma 0.95 with lr 0.0005 (Exp 9) was my worst run, even worse than the same learning rate with gamma 0.99 (Exp 3).
- **Episode length followed the reward:** in the runs that learned (1, 5, 7, 8, 10), episodes got longer because the agent survived longer. In the runs that failed (2, 3, 6, 9), episodes got shorter.

## My final configuration and why I chose it

**lr 0.00005, gamma 0.99, batch 32, epsilon 1.0 -> 0.05 with decay 50,000 (Exp 8).** It had the best 20-episode average and the best single game of my 10 runs, and it improved strongly in the last part of training. Its saved `dqn_model.pth` (taken at its best average) is my strongest model for `play.py`.

Exp 1 (lr 0.0001) is a close alternative: it had the highest *final* average (1,295) and was still improving when training stopped, so with more training steps it might catch up.

**What I would keep in mind:** 300,000 steps is short for Atari, and Air Raid scores vary a lot from one game to the next, so I would not over-interpret small differences between runs. The large gaps (for example Exp 8 vs Exp 9) are the conclusions I trust most. My next step would be to combine my best findings: a low learning rate (0.00005–0.0001) with batch 64, the faster epsilon decay, and more training steps.
