[![Play Demo](https://img.shields.io/badge/Play-Demo-blue?style=for-the-badge&logo=googlechrome)](https://katehuuh.github.io/curvecrash-RL/) [![Install Userscript](https://img.shields.io/badge/Userscript-green?style=for-the-badge&logo=tampermonkey)](https://raw.githubusercontent.com/Katehuuh/curvecrash-RL/main/CurveCrash-AI-v6.user.js)

curvecrash-RL is a reinforcement learning bot for experimentation on the [CurveCrash](https://curvecrash.com) environment (Achtung die Kurve), using PPO self-play and transfer learning, playable in browser. A 2.3M param CNN+GRU (IMPALA-CNN + Voronoi) with 3 discrete actions learns survival but not strategy. Breaking through probably needs a bigger model, continuous actions, or a different training algorithm.

**Demo**: open the link above, arrow keys to steer. **·** **Real site**: install [Tampermonkey](https://www.tampermonkey.net/), click the userscript badge, go to [curvecrash.com](https://curvecrash.com) and load `/models/v8.json`.

## Train your own

```bash
# Needs Python 3.10+, PyTorch, NumPy, Pygame.
python train_selfplay.py --arch impala --voronoi --gspp --num-envs 8 --total-timesteps 10000000 --checkpoint-dir checkpoints
python watch_agent_ffa.py --checkpoint checkpoints/agent_final.pt --gspp
```

## Files

`curvecrash_env_ffa.py` is the gym env, 512x512 arena with FFA and powerups. `train_selfplay.py` runs PPO with PFSP self-play. `experiments.py` has the model architectures (IMPALA-CNN, Voronoi wrapper). `watch_agent_ffa.py` is just a pygame viewer. `replay_pipeline.py` scrapes human replays and renders them into BC training data. `export_model.py` converts checkpoints to TF.js JSON. `CurveCrash-AI-v6.user.js` runs the model on the real site. `index.html` is the standalone browser demo.

<details>
<summary>11 versions trained, one real breakthrough (ego-centric rotation), the rest plateaued at ~65%</summary>

## What failed

| Version | What we tried | What happened |
|---------|--------------|---------------|
| v1 | NatureCNN 1.6M params, 256x256 obs | Entropy collapsed. Agent did clockwise circles forever. |
| v2 | Shrunk to 67K params, 128x128 | Stopped collapsing but survived 6s. Worse than playing with eyes closed. |
| v3 | Rotated obs so ego always faces right | 10x jump to 39s survival. The only change that actually mattered. |
| v4 | NatureCNN 669K, self-play pool | 24-29% WR. Camps territory but any human can box it in. |
| v5 | Added GRU(64), exploiter opponents | 33% WR vs v4. Picked up aggression, camping still loose. |
| v6 | GS++ powerups, GRU bumped to 128, CBAM | ~86s vs random. Works, nothing exciting. |
| v8 | IMPALA-CNN + Voronoi territory, 2.3M params | ~65% WR in self-play. Best we got. Survives but has zero strategy. |
| v9 | BC warmstart from 879K human replay frames | Wrecked the policy. 34% of actions became spins. PPO couldn't fix it because target_kl=0.015 kills updates after 1-2 epochs. We tried this twice, failed both times. |
| v9.1 | Same thing minus BC, target_kl raised to 0.03 | Plateaued at 25%. Voronoi fix made opponents stronger and the agent didn't adapt. |
| v10 | Territory + proximity rewards, 13 scalar obs (ray distances etc) | Hit 72% then collapsed to 45%. Turns out annealing rewards mid-training destroys what was learned. |
| v11 | Added minimap channels, dropped the annealing | Same 65% ceiling. Didn't collapse anymore but didn't improve either. |
| Safety shield | Tree search at inference (9-24 frame lookahead) | Useless. 3 discrete actions moving 5px/step in a 512px arena, there's nothing to differentiate. |
| BC-only | Pure supervised on human replays | 73% action accuracy. Drives straight into walls. It learned "go straight 44% of the time" from the data average instead of "go straight when it's safe." |

## What about the replay data?

`replay_pipeline.py` scrapes elite games (ELO 1600-2200) and renders 879K observation/action frames. We used it three ways, all failed.

BC warmstart corrupted the policy so badly that PPO couldn't recover across 5M steps. The action distribution shifted from a balanced 53/4/43 left/straight/right to 24% straight + 34% spin. Weight drift was 0.0066 total. Training basically froze.

BC as auxiliary loss during PPO did almost nothing. 38% WR vs 20% baseline at 1M steps, but the ceiling stayed the same. PPO gradients just overwhelm the BC signal.

BC alone got 73% accuracy imitating humans, then drove into walls. It memorized the average action distribution without learning when each action is appropriate. Makes sense, 3 actions with long-horizon consequences can't be captured frame-by-frame.

One thing we never tried: alternating between self-play and replay fine-tuning. Self-play learns survival (good at that), replays could teach strategy (humans are good at that), then self-play heals whatever BC broke. Might work, might just poison it again.

The data itself is fine. We had a bug where `turningDirection` always returned 0 so all labels were "straight," but that got fixed and re-rendered. The problem is imitation learning on this task, not the data.

## Architecture

```
Input: 7ch ego-centric 128x128 (self/enemy trails, prev frame, speed/erase powerups, Voronoi territory)
  IMPALA Block(7>16)   Conv3x3 + MaxPool + 2x Residual > 64x64
  IMPALA Block(16>32)  Conv3x3 + MaxPool + 2x Residual > 32x32
  IMPALA Block(32>32)  Conv3x3 + MaxPool + 2x Residual > 16x16
  CBAM(32)             channel + spatial attention
  Flatten(32x16x16=8192) > FC(8192>256)
  GRU(256>128)         temporal memory across frames
  Actor(128>3)         LEFT, STRAIGHT, RIGHT
  Critic(128>1)        value estimate
```

2,344,934 parameters.

## Lessons (save yourself time if you fork this)

Ego-centric rotation was the single biggest win across 11 versions. Before v3, the obs was world-aligned (fixed top-down) but actions were ego-relative (left/right/straight). The agent couldn't learn the mapping. Centering on the player and rotating so ego always faces right gave a 10x survival jump overnight.

Self-play cycling killed multiple training runs. Every time a new snapshot got added to the opponent pool, win rate crashed 30-50%. The agent develops strategy X, then plays against itself using strategy X, and can't beat its own clone. PFSP (prioritized fictitious self-play) helped, sampling opponents you lose to more often instead of 80/20 recent/old.

target_kl=0.015 freezes learning. PPO early-stops the epoch loop after 1-2 updates, so the policy barely moves. We wasted a 5M step v9 run before realizing this. 0.03 or higher lets the agent actually learn.

Watch for reward double-counting. We had kills getting 0.5 from the env + 0.1 from the training loop = 0.6 total, value normalization breaking GAE, and all 8 parallel envs facing the same opponent (correlated batch). These are easy to miss and hard to diagnose from metrics alone.

The .any() boolean downsampling from 512 to 128 matters more than you'd think. The env draws trails at 512x512 then pools 4x4 blocks with .any(), so a single trail pixel bloats to fill the whole obs block. If your browser code or export pipeline doesn't match this, the AI sees thinner trails than it trained on and crashes into everything.

Don't mmap random-access large files on Windows. Our 80GB BC obs file filled 32GB RAM through page faults. Load a random subset into RAM instead (~40K frames fits in ~4GB).

Voronoi territory is standard for Tron games. The BFS flood-fill channel showing "cells you can reach before any opponent" is what every competitive Tron bot uses (confirmed by University of Groningen's 2018 ICAART paper). Without it the agent has no spatial planning signal.

</details>
