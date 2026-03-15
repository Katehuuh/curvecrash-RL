[![Play Demo](https://img.shields.io/badge/Play-Demo-blue?style=for-the-badge&logo=googlechrome)](https://katehuuh.github.io/curvecrash-RL/) [![Install Userscript](https://img.shields.io/badge/Userscript-green?style=for-the-badge&logo=tampermonkey)](https://raw.githubusercontent.com/Katehuuh/curvecrash-RL/main/CurveCrash-AI-v6.user.js)

curvecrash-RL is a reinforcement learning bot for experimentation on the [CurveCrash](https://curvecrash.com) environment (Achtung die Kurve), using PPO self-play and transfer learning, playable in browser. A 2.3M param CNN+GRU (IMPALA-CNN + Voronoi) with 3 discrete actions learns survival but not strategy. Breaking through probably needs a bigger model, continuous actions, or a different training algorithm.

**Demo**: open the link above, arrow keys to steer. **·** **Real site**: install [Tampermonkey](https://www.tampermonkey.net/), click the userscript badge, go to [curvecrash.com](https://curvecrash.com) and load `/models/v8.json`.

## Train your own

```bash
# Needs Python 3.10+, PyTorch, NumPy, Pygame.
python train_selfplay.py --arch impala --voronoi --gspp --num-envs 8 --total-timesteps 10000000 --checkpoint-dir checkpoints
python watch_agent_ffa.py --checkpoint checkpoints/agent_final.pt --gspp
```

## What failed

| Version | Idea | Result |
|---------|------|--------|
| v1 | NatureCNN 1.6M params, 256x256 obs | Entropy collapsed. Clockwise circles. |
| v2 | Shrunk to 67K params, 128x128 | No collapse, but 6s survival (worse than eyes closed) |
| v3 | **Ego-centric rotation** | **10x breakthrough** — 39s survival. Only real architectural win. |
| v4 | NatureCNN 669K, self-play | 24-29% WR. Camps but gets trapped by humans. |
| v5 | Added GRU(64), exploiter training | 33% WR vs v4. Learned aggression but loose camping. |
| v6 | GS++ mode (powerups), GRU(128), CBAM | ~86s survival vs random. Solid but not strategic. |
| v8 | IMPALA-CNN + Voronoi, 2.3M params | **~65% WR self-play (best).** Survives but passive. |
| v9 | BC warmstart from human replays | **Poisoned action dist (34% spin).** PPO couldn't recover. |
| v9.1 | No BC, higher target_kl | Plateaued at 25% WR |
| v10 | Territory + proximity rewards, scalar obs | 72% peak then **collapsed to 45%** (reward annealing) |
| v11 | Minimap channels, no annealing | Same ~65% ceiling. Prevented collapse but no breakthrough. |
| Safety shield | Inference-time tree search | 3-action, 5px/step. Can't differentiate. |
| BC-only | Supervised learning from 879K human frames | 73% accuracy, drives straight into walls. |

## What about the replay data?

`replay_pipeline.py` scrapes elite human games (ELO 1600-2200) from curvecrash.com and renders 879K (obs, action) frames. We tried using this data three ways — all failed:

**BC warmstart** (pre-train on human data, then PPO): Corrupted the policy. Action distribution shifted from balanced 53/4/43 L/S/R to 24% straight + 34% spin. PPO with target_kl=0.015 early-stops after 1-2 epochs, so it couldn't undo the damage. Weight drift was only 0.0066 across 5M steps — training barely moved. We made this mistake twice (v9 + an earlier BC experiment).

**BC auxiliary loss** (small cross-entropy on human data during PPO): Minor effect. 38% WR vs 20% baseline at 1M steps, but didn't change the ceiling. The BC signal is too weak relative to PPO gradients.

**BC-only** (pure supervised): 73% action accuracy but the agent can't play. BC learns "go straight 44% of the time" from the data average, not "go straight WHEN it's safe." Context-dependent decisions require reasoning the model can't extract from single frames.

**Untested idea**: interleaved self-play and replay — let self-play learn survival (which it's good at), then periodically fine-tune on replays for strategy (which humans are good at), then back to self-play to "heal" the distribution. The theory: self-play alone learns survival but not strategy; replays alone learn strategy but not survival; alternating might get both. Risk: same BC poisoning problem if the replay signal is too strong.

The replay data itself is valid (physics-validated, correctly rendered after fixing a bug where all action labels were wrong). The problem is BC fundamentally — 3 discrete actions with long-horizon strategic dependencies can't be captured from imitation.

## Architecture

```
Input: 7ch ego-centric 128x128 (self/enemy trails, prev frame, speed/erase powerups, Voronoi territory)
  → IMPALA Block(7→16)   → Conv3x3 + MaxPool + 2×Residual → 64x64
  → IMPALA Block(16→32)  → Conv3x3 + MaxPool + 2×Residual → 32x32
  → IMPALA Block(32→32)  → Conv3x3 + MaxPool + 2×Residual → 16x16
  → CBAM(32)             → channel + spatial attention
  → Flatten(32×16×16=8192) → FC(8192→256)
  → GRU(256→128)         → temporal memory across frames
  → Actor(128→3)         → [LEFT, STRAIGHT, RIGHT]
  → Critic(128→1)        → value estimate
```

## Files

- `curvecrash_env_ffa.py` — module, Gym environment (512x512 arena, FFA, powerups)
- `train_selfplay.py` — PPO + PFSP self-play training loop
- `experiments.py` — Model architectures (IMPALA-CNN, Voronoi wrapper)
- `watch_agent_ffa.py` — UI only, Pygame viewer for local evaluation
- `replay_pipeline.py` — Scrape replays, render to observations, produce BC data
- `export_model.py` — Convert PyTorch checkpoint to TensorFlow.js
- `CurveCrash-AI-v6.user.js` — Browser userscript for live play
- `index.html` — Standalone browser demo (GitHub Pages)
