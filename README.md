[![Play Demo](https://img.shields.io/badge/Play-Demo-blue?style=for-the-badge&logo=googlechrome)](https://katehuuh.github.io/curvecrash-RL/) [![Install Userscript](https://img.shields.io/badge/Userscript-green?style=for-the-badge&logo=tampermonkey)](https://raw.githubusercontent.com/Katehuuh/curvecrash-RL/main/CurveCrash-AI-v6.user.js)

curvecrash-RL is a reinforcement learning bot for experimentation on the [CurveCrash](https://curvecrash.com) environment (Achtung die Kurve), using PPO self-play and transfer learning, playable in browser. A 2.3M param CNN+GRU (IMPALA-CNN + Voronoi) with 3 discrete actions learns survival but not strategy. Breaking through probably needs a bigger model, continuous actions, or a different training algorithm.

**Demo**: open the link above, arrow keys to steer. **·** **Real site**: install [Tampermonkey](https://www.tampermonkey.net/), click the userscript badge, go to [curvecrash.com](https://curvecrash.com).

## Train your own

```bash
# Needs Python 3.10+, PyTorch, NumPy, Pygame.
python train_selfplay.py --arch impala --voronoi --gspp --num-envs 8 --total-timesteps 10000000 --checkpoint-dir checkpoints
python watch_agent_ffa.py --checkpoint checkpoints/agent_final.pt --gspp
```

## What failed

| Version | Idea | Result |
|---------|------|--------|
| v8 | IMPALA-CNN + CBAM + GRU(128), 2.3M params | ~65% WR self-play. Beats random, loses to top-200 humans. |
| v9 | Behavior cloning warmstart | Poisoned action distribution. PPO couldn't recover. |
| v9.1 | Higher target_kl, fixed Voronoi | Plateaued at 25% WR |
| v10 | Territory + proximity rewards, scalar obs | 72% peak then collapsed to 45% (reward annealing) |
| v11 | Bilinear downsampling, fixed-orientation minimap, no annealing | Same ~65% ceiling. Prevented collapse but didn't break through. |
| Safety shield | Inference-time tree search / beam search | 3-action space moves ~5px/step in 512px arena. No differentiation. |

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

- `curvecrash_env_ffa.py` — Gym environment (512x512 arena, FFA, powerups)
- `train_selfplay.py` — PPO + PFSP self-play training loop
- `experiments.py` — Model architectures (IMPALA-CNN, Voronoi wrapper)
- `watch_agent_ffa.py` — Pygame viewer for local evaluation
- `replay_pipeline.py` — Scrape replays, render to observations, produce BC data
- `export_model.py` — Convert PyTorch checkpoint to TensorFlow.js
- `CurveCrash-AI-v6.user.js` — Browser userscript for live play
- `index.html` — Standalone browser demo (GitHub Pages)
