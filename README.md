# curvecrash-RL

RL bot for [CurveCrash](https://curvecrash.com) (Achtung die Kurve). PPO self-play, IMPALA-CNN + Voronoi, playable in browser.

[![Play Demo](https://img.shields.io/badge/Play-Demo-blue?style=for-the-badge&logo=googlechrome)](https://katehuuh.github.io/curvecrash-RL/) [![Install Userscript](https://img.shields.io/badge/Userscript-green?style=for-the-badge&logo=tampermonkey)](https://raw.githubusercontent.com/Katehuuh/curvecrash-RL/main/CurveCrash-AI-v6.user.js)

**Demo**: open the link, arrow keys to steer. **Real site**: install [Tampermonkey](https://www.tampermonkey.net/), click the userscript badge, go to [curvecrash.com](https://curvecrash.com).

## Train your own

```bash
python train_selfplay.py \
  --arch impala --voronoi --gspp \
  --num-envs 8 --total-timesteps 10000000 \
  --checkpoint-dir checkpoints

python watch_agent_ffa.py --checkpoint checkpoints/agent_final.pt --gspp
```

Needs Python 3.10+, PyTorch, NumPy, Pygame.

## What failed

| Version | Idea | Result |
|---------|------|--------|
| v8 | IMPALA-CNN + CBAM + GRU(128), 2.3M params | ~65% WR self-play. Beats random, loses to top-200 humans. |
| v9 | BC warmstart | Poisoned action dist. PPO couldn't recover. |
| v9.1 | Higher target_kl, fixed Voronoi | Plateaued at 25% WR |
| v10 | Territory + proximity rewards, scalar obs | 72% peak, collapsed to 45% from reward annealing |
| v11 | Bilinear DS, fixed minimap, no annealing | Same ~65% ceiling. No collapse but no breakthrough. |
| Safety shield | Inference-time tree/beam search | 3 actions at ~5px/step in 512px arena. Can't differentiate paths. |

2.3M param CNN+GRU with 3 discrete actions learns survival but not strategy. Breaking through probably needs a bigger model, continuous actions, or a different training algo.

## Architecture

```
7ch 128x128 (self/enemy trails, prev frame, speed/erase powerups, voronoi territory)
  > IMPALA(7>16)  Conv3x3 + MaxPool + 2xResidual > 64x64
  > IMPALA(16>32) Conv3x3 + MaxPool + 2xResidual > 32x32
  > IMPALA(32>32) Conv3x3 + MaxPool + 2xResidual > 16x16
  > CBAM(32)      channel + spatial attention
  > Flatten(8192) > FC(256) > GRU(128) > Actor(3) / Critic(1)
```

## Files

`curvecrash_env_ffa.py` env, `train_selfplay.py` training, `experiments.py` architectures, `watch_agent_ffa.py` viewer, `export_model.py` PyTorch to TF.js, `CurveCrash-AI-v6.user.js` userscript, `index.html` browser demo
