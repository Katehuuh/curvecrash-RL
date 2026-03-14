"""
Self-play PPO for CurveCrash FFA (v5.1 - NatureCNN + CBAM + SpatialAttn + GRU(128)).

Changes from v5:
  - GRU(128) (was 64): +69K params
  - CBAM on conv3 output: channel + spatial attention (~2.5K params)
  - Optional spatial self-attention on 6x6 feature map (~16K params)
  - Arena size variation (randomize physics per episode)
  - Larger opponent pool (100)
  - Frozen v5 opponents for AlphaStar iterative loop

Kept from v5:
  - NatureCNN backbone: 32->64->64 conv + 256 FC
  - 128x128 ego-centric rotated observations
  - Recurrent PPO (CleanRL ppo_atari_lstm.py)
  - Mirror augmentation, eps-smoothed categorical, target KL
  - AdamW, curriculum, self-play opponent pool, exploiter mode, gamma annealing

Usage:
    # v5.1 Phase 1: Main agent training
    python train_selfplay.py --mode main \
        --exploiter-pool checkpoints_v5_exploiter/agent_final.pt \
        --frozen-opponent checkpoints_v5_main/agent_best.pt \
        --exploiter-rate 0.2 --reward-gap-pass 0.03 \
        --total-timesteps 10000000 --checkpoint-dir checkpoints_v5_1_main

    # v5.1 Phase 2: Exploiter refresh
    python train_selfplay.py --mode exploiter \
        --frozen-opponent checkpoints_v5_1_main/agent_latest.pt \
        --reward-kill 0.5 --total-timesteps 3000000 --checkpoint-dir checkpoints_v5_1_exploiter

    tensorboard --logdir runs/
"""
import argparse
import copy
import gc
import os
import time
import random
from distutils.util import strtobool
from collections import deque

# Prevent OpenBLAS/MKL DLL conflict on Windows (heap corruption → segfault)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from curvecrash_env_ffa import CurveCrashFFAEnv, OBS_SIZE, FPS, MAX_AGENT_STEPS
from experiments import ImpalaCNNAgent, VoronoiWrapper, PFSPOpponentPool, ALL_SCRIPTED_BOTS


GRU_HIDDEN = 128
EMA_DECAY = 0.999
ANCHOR_STEP_TAG = -1  # Sentinel step_tag for best-ever anchor in opponent pool
NO_OPP_IDX = -1       # No valid PFSP opponent index (exploiter/uninitialized)
PERTURB_INTERVAL = 150_000  # Shrink & Perturb every N steps
PERTURB_ALPHA = 0.93        # Shrink factor (retain 93% of learned weights)
PERTURB_SIGMA = 0.002       # Noise scale


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default="curvecrash_v5_1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True)

    # Env
    p.add_argument("--min-players", type=int, default=2)
    p.add_argument("--max-players", type=int, default=6)

    # Training
    p.add_argument("--total-timesteps", type=int, default=50_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True)

    # PPO
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.015,
                    help="Early stop PPO epochs if approx KL exceeds this")
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Exploration
    p.add_argument("--eps-start", type=float, default=0.02,
                    help="Initial epsilon for smoothed categorical")
    p.add_argument("--eps-end", type=float, default=0.005,
                    help="Final epsilon for smoothed categorical")
    p.add_argument("--mirror-aug", type=lambda x: bool(strtobool(x)), default=True,
                    help="Mirror augmentation: flip obs + swap L/R actions")

    # Self-play
    p.add_argument("--snapshot-freq", type=int, default=200_000)
    p.add_argument("--opponent-update-freq", type=int, default=100_000)
    p.add_argument("--pool-size", type=int, default=100)

    # Anchor opponent (best-ever checkpoint always in pool)
    p.add_argument("--anchor-opponent", type=lambda x: bool(strtobool(x)), default=True,
                    help="Pin best-ever checkpoint in opponent pool (prevents forgetting)")

    # Value target normalization
    p.add_argument("--value-norm", type=lambda x: bool(strtobool(x)), default=True,
                    help="Normalize value targets with running mean/std")

    # Curriculum
    p.add_argument("--curriculum", type=lambda x: bool(strtobool(x)), default=True,
                    help="Staged player counts: 1v1 -> small FFA")

    # Checkpoints
    p.add_argument("--checkpoint-freq", type=int, default=500_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_v5")
    p.add_argument("--resume-from", type=str, default=None,
                    help="Path to v5 .pt checkpoint to resume weights from")

    # Mode (v5 new)
    p.add_argument("--mode", choices=["main", "exploiter"], default="main")
    p.add_argument("--frozen-opponent", type=str, default=None,
                    help="Path to frozen opponent checkpoint (exploiter mode)")
    p.add_argument("--frozen-is-legacy", type=lambda x: bool(strtobool(x)), default=False,
                    help="Frozen opponent uses old CNN-only architecture (no GRU)")

    # Exploiter pool (v5 new)
    p.add_argument("--exploiter-pool", type=str, nargs="*", default=None,
                    help="Paths to exploiter checkpoints for main training")
    p.add_argument("--exploiter-rate", type=float, default=0.3,
                    help="Probability of sampling from exploiter pool vs self-play pool")

    # Gamma annealing (v5 new)
    p.add_argument("--gamma-start", type=float, default=None,
                    help="Start gamma for annealing (None = use fixed --gamma)")
    p.add_argument("--gamma-end", type=float, default=None,
                    help="End gamma for annealing")

    # Kill bonus (v5 new)
    p.add_argument("--reward-kill", type=float, default=0.5,
                    help="Bonus reward per opponent death")

    # Gap pass bonus (v5.1 - reward for fully traversing a trail gap)
    p.add_argument("--reward-gap-pass", type=float, default=0.03,
                    help="Bonus reward per gap fully traversed (always active, never decayed)")

    # v5.1 architecture options
    p.add_argument("--no-spatial-attn", action="store_true",
                    help="Disable spatial self-attention (use CBAM + GRU(128) only)")
    p.add_argument("--arch", choices=["baseline", "impala"], default="baseline",
                    help="Architecture: baseline (NatureCNN) or impala (IMPALA-CNN residual)")
    p.add_argument("--voronoi", action="store_true",
                    help="Add Voronoi territory channel (+1 obs channel)")
    p.add_argument("--arena-variation", type=lambda x: bool(strtobool(x)), default=True,
                    help="Randomize arena physics per episode (default: True)")

    # v6: GS++ powerup mode
    p.add_argument("--gspp", action="store_true",
                    help="Enable GS++ mode (speed+erase powerups, 6ch observations)")
    p.add_argument("--powerup-spawn-interval", type=int, default=76,
                    help="Frames between powerup spawns (default: 76 = 1.27s)")
    p.add_argument("--reward-powerup-speed", type=float, default=0.2,
                    help="Bonus reward per speed powerup pickup")
    p.add_argument("--reward-powerup-erase", type=float, default=0.1,
                    help="Bonus reward per erase powerup pickup")

    # BC auxiliary loss (replay data mixed into self-play training)
    p.add_argument("--bc-data", type=str, default=None,
                    help="Path to pre-rendered NPZ with round_id/player_id for BC aux loss")
    p.add_argument("--bc-weight", type=float, default=0.1,
                    help="Weight for BC auxiliary loss (0=disabled)")
    p.add_argument("--bc-seq-len", type=int, default=128,
                    help="Truncated BPTT chunk size for BC sequences")
    p.add_argument("--bc-steps", type=int, default=4,
                    help="BC gradient steps per PPO update")

    # Legacy arg (kept for CLI compat, ignored — all rewards are constant now)
    p.add_argument("--aggression-decay-steps", type=int, default=0,
                    help="DEPRECATED: ignored. All reward shaping is constant.")

    # v9: ranking reward
    p.add_argument("--ranking-reward", action="store_true",
                    help="Use placement-based reward instead of binary win/lose")

    # v9: voronoi territory reward shaping
    p.add_argument("--voronoi-reward-coef", type=float, default=0.0,
                    help="Coefficient for voronoi territory potential-based shaping (0=disabled)")

    # v9: BC warm-start
    p.add_argument("--bc-warmstart-steps", type=int, default=0,
                    help="Pure BC training steps before PPO loop (0=disabled)")

    # v9: full checkpoint resume
    p.add_argument("--resume-training", type=str, default=None,
                    help="Path to training_state_*.pt for full resume (optimizer, pool, etc)")

    # v10: scalar observations
    p.add_argument("--use-scalar-obs", action="store_true",
                    help="Feed 13 scalar features (rays, speed, gap, territory) to model")

    # v10: dense territory reward
    p.add_argument("--territory-reward-coef", type=float, default=0.0,
                    help="Per-step reward for territory > fair share (0=disabled)")
    p.add_argument("--territory-anneal-steps", type=int, default=30_000_000,
                    help="Anneal territory reward to 0 over this many steps")

    # v10: aggressive proximity reward
    p.add_argument("--proximity-reward-coef", type=float, default=0.0,
                    help="Reward for being within 30px of opponent while facing them (0=disabled)")
    p.add_argument("--proximity-anneal-steps", type=int, default=20_000_000,
                    help="Anneal proximity reward to 0 over this many steps")

    # v10: scripted bot opponents
    p.add_argument("--scripted-bot-rate", type=float, default=0.0,
                    help="Fraction of envs using scripted bots instead of neural opponents")

    # v11: fixes from AI review
    p.add_argument("--no-reward-anneal", action="store_true",
                    help="Keep territory/proximity rewards constant (no annealing)")
    p.add_argument("--bilinear-ds", action="store_true",
                    help="Use mean-pool downsampling instead of boolean .any()")
    p.add_argument("--minimap", action="store_true",
                    help="Add 2 non-rotated minimap channels (global spatial reference)")

    return p.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class LegacyAgent(nn.Module):
    """CNN-only actor-critic (v4 architecture) for loading old checkpoints.
    No GRU -- used as frozen opponent in exploiter training."""

    def __init__(self, n_actions=3):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2304, 256)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(256, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def get_action_greedy(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        return logits.argmax(dim=1)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018).
    Channel attention (SE-style) + spatial attention. ~2.5K params for channels=64."""

    def __init__(self, channels=64, reduction=4):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        # Spatial attention
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.max(dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1)))
        x = x * spatial_att
        return x


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention on spatial feature map tokens.
    Input: (B, C=64, H=6, W=6) -> 36 tokens of dim 64.
    ~16K params with dim=64, num_heads=4."""

    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        qkv = self.qkv(tokens).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        # Residual connection + reshape back to spatial
        tokens = tokens + out
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class Agent(nn.Module):
    """NatureCNN + CBAM + SpatialAttn + GRU(128) actor-critic.
    Conv backbone -> CBAM(64) -> SpatialAttn(6x6) -> FC(256) -> GRU(128) -> actor(3)/critic(1).
    Supports 4ch (v5.1) or 6ch (v6 GS++) input."""

    def __init__(self, n_actions=3, use_spatial_attn=True, n_input_channels=4):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 32, 8, stride=4)),  # 128->31
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # 31->14
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2)),  # 14->6
            nn.ReLU(),
        )
        self.cbam = CBAM(64, reduction=4)
        self.use_spatial_attn = use_spatial_attn
        if use_spatial_attn:
            self.spatial_attn = SpatialSelfAttention(64, num_heads=4)
        self.fc = nn.Sequential(
            nn.Flatten(),                                 # 64*6*6=2304
            layer_init(nn.Linear(2304, 256)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(256, GRU_HIDDEN)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(GRU_HIDDEN, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(GRU_HIDDEN, 1), std=1.0)

    def _forward_gru(self, x, gru_state, done):
        """GRU forward handling both single-step (rollout) and multi-step (training).

        During rollout: x is (num_envs, C, H, W), single timestep.
        During training: x is (num_steps * envs_per_mb, C, H, W), reshaped into sequence.
        The batch_size is inferred from gru_state.shape[1].
        """
        x = self.network(x)        # (B, 64, 6, 6)
        x = self.cbam(x)           # (B, 64, 6, 6)
        if self.use_spatial_attn:
            x = self.spatial_attn(x)  # (B, 64, 6, 6)
        hidden = self.fc(x)        # (B, 256)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, 256))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            gru_state = (1.0 - d).view(1, -1, 1) * gru_state
            h, gru_state = self.gru(h.unsqueeze(0), gru_state)
            new_hidden.append(h)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_value(self, x, gru_state, done):
        hidden, _ = self._forward_gru(x, gru_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, gru_state, done, action=None, eps=0.0):
        hidden, new_gru_state = self._forward_gru(x, gru_state, done)
        logits = self.actor(hidden)
        if eps > 0:
            probs = torch.softmax(logits, dim=-1)
            probs = (1 - eps) * probs + eps / 3.0
            dist = torch.distributions.Categorical(probs=probs)
        else:
            dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden), new_gru_state

    def get_action_greedy(self, x, gru_state, done):
        hidden, new_gru_state = self._forward_gru(x, gru_state, done)
        logits = self.actor(hidden)
        return logits.argmax(dim=1), new_gru_state


class V5Agent(nn.Module):
    """v5 architecture (GRU(64), no CBAM/spatial) for loading frozen v5 opponents.
    Also supports ego inference in the viewer via get_action_and_value."""

    def __init__(self, n_actions=3):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2304, 256)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(256, 64)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(64, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def _forward_gru(self, x, gru_state, done):
        hidden = self.network(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, 256))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            gru_state = (1.0 - d).view(1, -1, 1) * gru_state
            h, gru_state = self.gru(h.unsqueeze(0), gru_state)
            new_hidden.append(h)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_action_and_value(self, x, gru_state, done, action=None, eps=0.0):
        hidden, new_gru_state = self._forward_gru(x, gru_state, done)
        logits = self.actor(hidden)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden), new_gru_state

    def get_action_greedy(self, x, gru_state, done):
        hidden, new_gru_state = self._forward_gru(x, gru_state, done)
        logits = self.actor(hidden)
        return logits.argmax(dim=1), new_gru_state


class BCReplayBuffer:
    """Loads pre-rendered replay data for BC auxiliary loss during self-play.
    Uses memory-mapped .npy files when available to avoid loading 18+ GB into RAM.

    Accepts a base path (with or without extension). Looks for:
        {base}_obs.npy, {base}_act.npy, {base}_round_id.npy, {base}_player_id.npy
    Falls back to {base}.npz if .npy files not found.
    """

    def __init__(self, npz_path):
        import time as _t
        # Strip any extension to get clean base path
        base = npz_path
        for ext in [".npz", ".npy"]:
            if base.endswith(ext):
                base = base[:-len(ext)]
                break

        obs_npy = base + "_obs.npy"
        act_npy = base + "_act.npy"
        rid_npy = base + "_round_id.npy"
        pid_npy = base + "_player_id.npy"

        use_mmap = all(os.path.isfile(f) for f in [obs_npy, act_npy, rid_npy, pid_npy])
        print(f"Loading BC replay data ({'mmap .npy' if use_mmap else 'npz'})...")
        t0 = _t.time()

        if use_mmap:
            self.obs = np.load(obs_npy, mmap_mode="r")  # memory-mapped, ~0 RAM
            self.act = np.load(act_npy)
            round_id = np.load(rid_npy)
            player_id = np.load(pid_npy)
        else:
            npz_file = base + ".npz"
            data = np.load(npz_file)
            self.obs = data["obs"]
            self.act = data["act"]
            round_id = data["round_id"]
            player_id = data["player_id"]

        t1 = _t.time()
        print(f"  Loaded {len(self.obs):,} samples in {t1-t0:.1f}s")

        # Build per-player-round sequences
        seq_map = {}
        for i in range(len(self.obs)):
            key = (int(round_id[i]), int(player_id[i]))
            if key not in seq_map:
                seq_map[key] = []
            seq_map[key].append(i)
        self.sequences = [np.array(v) for v in seq_map.values() if len(v) >= 2]
        print(f"  {len(self.sequences)} player-sequences for BC training")

    def sample_chunk(self, seq_len):
        """Sample a random chunk from a random sequence. Returns (indices,)."""
        seq = self.sequences[np.random.randint(len(self.sequences))]
        if len(seq) > seq_len:
            start = np.random.randint(0, len(seq) - seq_len)
            return seq[start:start + seq_len]
        return seq


def forward_sequential_bc(model, obs_t, h, target_channels=0):
    """Forward a sequence through CNN+GRU for BC loss. Returns (logits, h).
    Supports both NatureCNN (Agent) and IMPALA-CNN (ImpalaCNNAgent) architectures.
    If target_channels > obs_t channels, pads with zeros (e.g. 6ch BC data → 7ch model)."""
    if target_channels > 0 and obs_t.shape[1] < target_channels:
        pad = torch.zeros(
            obs_t.shape[0], target_channels - obs_t.shape[1],
            obs_t.shape[2], obs_t.shape[3],
            device=obs_t.device, dtype=obs_t.dtype
        )
        obs_t = torch.cat([obs_t, pad], dim=1)
    if hasattr(model, 'conv') and isinstance(model.conv, nn.Sequential):
        # ImpalaCNNAgent path: conv → relu → cbam → fc
        x = model.conv(obs_t)
        x = nn.functional.relu(x)
        if hasattr(model, 'cbam') and model.use_cbam:
            x = model.cbam(x)
        x = model.fc(x)
    else:
        # NatureCNN Agent path: network → cbam → spatial_attn → fc
        x = model.network(obs_t)
        x = model.cbam(x)
        if hasattr(model, 'use_spatial_attn') and model.use_spatial_attn:
            x = model.spatial_attn(x)
        x = model.fc(x)
    x = x.unsqueeze(1)              # (T, 1, 256)
    gru_out, h = model.gru(x, h)    # (T, 1, H)
    gru_out = gru_out.squeeze(1)     # (T, H)
    return model.actor(gru_out), h


def get_input_channels(state_dict):
    """Infer number of input channels from first conv layer weight shape."""
    key = "network.0.weight"
    if key in state_dict:
        return state_dict[key].shape[1]
    return 4


def detect_checkpoint_type(state_dict):
    """Detect checkpoint architecture: 'legacy' (v4), 'v5', 'v5_1', or 'v6'."""
    has_gru = any(k.startswith("gru.") for k in state_dict)
    has_cbam = any(k.startswith("cbam.") for k in state_dict)
    n_in = get_input_channels(state_dict)
    if not has_gru:
        return "legacy"
    elif has_cbam and n_in == 6:
        return "v6"
    elif has_cbam:
        return "v5_1"
    else:
        return "v5"


def get_gru_hidden_size(state_dict):
    """Infer GRU hidden size from checkpoint weights."""
    if "gru.weight_hh_l0" in state_dict:
        return state_dict["gru.weight_hh_l0"].shape[1]
    return 0


def load_agent(checkpoint_path, device='cpu'):
    """Load any checkpoint and return (agent, meta_dict).

    Handles legacy (v4), v5, v5.1, v6, and IMPALA checkpoints.
    Returns agent in eval mode + metadata dict with keys:
        type, n_input_ch, gru_hidden, has_gru, has_scalars
    """
    from experiments import ImpalaCNNAgent

    weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
    is_impala = any(k.startswith("conv.0.conv.") for k in weights)

    if is_impala:
        n_in = weights["conv.0.conv.weight"].shape[1]
        gh = weights["gru.weight_hh_l0"].shape[1]
        channels = []
        for i in range(10):
            k = f"conv.{i}.conv.weight"
            if k in weights:
                channels.append(weights[k].shape[0])
        n_scalar = weights["scalar_fc.0.weight"].shape[1] if "scalar_fc.0.weight" in weights else 0
        agent = ImpalaCNNAgent(
            n_input_channels=n_in, gru_hidden=gh,
            channels=tuple(channels),
            use_cbam=any(k.startswith("cbam.") for k in weights),
            n_scalar_inputs=n_scalar,
        ).to(device)
        agent.load_state_dict(weights)
        agent.eval()
        return agent, {
            'type': 'impala', 'n_input_ch': n_in, 'gru_hidden': gh,
            'has_gru': True, 'has_scalars': n_scalar > 0,
        }

    ct = detect_checkpoint_type(weights)
    gh = get_gru_hidden_size(weights)
    n_in = get_input_channels(weights)
    has_gru = ct in ("v5", "v5_1", "v6")

    if ct in ("v5_1", "v6"):
        has_sa = any(k.startswith("spatial_attn.") for k in weights)
        agent = Agent(use_spatial_attn=has_sa, n_input_channels=n_in).to(device)
    elif ct == "v5":
        agent = V5Agent().to(device)
    else:
        agent = LegacyAgent().to(device)

    agent.load_state_dict(weights)
    agent.eval()
    return agent, {
        'type': ct, 'n_input_ch': n_in, 'gru_hidden': gh,
        'has_gru': has_gru, 'has_scalars': False,
    }


class SelfPlayVecEnv:
    """Sync vector env with batched GPU opponent inference and GRU state tracking.
    v9: supports multiple opponent model slots for per-env opponent diversity."""

    MAX_OPP_PER_ENV = 10  # generous upper bound for opponent slots
    NUM_OPP_SLOTS = 4     # v9: number of concurrent opponent models

    SCRIPTED_SLOT = -1  # v10: marker for scripted bot opponents

    def __init__(self, env_fns, device, use_spatial_attn=True,
                 arch="baseline", n_channels=6, scripted_bot_rate=0.0):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.device = device
        self.use_spatial_attn = use_spatial_attn
        self.arch = arch
        self.n_channels = n_channels
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.scripted_bot_rate = scripted_bot_rate  # v10
        # v10: per-env assigned bot type (stable within an episode)
        self.env_scripted_bot = [None] * self.num_envs

        # Legacy single-opponent (backward compat)
        self.opponent_model = None
        self.opponent_type = None  # "legacy", "v5", "v5_1", or "impala"
        self.opponent_step_tag = 0
        self.opp_gru_hidden = 0  # current opponent's GRU hidden size

        # v9: Multi-slot opponents
        self.opponent_models = [None] * self.NUM_OPP_SLOTS
        self.opponent_types = [None] * self.NUM_OPP_SLOTS
        self.opp_n_channels_slots = [0] * self.NUM_OPP_SLOTS
        self.opp_gru_hidden_slots = [0] * self.NUM_OPP_SLOTS
        self.slot_pool_idx = [-1] * self.NUM_OPP_SLOTS  # PFSP index per slot
        self.env_opponent_slot = np.zeros(self.num_envs, dtype=np.int32)  # which slot each env uses
        self.multi_slot_mode = False  # True when using per-env opponents

        # GRU states for all opponent slots — allocated lazily based on opponent type
        self.opp_gru_states = torch.zeros(
            1, self.num_envs * self.MAX_OPP_PER_ENV, GRU_HIDDEN, device=device
        )

    def set_opponent(self, state_dict, step_tag=0, is_legacy=False):
        is_impala = any(k.startswith("conv.0.conv.") for k in state_dict)
        if is_impala:
            ckpt_type = "impala"
            n_in = state_dict["conv.0.conv.weight"].shape[1]
            gru_h = state_dict["gru.weight_hh_l0"].shape[1]
            channels = []
            for ci in range(10):
                key = f"conv.{ci}.conv.weight"
                if key in state_dict:
                    channels.append(state_dict[key].shape[0])
            # v10: detect scalar branch in checkpoint
            opp_n_scalar = state_dict["scalar_fc.0.weight"].shape[1] if "scalar_fc.0.weight" in state_dict else 0
            if not isinstance(self.opponent_model, ImpalaCNNAgent) or self.opponent_type != "impala":
                self.opponent_model = ImpalaCNNAgent(
                    n_input_channels=n_in, gru_hidden=gru_h,
                    channels=tuple(channels),
                    use_cbam=any(k.startswith("cbam.") for k in state_dict),
                    n_scalar_inputs=opp_n_scalar,
                ).to(self.device)
        else:
            ckpt_type = "legacy" if is_legacy else detect_checkpoint_type(state_dict)
            gru_h = get_gru_hidden_size(state_dict)
            n_in = get_input_channels(state_dict)

            # Instantiate correct model type
            if ckpt_type == "legacy":
                if not isinstance(self.opponent_model, LegacyAgent):
                    self.opponent_model = LegacyAgent().to(self.device)
            elif ckpt_type == "v5":
                if not isinstance(self.opponent_model, V5Agent) or self.opponent_type != "v5":
                    self.opponent_model = V5Agent().to(self.device)
            else:  # v5_1 or v6
                use_sa = any(k.startswith("spatial_attn.") for k in state_dict)
                if (not isinstance(self.opponent_model, Agent)
                        or self.opponent_type != ckpt_type):
                    self.opponent_model = Agent(
                        use_spatial_attn=use_sa, n_input_channels=n_in
                    ).to(self.device)

        self.opponent_model.load_state_dict(state_dict)
        self.opponent_model.eval()
        self.opponent_step_tag = step_tag
        self.opponent_type = ckpt_type
        self.opp_gru_hidden = gru_h
        self.opp_n_channels = n_in  # track opponent's expected input channels

        # Reallocate GRU states if hidden size changed
        if gru_h > 0 and self.opp_gru_states.shape[2] != gru_h:
            self.opp_gru_states = torch.zeros(
                1, self.num_envs * self.MAX_OPP_PER_ENV, gru_h, device=self.device
            )
        else:
            self.opp_gru_states.zero_()

    def set_opponent_slot(self, slot_idx, state_dict, step_tag=0, pool_idx=-1):
        """Load an opponent into a specific slot (v9 multi-slot mode)."""
        is_impala = any(k.startswith("conv.0.conv.") for k in state_dict)
        if is_impala:
            n_in = state_dict["conv.0.conv.weight"].shape[1]
            gru_h = state_dict["gru.weight_hh_l0"].shape[1]
            channels = []
            for ci in range(10):
                key = f"conv.{ci}.conv.weight"
                if key in state_dict:
                    channels.append(state_dict[key].shape[0])
            # v10: detect scalar branch in checkpoint
            opp_n_scalar = state_dict["scalar_fc.0.weight"].shape[1] if "scalar_fc.0.weight" in state_dict else 0
            model = ImpalaCNNAgent(
                n_input_channels=n_in, gru_hidden=gru_h,
                channels=tuple(channels),
                use_cbam=any(k.startswith("cbam.") for k in state_dict),
                n_scalar_inputs=opp_n_scalar,
            ).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.opponent_models[slot_idx] = model
            self.opponent_types[slot_idx] = "impala"
            self.opp_n_channels_slots[slot_idx] = n_in
            self.opp_gru_hidden_slots[slot_idx] = gru_h
        else:
            # For non-IMPALA, fall back to slot 0 single-model mode
            self.set_opponent(state_dict, step_tag)
            self.opponent_models[slot_idx] = self.opponent_model
            self.opponent_types[slot_idx] = self.opponent_type
            self.opp_n_channels_slots[slot_idx] = getattr(self, 'opp_n_channels', 4)
            self.opp_gru_hidden_slots[slot_idx] = self.opp_gru_hidden
        self.slot_pool_idx[slot_idx] = pool_idx
        self.multi_slot_mode = (any(m is not None for m in self.opponent_models)
                                or self.scripted_bot_rate > 0)

    def resample_env_opponent(self, env_i):
        """Assign env_i to a random active slot (called on episode reset).
        v10: with scripted_bot_rate probability, use scripted bots instead."""
        if self.scripted_bot_rate > 0 and random.random() < self.scripted_bot_rate:
            self.env_opponent_slot[env_i] = self.SCRIPTED_SLOT
            self.env_scripted_bot[env_i] = random.choice(ALL_SCRIPTED_BOTS)
            return
        active = [s for s in range(self.NUM_OPP_SLOTS) if self.opponent_models[s] is not None]
        if active:
            self.env_opponent_slot[env_i] = random.choice(active)

    def set_player_range(self, min_p, max_p):
        """Update player count range for curriculum."""
        for env in self.envs:
            env.min_players = min_p
            env.max_players = max_p

    def _reset_opp_gru(self, env_i):
        """Zero out GRU states for all opponent slots in env_i."""
        start = env_i * self.MAX_OPP_PER_ENV
        end = start + self.MAX_OPP_PER_ENV
        self.opp_gru_states[:, start:end, :] = 0

    def reset(self):
        obs_list, info_list = [], []
        for env in self.envs:
            o, i = env.reset()
            obs_list.append(o)
            info_list.append(i)
        self.opp_gru_states.zero_()
        return np.stack(obs_list), info_list

    def _infer_single_model(self, model, model_type, opp_n_channels, obs_list, gru_indices):
        """Batch-infer opponent actions for a single model. Returns actions array."""
        if not obs_list:
            return np.array([], dtype=np.int64)
        all_opp_arr = np.stack(obs_list)
        env_channels = all_opp_arr.shape[1]
        if env_channels > opp_n_channels:
            all_opp_arr = all_opp_arr[:, :opp_n_channels]
        elif env_channels < opp_n_channels:
            pad = np.zeros(
                (all_opp_arr.shape[0], opp_n_channels - env_channels,
                 all_opp_arr.shape[2], all_opp_arr.shape[3]),
                dtype=np.float32
            )
            all_opp_arr = np.concatenate([all_opp_arr, pad], axis=1)
        with torch.no_grad():
            obs_t = torch.tensor(all_opp_arr, dtype=torch.float32, device=self.device)
            if model_type == "legacy":
                actions = model.get_action_greedy(obs_t).cpu().numpy()
            else:
                gru_batch = self.opp_gru_states[:, gru_indices, :]
                done_zeros = torch.zeros(len(obs_list), device=self.device)
                acts, new_gru = model.get_action_greedy(obs_t, gru_batch, done_zeros)
                actions = acts.cpu().numpy()
                self.opp_gru_states[:, gru_indices, :] = new_gru
            del obs_t
        return actions

    def step(self, ego_actions):
        # 1. Collect opponent observations and GRU indices
        all_opp_obs = []
        gru_flat_indices = []
        env_meta = []
        # Also track which slot each opponent belongs to (for multi-slot mode)
        opp_slot_ids = []  # parallel to all_opp_obs
        # v10: cache env/player references for scripted bots (local, not persistent)
        scripted_env_cache = {}
        scripted_player_cache = {}
        scripted_env_idx = {}  # flat_idx -> env index i
        for i, env in enumerate(self.envs):
            opp_obs, live_mask = env.get_opponent_observations()
            slot = int(self.env_opponent_slot[i])
            flat_start = len(all_opp_obs)
            all_opp_obs.extend(opp_obs)
            opp_j = 0  # index into opp_obs (only alive opponents)
            for j, alive in enumerate(live_mask):
                if alive:
                    flat_idx = flat_start + opp_j
                    gru_flat_indices.append(i * self.MAX_OPP_PER_ENV + j)
                    opp_slot_ids.append(slot)
                    if slot == self.SCRIPTED_SLOT:
                        scripted_env_cache[flat_idx] = env
                        scripted_player_cache[flat_idx] = env.players[j + 1]
                        scripted_env_idx[flat_idx] = i
                    opp_j += 1
            env_meta.append((len(env.players) - 1, live_mask))

        # 2. Compute opponent actions
        if all_opp_obs:
            if self.multi_slot_mode:
                # Group by slot, infer per group, scatter back
                all_actions = np.zeros(len(all_opp_obs), dtype=np.int64)
                slot_groups = {}
                for flat_i, slot_id in enumerate(opp_slot_ids):
                    if slot_id not in slot_groups:
                        slot_groups[slot_id] = []
                    slot_groups[slot_id].append(flat_i)

                for slot_id, indices in slot_groups.items():
                    if slot_id == self.SCRIPTED_SLOT:
                        # v10: scripted bot opponents (bot type fixed per episode)
                        for idx in indices:
                            bot = self.env_scripted_bot[scripted_env_idx[idx]] or ALL_SCRIPTED_BOTS[0]
                            all_actions[idx] = bot.get_action(
                                scripted_env_cache[idx], scripted_player_cache[idx])
                        continue
                    model = self.opponent_models[slot_id]
                    if model is None:
                        # No model in this slot — random actions
                        for idx in indices:
                            all_actions[idx] = random.randint(0, 2)
                        continue
                    group_obs = [all_opp_obs[idx] for idx in indices]
                    group_gru = [gru_flat_indices[idx] for idx in indices]
                    group_acts = self._infer_single_model(
                        model, self.opponent_types[slot_id],
                        self.opp_n_channels_slots[slot_id],
                        group_obs, group_gru
                    )
                    for j, idx in enumerate(indices):
                        all_actions[idx] = group_acts[j]
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            elif self.opponent_model is not None:
                # Legacy single-model path
                all_actions = self._infer_single_model(
                    self.opponent_model, self.opponent_type,
                    getattr(self, 'opp_n_channels', 4),
                    all_opp_obs, gru_flat_indices
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            else:
                all_actions = np.random.randint(0, 3, size=len(all_opp_obs))
        else:
            all_actions = np.array([], dtype=np.int64)

        # 3. Step envs
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        act_idx = 0
        for i, (env, ego_a) in enumerate(zip(self.envs, ego_actions)):
            n_opp, live_mask = env_meta[i]

            opp_actions = np.ones(n_opp, dtype=np.int64)  # default straight
            for j, alive in enumerate(live_mask):
                if alive:
                    opp_actions[j] = all_actions[act_idx]
                    act_idx += 1

            o, r, te, tr, info = env.step(int(ego_a), opp_actions)

            # Track opponent deaths before potential reset
            opp_deaths = sum(1 for p in env.players[1:] if p.just_died)

            if te or tr:
                self._reset_opp_gru(i)
                final_info = info.copy()
                # v9: record which slot this env used for PFSP
                final_info["env_opp_slot"] = int(self.env_opponent_slot[i])
                o, _ = env.reset()
                info = final_info
                # v9: resample opponent slot on episode reset
                if self.multi_slot_mode:
                    self.resample_env_opponent(i)

            info["opponent_deaths"] = opp_deaths
            obs_list.append(o)
            rewards.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)

        return (
            np.stack(obs_list),
            np.array(rewards),
            np.array(terms),
            np.array(truncs),
            infos,
        )


class RunningMeanStd:
    """Welford's online algorithm for running mean/std."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0   # Warm prior: avoids near-zero std early (OpenAI baselines convention)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


def get_curriculum_range(global_step, total_steps):
    """Smooth curriculum: brief solo -> 1v1 -> ramp to full FFA."""
    progress = global_step / total_steps
    if progress < 0.02:
        return 1, 1     # Solo: basic wall avoidance only
    elif progress < 0.10:
        return 2, 3     # 1v1 to 1v2
    else:
        # Fix E: Smooth ramp from 3 to 6 players (no hard jump)
        max_p = min(6, 3 + int(3 * (progress - 0.10) / 0.30))
        return 2, max(3, max_p)


def get_exploiter_curriculum_range(global_step, total_steps):
    """Exploiter curriculum: pure 1v1 -> small groups."""
    if global_step < total_steps * 0.50:
        return 2, 2     # Pure 1v1
    elif global_step < total_steps * 0.80:
        return 2, 3
    else:
        return 2, 4


def main():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_scalar_obs and args.arch != "impala":
        raise ValueError("--use-scalar-obs requires --arch impala")

    # Limit PyTorch CPU threads to reduce contention with numpy
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set by experiments.py import

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(
        [f"|{k}|{v}|" for k, v in vars(args).items()]
    ))

    # Envs
    base_channels = 6 if args.gspp else 4
    n_channels = base_channels + (1 if args.voronoi else 0) + (2 if args.minimap else 0)
    env_cls = VoronoiWrapper if args.voronoi else CurveCrashFFAEnv

    def make_env(seed):
        def thunk():
            return env_cls(
                min_players=args.min_players, max_players=args.max_players,
                arena_variation=args.arena_variation,
                gspp=args.gspp,
                powerup_spawn_interval=args.powerup_spawn_interval,
                kill_reward=args.reward_kill,
                speed_pickup_reward=args.reward_powerup_speed,
                ranking_reward=args.ranking_reward,
                bilinear_ds=args.bilinear_ds,
                minimap=args.minimap,
            )
        return thunk

    use_spatial_attn = not args.no_spatial_attn
    envs = SelfPlayVecEnv(
        [make_env(args.seed + i) for i in range(args.num_envs)], device,
        use_spatial_attn=use_spatial_attn,
        arch=args.arch, n_channels=n_channels,
        scripted_bot_rate=args.scripted_bot_rate,
    )
    for i, env in enumerate(envs.envs):
        env.reset(seed=args.seed + i)

    n_scalar = 13 if args.use_scalar_obs else 0
    if args.arch == "impala":
        agent = ImpalaCNNAgent(
            n_input_channels=n_channels, gru_hidden=GRU_HIDDEN,
            channels=(16, 32, 32), use_cbam=True,
            n_scalar_inputs=n_scalar,
        ).to(device)
    else:
        agent = Agent(
            use_spatial_attn=use_spatial_attn, n_input_channels=n_channels
        ).to(device)
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location=device, weights_only=True)
        if not any(k.startswith("gru.") for k in sd):
            print("ERROR: Cannot resume from v4 checkpoint (no GRU). Architecture mismatch.")
            print("Use --frozen-opponent to train against v4 instead.")
            return
        # v11: pad first conv if loading checkpoint with fewer input channels (e.g. 7ch -> 9ch)
        old_conv = sd.get("conv.0.conv.weight")  # (out_ch, in_ch, kH, kW)
        if old_conv is not None and old_conv.shape[1] < n_channels:
            pad_ch = n_channels - old_conv.shape[1]
            pad = torch.zeros(old_conv.shape[0], pad_ch, old_conv.shape[2], old_conv.shape[3],
                              device=old_conv.device)
            sd["conv.0.conv.weight"] = torch.cat([old_conv, pad], dim=1)
            print(f"  Padded conv.0: {old_conv.shape[1]} -> {n_channels} input channels (+{pad_ch})")

        # v10: pad GRU weights if loading non-scalar checkpoint into scalar model
        need_strict_false = False
        if n_scalar > 0 and "scalar_fc.0.weight" not in sd:
            old_ih = sd["gru.weight_ih_l0"]  # (3*H, old_input_size)
            new_input_size = agent.gru.input_size  # 320 = 256 + 64
            if old_ih.shape[1] < new_input_size:
                pad = torch.zeros(old_ih.shape[0], new_input_size - old_ih.shape[1],
                                  device=old_ih.device)
                sd["gru.weight_ih_l0"] = torch.cat([old_ih, pad], dim=1)
                print(f"  Padded GRU weight_ih_l0: {old_ih.shape[1]} -> {new_input_size} cols")
            need_strict_false = True

        if need_strict_false:
            agent.load_state_dict(sd, strict=False)
            print(f"Resumed weights from: {args.resume_from} (scalar branch zero-initialized)")
        else:
            agent.load_state_dict(sd)
            print(f"Resumed weights from: {args.resume_from}")
    optimizer = optim.AdamW(
        agent.parameters(), lr=args.learning_rate, eps=1e-5,
        weight_decay=args.weight_decay
    )

    # Shrink & Perturb: save initialization weights for periodic regularization
    init_state_dict = {k: v.clone() for k, v in agent.state_dict().items()}

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Architecture: {args.arch}" + (" + voronoi" if args.voronoi else ""))
    print(f"Model params: {total_params:,}")
    print(f"Obs channels: {n_channels}")
    print(f"Obs size: {OBS_SIZE}x{OBS_SIZE}")

    # --- Exploiter mode setup ---
    if args.mode == "exploiter" and args.frozen_opponent:
        frozen_sd = torch.load(args.frozen_opponent, map_location="cpu", weights_only=True)
        is_legacy = args.frozen_is_legacy or not any(k.startswith("gru.") for k in frozen_sd)
        envs.set_opponent(frozen_sd, step_tag=0, is_legacy=is_legacy)
        ckpt_type = detect_checkpoint_type(frozen_sd)
        print(f"Exploiter mode: frozen opponent = {args.frozen_opponent} "
              f"(type={ckpt_type})")

    # --- Exploiter/frozen pool (for main mode) ---
    # Both --exploiter-pool and --frozen-opponent get merged into one pool.
    # At each opponent update, exploiter_rate % of the time we sample from this
    # pool (uniform random), otherwise from the self-play pool.
    exploiter_pool = []
    if args.exploiter_pool:
        for path in args.exploiter_pool:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            ckpt_type = detect_checkpoint_type(sd)
            is_legacy = (ckpt_type == "legacy")
            exploiter_pool.append((sd, path, is_legacy))
            print(f"  Pool opponent: {path} (type={ckpt_type})")
    if args.mode == "main" and args.frozen_opponent:
        frozen_sd = torch.load(args.frozen_opponent, map_location="cpu", weights_only=True)
        ckpt_type = detect_checkpoint_type(frozen_sd)
        is_legacy = args.frozen_is_legacy or (ckpt_type == "legacy")
        exploiter_pool.append((frozen_sd, args.frozen_opponent, is_legacy))
        print(f"  Pool opponent (frozen): {args.frozen_opponent} (type={ckpt_type})")
    if exploiter_pool:
        print(f"Opponent pool: {len(exploiter_pool)} frozen opponent(s) "
              f"(rate={args.exploiter_rate})")

    # --- Self-play pool (main mode only) ---
    opponent_pool = PFSPOpponentPool(max_size=args.pool_size) if args.mode == "main" else None

    # --- Storage ---
    batch_size = args.num_envs * args.num_steps
    assert args.num_envs % args.num_minibatches == 0, \
        f"num_envs ({args.num_envs}) must be divisible by num_minibatches ({args.num_minibatches})"
    envsperbatch = args.num_envs // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    obs = torch.zeros(
        (args.num_steps, args.num_envs, n_channels, OBS_SIZE, OBS_SIZE)
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # v10: scalar observations buffer
    if args.use_scalar_obs:
        scalar_obs_buf = torch.zeros(
            (args.num_steps, args.num_envs, n_scalar)
        ).to(device)

    # GRU hidden state: (1, num_envs, GRU_HIDDEN)
    gru_state = torch.zeros(1, args.num_envs, GRU_HIDDEN).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_np, reset_infos = envs.reset()
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # v10: initial scalar obs from env state after reset
    if args.use_scalar_obs:
        next_scalar = torch.zeros(args.num_envs, n_scalar, device=device)
        _scalar_staging = np.empty((args.num_envs, n_scalar), dtype=np.float32)
        for i, env in enumerate(envs.envs):
            _scalar_staging[i] = env.get_scalar_obs(env.ego)
        next_scalar.copy_(torch.as_tensor(_scalar_staging))
    next_checkpoint = args.checkpoint_freq
    next_snapshot = args.snapshot_freq
    next_opp_update = args.opponent_update_freq

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Metrics
    win_buffer = deque(maxlen=100)
    ep_len_buffer = deque(maxlen=100)
    ep_ret_buffer = deque(maxlen=100)

    # Fix B: Anchor opponent — track best-ever checkpoint
    best_win_rate = 0.0
    best_state_dict = None

    # Fix C: Value target normalization
    ret_rms = RunningMeanStd() if args.value_norm else None

    # Fix F: EMA opponent snapshots (kept on GPU; moved to CPU only at snapshot time)
    ema_state = {k: v.clone() for k, v in agent.state_dict().items()}

    # v9: Full checkpoint resume
    if args.resume_training:
        print(f"\n=== Resuming full training state from {args.resume_training} ===")
        ts = torch.load(args.resume_training, map_location=device, weights_only=False)
        agent.load_state_dict(ts["agent"])
        optimizer.load_state_dict(ts["optimizer"])
        global_step = ts["global_step"]
        ema_state = {k: v.to(device) for k, v in ts["ema_state"].items()}
        best_win_rate = ts["best_win_rate"]
        best_state_dict = ts.get("best_state_dict", None)
        win_buffer = deque(ts.get("win_buffer", []), maxlen=100)
        ep_ret_buffer = deque(ts.get("ep_ret_buffer", []), maxlen=100)
        ep_len_buffer = deque(ts.get("ep_len_buffer", []), maxlen=100)
        rank_buffer = deque(ts.get("rank_buffer", []), maxlen=200)
        next_checkpoint = ts.get("next_checkpoint", global_step + args.checkpoint_freq)
        next_snapshot = ts.get("next_snapshot", global_step + args.snapshot_freq)
        next_opp_update = ts.get("next_opp_update", global_step + args.opponent_update_freq)
        if ret_rms is not None and "ret_rms_mean" in ts:
            ret_rms.mean = ts["ret_rms_mean"]
            ret_rms.var = ts["ret_rms_var"]
            ret_rms.count = ts["ret_rms_count"]
        if opponent_pool is not None and "pool" in ts:
            pool_data = ts["pool"]
            for sd, st, w, g in zip(
                pool_data["entries"], pool_data["step_tags"],
                pool_data["wins"], pool_data["games"]
            ):
                opponent_pool.pool.append(sd)
                opponent_pool.step_tags.append(st)
                opponent_pool.wins.append(w)
                opponent_pool.games.append(g)
            print(f"  Restored opponent pool: {len(opponent_pool)} entries")
        # Adjust num_updates to account for already-completed steps
        updates_done = global_step // batch_size
        print(f"  Resumed at global_step={global_step:,}, updates_done={updates_done}, "
              f"best_wr={best_win_rate:.1%}, pool={len(opponent_pool) if opponent_pool else 0}")
        del ts
        gc.collect()

    # PFSP: track which pool opponent we're currently playing against
    current_opp_idx = NO_OPP_IDX

    print(f"Training {args.total_timesteps:,} timesteps, {num_updates} updates")
    print(f"Batch size: {batch_size}, Minibatch envs: {envsperbatch}")
    print(f"Players per episode: {args.min_players}-{args.max_players}")
    print(f"Mode: {args.mode}")
    if args.curriculum:
        if args.mode == "exploiter":
            print("Curriculum (exploiter): 1v1 -> 1v2 -> 1v3")
        else:
            print("Curriculum: solo -> 1v1 -> 2-4p -> 2-6p (FFA)")
    if args.mode == "main":
        print(f"Opponents: RANDOM until first snapshot at step {args.snapshot_freq:,}")
    print(f"Mirror augmentation: {args.mirror_aug}")
    print(f"Epsilon smoothing: {args.eps_start} -> {args.eps_end}")
    print(f"Target KL: {args.target_kl}")
    if args.gamma_start is not None and args.gamma_end is not None:
        print(f"Gamma annealing: {args.gamma_start} -> {args.gamma_end}")
    if args.reward_kill > 0:
        print(f"Kill bonus: +{args.reward_kill}")
    if args.reward_gap_pass > 0:
        print(f"Gap pass bonus: +{args.reward_gap_pass}")
    print(f"Reward shaping: constant (no decay)")
    print(f"v7 fixes: PFSP={args.mode=='main'}, anchor={args.anchor_opponent}, "
          f"value_norm={args.value_norm}, entropy_anneal=0.05->0.01, "
          f"smooth_curriculum=True, EMA_snapshots=True")
    print(f"Spatial self-attention: {use_spatial_attn}")
    print(f"Arena variation: {args.arena_variation}")
    print(f"GRU hidden: {GRU_HIDDEN}")
    if args.gspp:
        print(f"GS++ mode: ON (6ch, spawn_interval={args.powerup_spawn_interval})")
        if args.reward_powerup_speed > 0:
            print(f"  Speed pickup bonus: +{args.reward_powerup_speed}")
        if args.reward_powerup_erase > 0:
            print(f"  Erase pickup bonus: +{args.reward_powerup_erase}")

    # v10 diagnostics
    if args.use_scalar_obs:
        print(f"v10 scalar observations: ON ({n_scalar} features)")
    if args.territory_reward_coef > 0:
        print(f"v10 territory reward: coef={args.territory_reward_coef}, "
              f"anneal={args.territory_anneal_steps:,} steps")
    if args.proximity_reward_coef > 0:
        print(f"v10 proximity reward: coef={args.proximity_reward_coef}, "
              f"anneal={args.proximity_anneal_steps:,} steps")
    if args.scripted_bot_rate > 0:
        print(f"v10 scripted bots: rate={args.scripted_bot_rate}")

    # Load BC replay data for auxiliary loss
    bc_buffer = None
    if args.bc_data:
        bc_buffer = BCReplayBuffer(args.bc_data)
        print(f"BC auxiliary loss: weight={args.bc_weight}, "
              f"steps/update={args.bc_steps}, seq_len={args.bc_seq_len}")

    # v9: voronoi territory shaping state
    prev_territory_frac = np.zeros(args.num_envs, dtype=np.float32)
    if args.voronoi_reward_coef > 0:
        print(f"Voronoi territory shaping: coef={args.voronoi_reward_coef}")
    if args.ranking_reward:
        print("Ranking reward: ON (placement-based)")
    rank_buffer = deque(maxlen=200)

    # v9: BC warm-start — pure BC training before PPO loop
    if bc_buffer is not None and args.bc_warmstart_steps > 0:
        print(f"\n=== BC Warm-Start: {args.bc_warmstart_steps} steps ===")
        agent.train()
        bc_warmstart_losses = []
        for ws_step in range(args.bc_warmstart_steps):
            chunk_idx = bc_buffer.sample_chunk(args.bc_seq_len)
            bc_obs = np.ascontiguousarray(bc_buffer.obs[chunk_idx], dtype=np.float32) / 255.0
            bc_act = np.ascontiguousarray(bc_buffer.act[chunk_idx], dtype=np.int64)
            bc_obs_t = torch.from_numpy(bc_obs).to(device)
            bc_act_t = torch.from_numpy(bc_act).to(device)
            h_bc = torch.zeros(1, 1, GRU_HIDDEN, device=device)
            bc_logits, _ = forward_sequential_bc(agent, bc_obs_t, h_bc, target_channels=n_channels)
            bc_loss = nn.functional.cross_entropy(bc_logits, bc_act_t)
            optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            bc_warmstart_losses.append(bc_loss.item())
            if (ws_step + 1) % 500 == 0 or ws_step == 0:
                avg_loss = np.mean(bc_warmstart_losses[-500:])
                print(f"  BC warmstart step {ws_step+1}/{args.bc_warmstart_steps}: loss={avg_loss:.4f}")
        print(f"=== BC Warm-Start complete. Final loss={np.mean(bc_warmstart_losses[-100:]):.4f} ===\n")

    start_update = max(1, global_step // batch_size + 1) if global_step > 0 else 1
    for update in range(start_update, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Epsilon schedule
        progress = min(1.0, global_step / args.total_timesteps)
        current_eps = args.eps_start + (args.eps_end - args.eps_start) * progress

        # Gamma annealing
        if args.gamma_start is not None and args.gamma_end is not None:
            current_gamma = args.gamma_start + (args.gamma_end - args.gamma_start) * progress
        else:
            current_gamma = args.gamma

        # Curriculum
        if args.curriculum:
            if args.mode == "exploiter":
                min_p, max_p = get_exploiter_curriculum_range(global_step, args.total_timesteps)
            else:
                min_p, max_p = get_curriculum_range(global_step, args.total_timesteps)
            envs.set_player_range(min_p, max_p)

        # Save initial GRU state for recurrent PPO training
        initial_gru_state = gru_state.clone()

        # Constant aggression factor — all reward shaping is permanent.
        # Decay was causing agents to lose kill/speed-seeking behavior.
        aggr_factor = 1.0

        # Push aggression factor into envs for internal combo rewards
        for env in envs.envs:
            env.aggression_bonus = aggr_factor

        # v10/v11: pre-compute reward coefficients
        terr_coef = 0.0
        if args.territory_reward_coef > 0 and args.voronoi:
            if args.no_reward_anneal:
                terr_coef = args.territory_reward_coef
            else:
                t_anneal = max(0.0, 1.0 - global_step / args.territory_anneal_steps)
                terr_coef = args.territory_reward_coef * t_anneal
        prox_coef = 0.0
        if args.proximity_reward_coef > 0:
            if args.no_reward_anneal:
                prox_coef = args.proximity_reward_coef
            else:
                p_anneal = max(0.0, 1.0 - global_step / args.proximity_anneal_steps)
                prox_coef = args.proximity_reward_coef * p_anneal

        # --- Rollout ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            if args.use_scalar_obs:
                scalar_obs_buf[step] = next_scalar

            with torch.no_grad():
                scalars_in = next_scalar if args.use_scalar_obs else None
                action, logprob, _, value, gru_state = agent.get_action_and_value(
                    next_obs, gru_state, next_done, eps=current_eps, scalars=scalars_in
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Ensure all async CUDA ops complete before touching numpy/Python heap
            # (RTX 5090 + Windows: async CUDA memory ops corrupt Python heap)
            if device.type == "cuda":
                torch.cuda.synchronize()

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(
                action.cpu().numpy()
            )

            # Reward shaping — constant, no decay
            # NOTE: kill_reward and speed_pickup_reward are already applied inside
            # the env (curvecrash_env_ffa.py line 705-711) via aggression_bonus.
            # Only add rewards NOT handled by the env here.
            for i, info in enumerate(infos):
                if args.reward_powerup_erase > 0:
                    reward_np[i] += args.reward_powerup_erase * info.get("ego_erase_pickups", 0)
                if args.reward_gap_pass > 0:
                    reward_np[i] += args.reward_gap_pass * info.get("ego_gap_passes", 0)

                # Survival-scaled kill bonus at episode end:
                # Kill+survive = big reward, kill+die immediately = small reward
                # Prevents kamikaze: grab speed → kill one → die
                if "episode" in info:
                    ep_kills = info.get("ego_kills", 0)
                    ep_len = info["episode"]["l"]
                    survival_ratio = min(1.0, ep_len / MAX_AGENT_STEPS)
                    reward_np[i] += ep_kills * survival_ratio * args.reward_kill

            # v9: Voronoi territory potential-based shaping: F = γΦ(s') - Φ(s)
            if args.voronoi_reward_coef > 0 and args.voronoi:
                for i, env in enumerate(envs.envs):
                    curr_frac = getattr(env, '_ego_territory_frac', 0.0)
                    done_i = term_np[i] or trunc_np[i]
                    if done_i:
                        # Reset on episode boundary
                        shaping = -prev_territory_frac[i]  # γ·0 - Φ(s)
                        prev_territory_frac[i] = 0.0
                    else:
                        shaping = current_gamma * curr_frac - prev_territory_frac[i]
                        prev_territory_frac[i] = curr_frac
                    reward_np[i] += args.voronoi_reward_coef * shaping

            # v10: Dense territory reward (annealed, coef hoisted above loop)
            if terr_coef > 0:
                for i, env in enumerate(envs.envs):
                    ego_frac = getattr(env, '_ego_territory_frac', 0.0)
                    alive_count = sum(1 for p in env.players if p.alive)
                    fair_share = 1.0 / max(alive_count, 1)
                    reward_np[i] += terr_coef * (ego_frac - fair_share)

            # v10: Aggressive proximity reward (annealed)
            if prox_coef > 0:
                for i, env in enumerate(envs.envs):
                    ego = env.ego
                    if not ego.alive:
                        continue
                    for p in env.players[1:]:
                        if not p.alive:
                            continue
                        dx = p.x - ego.x
                        dy = p.y - ego.y
                        dist_sq = dx * dx + dy * dy
                        if dist_sq < 900:  # 30px threshold
                            angle_to_opp = np.arctan2(dy, dx)
                            angle_diff = (angle_to_opp - ego.angle + np.pi) % (2 * np.pi) - np.pi
                            if abs(angle_diff) < np.pi / 3:  # within 60°
                                dist = dist_sq ** 0.5
                                reward_np[i] += prox_coef * (30 - dist) / 30

            rewards[step].copy_(torch.as_tensor(reward_np, device=device))
            next_obs.copy_(torch.as_tensor(next_obs_np, device=device))
            next_done.copy_(torch.as_tensor(
                np.logical_or(term_np, trunc_np).astype(np.float32), device=device
            ))

            # v10: update scalar obs from current env state (post-reset if terminal)
            if args.use_scalar_obs:
                for i, env in enumerate(envs.envs):
                    _scalar_staging[i] = env.get_scalar_obs(env.ego)
                next_scalar.copy_(torch.as_tensor(_scalar_staging))

            for info in infos:
                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    ep_l = info["episode"]["l"]
                    is_win = info.get("win", False)
                    n_pl = info.get("n_players", 0)
                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episode_length", ep_l, global_step)
                    writer.add_scalar("charts/n_players", n_pl, global_step)
                    writer.add_scalar("charts/win", float(is_win), global_step)
                    kills = info.get("ego_kills", 0)
                    writer.add_scalar("charts/ego_kills", kills, global_step)
                    ego_rank = info.get("ego_rank", 0)
                    if ego_rank > 0:
                        writer.add_scalar("charts/ego_rank", ego_rank, global_step)
                        rank_buffer.append(ego_rank)
                    win_buffer.append(float(is_win))
                    ep_len_buffer.append(ep_l)
                    ep_ret_buffer.append(ep_r)
                    # PFSP: record game result against current opponent
                    if opponent_pool is not None and n_pl > 1:
                        if envs.multi_slot_mode:
                            # v9: per-env slot tracking
                            slot = info.get("env_opp_slot", 0)
                            pool_idx = envs.slot_pool_idx[slot]
                            if pool_idx >= 0:
                                opponent_pool.record_result(pool_idx, is_win)
                        elif current_opp_idx >= 0:
                            opponent_pool.record_result(current_opp_idx, is_win)
                    print(
                        f"step={global_step:>10,}  "
                        f"ep_r={ep_r:.2f}  ep_l={ep_l}  "
                        f"players={n_pl}  win={is_win}  kills={kills}  "
                        f"opp_pool={len(opponent_pool) if opponent_pool else 'N/A'}"
                    )

        # --- Fix F: Update EMA weights (in-place on GPU, no PCIe transfer) ---
        with torch.no_grad():
            for k, v in agent.state_dict().items():
                ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)

        # --- Shrink & Perturb (maintain plasticity, Ash & Adams 2020) ---
        if global_step % PERTURB_INTERVAL < args.num_envs * args.num_steps and global_step > 1_500_000:
            with torch.no_grad():
                for k, v in agent.state_dict().items():
                    init_v = init_state_dict[k].to(device)
                    noise = torch.randn_like(v) * PERTURB_SIGMA
                    v.mul_(PERTURB_ALPHA).add_(init_v, alpha=1 - PERTURB_ALPHA).add_(noise)
            print(f"  >> Shrink & Perturb applied (alpha={PERTURB_ALPHA}, sigma={PERTURB_SIGMA})")

        # --- Fix B: Track best-ever checkpoint by rolling win rate ---
        if args.anchor_opponent and win_buffer and len(win_buffer) >= 20:
            current_wr = sum(win_buffer) / len(win_buffer)
            if current_wr > best_win_rate:
                best_win_rate = current_wr
                best_state_dict = {k: v.cpu().clone() for k, v in ema_state.items()}
                print(f"  >> New best checkpoint! win_rate={current_wr:.1%}")

        # --- Opponent pool management (main mode only) ---
        if args.mode == "main" and opponent_pool is not None:
            if global_step >= next_snapshot:
                # Fix F: Snapshot EMA weights (GPU → CPU) instead of raw weights
                ema_cpu = {k: v.cpu() for k, v in ema_state.items()}
                if len(opponent_pool) >= opponent_pool.max_size:
                    current_opp_idx = NO_OPP_IDX  # Deque eviction shifts indices
                opponent_pool.add(ema_cpu, global_step)
                print(f"  >> EMA snapshot added to pool (size={len(opponent_pool)})")
                # Fix B: Re-add anchor if it was evicted by deque maxlen
                if args.anchor_opponent and best_state_dict is not None:
                    if ANCHOR_STEP_TAG not in opponent_pool.step_tags:
                        if len(opponent_pool) >= opponent_pool.max_size:
                            current_opp_idx = NO_OPP_IDX
                        opponent_pool.add(
                            {k: v.clone() for k, v in best_state_dict.items()},
                            ANCHOR_STEP_TAG,
                        )
                        print(f"  >> Anchor re-added to pool (best_wr={best_win_rate:.1%})")
                next_snapshot += args.snapshot_freq

            if global_step >= next_opp_update and (len(opponent_pool) > 0 or exploiter_pool):
                if len(opponent_pool) >= 2:
                    # v9: Fill 4 opponent slots for per-env diversity
                    # Slot 0: anchor (best-ever)
                    if args.anchor_opponent and best_state_dict is not None:
                        envs.set_opponent_slot(0, {k: v.clone() for k, v in best_state_dict.items()},
                                               ANCHOR_STEP_TAG, pool_idx=-1)
                        slot_desc = [f"slot0=ANCHOR(wr={best_win_rate:.1%})"]
                    else:
                        sd, idx = opponent_pool.sample()
                        envs.set_opponent_slot(0, sd, 0, pool_idx=idx)
                        slot_desc = [f"slot0=PFSP[{idx}]"]
                    # Slot 1: PFSP (hardest opponent)
                    sd, idx = opponent_pool.sample()
                    envs.set_opponent_slot(1, sd, 0, pool_idx=idx)
                    slot_desc.append(f"slot1=PFSP[{idx}]")
                    # Slot 2: recent (last few snapshots)
                    n = len(opponent_pool)
                    idx = random.randint(max(0, n - min(4, n)), n - 1)
                    envs.set_opponent_slot(2, opponent_pool.pool[idx], 0, pool_idx=idx)
                    slot_desc.append(f"slot2=RECENT[{idx}]")
                    # Slot 3: random from pool
                    idx = random.randint(0, n - 1)
                    envs.set_opponent_slot(3, opponent_pool.pool[idx], 0, pool_idx=idx)
                    slot_desc.append(f"slot3=RAND[{idx}]")
                    print(f"  >> Opponent slots refreshed: {', '.join(slot_desc)}")
                    current_opp_idx = NO_OPP_IDX  # no longer tracking single opponent
                else:
                    # Fallback: not enough pool entries for multi-slot, use single model
                    use_exploiter = (
                        exploiter_pool
                        and random.random() < args.exploiter_rate
                    )
                    if use_exploiter:
                        sd, path, is_legacy = random.choice(exploiter_pool)
                        envs.set_opponent(sd, step_tag=-1, is_legacy=is_legacy)
                        current_opp_idx = NO_OPP_IDX
                        print(f"  >> Opponent = EXPLOITER: {os.path.basename(path)}")
                    elif len(opponent_pool) > 0:
                        sd, opp_idx = opponent_pool.sample()
                        envs.set_opponent(sd, opp_idx, is_legacy=False)
                        current_opp_idx = opp_idx
                        print(f"  >> Opponent = PFSP pool[{opp_idx}]")
                next_opp_update += args.opponent_update_freq

        # --- GAE (with gamma annealing) ---
        with torch.no_grad():
            gae_scalars = next_scalar if args.use_scalar_obs else None
            next_value = agent.get_value(
                next_obs, gru_state, next_done, scalars=gae_scalars
            ).reshape(1, -1)

            # When value_norm is active, critic predicts normalized-scale values.
            # Unnormalize before GAE so advantages are computed in raw reward scale.
            if ret_rms is not None:
                values_for_gae = values * ret_rms.std + ret_rms.mean
                next_value_for_gae = next_value * ret_rms.std + ret_rms.mean
            else:
                values_for_gae = values
                next_value_for_gae = next_value

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value_for_gae
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values_for_gae[t + 1]
                delta = (
                    rewards[t]
                    + current_gamma * nextvalues * nextnonterminal
                    - values_for_gae[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + current_gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_for_gae

            # Value target normalization: normalize returns for critic training
            if ret_rms is not None:
                ret_rms.update(returns.cpu().numpy().flatten())
                normalized_returns = (returns - ret_rms.mean) / ret_rms.std
            else:
                normalized_returns = returns

        # Flatten for indexing
        b_obs = obs.reshape((-1, n_channels, OBS_SIZE, OBS_SIZE))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_normalized_returns = normalized_returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)
        if args.use_scalar_obs:
            b_scalars = scalar_obs_buf.reshape((-1, n_scalar))

        # --- PPO update (sequence-based minibatching for recurrent) ---
        # Fix D: Entropy annealing (0.05 → 0.01 over training)
        current_ent_coef = args.ent_coef * (1.0 - 0.8 * progress)
        clipfracs = []
        kl_exceeded = False

        for epoch in range(args.update_epochs):
            envinds = np.arange(args.num_envs)
            np.random.shuffle(envinds)

            for mb_start in range(0, args.num_envs, envsperbatch):
                mbenvinds = envinds[mb_start:mb_start + envsperbatch]

                # Build indices in step-major order for sequence reconstruction
                mb_inds = np.array([
                    t * args.num_envs + e
                    for t in range(args.num_steps)
                    for e in mbenvinds
                ])

                mb_obs = b_obs[mb_inds].clone()
                mb_actions = b_actions.long()[mb_inds].clone()
                mb_dones = b_dones[mb_inds]
                mb_scalars = b_scalars[mb_inds].clone() if args.use_scalar_obs else None

                # Mirror augmentation: consistent per env across all timesteps
                if args.mirror_aug:
                    mirror_envs = torch.rand(envsperbatch, device=device) < 0.5
                    if mirror_envs.any():
                        # Expand to all timesteps (step-major order)
                        mirror_mask = mirror_envs.repeat(args.num_steps)
                        mb_obs[mirror_mask] = mb_obs[mirror_mask].flip(-1)
                        acts = mb_actions[mirror_mask]
                        swapped = acts.clone()
                        swapped[acts == 0] = 2
                        swapped[acts == 2] = 0
                        mb_actions[mirror_mask] = swapped
                        # v10: mirror scalar ray distances
                        # Rays 0-7: 0=fwd, 1-7 clockwise. Mirror swaps 1↔7, 2↔6, 3↔5
                        if mb_scalars is not None:
                            m = mb_scalars[mirror_mask]
                            tmp = m[:, [1, 2, 3]].clone()
                            m[:, [1, 2, 3]] = m[:, [7, 6, 5]]
                            m[:, [7, 6, 5]] = tmp
                            mb_scalars[mirror_mask] = m

                # Reconstruct GRU states from initial state for this env subset
                mb_gru_state = initial_gru_state[:, mbenvinds].contiguous()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs, mb_gru_state, mb_dones, mb_actions, eps=current_eps,
                    scalars=mb_scalars
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                # Fix C: Use normalized returns for value loss
                v_loss = 0.5 * ((newvalue - b_normalized_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - current_ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Target KL early stopping
            if args.target_kl is not None and approx_kl > args.target_kl:
                kl_exceeded = True
                break

        # --- BC auxiliary loss (interleaved with PPO) ---
        bc_loss_val = 0.0
        if bc_buffer is not None and args.bc_weight > 0:
            agent.train()
            for _ in range(args.bc_steps):
                chunk_idx = bc_buffer.sample_chunk(args.bc_seq_len)
                # .copy() ensures contiguous RAM (not mmap-backed) before CUDA transfer
                bc_obs = np.ascontiguousarray(bc_buffer.obs[chunk_idx], dtype=np.float32) / 255.0
                bc_act = np.ascontiguousarray(bc_buffer.act[chunk_idx], dtype=np.int64)
                bc_obs_t = torch.from_numpy(bc_obs).to(device)
                bc_act_t = torch.from_numpy(bc_act).to(device)
                h_bc = torch.zeros(1, 1, GRU_HIDDEN, device=device)
                bc_logits, _ = forward_sequential_bc(agent, bc_obs_t, h_bc, target_channels=n_channels)
                bc_loss = args.bc_weight * nn.functional.cross_entropy(bc_logits, bc_act_t)
                optimizer.zero_grad()
                bc_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                bc_loss_val += bc_loss.item()
            bc_loss_val /= args.bc_steps

        # --- Logging ---
        y_pred = b_values.cpu().numpy()
        # Critic predicts normalized values, compare against normalized returns
        y_true = b_normalized_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/entropy_coef", current_ent_coef, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if ret_rms is not None:
            writer.add_scalar("debug/value_norm_mean", ret_rms.mean, global_step)
            writer.add_scalar("debug/value_norm_std", ret_rms.std, global_step)
        if best_win_rate > 0:
            writer.add_scalar("selfplay/best_win_rate", best_win_rate, global_step)
        writer.add_scalar("explore/epsilon", current_eps, global_step)
        writer.add_scalar("explore/kl_exceeded", float(kl_exceeded), global_step)
        writer.add_scalar("explore/gamma", current_gamma, global_step)
        writer.add_scalar("debug/gru_hidden_norm", gru_state.norm().item(), global_step)
        if args.territory_reward_coef > 0:
            writer.add_scalar("v10/territory_coef", terr_coef, global_step)
        if args.proximity_reward_coef > 0:
            writer.add_scalar("v10/proximity_coef", prox_coef, global_step)
        if bc_buffer is not None:
            writer.add_scalar("losses/bc_loss", bc_loss_val, global_step)

        if win_buffer:
            wr = sum(win_buffer) / len(win_buffer)
            writer.add_scalar("charts/win_rate_100", wr, global_step)
        if ep_len_buffer:
            avg_len = sum(ep_len_buffer) / len(ep_len_buffer)
            writer.add_scalar("charts/avg_episode_length_100", avg_len, global_step)
        if ep_ret_buffer:
            avg_ret = sum(ep_ret_buffer) / len(ep_ret_buffer)
            writer.add_scalar("charts/avg_return_100", avg_ret, global_step)

        if opponent_pool is not None:
            writer.add_scalar("selfplay/pool_size", len(opponent_pool), global_step)
            # PFSP diagnostics: log opponent win rates
            if hasattr(opponent_pool, 'games') and len(opponent_pool) > 0:
                total_games = sum(opponent_pool.games)
                if total_games > 0:
                    total_wins = sum(opponent_pool.wins)
                    writer.add_scalar("selfplay/pfsp_avg_winrate",
                                      total_wins / total_games, global_step)
        writer.add_scalar("selfplay/opponent_step", envs.opponent_step_tag, global_step)
        if args.curriculum:
            if args.mode == "exploiter":
                _, max_p = get_exploiter_curriculum_range(global_step, args.total_timesteps)
            else:
                _, max_p = get_curriculum_range(global_step, args.total_timesteps)
            writer.add_scalar("curriculum/max_players", max_p, global_step)

        if update % 10 == 0:
            wr_str = f"{sum(win_buffer)/len(win_buffer):.1%}" if win_buffer else "N/A"
            ent_str = f"{entropy_loss.item():.3f}"
            eps_str = f"{current_eps:.4f}"
            gam_str = f"{current_gamma:.4f}"
            bc_str = f"  bc={bc_loss_val:.4f}" if bc_buffer is not None else ""
            print(
                f"update {update}/{num_updates}  step={global_step:,}  SPS={sps}  "
                f"pg={pg_loss.item():.4f}  vf={v_loss.item():.4f}  "
                f"ent={ent_str}  eps={eps_str}  gamma={gam_str}  kl={approx_kl:.4f}  "
                f"win_rate={wr_str}{bc_str}"
            )

        # Periodic lightweight GC (every 50 updates)
        if update % 50 == 0:
            gc.collect()

        # Checkpoint
        if global_step >= next_checkpoint:
            path = os.path.join(args.checkpoint_dir, f"agent_{global_step}.pt")
            torch.save(agent.state_dict(), path)
            # v9: Full training state for resuming
            ts_path = os.path.join(args.checkpoint_dir, f"training_state_{global_step}.pt")
            training_state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "ema_state": {k: v.cpu().clone() for k, v in ema_state.items()},
                "best_win_rate": best_win_rate,
                "best_state_dict": best_state_dict,
                "win_buffer": list(win_buffer),
                "ep_ret_buffer": list(ep_ret_buffer),
                "ep_len_buffer": list(ep_len_buffer),
                "rank_buffer": list(rank_buffer),
                "next_checkpoint": next_checkpoint + args.checkpoint_freq,
                "next_snapshot": next_snapshot,
                "next_opp_update": next_opp_update,
            }
            if ret_rms is not None:
                training_state["ret_rms_mean"] = ret_rms.mean
                training_state["ret_rms_var"] = ret_rms.var
                training_state["ret_rms_count"] = ret_rms.count
            if opponent_pool is not None:
                training_state["pool"] = {
                    "entries": [sd for sd in opponent_pool.pool],
                    "step_tags": list(opponent_pool.step_tags),
                    "wins": list(opponent_pool.wins),
                    "games": list(opponent_pool.games),
                }
            torch.save(training_state, ts_path)
            print(f"Checkpoint saved: {path} + {ts_path}")
            next_checkpoint += args.checkpoint_freq
            # Periodic memory cleanup — prevents heap fragmentation crash on
            # long runs (RTX 5090 + Windows + numpy 1.26)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    final_path = os.path.join(args.checkpoint_dir, "agent_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
