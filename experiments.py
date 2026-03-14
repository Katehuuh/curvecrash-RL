"""
Ablation experiments for CurveCrash AI.

Tests architecture, observation, and self-play variants against a shared baseline.
Each experiment runs a short training session and logs metrics for comparison.

Usage:
    # List all experiments
    python experiments.py --list

    # Run a single experiment
    python experiments.py --run baseline --steps 1000000

    # Run all experiments sequentially
    python experiments.py --run-all --steps 1000000

    # Run specific experiments
    python experiments.py --run impala_cnn --run impala_cnn_voronoi --steps 1000000

    # Compare results
    python experiments.py --compare

This script is TEMPORARY — delete once the best config is found.
"""
import argparse
import copy
import csv
import gc
import math
import os
import random
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_interop_threads(1)

from curvecrash_env_ffa import (
    CurveCrashFFAEnv, OBS_SIZE, FPS, ARENA_SIM,
    FRAME_SKIP, MAX_AGENT_STEPS,
)

RESULTS_DIR = Path(__file__).parent / "experiment_results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
BC_DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# Behavioral Cloning Data Pool
# ============================================================================

class BCDataPool:
    """Chunk-based BC data sampler. Keeps only 1 chunk in RAM at a time.
    Each chunk is ~5-17 MB (vs 47 GB for all 500K samples). Rotates through
    chunks randomly, sampling within each chunk before moving to the next."""

    def __init__(self, data_dir, device):
        self.device = device
        self.files = sorted(Path(data_dir).glob("bc_data_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No bc_data_*.npz in {data_dir}")

        # Count total samples without loading obs
        self.total_samples = 0
        for f in self.files:
            d = np.load(f)
            self.total_samples += len(d["act"])
            d.close()

        # Shuffle file order, load first chunk
        self._file_order = list(range(len(self.files)))
        random.shuffle(self._file_order)
        self._file_idx = 0
        self._load_chunk(self._file_order[0])
        print(f"  BC data: {self.total_samples:,} samples across {len(self.files)} chunks (lazy loading)")

    def _load_chunk(self, chunk_idx):
        """Load a single NPZ chunk into RAM (~5-17 MB)."""
        d = np.load(self.files[chunk_idx])
        self._obs = d["obs"]    # (N, 6, 128, 128) uint8
        self._act = d["act"]    # (N,) uint8
        self._n = len(self._act)
        self._perm = np.random.permutation(self._n)
        self._pos = 0
        d.close()

    def sample(self, batch_size):
        """Return (obs_tensor, act_tensor) on device. obs is float32 [0,1]."""
        if self._pos + batch_size > self._n:
            # Move to next chunk
            self._file_idx = (self._file_idx + 1) % len(self._file_order)
            if self._file_idx == 0:
                random.shuffle(self._file_order)
            self._load_chunk(self._file_order[self._file_idx])

        idx = self._perm[self._pos:self._pos + batch_size]
        self._pos += batch_size

        obs = torch.tensor(
            self._obs[idx].astype(np.float32) / 255.0, device=self.device
        )
        act = torch.tensor(self._act[idx].astype(np.int64), device=self.device)
        return obs, act


# ============================================================================
# Architecture Variants
# ============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ResBlock(nn.Module):
    """Residual block for IMPALA-CNN."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + x


class ConvSequence(nn.Module):
    """Conv -> MaxPool -> ResBlock -> ResBlock (IMPALA-CNN building block)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels=64, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.max(dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1)))
        x = x * spatial_att
        return x


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention on spatial feature map tokens."""
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(tokens).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        tokens = tokens + out
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# --- Baseline Agent (NatureCNN + CBAM + SpatialAttn + GRU) ---

class BaselineAgent(nn.Module):
    """Current v5.1/v6 architecture. NatureCNN backbone."""
    def __init__(self, n_actions=3, use_spatial_attn=True, n_input_channels=6,
                 gru_hidden=128):
        super().__init__()
        self.gru_hidden = gru_hidden
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
            nn.Flatten(),
            layer_init(nn.Linear(2304, 256)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(256, gru_hidden)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(gru_hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(gru_hidden, 1), std=1.0)

    def _forward_gru(self, x, gru_state, done):
        x = self.network(x)
        x = self.cbam(x)
        if self.use_spatial_attn:
            x = self.spatial_attn(x)
        hidden = self.fc(x)
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


# --- IMPALA-CNN Agent ---

class ImpalaCNNAgent(nn.Module):
    """IMPALA-CNN backbone with residual blocks. Replaces NatureCNN.
    Conv->MaxPool->Res->Res at each stage. 15-layer deep vs NatureCNN's 3.
    ~1.7M params vs 836K baseline.

    Architecture (128x128 input):
        ConvSeq(in->16): 128->64
        ConvSeq(16->32): 64->32
        ConvSeq(32->32): 32->16
        ReLU -> Flatten(32*16*16=8192) -> FC(256) -> GRU -> actor/critic
    """
    def __init__(self, n_actions=3, n_input_channels=6, gru_hidden=128,
                 channels=(16, 32, 32), use_cbam=True, n_scalar_inputs=0):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.n_input_channels = n_input_channels
        self.n_scalar_inputs = n_scalar_inputs

        # Build conv sequences
        layers = []
        in_ch = n_input_channels
        for out_ch in channels:
            layers.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # After 3 stages of stride-2 pooling: 128->64->32->16
        # Feature map: channels[-1] * 16 * 16
        self._feat_size = channels[-1] * 16 * 16

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(channels[-1], reduction=4)

        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(self._feat_size, 256)),
            nn.ReLU(),
        )

        # Scalar input branch (ray distances, speed, gap, territory, etc.)
        self._cnn_feat_dim = 256  # output of self.fc
        if n_scalar_inputs > 0:
            self._scalar_feat_dim = 64
            self.scalar_fc = nn.Sequential(
                layer_init(nn.Linear(n_scalar_inputs, self._scalar_feat_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self._scalar_feat_dim, self._scalar_feat_dim)),
                nn.ReLU(),
            )
            gru_input_size = self._cnn_feat_dim + self._scalar_feat_dim
        else:
            self._scalar_feat_dim = 0
            gru_input_size = self._cnn_feat_dim
        self._gru_input_size = gru_input_size

        self.gru = nn.GRU(gru_input_size, gru_hidden)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(gru_hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(gru_hidden, 1), std=1.0)

    def _forward_gru(self, x, gru_state, done, scalars=None):
        x = self.conv(x)
        x = nn.functional.relu(x)
        if self.use_cbam:
            x = self.cbam(x)
        hidden = self.fc(x)  # (B*T, 256)

        # Concatenate scalar features if available
        if self.n_scalar_inputs > 0:
            if scalars is not None:
                scalar_feat = self.scalar_fc(scalars)  # (B*T, 64)
            else:
                scalar_feat = torch.zeros(
                    hidden.shape[0], self._scalar_feat_dim,
                    device=hidden.device, dtype=hidden.dtype)
            hidden = torch.cat([hidden, scalar_feat], dim=-1)  # (B*T, 320)

        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self._gru_input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            gru_state = (1.0 - d).view(1, -1, 1) * gru_state
            h, gru_state = self.gru(h.unsqueeze(0), gru_state)
            new_hidden.append(h)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_value(self, x, gru_state, done, scalars=None):
        hidden, _ = self._forward_gru(x, gru_state, done, scalars=scalars)
        return self.critic(hidden)

    def get_action_and_value(self, x, gru_state, done, action=None, eps=0.0, scalars=None):
        hidden, new_gru_state = self._forward_gru(x, gru_state, done, scalars=scalars)
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

    def get_action_greedy(self, x, gru_state, done, scalars=None):
        hidden, new_gru_state = self._forward_gru(x, gru_state, done, scalars=scalars)
        logits = self.actor(hidden)
        return logits.argmax(dim=1), new_gru_state


# ============================================================================
# Scripted Bot Opponents (for diverse self-play)
# ============================================================================

class ScriptedBot:
    """Base class for scripted opponents. get_action(env, player) → 0/1/2."""
    def get_action(self, env, player):
        raise NotImplementedError


class WallHugger(ScriptedBot):
    """Follows nearest wall, claims perimeter territory."""

    def get_action(self, env, player):
        best_act, best_wall_dist = 1, float('inf')
        px, py, angle = player.x, player.y, player.angle
        speed = env._speed / FPS
        tr = env._turn_rate / FPS

        for act in [0, 1, 2]:
            a = angle
            x, y = px, py
            alive = True
            for _ in range(6):
                if act == 0:
                    a -= tr
                elif act == 2:
                    a += tr
                x += math.cos(a) * speed
                y += math.sin(a) * speed
                ix, iy = int(round(x)), int(round(y))
                if (ix < 0 or ix >= ARENA_SIM or iy < env._offset_y or
                        iy >= env._offset_y + env._arena_h):
                    alive = False
                    break
                if 0 <= ix < ARENA_SIM and 0 <= iy < ARENA_SIM:
                    if env.trail_owner[iy, ix] > 0:
                        alive = False
                        break
            if not alive:
                continue
            wall_dist = min(x, ARENA_SIM - x,
                            y - env._offset_y,
                            env._offset_y + env._arena_h - y)
            if wall_dist < best_wall_dist:
                best_wall_dist = wall_dist
                best_act = act
        return best_act


class AggressiveCutter(ScriptedBot):
    """Steers toward nearest alive opponent to cut them off."""

    def get_action(self, env, player):
        # Find nearest alive opponent
        nearest_dist = float('inf')
        target = None
        for p in env.players:
            if p.id == player.id or not p.alive:
                continue
            d = math.hypot(p.x - player.x, p.y - player.y)
            if d < nearest_dist:
                nearest_dist = d
                target = p

        if target is None:
            return 1  # no target, go straight

        # Predict target position 10 frames ahead
        tx = target.x + math.cos(target.angle) * (env._speed / FPS) * 10
        ty = target.y + math.sin(target.angle) * (env._speed / FPS) * 10

        # Compute desired angle to intercept point
        dx, dy = tx - player.x, ty - player.y
        desired = math.atan2(dy, dx)
        diff = desired - player.angle
        # Normalize to [-pi, pi]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        if diff < -0.05:
            return 0  # turn left
        elif diff > 0.05:
            return 2  # turn right
        return 1  # go straight


class TerritoryMaximizer(ScriptedBot):
    """Picks action that moves toward largest open space."""

    def get_action(self, env, player):
        px, py, angle = player.x, player.y, player.angle
        speed = env._speed / FPS
        tr = env._turn_rate / FPS
        best_act, best_clear = 1, -1

        for act in [0, 1, 2]:
            a = angle
            x, y = px, py
            clear_dist = 0
            for step in range(1, 30):
                if act == 0:
                    a -= tr
                elif act == 2:
                    a += tr
                x += math.cos(a) * speed
                y += math.sin(a) * speed
                ix, iy = int(round(x)), int(round(y))
                if (ix < 0 or ix >= ARENA_SIM or iy < env._offset_y or
                        iy >= env._offset_y + env._arena_h):
                    break
                if 0 <= ix < ARENA_SIM and 0 <= iy < ARENA_SIM:
                    if env.trail_owner[iy, ix] > 0:
                        break
                clear_dist = step
            if clear_dist > best_clear:
                best_clear = clear_dist
                best_act = act
        return best_act


class BeamSearchBot(ScriptedBot):
    """3-step macro search over action sequences. Strongest scripted bot."""
    MACRO_LEN = 8  # frames per macro-action
    ACTIONS = [0, 1, 2]

    def get_action(self, env, player):
        px, py, angle = player.x, player.y, player.angle
        speed = env._speed / FPS
        tr = env._turn_rate / FPS
        best_score = -float('inf')
        best_first = 1

        # Search all 3-step sequences (3^3 = 27)
        for a1 in self.ACTIONS:
            for a2 in self.ACTIONS:
                for a3 in self.ACTIONS:
                    x, y, a = px, py, angle
                    alive = True
                    dist_traveled = 0
                    min_wall = float('inf')

                    for act in [a1, a2, a3]:
                        for _ in range(self.MACRO_LEN):
                            if act == 0:
                                a -= tr
                            elif act == 2:
                                a += tr
                            x += math.cos(a) * speed
                            y += math.sin(a) * speed
                            ix, iy = int(round(x)), int(round(y))
                            if (ix < 0 or ix >= ARENA_SIM or
                                    iy < env._offset_y or
                                    iy >= env._offset_y + env._arena_h):
                                alive = False
                                break
                            if 0 <= ix < ARENA_SIM and 0 <= iy < ARENA_SIM:
                                if env.trail_owner[iy, ix] > 0:
                                    alive = False
                                    break
                            wd = min(x, ARENA_SIM - x,
                                     y - env._offset_y,
                                     env._offset_y + env._arena_h - y)
                            min_wall = min(min_wall, wd)
                            dist_traveled += speed
                        if not alive:
                            break

                    if not alive:
                        score = -1000 + dist_traveled
                    else:
                        # Reward: survival + distance from walls + territory
                        score = dist_traveled + min_wall * 0.5
                        # Bonus for being near opponents (aggressive)
                        for p in env.players:
                            if p.id != player.id and p.alive:
                                d = math.hypot(x - p.x, y - p.y)
                                if d < 60:
                                    score += (60 - d) * 0.3

                    if score > best_score:
                        best_score = score
                        best_first = a1
        return best_first


ALL_SCRIPTED_BOTS = [WallHugger(), AggressiveCutter(), TerritoryMaximizer(), BeamSearchBot()]


# ============================================================================
# Observation Wrappers (add channels to base env)
# ============================================================================

class VoronoiWrapper(CurveCrashFFAEnv):
    """Adds Voronoi territory channel to observation.
    BFS flood fill from each alive player on the OBS_SIZE grid.
    Territory = binary: 1 where ego reaches first, 0 otherwise.

    This is THE key feature from Tron AI literature. Every winning
    Tron bot uses Voronoi territory as its primary evaluation signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_ch = (6 if self.gspp else 4) + (2 if self._minimap else 0)
        self._base_channels = base_ch
        n_channels = base_ch + 1  # +voronoi
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (n_channels, OBS_SIZE, OBS_SIZE), dtype=np.float32
        )

    def _compute_voronoi_territory(self):
        """BFS flood fill at observation resolution. Returns raw territory map.
        territory[y,x] = player_id of first to reach, 0 = unclaimed/blocked.
        Cached as self._last_territory for opponent obs and reward shaping."""
        f = self._ds_factor
        blocked_full = (self.trail_owner > 0)
        np.maximum(blocked_full, self._wall_mask_ds is not None, out=blocked_full)
        blocked_ds = blocked_full.reshape(OBS_SIZE, f, OBS_SIZE, f).any(axis=(1, 3))

        territory = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.int16)
        visited = blocked_ds.copy()

        queues = {}
        for p in self.players:
            if not p.alive:
                continue
            px = max(0, min(OBS_SIZE - 1, int(p.x / f)))
            py = max(0, min(OBS_SIZE - 1, int(p.y / f)))
            if not visited[py, px]:
                territory[py, px] = p.id
                visited[py, px] = True
                queues[p.id] = deque([(py, px)])
            else:
                queues[p.id] = deque()

        active = True
        while active:
            active = False
            for pid, q in queues.items():
                next_q = deque()
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < OBS_SIZE and 0 <= nx < OBS_SIZE and not visited[ny, nx]:
                            visited[ny, nx] = True
                            territory[ny, nx] = pid
                            next_q.append((ny, nx))
                            active = True
                queues[pid] = next_q

        self._last_territory = territory
        # Territory fraction for reward shaping
        total_cells = max(1, int(np.sum(territory > 0)))
        ego_cells = int(np.sum(territory == self.ego.id))
        self._ego_territory_frac = ego_cells / total_cells
        return territory

    def _compute_voronoi_ds(self):
        """Backward-compat wrapper: returns ego binary territory."""
        territory = self._compute_voronoi_territory()
        return (territory == self.ego.id).astype(np.float32)

    def _get_player_obs(self, player):
        """Override to append Voronoi channel."""
        base_obs = super()._get_player_obs(player)
        voronoi = self._compute_voronoi_ds()

        src_ri, src_ci, valid = self._compute_rotation_map(player)
        voronoi_rot = self._rotate_channel(voronoi, src_ri, src_ci, valid, 0.0)

        return np.concatenate([base_obs, voronoi_rot[np.newaxis]], axis=0)

    def get_opponent_observations(self):
        """Give each opponent their own rotated voronoi territory channel."""
        obs_list, live_mask = super().get_opponent_observations()
        territory = getattr(self, '_last_territory', None)
        if territory is None:
            territory = self._compute_voronoi_territory()

        result = []
        obs_idx = 0
        for j, alive in enumerate(live_mask):
            if not alive:
                continue
            player = self.players[j + 1]  # players[0] is ego
            player_territory = (territory == player.id).astype(np.float32)
            src_ri, src_ci, valid = self._compute_rotation_map(player)
            voronoi_rot = self._rotate_channel(player_territory, src_ri, src_ci, valid, 0.0)
            result.append(np.concatenate([obs_list[obs_idx], voronoi_rot[np.newaxis]], axis=0))
            obs_idx += 1
        return result, live_mask


class DistanceWrapper(CurveCrashFFAEnv):
    """Adds distance-to-nearest-obstacle channel.
    Normalized float [0, 1] where 0 = on obstacle, 1 = far away.
    Gives the agent gradient signal about wall/trail proximity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_ch = 6 if self.gspp else 4
        self._base_channels = base_ch
        n_channels = base_ch + 1  # +distance
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (n_channels, OBS_SIZE, OBS_SIZE), dtype=np.float32
        )

    def _compute_distance_ds(self):
        """Distance transform at observation resolution.
        Returns (OBS_SIZE, OBS_SIZE) float32 normalized to [0, 1]."""
        from scipy.ndimage import distance_transform_edt
        f = self._ds_factor
        # All obstacles at obs resolution
        blocked = (self.trail_owner > 0).reshape(
            OBS_SIZE, f, OBS_SIZE, f
        ).any(axis=(1, 3)).astype(np.float32)
        np.maximum(blocked, self._wall_mask_ds, out=blocked)

        # EDT on free space (invert blocked)
        free = 1.0 - blocked
        dist = distance_transform_edt(free).astype(np.float32)

        # Normalize: max possible distance is ~OBS_SIZE/2
        max_dist = OBS_SIZE / 2.0
        dist = np.clip(dist / max_dist, 0.0, 1.0)
        return dist

    def _get_player_obs(self, player):
        """Override to append distance channel."""
        base_obs = super()._get_player_obs(player)
        dist_ch = self._compute_distance_ds()

        src_ri, src_ci, valid = self._compute_rotation_map(player)
        dist_rot = self._rotate_channel(dist_ch, src_ri, src_ci, valid, 0.0)

        return np.concatenate([base_obs, dist_rot[np.newaxis]], axis=0)

    def get_opponent_observations(self):
        """Pad opponent obs with zero distance channel."""
        obs_list, live_mask = super().get_opponent_observations()
        return [np.concatenate([o, np.zeros((1, OBS_SIZE, OBS_SIZE), dtype=np.float32)], axis=0)
                for o in obs_list], live_mask


class VoronoiDistanceWrapper(CurveCrashFFAEnv):
    """Adds both Voronoi territory AND distance-to-obstacle channels.
    8 total channels for GS++ mode (6 base + voronoi + distance)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_ch = 6 if self.gspp else 4
        self._base_channels = base_ch
        n_channels = base_ch + 2  # +voronoi +distance
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (n_channels, OBS_SIZE, OBS_SIZE), dtype=np.float32
        )

    def _compute_voronoi_territory(self):
        """BFS flood fill. Returns raw territory map, caches for opponent obs."""
        f = self._ds_factor
        blocked_ds = (self.trail_owner > 0).reshape(
            OBS_SIZE, f, OBS_SIZE, f
        ).any(axis=(1, 3))

        territory = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.int16)
        visited = blocked_ds.copy()

        queues = {}
        for p in self.players:
            if not p.alive:
                continue
            px = max(0, min(OBS_SIZE - 1, int(p.x / f)))
            py = max(0, min(OBS_SIZE - 1, int(p.y / f)))
            if not visited[py, px]:
                territory[py, px] = p.id
                visited[py, px] = True
                queues[p.id] = deque([(py, px)])
            else:
                queues[p.id] = deque()

        active = True
        while active:
            active = False
            for pid, q in queues.items():
                next_q = deque()
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < OBS_SIZE and 0 <= nx < OBS_SIZE and not visited[ny, nx]:
                            visited[ny, nx] = True
                            territory[ny, nx] = pid
                            next_q.append((ny, nx))
                            active = True
                queues[pid] = next_q

        self._last_territory = territory
        total_cells = max(1, int(np.sum(territory > 0)))
        self._ego_territory_frac = int(np.sum(territory == self.ego.id)) / total_cells
        return territory

    def _compute_voronoi_ds(self):
        territory = self._compute_voronoi_territory()
        return (territory == self.ego.id).astype(np.float32)

    def _compute_distance_ds(self):
        from scipy.ndimage import distance_transform_edt
        f = self._ds_factor
        blocked = (self.trail_owner > 0).reshape(
            OBS_SIZE, f, OBS_SIZE, f
        ).any(axis=(1, 3)).astype(np.float32)
        np.maximum(blocked, self._wall_mask_ds, out=blocked)
        free = 1.0 - blocked
        dist = distance_transform_edt(free).astype(np.float32)
        return np.clip(dist / (OBS_SIZE / 2.0), 0.0, 1.0)

    def _get_player_obs(self, player):
        base_obs = super()._get_player_obs(player)
        voronoi = self._compute_voronoi_ds()
        dist_ch = self._compute_distance_ds()

        src_ri, src_ci, valid = self._compute_rotation_map(player)
        voronoi_rot = self._rotate_channel(voronoi, src_ri, src_ci, valid, 0.0)
        dist_rot = self._rotate_channel(dist_ch, src_ri, src_ci, valid, 0.0)

        return np.concatenate([
            base_obs,
            voronoi_rot[np.newaxis],
            dist_rot[np.newaxis],
        ], axis=0)

    def get_opponent_observations(self):
        """Give opponents real voronoi + zero distance channel."""
        obs_list, live_mask = super().get_opponent_observations()
        territory = getattr(self, '_last_territory', None)
        if territory is None:
            territory = self._compute_voronoi_territory()

        result = []
        obs_idx = 0
        for j, alive in enumerate(live_mask):
            if not alive:
                continue
            player = self.players[j + 1]
            player_territory = (territory == player.id).astype(np.float32)
            src_ri, src_ci, valid = self._compute_rotation_map(player)
            voronoi_rot = self._rotate_channel(player_territory, src_ri, src_ci, valid, 0.0)
            dist_zero = np.zeros((1, OBS_SIZE, OBS_SIZE), dtype=np.float32)
            result.append(np.concatenate([obs_list[obs_idx], voronoi_rot[np.newaxis], dist_zero], axis=0))
            obs_idx += 1
        return result, live_mask


# Need gym import for observation space override
import gymnasium as gym


# ============================================================================
# Self-Play: PFSP (Prioritized Fictitious Self-Play)
# ============================================================================

class PFSPOpponentPool:
    """AlphaStar-style prioritized opponent sampling.
    Samples opponents proportional to how much we struggle against them.
    f(x) = (1-x)^p where x = win rate against that opponent."""

    def __init__(self, max_size=100, priority_exponent=2.0):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.step_tags = deque(maxlen=max_size)
        self.wins = deque(maxlen=max_size)   # wins against each opponent
        self.games = deque(maxlen=max_size)  # total games against each
        self.priority_exp = priority_exponent

    def add(self, state_dict, step):
        self.pool.append(state_dict)
        self.step_tags.append(step)
        self.wins.append(0)
        self.games.append(0)

    def record_result(self, idx, won):
        """Record a game result against opponent at index idx."""
        if 0 <= idx < len(self.pool):
            self.games[idx] += 1
            if won:
                self.wins[idx] += 1

    def sample(self):
        """PFSP sampling: prioritize opponents we lose to more."""
        if not self.pool:
            return None, -1
        n = len(self.pool)

        # Compute priorities
        priorities = np.zeros(n)
        for i in range(n):
            if self.games[i] < 5:
                # Not enough data — give high priority (explore)
                priorities[i] = 1.0
            else:
                win_rate = self.wins[i] / self.games[i]
                # f(x) = (1-x)^p: higher priority when we lose more
                priorities[i] = (1.0 - win_rate) ** self.priority_exp

        # Normalize to probability distribution
        total = priorities.sum()
        if total < 1e-8:
            # All win rates ~100%, sample uniformly from recent
            idx = random.randint(max(0, n - max(1, n // 3)), n - 1)
        else:
            probs = priorities / total
            idx = np.random.choice(n, p=probs)

        return self.pool[idx], idx

    def __len__(self):
        return len(self.pool)


class UniformOpponentPool:
    """Current approach: 80% recent, 20% uniform."""
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.step_tags = deque(maxlen=max_size)

    def add(self, state_dict, step):
        self.pool.append(state_dict)
        self.step_tags.append(step)

    def record_result(self, idx, won):
        pass  # no-op for uniform pool

    def sample(self):
        if not self.pool:
            return None, -1
        n = len(self.pool)
        recent_start = max(0, n - max(1, n // 3))
        if random.random() < 0.8 and recent_start < n:
            idx = random.randint(recent_start, n - 1)
        else:
            idx = random.randint(0, n - 1)
        return self.pool[idx], idx

    def __len__(self):
        return len(self.pool)


# ============================================================================
# Experiment Configs
# ============================================================================

EXPERIMENTS = {
    # --- Baseline ---
    "baseline": {
        "desc": "Current v6: NatureCNN+CBAM+SpatialAttn+GRU(128), 6ch, uniform pool",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
    },

    # --- Architecture ablations ---
    "no_spatial_attn": {
        "desc": "NatureCNN+CBAM+GRU(128) — remove spatial self-attention",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": False},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
    },
    "gru64": {
        "desc": "NatureCNN+CBAM+SpatialAttn+GRU(64) — smaller memory",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 64, "use_spatial_attn": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
    },
    "gru256": {
        "desc": "NatureCNN+CBAM+SpatialAttn+GRU(256) — bigger memory",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 256, "use_spatial_attn": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
    },
    "impala_cnn": {
        "desc": "IMPALA-CNN(16,32,32)+CBAM+GRU(128) — residual backbone",
        "agent_cls": ImpalaCNNAgent,
        "agent_kwargs": {"gru_hidden": 128, "channels": (16, 32, 32), "use_cbam": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
    },
    "impala_cnn_wide": {
        "desc": "IMPALA-CNN(32,64,64)+CBAM+GRU(128) — wider residual backbone",
        "agent_cls": ImpalaCNNAgent,
        "agent_kwargs": {"gru_hidden": 128, "channels": (32, 64, 64), "use_cbam": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
    },

    # --- Observation ablations ---
    "voronoi": {
        "desc": "NatureCNN+CBAM+SpatialAttn+GRU(128), 7ch (+Voronoi territory)",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": VoronoiWrapper,
        "pool_cls": UniformOpponentPool,
        "n_channels": 7,
    },
    "distance": {
        "desc": "NatureCNN+CBAM+SpatialAttn+GRU(128), 7ch (+distance field)",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": DistanceWrapper,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6 + 1,
    },
    "voronoi_distance": {
        "desc": "NatureCNN+CBAM+SpatialAttn+GRU(128), 8ch (+Voronoi+distance)",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": VoronoiDistanceWrapper,
        "pool_cls": UniformOpponentPool,
        "n_channels": 8,
    },

    # --- Combined: best architecture + best observations ---
    "impala_voronoi": {
        "desc": "IMPALA-CNN(16,32,32)+CBAM+GRU(128), 7ch (+Voronoi)",
        "agent_cls": ImpalaCNNAgent,
        "agent_kwargs": {"gru_hidden": 128, "channels": (16, 32, 32), "use_cbam": True},
        "env_cls": VoronoiWrapper,
        "pool_cls": UniformOpponentPool,
        "n_channels": 7,
    },
    "impala_voronoi_distance": {
        "desc": "IMPALA-CNN(16,32,32)+CBAM+GRU(128), 8ch (+Voronoi+distance)",
        "agent_cls": ImpalaCNNAgent,
        "agent_kwargs": {"gru_hidden": 128, "channels": (16, 32, 32), "use_cbam": True},
        "env_cls": VoronoiDistanceWrapper,
        "pool_cls": UniformOpponentPool,
        "n_channels": 8,
    },

    # --- Self-play ablations ---
    "pfsp": {
        "desc": "NatureCNN+CBAM+SpatialAttn+GRU(128), 6ch, PFSP pool",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": PFSPOpponentPool,
        "n_channels": 6,
    },

    # --- The kitchen sink: all improvements combined ---
    "full_upgrade": {
        "desc": "IMPALA-CNN(16,32,32)+CBAM+GRU(128), 8ch (Voronoi+dist), PFSP",
        "agent_cls": ImpalaCNNAgent,
        "agent_kwargs": {"gru_hidden": 128, "channels": (16, 32, 32), "use_cbam": True},
        "env_cls": VoronoiDistanceWrapper,
        "pool_cls": PFSPOpponentPool,
        "n_channels": 8,
    },

    # --- BC (behavioral cloning) auxiliary loss experiments ---
    # BC data is 6ch only, so these use base env (no voronoi/distance)
    "bc_baseline": {
        "desc": "Baseline + BC aux loss (elite replay data, bc_coef=0.1)",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
        "bc_coef": 0.1,
    },
    "bc_impala": {
        "desc": "IMPALA-CNN + BC aux loss (elite replay data, bc_coef=0.1)",
        "agent_cls": ImpalaCNNAgent,
        "agent_kwargs": {"gru_hidden": 128, "channels": (16, 32, 32), "use_cbam": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": UniformOpponentPool,
        "n_channels": 6,
        "bc_coef": 0.1,
    },
    "bc_pfsp": {
        "desc": "Baseline + BC aux loss + PFSP (AlphaStar-style)",
        "agent_cls": BaselineAgent,
        "agent_kwargs": {"gru_hidden": 128, "use_spatial_attn": True},
        "env_cls": CurveCrashFFAEnv,
        "pool_cls": PFSPOpponentPool,
        "n_channels": 6,
        "bc_coef": 0.1,
    },
}


# ============================================================================
# Simplified Self-Play Vec Env (for experiments)
# ============================================================================

class ExperimentVecEnv:
    """Lightweight vec env for experiments. Opponents use random actions
    initially, then self-play snapshots once pool has entries."""

    MAX_OPP_PER_ENV = 10

    def __init__(self, envs, device, agent_cls, agent_kwargs, n_channels):
        self.envs = envs
        self.num_envs = len(envs)
        self.device = device
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

        self.opponent_model = None
        self._agent_cls = agent_cls
        self._agent_kwargs = agent_kwargs
        self._n_channels = n_channels
        self.opp_gru_states = None
        self._current_pool_idx = -1

    def set_opponent(self, state_dict, pool_idx=-1):
        gru_h = self._agent_kwargs.get("gru_hidden", 128)
        if self.opponent_model is None:
            self.opponent_model = self._agent_cls(
                n_input_channels=self._n_channels, **self._agent_kwargs
            ).to(self.device)
            self.opp_gru_states = torch.zeros(
                1, self.num_envs * self.MAX_OPP_PER_ENV, gru_h, device=self.device
            )
        self.opponent_model.load_state_dict(state_dict)
        self.opponent_model.eval()
        self._current_pool_idx = pool_idx
        self.opp_gru_states.zero_()

    def reset(self):
        obs_list = []
        for env in self.envs:
            o, _ = env.reset()
            obs_list.append(o)
        if self.opp_gru_states is not None:
            self.opp_gru_states.zero_()
        return np.stack(obs_list)

    def step(self, ego_actions):
        # Collect opponent obs
        all_opp_obs = []
        gru_indices = []
        env_meta = []
        for i, env in enumerate(self.envs):
            opp_obs, live_mask = env.get_opponent_observations()
            all_opp_obs.extend(opp_obs)
            for j, alive in enumerate(live_mask):
                if alive:
                    gru_indices.append(i * self.MAX_OPP_PER_ENV + j)
            env_meta.append((len(env.players) - 1, live_mask))

        # Opponent actions
        if all_opp_obs and self.opponent_model is not None:
            gru_h = self._agent_kwargs.get("gru_hidden", 128)
            with torch.no_grad():
                obs_t = torch.tensor(
                    np.stack(all_opp_obs), dtype=torch.float32, device=self.device
                )
                gru_batch = self.opp_gru_states[:, gru_indices, :]
                done_zeros = torch.zeros(len(all_opp_obs), device=self.device)
                acts, new_gru = self.opponent_model.get_action_greedy(
                    obs_t, gru_batch, done_zeros
                )
                all_actions = acts.cpu().numpy()
                self.opp_gru_states[:, gru_indices, :] = new_gru
                del obs_t
        elif all_opp_obs:
            all_actions = np.random.randint(0, 3, size=len(all_opp_obs))
        else:
            all_actions = np.array([], dtype=np.int64)

        # Step envs
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        act_idx = 0
        for i, (env, ego_a) in enumerate(zip(self.envs, ego_actions)):
            n_opp, live_mask = env_meta[i]
            opp_actions = np.ones(n_opp, dtype=np.int64)
            for j, alive in enumerate(live_mask):
                if alive:
                    opp_actions[j] = all_actions[act_idx]
                    act_idx += 1

            o, r, te, tr, info = env.step(int(ego_a), opp_actions)

            if te or tr:
                start = i * self.MAX_OPP_PER_ENV
                end = start + self.MAX_OPP_PER_ENV
                if self.opp_gru_states is not None:
                    self.opp_gru_states[:, start:end, :] = 0
                final_info = info.copy()
                o, _ = env.reset()
                info = final_info

            obs_list.append(o)
            rewards.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)

        return (
            np.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terms),
            np.array(truncs),
            infos,
        )


# ============================================================================
# Training Loop (stripped-down PPO for experiments)
# ============================================================================

def run_experiment(name, config, total_steps, seed=1, num_envs=8, num_steps=128,
                   resume_from=None):
    """Run a single experiment and return metrics dict."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {config['desc']}")
    print(f"  Steps: {total_steps:,}  Envs: {num_envs}  Seed: {seed}")
    print(f"{'='*70}\n")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)

    n_channels = config["n_channels"]
    env_cls = config["env_cls"]
    agent_cls = config["agent_cls"]
    agent_kwargs = config["agent_kwargs"].copy()
    pool_cls = config["pool_cls"]
    bc_coef = config.get("bc_coef", 0.0)

    # Load BC data if experiment uses it
    bc_pool = None
    if bc_coef > 0:
        bc_pool = BCDataPool(BC_DATA_DIR, device)

    # Create envs
    def make_env(s):
        return env_cls(
            min_players=2, max_players=6,
            arena_variation=True, gspp=True,
            powerup_spawn_interval=38,
            kill_reward=0.5, speed_pickup_reward=0.2,
        )
    envs = ExperimentVecEnv(
        [make_env(seed + i) for i in range(num_envs)],
        device, agent_cls, agent_kwargs, n_channels,
    )
    for i, env in enumerate(envs.envs):
        env.reset(seed=seed + i)

    # Create agent
    agent = agent_cls(n_input_channels=n_channels, **agent_kwargs).to(device)
    if resume_from:
        sd = torch.load(resume_from, map_location=device, weights_only=True)
        agent.load_state_dict(sd, strict=False)
        print(f"  Resumed from: {resume_from}")

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Model params: {total_params:,}")

    gru_hidden = agent_kwargs.get("gru_hidden", 128)
    optimizer = optim.AdamW(agent.parameters(), lr=2.5e-4, eps=1e-5, weight_decay=1e-4)

    # Opponent pool
    pool = pool_cls(max_size=50)

    # Storage
    batch_size = num_envs * num_steps
    num_updates = total_steps // batch_size

    obs_buf = torch.zeros(
        (num_steps, num_envs, n_channels, OBS_SIZE, OBS_SIZE), device=device
    )
    actions_buf = torch.zeros((num_steps, num_envs), device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)

    gru_state = torch.zeros(1, num_envs, gru_hidden, device=device)
    next_obs_np = envs.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    # Metrics
    global_step = 0
    start_time = time.time()
    win_buffer = deque(maxlen=200)
    ep_ret_buffer = deque(maxlen=200)
    ep_len_buffer = deque(maxlen=200)
    kill_buffer = deque(maxlen=200)
    speed_pickup_buffer = deque(maxlen=200)

    snapshot_freq = 200_000
    opp_update_freq = 100_000
    next_snapshot = snapshot_freq
    next_opp_update = opp_update_freq

    # PPO hyperparams
    gamma = 0.995
    gae_lambda = 0.95
    clip_coef = 0.1
    ent_coef = 0.05
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.015
    eps_start = 0.02
    eps_end = 0.005
    envsperbatch = num_envs // 4

    for update in range(1, num_updates + 1):
        # LR anneal
        frac = 1.0 - (update - 1) / num_updates
        optimizer.param_groups[0]["lr"] = frac * 2.5e-4

        progress = min(1.0, global_step / total_steps)
        current_eps = eps_start + (eps_end - eps_start) * progress

        # Constant aggression — no decay
        for env in envs.envs:
            env.aggression_bonus = 1.0

        # Curriculum
        if global_step < total_steps * 0.02:
            min_p, max_p = 1, 1
        elif global_step < total_steps * 0.10:
            min_p, max_p = 2, 3
        elif global_step < total_steps * 0.25:
            min_p, max_p = 2, 4
        else:
            min_p, max_p = 2, 6
        for env in envs.envs:
            env.min_players = min_p
            env.max_players = max_p

        initial_gru_state = gru_state.clone()

        # Rollout
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, gru_state = agent.get_action_and_value(
                    next_obs, gru_state, next_done, eps=current_eps
                )
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            if device.type == "cuda":
                torch.cuda.synchronize()

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(
                action.cpu().numpy()
            )

            # Reward shaping — constant, no decay
            # Instant small signals for pickups/gaps
            for i, info in enumerate(infos):
                reward_np[i] += 0.2 * info.get("ego_speed_pickups", 0)
                reward_np[i] += 0.1 * info.get("ego_erase_pickups", 0)
                reward_np[i] += 0.03 * info.get("ego_gap_passes", 0)
                # Small instant kill signal (reduced from 0.5 → 0.1)
                reward_np[i] += 0.1 * info.get("step_kills", 0)

                # Survival-scaled kill bonus at episode end:
                # Kill+survive = big reward, kill+die immediately = small reward
                # This prevents kamikaze: grab speed → kill one → die
                if "episode" in info:
                    ep_kills = info.get("ego_kills", 0)
                    ep_len = info["episode"]["l"]
                    survival_ratio = min(1.0, ep_len / MAX_AGENT_STEPS)
                    reward_np[i] += ep_kills * survival_ratio * 0.5

            rewards_buf[step].copy_(torch.as_tensor(reward_np, device=device))
            next_obs.copy_(torch.as_tensor(next_obs_np, device=device))
            next_done.copy_(torch.as_tensor(
                np.logical_or(term_np, trunc_np).astype(np.float32), device=device
            ))

            for info in infos:
                if "episode" in info:
                    ep_r = info["episode"]["r"]
                    ep_l = info["episode"]["l"]
                    is_win = info.get("win", False)
                    kills = info.get("ego_kills", 0)
                    speed = info.get("ego_speed_pickups", 0)
                    win_buffer.append(float(is_win))
                    ep_ret_buffer.append(ep_r)
                    ep_len_buffer.append(ep_l)
                    kill_buffer.append(kills)
                    speed_pickup_buffer.append(speed)

                    # Record result for PFSP
                    if hasattr(pool, 'record_result') and envs._current_pool_idx >= 0:
                        pool.record_result(envs._current_pool_idx, is_win)

        # Opponent pool management
        if global_step >= next_snapshot:
            pool.add(
                {k: v.cpu() for k, v in agent.state_dict().items()},
                global_step,
            )
            next_snapshot += snapshot_freq

        if global_step >= next_opp_update and len(pool) > 0:
            sd, idx = pool.sample()
            envs.set_opponent(sd, pool_idx=idx)
            next_opp_update += opp_update_freq

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs, gru_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # PPO update
        b_obs = obs_buf.reshape((-1, n_channels, OBS_SIZE, OBS_SIZE))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_dones = dones_buf.reshape(-1)

        for epoch in range(4):
            envinds = np.arange(num_envs)
            np.random.shuffle(envinds)

            for mb_start in range(0, num_envs, envsperbatch):
                mbenvinds = envinds[mb_start:mb_start + envsperbatch]
                mb_inds = np.array([
                    t * num_envs + e
                    for t in range(num_steps) for e in mbenvinds
                ])

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions.long()[mb_inds]
                mb_dones = b_dones[mb_inds]
                mb_gru_state = initial_gru_state[:, mbenvinds].contiguous()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs, mb_gru_state, mb_dones, mb_actions, eps=current_eps
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                # BC auxiliary loss: cross-entropy on expert demonstrations
                if bc_pool is not None:
                    bc_batch = min(256, len(mb_inds))
                    bc_obs, bc_act = bc_pool.sample(bc_batch)
                    bc_gru = torch.zeros(1, bc_batch, gru_hidden, device=device)
                    bc_done = torch.zeros(bc_batch, device=device)
                    bc_hidden, _ = agent._forward_gru(bc_obs, bc_gru, bc_done)
                    bc_logits = agent.actor(bc_hidden)
                    bc_loss = F.cross_entropy(bc_logits, bc_act)
                    loss = loss + bc_coef * bc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        # Logging
        if update % 10 == 0:
            sps = int(global_step / (time.time() - start_time))
            wr = f"{sum(win_buffer)/len(win_buffer):.1%}" if win_buffer else "N/A"
            avg_kills = f"{sum(kill_buffer)/len(kill_buffer):.2f}" if kill_buffer else "N/A"
            bc_str = f"  bc={bc_loss.item():.4f}" if bc_pool is not None else ""
            print(
                f"  [{name}] step={global_step:>10,}  SPS={sps}  "
                f"pg={pg_loss.item():.4f}  vf={v_loss.item():.4f}  "
                f"ent={entropy_loss.item():.3f}{bc_str}  win={wr}  kills={avg_kills}"
            )

        if update % 50 == 0:
            gc.collect()

    # Final metrics
    elapsed = time.time() - start_time
    results = {
        "experiment": name,
        "description": config["desc"],
        "total_steps": total_steps,
        "params": total_params,
        "n_channels": n_channels,
        "gru_hidden": gru_hidden,
        "elapsed_s": round(elapsed, 1),
        "sps": int(global_step / elapsed),
        "win_rate": round(sum(win_buffer) / len(win_buffer), 4) if win_buffer else 0.0,
        "avg_return": round(sum(ep_ret_buffer) / len(ep_ret_buffer), 4) if ep_ret_buffer else 0.0,
        "avg_length": round(sum(ep_len_buffer) / len(ep_len_buffer), 1) if ep_len_buffer else 0.0,
        "avg_kills": round(sum(kill_buffer) / len(kill_buffer), 3) if kill_buffer else 0.0,
        "avg_speed_pickups": round(sum(speed_pickup_buffer) / len(speed_pickup_buffer), 3) if speed_pickup_buffer else 0.0,
        "bc_coef": bc_coef,
    }

    # Save checkpoint
    exp_dir = RESULTS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / "agent_final.pt"
    torch.save(agent.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # Append to CSV
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    print(f"\n  Results: win={results['win_rate']:.1%}  "
          f"return={results['avg_return']:.3f}  "
          f"kills={results['avg_kills']:.2f}  "
          f"speed={results['avg_speed_pickups']:.2f}  "
          f"SPS={results['sps']}")

    # Cleanup
    del agent, envs, optimizer
    if bc_pool is not None:
        del bc_pool
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def print_comparison():
    """Print results comparison table from CSV."""
    if not RESULTS_CSV.exists():
        print("No results yet. Run experiments first.")
        return

    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No results yet.")
        return

    # Print table
    print(f"\n{'='*120}")
    print(f"{'EXPERIMENT RESULTS':^120}")
    print(f"{'='*120}")
    print(f"{'Experiment':<28} {'Params':>8} {'Ch':>3} {'GRU':>4} "
          f"{'Win%':>6} {'Return':>8} {'Length':>7} {'Kills':>6} {'Speed':>6} "
          f"{'SPS':>6} {'Time':>8}")
    print(f"{'-'*120}")

    for r in rows:
        elapsed = float(r.get('elapsed_s', 0))
        time_str = f"{elapsed/60:.0f}m" if elapsed > 0 else "?"
        print(
            f"{r['experiment']:<28} "
            f"{int(r.get('params', 0)):>8,} "
            f"{r.get('n_channels', '?'):>3} "
            f"{r.get('gru_hidden', '?'):>4} "
            f"{float(r.get('win_rate', 0))*100:>5.1f}% "
            f"{float(r.get('avg_return', 0)):>8.3f} "
            f"{float(r.get('avg_length', 0)):>7.1f} "
            f"{float(r.get('avg_kills', 0)):>6.2f} "
            f"{float(r.get('avg_speed_pickups', 0)):>6.2f} "
            f"{int(r.get('sps', 0)):>6} "
            f"{time_str:>8}"
        )
    print(f"{'='*120}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="CurveCrash AI Ablation Experiments")
    p.add_argument("--list", action="store_true", help="List all experiments")
    p.add_argument("--run", action="append", default=[], help="Run specific experiment(s)")
    p.add_argument("--run-all", action="store_true", help="Run all experiments")
    p.add_argument("--compare", action="store_true", help="Print comparison table")
    p.add_argument("--steps", type=int, default=1_000_000, help="Training steps per experiment")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume all experiments from this checkpoint (warm-start)")
    args = p.parse_args()

    if args.list:
        print(f"\nAvailable experiments ({len(EXPERIMENTS)}):\n")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:<28} {cfg['desc']}")
        print()
        return

    if args.compare:
        print_comparison()
        return

    experiments_to_run = []
    if args.run_all:
        experiments_to_run = list(EXPERIMENTS.keys())
    elif args.run:
        for name in args.run:
            if name not in EXPERIMENTS:
                print(f"Unknown experiment: {name}")
                print(f"Available: {', '.join(EXPERIMENTS.keys())}")
                return
        experiments_to_run = args.run

    if not experiments_to_run:
        p.print_help()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(experiments_to_run)} experiment(s), {args.steps:,} steps each")
    print(f"Results will be saved to: {RESULTS_CSV}")

    all_results = []
    for name in experiments_to_run:
        result = run_experiment(
            name, EXPERIMENTS[name], args.steps,
            seed=args.seed, num_envs=args.num_envs,
            resume_from=args.resume_from,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print_comparison()


if __name__ == "__main__":
    main()
