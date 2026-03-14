"""
FFA CurveCrash Environment for self-play RL (v3 - ego-centric rotation).

Observation: (4, 128, 128) ego-centric rotated binary
  Centered on ego, rotated so ego faces RIGHT (+x).
  Ch 0: self trail + walls (current, rotated)
  Ch 1: enemy trails + arrows (current, rotated)
  Ch 2-3: previous frame of ch 0-1 (rotated with CURRENT heading)
  Out-of-arena pixels: ch0=1.0 (wall/death), ch1=0.0 (no enemies)

Action:  Discrete(3) — 0=left, 1=straight, 2=right
Reward:  +0.01/step alive, -1.0 on death (rescaled for PPO stability)
Players: 2-11 (random per episode)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# === PHYSICS: "None" mode (measured from 256x256 square arena, curvecrash_sim.py) ===
ARENA_SIM = 512
FPS = 60
SPEED_RATIO = 0.1289       # 33 px/s at 256 → 66 px/s at 512
TURN_RADIUS_RATIO = 0.0486 # 12.5 px at 256 → 24.9 px at 512
TRAIL_WIDTH_RATIO = 0.008  # 2.0 px at 256 → 4.1 px at 512

SPEED = SPEED_RATIO * ARENA_SIM
TURN_RADIUS = TURN_RADIUS_RATIO * ARENA_SIM
TRAIL_WIDTH = TRAIL_WIDTH_RATIO * ARENA_SIM
TURN_RATE = SPEED / TURN_RADIUS

# === PHYSICS: "GS++" mode (from replay data: 922x763 field) ===
# Actual values from ffa_gspp.ndjson: speed=180, tr=38.5, size=6.3, hs=22.4
# Ratios derived as: raw_value / max_field_dimension (922).
_GSPP_FW = 922
SPEED_RATIO_GSPP = 180.0 / _GSPP_FW          # ≈ 0.1952 → 100 px/s at 512
TURN_RADIUS_RATIO_GSPP = 38.5 / _GSPP_FW     # ≈ 0.04175 → 21.4 px at 512
TRAIL_WIDTH_RATIO_GSPP = 6.3 / _GSPP_FW      # ≈ 0.006832 → 3.5 px at 512
GAP_LENGTH_RATIO_GSPP = 22.4 / _GSPP_FW      # ≈ 0.02429 → 12.4 px at 512

# Rectangular field for GS++ (real game: 922x763)
_GSPP_FH = 763
ARENA_H_GSPP = int(round(_GSPP_FH * (ARENA_SIM / _GSPP_FW)))  # 424
ARENA_OFFSET_Y_GSPP = (ARENA_SIM - ARENA_H_GSPP) // 2         # 44

GAP_DURATION = 0.159  # None mode gap duration (seconds)
GAP_INTERVAL_MIN = 1.1
GAP_INTERVAL_MAX = 2.0

# Gap-pass detection
GAP_PASS_DETECT_WIDTH = TRAIL_WIDTH * 3  # perpendicular detection zone
GAP_PASS_MAX_AGE = 30.0  # seconds before gap zone expires

# === ENV PARAMS ===
OBS_SIZE = 128
FRAME_SKIP = 3
MAX_AGENT_STEPS = 3600
SPAWN_SECONDS = 3.0
N_SCALAR_OBS = 13  # ray distances (8) + speed_boost + gap_progress + alive_frac + territory_frac + speed_multiplier

# Reward: rescaled from 1.0→0.01 so entropy bonus is meaningful relative to returns.
# Old: 100-step episode = 100 return → ent_coef*entropy is negligible → collapse.
# New: 100-step episode = 1.0 return → ent_coef=0.05 actually matters.
REWARD_ALIVE = 0.01   # Solo only: survival teaches basic movement
REWARD_DEATH = -1.0   # Multiplayer + solo: strong "don't die" signal
REWARD_WIN = 1.0      # Multiplayer only: last alive, episode ends immediately
# FFA reward design: solo uses alive drip (learn basics), multiplayer uses
# only win(+1)/death(-1) terminal signals. No per-step reward in FFA —
# the agent should play to WIN, not to survive longest.


class PlayerState:
    __slots__ = [
        'id', 'x', 'y', 'angle', 'alive', 'spawning', 'drawing',
        'time_alive', 'gap_timer', 'next_gap_at', 'gap_remaining',
        'just_died', 'prev_channels',
        'gap_start_x', 'gap_start_y', 'gap_just_closed',
        'speed_boosts',
        'prev_draw_x', 'prev_draw_y',
        'prev_move_x', 'prev_move_y',
        'death_order',
    ]

    def __init__(self, player_id, x, y, angle, next_gap_at):
        self.id = player_id
        self.x, self.y = x, y
        self.angle = angle
        self.alive = True
        self.spawning = True
        self.drawing = False
        self.time_alive = 0.0
        self.gap_timer = 0.0
        self.next_gap_at = next_gap_at
        self.gap_remaining = 0.0
        self.just_died = False
        self.prev_channels = np.zeros((2, OBS_SIZE, OBS_SIZE), dtype=np.float32)
        self.gap_start_x = 0.0
        self.gap_start_y = 0.0
        self.gap_just_closed = False
        self.speed_boosts = []  # [(expire_frame, multiplier), ...]
        self.prev_draw_x = x
        self.prev_draw_y = y
        self.prev_move_x = x
        self.prev_move_y = y
        self.death_order = 0


class Powerup:
    """A GS++ powerup on the field."""
    __slots__ = ['ptype', 'x', 'y', 'radius']
    def __init__(self, ptype, x, y, radius):
        self.ptype = ptype   # 'speed' or 'erase'
        self.x = x
        self.y = y
        self.radius = radius


class GapZone:
    """Tracks a gap in a trail for gap-pass detection.
    A gap pass = another player enters the gap zone from one side
    and exits alive on the other side (fully traversed)."""
    __slots__ = ['owner_id', 'cx', 'cy', 'tx', 'ty', 'nx', 'ny',
                 'half_len', 'tracking', 'frame']

    def __init__(self, owner_id, sx, sy, ex, ey, frame):
        self.owner_id = owner_id
        self.cx = (sx + ex) / 2
        self.cy = (sy + ey) / 2
        length = math.hypot(ex - sx, ey - sy)
        # Half-length along trail + margin for detection
        self.half_len = max(length / 2 + TRAIL_WIDTH, TRAIL_WIDTH * 2)
        if length > 0.01:
            self.tx = (ex - sx) / length  # trail direction (unit)
            self.ty = (ey - sy) / length
        else:
            self.tx, self.ty = 1.0, 0.0
        self.nx = -self.ty  # normal (perpendicular to trail)
        self.ny = self.tx
        self.tracking = {}  # player_id -> entry side sign (+1 or -1)
        self.frame = frame


class CurveCrashFFAEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, min_players=2, max_players=11, render_mode=None,
                 arena_variation=False, gspp=False,
                 powerup_spawn_interval=76,
                 powerup_speed_weight=2, powerup_erase_weight=1,
                 aggression_bonus=0.0, kill_reward=0.0,
                 speed_pickup_reward=0.0,
                 ranking_reward=False,
                 bilinear_ds=False, minimap=False):
        """
        Args:
            aggression_bonus: Multiplier for aggression rewards (kills, pickups).
                Set externally and decay over training: 1.0 early → 0.0 late.
                When 0.0 (default), no aggression shaping — pure win/death signal.
            kill_reward: Base reward per kill (scaled by aggression_bonus).
            speed_pickup_reward: Base reward for picking up speed boost
                (scaled by aggression_bonus).
        """
        super().__init__()
        self.min_players = min_players
        self.max_players = max_players
        self.render_mode = render_mode
        self.arena_variation = arena_variation
        self.gspp = gspp
        self.aggression_bonus = aggression_bonus
        self._kill_reward = kill_reward
        self._speed_pickup_reward = speed_pickup_reward
        self._ranking_reward = ranking_reward
        self._bilinear_ds = bilinear_ds
        self._minimap = minimap
        self._arena_sim = ARENA_SIM

        # Rectangular field for GS++ (matching real game aspect ratio)
        if self.gspp:
            self._arena_h = ARENA_H_GSPP
            self._offset_y = ARENA_OFFSET_Y_GSPP
        else:
            self._arena_h = ARENA_SIM
            self._offset_y = 0

        # Select physics ratios based on game mode
        if self.gspp:
            self._speed_ratio = SPEED_RATIO_GSPP
            self._tr_ratio = TURN_RADIUS_RATIO_GSPP
            self._tw_ratio = TRAIL_WIDTH_RATIO_GSPP
        else:
            self._speed_ratio = SPEED_RATIO
            self._tr_ratio = TURN_RADIUS_RATIO
            self._tw_ratio = TRAIL_WIDTH_RATIO

        # Physics (recomputed per episode if arena_variation=True)
        self._speed = self._speed_ratio * ARENA_SIM
        self._turn_radius = self._tr_ratio * ARENA_SIM
        self._trail_width = self._tw_ratio * ARENA_SIM
        self._turn_rate = self._speed / self._turn_radius
        self._gap_detect_width = self._trail_width * 3
        if self.gspp:
            self._gap_length = GAP_LENGTH_RATIO_GSPP * ARENA_SIM  # GS++ hs=22.4 → fixed distance
        else:
            self._gap_length = GAP_DURATION * self._speed  # None mode: time-based

        # GS++ powerup parameters
        self._powerup_spawn_interval = powerup_spawn_interval
        self._powerup_speed_weight = powerup_speed_weight
        self._powerup_erase_weight = powerup_erase_weight
        self._powerup_radius = 16  # 16px at 256-field = 6.25% ratio → 32px diam at 512
        self._powerup_first_spawn = int(3.5 * FPS)  # ~frame 210

        self.action_space = spaces.Discrete(3)
        n_channels = (6 if self.gspp else 4) + (2 if self._minimap else 0)
        self.observation_space = spaces.Box(
            0.0, 1.0, (n_channels, OBS_SIZE, OBS_SIZE), dtype=np.float32
        )
        self._ds_factor = ARENA_SIM // OBS_SIZE  # 4
        self._wall_w = max(1, int(round(self._trail_width / 2)) + 1)
        self._head_r = max(1, int(round(self._trail_width / 2)))
        self._half_trail = max(1, int(round(self._trail_width / 2)))
        self._arrow_len_ds = 5  # arrow length in obs-resolution pixels
        self._powerup_radius_ds = max(1, self._powerup_radius // self._ds_factor)  # ~4

        self.active_gaps = []
        self._gap_max_age_frames = int(GAP_PASS_MAX_AGE * FPS)

        # Cached wall mask at observation resolution
        f = self._ds_factor
        self._wall_mask_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        if self.gspp:
            # Rectangular field: wall bands at top/bottom (matching replay renderer)
            top_ds = self._offset_y // f
            bot_ds = min(OBS_SIZE, (self._offset_y + self._arena_h + f - 1) // f)
            if top_ds > 0:
                self._wall_mask_ds[:top_ds, :] = 1.0
            if bot_ds < OBS_SIZE:
                self._wall_mask_ds[bot_ds:, :] = 1.0
            # 1-pixel border at field edges
            if 0 <= top_ds < OBS_SIZE:
                self._wall_mask_ds[top_ds, :] = 1.0
            if 0 < bot_ds <= OBS_SIZE:
                self._wall_mask_ds[bot_ds - 1, :] = 1.0
            self._wall_mask_ds[top_ds:bot_ds, 0] = 1.0
            self._wall_mask_ds[top_ds:bot_ds, -1] = 1.0
        else:
            w_ds = max(1, (self._wall_w + f - 1) // f)
            self._wall_mask_ds[:w_ds, :] = 1.0
            self._wall_mask_ds[-w_ds:, :] = 1.0
            self._wall_mask_ds[:, :w_ds] = 1.0
            self._wall_mask_ds[:, -w_ds:] = 1.0

        # Ego-centric rotation: precompute offset grids
        # For each output pixel (row, col), offset from center of the frame
        half = OBS_SIZE / 2.0
        oy = np.arange(OBS_SIZE, dtype=np.float32) - half + 0.5  # row offsets (dy)
        ox = np.arange(OBS_SIZE, dtype=np.float32) - half + 0.5  # col offsets (dx)
        self._ego_dy, self._ego_dx = np.meshgrid(oy, ox, indexing='ij')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Arena size variation: randomize physics to simulate different effective
        # arena sizes (480-544) while keeping grid fixed at 512 for clean downsampling.
        # Scaling physics by k = A_eff/512 is equivalent to playing on a different-sized arena.
        if self.arena_variation:
            A_eff = float(rng.uniform(480, 544))
        else:
            A_eff = float(ARENA_SIM)
        self._arena_sim = ARENA_SIM  # grid always 512

        # Recompute physics from effective arena size
        self._speed = self._speed_ratio * A_eff
        self._turn_radius = self._tr_ratio * A_eff
        self._trail_width = self._tw_ratio * A_eff
        self._turn_rate = self._speed / self._turn_radius
        self._wall_w = max(1, int(round(self._trail_width / 2)) + 1)
        self._head_r = max(1, int(round(self._trail_width / 2)))
        self._half_trail = max(1, int(round(self._trail_width / 2)))

        # Rebuild wall mask (trail width may have changed) — None mode only
        # GS++ wall mask is fixed (rectangular bounds don't change with arena_variation)
        if not self.gspp:
            f = self._ds_factor
            w_ds = max(1, (self._wall_w + f - 1) // f)
            self._wall_mask_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
            self._wall_mask_ds[:w_ds, :] = 1.0
            self._wall_mask_ds[-w_ds:, :] = 1.0
            self._wall_mask_ds[:, :w_ds] = 1.0
            self._wall_mask_ds[:, -w_ds:] = 1.0

        # Gap pass detection width scales with trail width
        self._gap_detect_width = self._trail_width * 3
        if self.gspp:
            self._gap_length = GAP_LENGTH_RATIO_GSPP * A_eff  # GS++ hs=22.4 → fixed distance
        else:
            self._gap_length = GAP_DURATION * self._speed  # None mode: time-based

        n_players = int(rng.integers(self.min_players, self.max_players + 1))

        A = ARENA_SIM
        self.trail_owner = np.zeros((A, A), dtype=np.int16)
        self.trail_frame = np.zeros((A, A), dtype=np.int32)

        # Spawn players with spacing (rectangular bounds for GS++)
        margin = A * 0.15
        min_dist = A * 0.08
        self.players = []
        for i in range(n_players):
            for _ in range(200):
                x = float(rng.uniform(margin, A - margin))
                y = float(rng.uniform(self._offset_y + margin,
                                      self._offset_y + self._arena_h - margin))
                if all(math.hypot(x - p.x, y - p.y) >= min_dist for p in self.players):
                    break
            angle = float(rng.uniform(0, 2 * math.pi))
            gap_at = float(rng.uniform(GAP_INTERVAL_MIN, GAP_INTERVAL_MAX))
            self.players.append(PlayerState(i + 1, x, y, angle, gap_at))

        self.ego = self.players[0]
        self.active_gaps = []
        self.powerups = []
        self._next_powerup_frame = self._powerup_first_spawn if self.gspp else float('inf')
        self.frame_count = 0
        self.agent_steps = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.ego_kills = 0
        self.n_opponents_start = n_players - 1
        self._death_count = 0

        obs = self._get_player_obs(self.ego)
        info = {"n_players": n_players, "scalar_obs": self.get_scalar_obs(self.ego)}
        return obs, info

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _sim_step_player(self, player, action_int):
        dt = 1.0 / FPS
        player.time_alive += dt

        # Expire old speed boosts and compute multiplier
        if player.speed_boosts:
            player.speed_boosts = [
                (f, m) for f, m in player.speed_boosts
                if f > self.frame_count
            ]
        speed_mult = 1.0
        for _, m in player.speed_boosts:
            speed_mult *= m
        effective_speed = self._speed * speed_mult

        steer = action_int - 1
        if steer != 0:
            # Cube-root turn scaling: angular rate increases with speed^(1/3)
            # At 4x speed → 1.59x angular rate → circle diameter ~21% of field
            # At 1x speed → unchanged (1^(1/3) = 1)
            effective_turn_rate = self._turn_rate * (speed_mult ** (1.0 / 3.0))
            player.angle += steer * effective_turn_rate * dt

        player.prev_move_x = player.x
        player.prev_move_y = player.y
        player.x += math.cos(player.angle) * effective_speed * dt
        player.y += math.sin(player.angle) * effective_speed * dt

        if player.spawning:
            if player.time_alive >= SPAWN_SECONDS:
                player.spawning = False
                player.drawing = True
                player.gap_timer = 0.0
                player.prev_draw_x = player.x
                player.prev_draw_y = player.y
            return

        player.gap_timer += dt
        if player.gap_remaining > 0:
            # Gap tracks DISTANCE remaining, not time — constant size at any speed
            player.gap_remaining -= effective_speed * dt
            if player.gap_remaining <= 0:
                player.drawing = True
                player.gap_just_closed = True
                player.gap_timer = 0
                player.prev_draw_x = player.x
                player.prev_draw_y = player.y
                player.next_gap_at = float(
                    self.np_random.uniform(GAP_INTERVAL_MIN, GAP_INTERVAL_MAX)
                )
        else:
            if player.gap_timer >= player.next_gap_at:
                player.drawing = False
                player.gap_start_x = player.x
                player.gap_start_y = player.y
                # Gap is a fixed DISTANCE (pixels), not time
                jitter = float(self.np_random.uniform(-0.016, 0.016))
                player.gap_remaining = self._gap_length * (1.0 + jitter)

    def _check_collision(self, player):
        """Check collision. Returns (hit: bool, killer_id: int).
        killer_id is trail owner for enemy trail kills, 0 for wall/self."""
        tw = self._trail_width
        wm = tw / 2 + 1
        A = ARENA_SIM

        # Wall check (rectangular bounds for GS++)
        if (player.x < wm or player.x >= A - wm or
                player.y < self._offset_y + wm or
                player.y >= self._offset_y + self._arena_h - wm):
            return True, 0

        skip_frames = max(8, int(tw * 3))
        min_frame = self.frame_count - skip_frames
        cr = max(1, int(tw / 2) - 1)
        pid = player.id

        # Sweep from prev to current position to prevent tunneling at high speed
        mx = player.x - player.prev_move_x
        my = player.y - player.prev_move_y
        dist = math.hypot(mx, my)
        # Check every ~1 pixel along movement (at normal speed dist≈1.7, so 2 checks)
        n_checks = max(1, int(math.ceil(dist)))

        for s in range(n_checks + 1):
            t = s / n_checks if n_checks > 0 else 1.0
            sx = player.prev_move_x + mx * t
            sy = player.prev_move_y + my * t
            px_i = int(round(sx))
            py_i = int(round(sy))

            for dy in range(-cr, cr + 1):
                for dx in range(-cr, cr + 1):
                    if dx * dx + dy * dy > cr * cr:
                        continue
                    cx, cy = px_i + dx, py_i + dy
                    if 0 <= cx < A and 0 <= cy < A:
                        owner = self.trail_owner[cy, cx]
                        if owner > 0:
                            if owner == pid:
                                if self.trail_frame[cy, cx] < min_frame:
                                    return True, 0  # self-collision
                            else:
                                return True, owner  # enemy trail kill
        return False, 0

    def _draw_trail(self, player):
        hw = self._half_trail
        pid = player.id
        fc = self.frame_count
        A = ARENA_SIM

        # Vector from previous draw position to current
        mx = player.x - player.prev_draw_x
        my = player.y - player.prev_draw_y
        dist = math.hypot(mx, my)

        # Interpolate stamps along the movement to fill gaps at high speed
        n_stamps = max(1, int(math.ceil(dist)))
        for s in range(n_stamps + 1):
            t = s / n_stamps if n_stamps > 0 else 1.0
            sx = player.prev_draw_x + mx * t
            sy = player.prev_draw_y + my * t

            # Perpendicular direction: use movement dir if available, else angle
            if dist > 0.5:
                perp_x = -my / dist
                perp_y = mx / dist
            else:
                perp_x = -math.sin(player.angle)
                perp_y = math.cos(player.angle)

            for i in range(-hw, hw + 1):
                cx = int(round(sx + perp_x * i))
                cy = int(round(sy + perp_y * i))
                if 0 <= cx < A and 0 <= cy < A:
                    self.trail_owner[cy, cx] = pid
                    self.trail_frame[cy, cx] = fc

        # Head circle at final position
        ix, iy = int(round(player.x)), int(round(player.y))
        r = self._head_r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    cx, cy = ix + dx, iy + dy
                    if 0 <= cx < A and 0 <= cy < A:
                        self.trail_owner[cy, cx] = pid
                        self.trail_frame[cy, cx] = fc

        # Update previous draw position
        player.prev_draw_x = player.x
        player.prev_draw_y = player.y

    # ------------------------------------------------------------------
    # GS++ Powerups
    # ------------------------------------------------------------------
    def _spawn_powerup(self):
        """Spawn a random powerup at a random location (within rectangular bounds)."""
        margin = self._powerup_radius * 2
        x = float(self.np_random.uniform(margin, ARENA_SIM - margin))
        y = float(self.np_random.uniform(self._offset_y + margin,
                                         self._offset_y + self._arena_h - margin))
        total_w = self._powerup_speed_weight + self._powerup_erase_weight
        if float(self.np_random.random()) < self._powerup_speed_weight / total_w:
            ptype = 'speed'
        else:
            ptype = 'erase'
        self.powerups.append(Powerup(ptype, x, y, self._powerup_radius))

    def _check_powerup_pickups(self):
        """Check if any alive player picks up a powerup. Returns pickup info."""
        remaining = []
        ego_speed = 0
        ego_erase = 0
        erase_events = 0
        pickup_radius = self._powerup_radius + self._trail_width / 2

        for pup in self.powerups:
            picked_up = False
            for player in self.players:
                if not player.alive or player.spawning:
                    continue
                dist = math.hypot(player.x - pup.x, player.y - pup.y)
                if dist < pickup_radius:
                    if pup.ptype == 'speed':
                        player.speed_boosts.append(
                            (self.frame_count + int(4 * FPS), 2.0)
                        )
                        if player.id == self.ego.id:
                            ego_speed += 1
                    elif pup.ptype == 'erase':
                        self.trail_owner[:] = 0
                        self.trail_frame[:] = 0
                        self.active_gaps.clear()
                        erase_events += 1
                        if player.id == self.ego.id:
                            ego_erase += 1
                    picked_up = True
                    break  # Each powerup picked up once
            if not picked_up:
                remaining.append(pup)
        self.powerups = remaining
        return ego_speed, ego_erase, erase_events

    def _build_powerup_grids_ds(self):
        """Build powerup location grids at observation resolution (128x128)."""
        f = self._ds_factor
        r_ds = self._powerup_radius_ds
        speed_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        erase_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)

        for pup in self.powerups:
            px = int(pup.x / f)
            py = int(pup.y / f)
            grid = speed_ds if pup.ptype == 'speed' else erase_ds
            for dy in range(-r_ds, r_ds + 1):
                for dx in range(-r_ds, r_ds + 1):
                    if dx * dx + dy * dy <= r_ds * r_ds:
                        cx, cy = px + dx, py + dy
                        if 0 <= cx < OBS_SIZE and 0 <= cy < OBS_SIZE:
                            grid[cy, cx] = 1.0
        return speed_ds, erase_ds

    def _detect_gap_passes(self):
        """Detect players that fully traversed a gap (entered one side, exited other).
        Returns dict {player_id: pass_count}."""
        # Expire old gap zones
        cutoff = self.frame_count - self._gap_max_age_frames
        self.active_gaps = [g for g in self.active_gaps if g.frame > cutoff]

        passes = {}
        for gap in self.active_gaps:
            for player in self.players:
                if not player.alive or player.id == gap.owner_id:
                    continue

                dx = player.x - gap.cx
                dy = player.y - gap.cy

                # Project onto trail direction and perpendicular
                along = dx * gap.tx + dy * gap.ty
                cross = dx * gap.nx + dy * gap.ny

                in_zone = (abs(along) <= gap.half_len
                           and abs(cross) <= self._gap_detect_width)

                if not in_zone:
                    # Left the zone — clean up tracking
                    gap.tracking.pop(player.id, None)
                    continue

                # Inside the gap detection zone
                cross_sign = 1 if cross > 0 else -1
                if player.id not in gap.tracking:
                    # Just entered — record which side
                    gap.tracking[player.id] = cross_sign
                elif gap.tracking[player.id] != cross_sign:
                    # Switched sides while inside zone = gap pass!
                    passes[player.id] = passes.get(player.id, 0) + 1
                    del gap.tracking[player.id]

            # Also clean dead players from tracking
            dead_ids = [pid for pid in gap.tracking
                        if not any(p.id == pid and p.alive for p in self.players)]
            for pid in dead_ids:
                del gap.tracking[pid]

        return passes

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, ego_action, opponent_actions=None):
        self.agent_steps += 1
        self.episode_length += 1

        n_opp = len(self.players) - 1
        if opponent_actions is None:
            opponent_actions = self.np_random.integers(0, 3, size=n_opp)

        all_actions = [int(ego_action)] + [int(a) for a in opponent_actions]

        for p in self.players:
            p.just_died = False

        step_ego_speed = 0
        step_ego_erase = 0
        step_erase_events = 0
        step_kills = 0

        for _ in range(FRAME_SKIP):
            self.frame_count += 1

            # GS++ powerup spawning
            if self.gspp and self.frame_count >= self._next_powerup_frame:
                self._spawn_powerup()
                self._next_powerup_frame = (
                    self.frame_count + self._powerup_spawn_interval
                )

            for player, act in zip(self.players, all_actions):
                if player.alive:
                    self._sim_step_player(player, act)
                    # Record gap zones when a gap just closed
                    if player.gap_just_closed:
                        self.active_gaps.append(GapZone(
                            player.id,
                            player.gap_start_x, player.gap_start_y,
                            player.x, player.y,
                            self.frame_count,
                        ))
                        player.gap_just_closed = False

            # GS++ powerup pickup (after moving, before collision)
            if self.gspp:
                es, ee, ev = self._check_powerup_pickups()
                step_ego_speed += es
                step_ego_erase += ee
                step_erase_events += ev

            # Collect deaths before applying (simultaneous)
            deaths = []
            for player in self.players:
                if player.alive and not player.spawning:
                    hit, killer = self._check_collision(player)
                    if hit:
                        deaths.append((player, killer))
            for player, killer in deaths:
                player.alive = False
                player.just_died = True
                self._death_count += 1
                player.death_order = self._death_count
                # Track kills: ego gets credit when enemy hits ego's trail
                if killer == self.ego.id and player.id != self.ego.id:
                    self.ego_kills += 1
                    step_kills += 1

            for player in self.players:
                if player.alive and player.drawing:
                    self._draw_trail(player)

            if not self.ego.alive:
                break

        # Detect gap passes (full traversals through trail gaps)
        gap_passes = self._detect_gap_passes()
        ego_gap_passes = gap_passes.get(self.ego.id, 0)

        # Reward: survival during play + win/lose terminal signal
        reward = 0.0
        terminated = False
        truncated = False
        won = False

        multiplayer = len(self.players) > 1

        if not self.ego.alive:
            if self._ranking_reward and multiplayer:
                # Placement-based: count how many opponents are still alive
                n = len(self.players)
                alive_opponents = sum(1 for p in self.players[1:] if p.alive)
                # ego_rank: 1-indexed from last (worst). More alive = worse rank.
                ego_rank = alive_opponents + 1  # +1 for ego's own position
                # Linear: 1st=+1.0, last=-1.0
                reward += -1.0 + 2.0 * (n - ego_rank) / (n - 1)
            else:
                reward += REWARD_DEATH
            terminated = True
        elif multiplayer and all(not p.alive for p in self.players[1:]):
            # FFA/1v1 win: last alive → reward + end episode immediately
            reward += REWARD_WIN
            won = True
            truncated = True
        elif not multiplayer:
            # Solo: survival reward teaches basic movement/wall avoidance
            reward += REWARD_ALIVE
        # else: multiplayer, still fighting — no alive drip.
        # Win/lose is the only signal. Duration doesn't matter.

        # Aggression bonus: decaying reward shaping for kills & pickups.
        # aggression_bonus is set externally (1.0 early → 0.0 late in training).
        # Encourages: pick up speed → use speed to get kills → combo bonus.
        if self.aggression_bonus > 0:
            ab = self.aggression_bonus
            # Reward picking up speed (encourages seeking powerups)
            reward += ab * self._speed_pickup_reward * step_ego_speed
            # Reward kills (base kill reward)
            reward += ab * self._kill_reward * step_kills
            # Combo: kills while boosted get extra multiplier (speed + kill = risky aggressive play)
            ego_boosted = len(self.ego.speed_boosts) > 0 if self.ego.alive else False
            if ego_boosted and step_kills > 0:
                reward += ab * self._kill_reward * step_kills  # 2x kill reward when boosted

        if not terminated:
            truncated = truncated if won else (
                self.agent_steps >= MAX_AGENT_STEPS and self.ego.alive
            )
        self.episode_return += reward

        info = {"ego_gap_passes": ego_gap_passes, "ego_kills": self.ego_kills,
                "step_kills": step_kills}
        if self.gspp:
            info["ego_speed_pickups"] = step_ego_speed
            info["ego_erase_pickups"] = step_ego_erase
            info["erase_events"] = step_erase_events
            info["num_powerups"] = len(self.powerups)
        if terminated or truncated:
            n = len(self.players)
            if won:
                info["ego_rank"] = 1
            elif not self.ego.alive:
                alive_opponents = sum(1 for p in self.players[1:] if p.alive)
                info["ego_rank"] = alive_opponents + 1
            else:
                info["ego_rank"] = 1  # truncated while alive = effectively 1st
            info["episode"] = {"r": self.episode_return, "l": self.episode_length}
            info["win"] = won
            info["n_players"] = len(self.players)

        obs = self._get_player_obs(self.ego)
        info["scalar_obs"] = self.get_scalar_obs(self.ego)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _draw_arrow_ds(self, channel, x, y, angle):
        """Draw direction arrow directly at observation resolution (64x64).
        5 pixels at ds = 40px at sim resolution — visible heading indicator."""
        f = self._ds_factor
        px_ds = x / f
        py_ds = y / f
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for t in range(self._arrow_len_ds):
            cx = int(round(px_ds + cos_a * t))
            cy = int(round(py_ds + sin_a * t))
            if 0 <= cx < OBS_SIZE and 0 <= cy < OBS_SIZE:
                channel[cy, cx] = 1.0

    def _compute_rotation_map(self, player):
        """Compute source coordinates for ego-centric rotation.
        Returns (src_ri, src_ci, valid) for fancy-index lookup."""
        f = self._ds_factor
        cos_a = math.cos(player.angle)
        sin_a = math.sin(player.angle)
        ego_col = player.x / f
        ego_row = player.y / f

        # Map output (ego-frame) pixels to input (world-frame) pixels
        src_col = ego_col + cos_a * self._ego_dx - sin_a * self._ego_dy
        src_row = ego_row + sin_a * self._ego_dx + cos_a * self._ego_dy

        src_ci = np.round(src_col).astype(np.intp)
        src_ri = np.round(src_row).astype(np.intp)

        valid = (src_ci >= 0) & (src_ci < OBS_SIZE) & (src_ri >= 0) & (src_ri < OBS_SIZE)
        np.clip(src_ci, 0, OBS_SIZE - 1, out=src_ci)
        np.clip(src_ri, 0, OBS_SIZE - 1, out=src_ri)
        return src_ri, src_ci, valid

    def _rotate_channel(self, ch, src_ri, src_ci, valid, fill):
        """Lookup rotated pixels, fill out-of-bounds with `fill`."""
        out = ch[src_ri, src_ci]
        out[~valid] = fill
        return out

    def _compute_ray_distances(self, player, n_rays=8, max_dist=256.0):
        """Cast rays from player in ego-relative directions.
        Returns normalized distances (0=touching obstacle, 1=max_dist away).
        Rays: 0=ahead, 1=ahead-right, 2=right, ..., 7=ahead-left (45° increments)."""
        A = ARENA_SIM
        dists = np.ones(n_rays, dtype=np.float32)
        px, py = player.x, player.y
        angle = player.angle
        tw = self._trail_width
        skip_frames = max(8, int(tw * 3))
        min_frame = self.frame_count - skip_frames
        pid = player.id
        step_size = 2.0  # pixels per step

        for i in range(n_rays):
            ray_angle = angle + i * (2.0 * math.pi / n_rays)
            dx = math.cos(ray_angle) * step_size
            dy = math.sin(ray_angle) * step_size
            rx, ry = px, py

            for s in range(1, int(max_dist / step_size) + 1):
                rx += dx
                ry += dy
                ix, iy = int(round(rx)), int(round(ry))

                # Wall check
                if (ix < 0 or ix >= A or iy < self._offset_y or
                        iy >= self._offset_y + self._arena_h):
                    dists[i] = (s * step_size) / max_dist
                    break

                # Trail check
                if 0 <= ix < A and 0 <= iy < A:
                    owner = self.trail_owner[iy, ix]
                    if owner > 0:
                        if owner == pid:
                            if self.trail_frame[iy, ix] < min_frame:
                                dists[i] = (s * step_size) / max_dist
                                break
                        else:
                            dists[i] = (s * step_size) / max_dist
                            break
        return dists

    def get_scalar_obs(self, player):
        """Structured scalar observations for a player.
        Returns float32 array of N_SCALAR_OBS values, all in [0, 1]."""
        scalars = np.zeros(N_SCALAR_OBS, dtype=np.float32)

        # Ray distances (8 directions, ego-relative)
        scalars[0:8] = self._compute_ray_distances(player)

        # Speed boost active (0 or 1)
        scalars[8] = 1.0 if len(player.speed_boosts) > 0 else 0.0

        # Gap progress: how close to next gap (0=just had gap, 1=gap imminent)
        if player.next_gap_at > 0:
            scalars[9] = min(1.0, player.gap_timer / player.next_gap_at)
        else:
            scalars[9] = 0.0

        # Alive opponent fraction
        n_total = len(self.players) - 1
        if n_total > 0:
            alive = sum(1 for p in self.players[1:] if p.alive) if player.id == self.ego.id else \
                    sum(1 for p in self.players if p.alive and p.id != player.id)
            scalars[10] = alive / n_total
        else:
            scalars[10] = 0.0

        # Territory fraction (set by voronoi wrapper, default 0)
        scalars[11] = getattr(self, '_ego_territory_frac', 0.0) if player.id == self.ego.id else 0.0

        # Speed multiplier (normalized: 1.0=normal, higher=boosted)
        speed_mult = 1.0
        for expire_frame, mult in player.speed_boosts:
            if expire_frame > self.frame_count:
                speed_mult *= mult
        scalars[12] = min(1.0, speed_mult / 4.0)  # normalize: 4x = 1.0

        return scalars

    # ------------------------------------------------------------------
    # Safety shield — tree-search action safety
    # ------------------------------------------------------------------
    def _sim_forward(self, x, y, angle, action, n_frames, pid,
                     eff_speed, eff_turn_rate, wm, cr, min_frame, A, oy, ah):
        """Simulate one action for n_frames. Returns (x, y, angle) or None if dead."""
        dt = 1.0 / FPS
        steer = action - 1
        for _ in range(n_frames):
            prev_x, prev_y = x, y
            if steer != 0:
                angle += steer * eff_turn_rate * dt
            x += math.cos(angle) * eff_speed * dt
            y += math.sin(angle) * eff_speed * dt

            # Wall check
            if x < wm or x >= A - wm or y < oy + wm or y >= oy + ah - wm:
                return None

            # Trail check (swept)
            mx, my = x - prev_x, y - prev_y
            dist = math.hypot(mx, my)
            n_checks = max(1, int(math.ceil(dist)))
            for s in range(n_checks + 1):
                t = s / n_checks if n_checks > 0 else 1.0
                sx = prev_x + mx * t
                sy = prev_y + my * t
                px_i = int(round(sx))
                py_i = int(round(sy))
                for dy in range(-cr, cr + 1):
                    for dx in range(-cr, cr + 1):
                        if dx * dx + dy * dy > cr * cr:
                            continue
                        cx, cy = px_i + dx, py_i + dy
                        if 0 <= cx < A and 0 <= cy < A:
                            owner = self.trail_owner[cy, cx]
                            if owner > 0:
                                if owner == pid:
                                    if self.trail_frame[cy, cx] < min_frame:
                                        return None
                                else:
                                    return None
        return (x, y, angle)

    def get_safe_action_mask(self, player, depth=7, frames_per_step=3):
        """Tree-search safety: action is safe if ANY follow-up sequence survives.

        depth=7 at 3 frames/step = 21 frames = 0.35s lookahead.
        Searches 3^depth paths with early termination.
        """
        if not player.alive:
            return [False, False, False]

        sim_args = self._shield_sim_args(player)

        def _has_surviving_path(x, y, angle, remaining_depth):
            if remaining_depth == 0:
                return True
            for a in range(3):
                result = self._sim_forward(
                    x, y, angle, a, frames_per_step, *sim_args)
                if result is not None:
                    if _has_surviving_path(*result, remaining_depth - 1):
                        return True
            return False

        mask = []
        for first_action in range(3):
            result = self._sim_forward(
                player.x, player.y, player.angle,
                first_action, frames_per_step, *sim_args)
            if result is None:
                mask.append(False)
            else:
                mask.append(_has_surviving_path(*result, depth - 1))
        return mask

    def get_action_survival_scores(self, player, depth=5, frames_per_step=3):
        """Count surviving leaf paths for each first action.

        Returns list of 3 floats in [0, 1] — fraction of reachable leaves
        that survive. Higher = more escape routes = safer direction.
        depth=5 → 3^4=81 paths per action, ~1-3ms total.
        """
        if not player.alive:
            return [0.0, 0.0, 0.0]

        sim_args = self._shield_sim_args(player)

        def _count_surviving(x, y, angle, remaining_depth):
            if remaining_depth == 0:
                return 1
            total = 0
            for a in range(3):
                result = self._sim_forward(
                    x, y, angle, a, frames_per_step, *sim_args)
                if result is not None:
                    total += _count_surviving(*result, remaining_depth - 1)
            return total

        max_leaves = 3 ** (depth - 1)
        scores = []
        for first_action in range(3):
            result = self._sim_forward(
                player.x, player.y, player.angle,
                first_action, frames_per_step, *sim_args)
            if result is None:
                scores.append(0.0)
            else:
                count = _count_surviving(*result, depth - 1)
                scores.append(count / max_leaves)
        return scores

    def search_action_scores(self, player, macro_len=8, depth=3,
                             n_rays=12, ray_dist=80):
        """Macro beam search: score each first action by best reachable leaf.

        depth=3, macro_len=8 → 3^3=27 sequences, 24 frames (0.4s) lookahead.
        Leaf score = survival + openness (ray sum) + wall distance.
        Returns: list of 3 floats (score per first action, higher = better).
        """
        if not player.alive:
            return [0.0, 0.0, 0.0]

        dt = 1.0 / FPS
        speed_mult = 1.0
        for _, m in player.speed_boosts:
            speed_mult *= m
        eff_speed = self._speed * speed_mult
        eff_tr = self._turn_rate * (speed_mult ** (1.0 / 3.0))
        step_speed = eff_speed * dt
        step_tr = eff_tr * dt

        A = ARENA_SIM
        oy = self._offset_y
        ah = self._arena_h
        pid = player.id
        tw = self._trail_width
        skip_frames = max(8, int(tw * 3))
        min_frame = self.frame_count - skip_frames
        trail_owner = self.trail_owner
        trail_frame = self.trail_frame

        # Generate all action sequences
        actions_list = []
        for a1 in range(3):
            for a2 in range(3):
                for a3 in range(3):
                    actions_list.append((a1, a2, a3))

        best_per_first = [float('-inf'), float('-inf'), float('-inf')]

        for seq in actions_list:
            x, y, angle = player.x, player.y, player.angle
            alive = True
            dist_traveled = 0.0
            min_wall = float('inf')

            for act in seq:
                steer = act - 1
                for _ in range(macro_len):
                    if steer != 0:
                        angle += steer * step_tr
                    prev_x, prev_y = x, y
                    x += math.cos(angle) * step_speed
                    y += math.sin(angle) * step_speed

                    # Wall check
                    wd = min(x, A - x, y - oy, oy + ah - y)
                    if wd < tw / 2 + 1:
                        alive = False
                        break
                    min_wall = min(min_wall, wd)

                    # Trail check at new position
                    ix, iy = int(round(x)), int(round(y))
                    if 0 <= ix < A and 0 <= iy < A:
                        owner = trail_owner[iy, ix]
                        if owner > 0:
                            if owner == pid:
                                if trail_frame[iy, ix] < min_frame:
                                    alive = False
                                    break
                            else:
                                alive = False
                                break

                    dist_traveled += step_speed
                if not alive:
                    break

            if not alive:
                score = -1000.0 + dist_traveled
            else:
                # Openness: cast rays from leaf position
                ray_total = 0.0
                for i in range(n_rays):
                    ang = 6.283185307 * i / n_rays
                    dx = math.cos(ang)
                    dy = math.sin(ang)
                    for s in range(1, ray_dist + 1):
                        cx = int(round(x + dx * s))
                        cy = int(round(y + dy * s))
                        if (cx < 0 or cx >= A
                                or cy < oy or cy >= oy + ah):
                            ray_total += s
                            break
                        if trail_owner[cy, cx] > 0:
                            ray_total += s
                            break
                    else:
                        ray_total += ray_dist

                score = dist_traveled + min_wall * 0.3 + ray_total * 0.1

            first = seq[0]
            if score > best_per_first[first]:
                best_per_first[first] = score

        return best_per_first

    def get_action_openness(self, player, sim_frames=15, n_rays=16, max_dist=100):
        """Simulate each action for sim_frames, then measure open space at endpoint.

        Returns 3 floats: total traversable distance in n_rays directions.
        Higher = more open space = safer strategic choice.
        Cost: 3 endpoints × 16 rays × 80 steps ≈ 3840 pixel lookups (~0.5ms).
        """
        if not player.alive:
            return [0.0, 0.0, 0.0]

        sim_args = self._shield_sim_args(player)
        A = ARENA_SIM
        oy = self._offset_y
        ah = self._arena_h
        trail_owner = self.trail_owner
        max_total = n_rays * max_dist  # normalization factor

        scores = []
        for action in range(3):
            result = self._sim_forward(
                player.x, player.y, player.angle,
                action, sim_frames, *sim_args)
            if result is None:
                scores.append(0.0)
                continue

            x, y, _ = result
            total = 0
            for i in range(n_rays):
                ang = 6.283185307 * i / n_rays  # 2*pi
                dx = math.cos(ang)
                dy = math.sin(ang)
                for step in range(1, max_dist + 1):
                    cx = int(round(x + dx * step))
                    cy = int(round(y + dy * step))
                    if (cx < 0 or cx >= A
                            or cy < oy or cy >= oy + ah):
                        total += step
                        break
                    if trail_owner[cy, cx] > 0:
                        total += step
                        break
                else:
                    total += max_dist
            scores.append(total / max_total)
        return scores

    def _shield_sim_args(self, player):
        """Pre-compute shared simulation constants for shield methods."""
        speed_mult = 1.0
        for _, m in player.speed_boosts:
            speed_mult *= m
        eff_speed = self._speed * speed_mult
        eff_turn_rate = self._turn_rate * (speed_mult ** (1.0 / 3.0))
        tw = self._trail_width
        wm = tw / 2 + 1
        A = ARENA_SIM
        skip_frames = max(8, int(tw * 3))
        min_frame = self.frame_count - skip_frames
        cr = max(1, int(tw / 2) - 1)
        pid = player.id
        oy = self._offset_y
        ah = self._arena_h
        return (pid, eff_speed, eff_turn_rate, wm, cr, min_frame, A, oy, ah)

    def _get_player_obs(self, player):
        """Ego-centric rotated observation at 128x128.
        Centered on player, rotated so player faces RIGHT (+x).
        Previous frame also rotated with CURRENT heading for consistent temporal info."""
        pid = player.id
        f = self._ds_factor

        # Bool masks at arena resolution
        self_mask = (self.trail_owner == pid)
        enemy_mask = (self.trail_owner > 0) & ~self_mask

        # Downsample arena -> OBS_SIZE
        if self._bilinear_ds:
            # v11: mean-pool — gaps become dim regions instead of invisible
            # sum booleans (int) then divide, avoiding full 512x512 float32 cast
            inv_area = np.float32(1.0 / (f * f))
            self_ds = self_mask.reshape(
                OBS_SIZE, f, OBS_SIZE, f).sum(axis=(1, 3)).astype(np.float32) * inv_area
            enemy_ds = enemy_mask.reshape(
                OBS_SIZE, f, OBS_SIZE, f).sum(axis=(1, 3)).astype(np.float32) * inv_area
        else:
            # v8-v10: boolean max-pool (.any())
            self_ds = self_mask.reshape(OBS_SIZE, f, OBS_SIZE, f).any(
                axis=(1, 3)).astype(np.float32)
            enemy_ds = enemy_mask.reshape(OBS_SIZE, f, OBS_SIZE, f).any(
                axis=(1, 3)).astype(np.float32)

        # Add walls (world frame)
        np.maximum(self_ds, self._wall_mask_ds, out=self_ds)

        # v11: save non-rotated minimap BEFORE arrows are drawn
        if self._minimap:
            minimap_self = self_ds.copy()
            minimap_enemy = enemy_ds.copy()

        # Direction arrows (world frame, after minimap copy)
        for p in self.players:
            if p.alive:
                ch = self_ds if p.id == pid else enemy_ds
                self._draw_arrow_ds(ch, p.x, p.y, p.angle)

        # Ego-centric rotation
        src_ri, src_ci, valid = self._compute_rotation_map(player)

        self_rot = self._rotate_channel(self_ds, src_ri, src_ci, valid, 1.0)
        enemy_rot = self._rotate_channel(enemy_ds, src_ri, src_ci, valid, 0.0)

        # Previous frame rotated with CURRENT heading (consistent perspective)
        prev_self_rot = self._rotate_channel(player.prev_channels[0], src_ri, src_ci, valid, 1.0)
        prev_enemy_rot = self._rotate_channel(player.prev_channels[1], src_ri, src_ci, valid, 0.0)

        if self.gspp:
            speed_ds, erase_ds = self._build_powerup_grids_ds()
            speed_rot = self._rotate_channel(speed_ds, src_ri, src_ci, valid, 0.0)
            erase_rot = self._rotate_channel(erase_ds, src_ri, src_ci, valid, 0.0)
            obs = np.stack([self_rot, enemy_rot, prev_self_rot, prev_enemy_rot,
                            speed_rot, erase_rot])
        else:
            obs = np.stack([self_rot, enemy_rot, prev_self_rot, prev_enemy_rot])

        # v11: append non-rotated minimap channels (stable global reference)
        if self._minimap:
            obs = np.concatenate([obs, minimap_self[np.newaxis],
                                  minimap_enemy[np.newaxis]], axis=0)

        # Store WORLD-FRAME channels for next step (unrotated)
        player.prev_channels = np.stack([self_ds, enemy_ds])
        return obs

    def get_opponent_observations(self):
        """Vectorized obs for all live opponents with ego-centric rotation."""
        live_opps = []
        live_mask = []
        for player in self.players[1:]:
            if player.alive:
                live_opps.append(player)
                live_mask.append(True)
            else:
                live_mask.append(False)

        if not live_opps:
            return [], live_mask

        n = len(live_opps)
        f = self._ds_factor
        pids = np.array([p.id for p in live_opps], dtype=np.int16)

        # Vectorized bool masks at 512x512
        total_mask = self.trail_owner > 0
        self_masks = self.trail_owner[None, :, :] == pids[:, None, None]
        enemy_masks = total_mask & ~self_masks

        # Downsample: bilinear (mean-pool) or boolean (any-pool)
        if self._bilinear_ds:
            inv_area = np.float32(1.0 / (f * f))
            self_f = self_masks.reshape(n, OBS_SIZE, f, OBS_SIZE, f).sum(axis=(2, 4)).astype(np.float32) * inv_area
            enemy_f = enemy_masks.reshape(n, OBS_SIZE, f, OBS_SIZE, f).sum(axis=(2, 4)).astype(np.float32) * inv_area
        else:
            self_f = self_masks.reshape(n, OBS_SIZE, f, OBS_SIZE, f).any(axis=(2, 4)).astype(np.float32)
            enemy_f = enemy_masks.reshape(n, OBS_SIZE, f, OBS_SIZE, f).any(axis=(2, 4)).astype(np.float32)

        # Walls (cached, broadcast)
        np.maximum(self_f, self._wall_mask_ds[None, :, :], out=self_f)

        # v11: save non-rotated minimap BEFORE arrows are drawn
        if self._minimap:
            minimap_self_f = self_f.copy()   # (n, 128, 128)
            minimap_enemy_f = enemy_f.copy()

        # Direction arrows (world frame)
        for p in self.players:
            if not p.alive:
                continue
            for i, obs_player in enumerate(live_opps):
                ch = self_f[i] if p.id == obs_player.id else enemy_f[i]
                self._draw_arrow_ds(ch, p.x, p.y, p.angle)

        # Build powerup grids once if GS++ (shared across all opponents)
        if self.gspp:
            speed_ds, erase_ds = self._build_powerup_grids_ds()

        # Per-opponent ego-centric rotation + frame stack
        obs_list = []
        for i, player in enumerate(live_opps):
            src_ri, src_ci, valid = self._compute_rotation_map(player)

            sr = self._rotate_channel(self_f[i], src_ri, src_ci, valid, 1.0)
            er = self._rotate_channel(enemy_f[i], src_ri, src_ci, valid, 0.0)
            psr = self._rotate_channel(player.prev_channels[0], src_ri, src_ci, valid, 1.0)
            per = self._rotate_channel(player.prev_channels[1], src_ri, src_ci, valid, 0.0)

            if self.gspp:
                spr = self._rotate_channel(speed_ds, src_ri, src_ci, valid, 0.0)
                err = self._rotate_channel(erase_ds, src_ri, src_ci, valid, 0.0)
                obs = np.stack([sr, er, psr, per, spr, err])
            else:
                obs = np.stack([sr, er, psr, per])

            # v11: append non-rotated minimap channels
            if self._minimap:
                obs = np.concatenate([obs, minimap_self_f[i:i+1],
                                      minimap_enemy_f[i:i+1]], axis=0)
            obs_list.append(obs)

            # Store world-frame for next step
            player.prev_channels = np.stack([self_f[i], enemy_f[i]])

        return obs_list, live_mask

    def get_live_opponent_count(self):
        return sum(1 for p in self.players[1:] if p.alive)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        img = np.zeros((ARENA_SIM, ARENA_SIM, 3), dtype=np.uint8)
        ego_mask = self.trail_owner == self.ego.id
        enemy_mask = (self.trail_owner > 0) & (~ego_mask)
        img[ego_mask] = [255, 255, 255]
        img[enemy_mask] = [128, 128, 128]
        w = self._wall_w
        if self.gspp:
            # Rectangular field walls
            oy = self._offset_y
            ah = self._arena_h
            img[:oy + w, :] = [180, 180, 180]      # top wall band
            img[oy + ah - w:, :] = [180, 180, 180]  # bottom wall band
            img[oy:oy + ah, :w] = [180, 180, 180]
            img[oy:oy + ah, -w:] = [180, 180, 180]
        else:
            img[:w, :] = [180, 180, 180]
            img[-w:, :] = [180, 180, 180]
            img[:, :w] = [180, 180, 180]
            img[:, -w:] = [180, 180, 180]

        # Draw powerups
        for pup in self.powerups:
            ix, iy = int(round(pup.x)), int(round(pup.y))
            r = pup.radius
            if pup.ptype == 'speed':
                color = [76, 175, 80]  # green
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            cx, cy = ix + dx, iy + dy
                            if 0 <= cx < ARENA_SIM and 0 <= cy < ARENA_SIM:
                                img[cy, cx] = color
            else:
                # Erase: blue circle with white fill
                r_inner = max(1, r - 2)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        d2 = dx * dx + dy * dy
                        if d2 <= r * r:
                            cx, cy = ix + dx, iy + dy
                            if 0 <= cx < ARENA_SIM and 0 <= cy < ARENA_SIM:
                                if d2 <= r_inner * r_inner:
                                    img[cy, cx] = [255, 255, 255]
                                else:
                                    img[cy, cx] = [50, 100, 255]
        return img
