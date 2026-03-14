#!/usr/bin/env python3
"""CurveCrash replay data pipeline - scrape, validate, analyze, render, watch."""

import argparse
import json
import math
import os
import re
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# NOTE: heavy deps (playwright, asyncio, pygame, requests, bs4) imported only
# inside their subcommand functions to keep the module lightweight for imports.

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# Core: Replay Renderer Engine (from replay_renderer.py)
# ============================================================

# Match env constants
SIM_SIZE = 512
OBS_SIZE = 128
DS_FACTOR = SIM_SIZE // OBS_SIZE  # 4


class ReplayPlayer:
    __slots__ = [
        "id", "x", "y", "prev_x", "prev_y", "angle", "alive", "turning",
        "drawing", "gap_remaining", "prev_self_ds", "prev_enemy_ds",
        "speed_boosts",
    ]

    def __init__(self, pid, x, y, angle_deg):
        self.id = pid
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.angle = angle_deg * math.pi / 180.0
        self.alive = True
        self.turning = 0  # -1, 0, 1
        self.drawing = False  # starts False during spawn grace
        self.gap_remaining = 0.0
        self.prev_self_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        self.prev_enemy_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        self.speed_boosts = []  # [(expire_frame, multiplier)]


class ReplayRenderer:
    """Simulates one round of a CurveCrash replay, producing observations.

    Args:
        settings: Game settings dict with fw, fh, fps, speed, tr, size, hs.
        round_data: Round dict with nf, sp, tu, de, ho, pu.
        gspp: If True, produce 6-channel observations with powerup channels.
              If False, produce 4-channel observations (classic None mode).
    """

    def __init__(self, settings, round_data, gspp=False):
        self.fw = settings["fw"]
        self.fh = settings["fh"]
        self.fps = settings["fps"]
        self.speed_raw = settings["speed"]
        self.turn_radius_raw = settings["tr"]
        self.curve_size_raw = settings["size"]
        self.hole_size_raw = settings["hs"]
        self.gspp = gspp

        # Scale factor: replay coords -> sim grid (isotropic, preserve aspect)
        self.scale = SIM_SIZE / max(self.fw, self.fh)
        self.sim_w = int(round(self.fw * self.scale))
        self.sim_h = int(round(self.fh * self.scale))
        # Center the field in the 512x512 grid
        self.offset_x = (SIM_SIZE - self.sim_w) // 2
        self.offset_y = (SIM_SIZE - self.sim_h) // 2

        # Scaled physics (per frame in replay's fps)
        self.speed_per_frame = (self.speed_raw / self.fps) * self.scale
        self.turn_rate_per_frame = self.speed_per_frame / (self.turn_radius_raw * self.scale)
        # Trail width: use collision size (5.53 game units -> 3 sim px)
        # trail_hw=1 -> width=3 pixels, matching collision hitbox
        self.trail_hw = max(1, int(round(self.curve_size_raw * self.scale / 2 - 0.5)))
        self._trail_offsets = np.arange(-self.trail_hw, self.trail_hw + 1, dtype=np.float64)
        self.gap_duration_frames = self.hole_size_raw / (self.speed_raw / self.fps)

        # Round data
        self.num_frames = round_data["nf"]
        self.spawns = round_data["sp"]  # [curveId, x, y, angleDeg]

        # Build death lookup: frame -> set of curveIds that die
        self.death_map = {}
        for d in round_data["de"]:
            frame, cid = d[0], d[1]
            if frame not in self.death_map:
                self.death_map[frame] = set()
            self.death_map[frame].add(cid)

        # Build turn action lookup: frame -> [(playerId, direction, subMillis)]
        # subMillis gives sub-frame timing precision -- critical for reducing
        # physics drift in position reconstruction.
        self.turn_map = {}
        for t in round_data["tu"]:
            frame, pid, direction = t[0], t[1], t[2]
            sub_ms = t[3] if len(t) > 3 else 0
            if frame not in self.turn_map:
                self.turn_map[frame] = []
            self.turn_map[frame].append((pid, direction, sub_ms))

        # Build hole (gap) events: frame -> set of curveIds starting a gap
        self.hole_map = {}
        for h in round_data["ho"]:
            frame, cid = h[0], h[1]
            if frame not in self.hole_map:
                self.hole_map[frame] = set()
            self.hole_map[frame].add(cid)

        # GS++ powerup spawn map: frame -> [(sim_x, sim_y, ptype)]
        self.powerup_spawn_map = {}
        self.field_powerups = []  # active powerups on field: [(x, y, ptype)]
        # Real game powerup: ~18 game units radius ~ 10 sim pixels at our scale
        self._powerup_radius = 10  # visual AND collision radius in sim pixels
        self._powerup_radius_ds = max(1, self._powerup_radius // DS_FACTOR)
        # Pickup = powerup radius + player body radius (tight, matching visual)
        self._pickup_radius = self._powerup_radius + self.trail_hw + 1

        for pu in round_data.get("pu", []):
            frame = pu[0]
            sx = pu[1] * self.scale + self.offset_x
            sy = pu[2] * self.scale + self.offset_y
            pid = pu[3]
            ptype = 'speed' if pid == 1 else 'erase'
            if frame not in self.powerup_spawn_map:
                self.powerup_spawn_map[frame] = []
            self.powerup_spawn_map[frame].append((sx, sy, ptype))

        # Per-frame state data (v2/v3 format): eliminates physics simulation
        # v2: 5 vals per player [x, y, angle, alive, turning]
        # v3: 7 vals per player [x, y, angle, alive, turning, holeLeft, speed]
        #     + ef: [frame,...] erase pickup frames
        self._pos_data = None  # frame -> {pid: (x, y, angle, alive, turning, holeLeft, speed)}
        self._pos_sample_every = 3
        self._pos_sampled_frames = []
        self._erase_frames = set()  # frames where erase powerup was picked up
        self._state_version = 2  # default
        st = round_data.get("st")
        if st and "d" in st and len(st["d"]) > 0:
            self._pos_data = {}
            self._pos_sample_every = st.get("se", 3)
            self._state_version = st.get("v", 2)
            self._erase_frames = set(st.get("ef", []))
            pids = st["pids"]
            vals_per_player = 7 if self._state_version >= 3 else 5
            for entry in st["d"]:
                frame_num = entry[0]
                row = entry[1]
                frame_state = {}
                for i, pid in enumerate(pids):
                    off = i * vals_per_player
                    if off + vals_per_player <= len(row):
                        gx = row[off]
                        gy = row[off + 1]
                        angle = row[off + 2]
                        alive = row[off + 3]
                        turning = row[off + 4]
                        holeLeft = row[off + 5] if vals_per_player >= 7 else -1.0
                        speed = row[off + 6] if vals_per_player >= 7 else -1.0
                        sx = gx * self.scale + self.offset_x
                        sy = gy * self.scale + self.offset_y
                        frame_state[pid] = (sx, sy, angle, int(alive), int(turning), holeLeft, speed)
                    elif off + 5 <= len(row):
                        # Fallback for v2 data (5 values)
                        gx, gy = row[off], row[off + 1]
                        angle, alive, turning = row[off + 2], row[off + 3], row[off + 4]
                        sx = gx * self.scale + self.offset_x
                        sy = gy * self.scale + self.offset_y
                        frame_state[pid] = (sx, sy, angle, int(alive), int(turning), -1.0, -1.0)
                self._pos_data[frame_num] = frame_state
            self._pos_sampled_frames = sorted(self._pos_data.keys())

        # Initialize players with field centering
        self.players = {}
        for sp in self.spawns:
            cid, x, y, angle_deg = sp[0], sp[1], sp[2], sp[3]
            sx = x * self.scale + self.offset_x
            sy = y * self.scale + self.offset_y
            self.players[cid] = ReplayPlayer(cid, sx, sy, angle_deg)

        # Trail grid
        self.trail_owner = np.zeros((SIM_SIZE, SIM_SIZE), dtype=np.int16)
        self.trail_frame = np.zeros((SIM_SIZE, SIM_SIZE), dtype=np.int32)

        # Wall mask at observation resolution: mark actual field boundaries
        self._wall_mask_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        top_ds = self.offset_y // DS_FACTOR
        bot_ds = min(OBS_SIZE, (self.offset_y + self.sim_h + DS_FACTOR - 1) // DS_FACTOR)
        left_ds = self.offset_x // DS_FACTOR
        right_ds = min(OBS_SIZE, (self.offset_x + self.sim_w + DS_FACTOR - 1) // DS_FACTOR)
        # Mark everything outside the field as wall
        if top_ds > 0:
            self._wall_mask_ds[:top_ds, :] = 1.0
        if bot_ds < OBS_SIZE:
            self._wall_mask_ds[bot_ds:, :] = 1.0
        if left_ds > 0:
            self._wall_mask_ds[:, :left_ds] = 1.0
        if right_ds < OBS_SIZE:
            self._wall_mask_ds[:, right_ds:] = 1.0
        # 1-pixel border at field edges
        if 0 <= top_ds < OBS_SIZE:
            self._wall_mask_ds[top_ds, left_ds:right_ds] = 1.0
        if 0 < bot_ds <= OBS_SIZE:
            self._wall_mask_ds[bot_ds - 1, left_ds:right_ds] = 1.0
        if 0 <= left_ds < OBS_SIZE:
            self._wall_mask_ds[top_ds:bot_ds, left_ds] = 1.0
        if 0 < right_ds <= OBS_SIZE:
            self._wall_mask_ds[top_ds:bot_ds, right_ds - 1] = 1.0

        # Ego-centric rotation grid (precomputed)
        half = OBS_SIZE / 2.0
        oy = np.arange(OBS_SIZE, dtype=np.float32) - half + 0.5
        ox = np.arange(OBS_SIZE, dtype=np.float32) - half + 0.5
        self._ego_dy, self._ego_dx = np.meshgrid(oy, ox, indexing='ij')

        self._spawn_grace_frames = int(self.fps * 1.5)  # ~1.5s grace
        self._ms_per_frame = 1000.0 / self.fps  # 16.667 for 60fps
        self.frame = 0

    def _move_player_subframe(self, player, turn_dir, frac, eff_speed, eff_turn_rate):
        """Move player for a fraction of a frame using exact arc integration.

        For turning: exact circular arc (zero integration error).
        For straight: trivial linear movement.
        """
        if turn_dir != 0:
            # Exact arc: player follows circle of radius R = eff_speed / eff_turn_rate
            # scaled by fraction of frame.
            alpha = turn_dir * eff_turn_rate * frac  # angle change
            theta0 = player.angle
            theta1 = theta0 + alpha
            R_eff = eff_speed / eff_turn_rate  # effective circle radius
            # Exact position on circular arc:
            # x' = x + d * R * (sin(theta') - sin(theta))
            # y' = y - d * R * (cos(theta') - cos(theta))
            player.x += turn_dir * R_eff * (math.sin(theta1) - math.sin(theta0))
            player.y -= turn_dir * R_eff * (math.cos(theta1) - math.cos(theta0))
            player.angle = theta1
        else:
            # Straight line -- no integration error
            player.x += math.cos(player.angle) * eff_speed * frac
            player.y += math.sin(player.angle) * eff_speed * frac

    def _draw_trail_pixel(self, player):
        """Draw trail from prev position to current position (interpolated).

        Interpolates along the movement path so trails are solid continuous
        lines even at high speed (no gaps between frames).
        Clips to rectangular field bounds.
        """
        hw = self.trail_hw
        pid = player.id + 1  # owner uses 1-indexed (0 = no trail)

        dx = player.x - player.prev_x
        dy = player.y - player.prev_y
        dist = math.hypot(dx, dy)
        n_stamps = max(1, int(math.ceil(dist)))

        x_min = self.offset_x
        x_max = self.offset_x + self.sim_w - 1
        y_min = self.offset_y
        y_max = self.offset_y + self.sim_h - 1

        perp_x = -math.sin(player.angle)
        perp_y = math.cos(player.angle)
        offsets = self._trail_offsets

        for i in range(n_stamps):
            t = i / max(1, n_stamps - 1) if n_stamps > 1 else 1.0
            px = player.prev_x + dx * t
            py = player.prev_y + dy * t

            cx = np.round(px + perp_x * offsets).astype(np.intp)
            cy = np.round(py + perp_y * offsets).astype(np.intp)
            valid = ((cx >= x_min) & (cx <= x_max) &
                     (cy >= y_min) & (cy <= y_max))
            cx, cy = cx[valid], cy[valid]
            self.trail_owner[cy, cx] = pid
            self.trail_frame[cy, cx] = self.frame

    def _draw_arrow_ds(self, channel, x, y, angle):
        """Draw 5-pixel direction arrow at observation resolution."""
        px_ds = x / DS_FACTOR
        py_ds = y / DS_FACTOR
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for t in range(5):
            cx = int(round(px_ds + cos_a * t))
            cy = int(round(py_ds + sin_a * t))
            if 0 <= cx < OBS_SIZE and 0 <= cy < OBS_SIZE:
                channel[cy, cx] = 1.0

    def _build_powerup_grids_ds(self):
        """Build powerup location grids at observation resolution (128x128)."""
        r_ds = self._powerup_radius_ds
        speed_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        erase_ds = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)

        for px, py, ptype in self.field_powerups:
            cx_ds = int(px / DS_FACTOR)
            cy_ds = int(py / DS_FACTOR)
            grid = speed_ds if ptype == 'speed' else erase_ds
            for dy in range(-r_ds, r_ds + 1):
                for dx in range(-r_ds, r_ds + 1):
                    if dx * dx + dy * dy <= r_ds * r_ds:
                        gx, gy = cx_ds + dx, cy_ds + dy
                        if 0 <= gx < OBS_SIZE and 0 <= gy < OBS_SIZE:
                            grid[gy, gx] = 1.0
        return speed_ds, erase_ds

    def _get_ego_obs(self, player):
        """Render ego-centric rotated observation for one player.

        Returns (4, 128, 128) when gspp=False, (6, 128, 128) when gspp=True.
        """
        pid = player.id + 1  # 1-indexed owner
        f = DS_FACTOR

        # Bool masks at 512x512
        self_mask = (self.trail_owner == pid)
        enemy_mask = (self.trail_owner > 0) & ~self_mask

        # Downsample to 128x128
        self_ds = self_mask.reshape(OBS_SIZE, f, OBS_SIZE, f).any(
            axis=(1, 3)).astype(np.float32)
        enemy_ds = enemy_mask.reshape(OBS_SIZE, f, OBS_SIZE, f).any(
            axis=(1, 3)).astype(np.float32)

        # Add walls
        np.maximum(self_ds, self._wall_mask_ds, out=self_ds)

        # Add direction arrows for all alive players
        for p in self.players.values():
            if p.alive:
                ch = self_ds if p.id == player.id else enemy_ds
                self._draw_arrow_ds(ch, p.x, p.y, p.angle)

        # Ego-centric rotation (centered on player, facing right)
        cos_a = math.cos(player.angle)
        sin_a = math.sin(player.angle)
        ego_col = player.x / f
        ego_row = player.y / f

        src_col = ego_col + cos_a * self._ego_dx - sin_a * self._ego_dy
        src_row = ego_row + sin_a * self._ego_dx + cos_a * self._ego_dy

        src_ci = np.round(src_col).astype(np.intp)
        src_ri = np.round(src_row).astype(np.intp)

        valid = (src_ci >= 0) & (src_ci < OBS_SIZE) & (src_ri >= 0) & (src_ri < OBS_SIZE)
        np.clip(src_ci, 0, OBS_SIZE - 1, out=src_ci)
        np.clip(src_ri, 0, OBS_SIZE - 1, out=src_ri)

        def rotate_ch(ch, fill):
            out = ch[src_ri, src_ci]
            out[~valid] = fill
            return out

        self_rot = rotate_ch(self_ds, 1.0)  # out of bounds = wall
        enemy_rot = rotate_ch(enemy_ds, 0.0)
        prev_self_rot = rotate_ch(player.prev_self_ds, 1.0)
        prev_enemy_rot = rotate_ch(player.prev_enemy_ds, 0.0)

        if self.gspp:
            speed_ds, erase_ds = self._build_powerup_grids_ds()
            speed_rot = rotate_ch(speed_ds, 0.0)
            erase_rot = rotate_ch(erase_ds, 0.0)
            obs = np.stack([self_rot, enemy_rot, prev_self_rot, prev_enemy_rot,
                            speed_rot, erase_rot])
        else:
            obs = np.stack([self_rot, enemy_rot, prev_self_rot, prev_enemy_rot])

        # Store world-frame channels for next frame
        player.prev_self_ds = self_ds.copy()
        player.prev_enemy_ds = enemy_ds.copy()

        return obs

    def step_frame(self, skip_obs=False):
        """Advance one frame. Returns dict of {curveId: (obs, action)} for alive players.

        Args:
            skip_obs: If True, skip expensive observation rendering. Returns
                     actions only (used for fast validation).
        """
        if self.frame >= self.num_frames:
            return None

        # Spawn powerups from replay data
        if self.frame in self.powerup_spawn_map:
            for sx, sy, ptype in self.powerup_spawn_map[self.frame]:
                self.field_powerups.append((sx, sy, ptype))

        # === POSITION MODE: Interpolate game positions at every frame (v2/v3 data) ===
        if self._pos_data is not None:
            # Find surrounding sampled frames for interpolation
            se = self._pos_sample_every
            f_prev = (self.frame // se) * se
            f_next = f_prev + se
            t = (self.frame - f_prev) / se if se > 0 else 0.0

            state_prev = self._pos_data.get(f_prev, {})
            state_next = self._pos_data.get(f_next, state_prev)

            # Step 1: Erase powerup -- exact frame from game engine (v3)
            if self.frame in self._erase_frames:
                self.trail_owner[:] = 0
                self.trail_frame[:] = 0
                # Remove the erase powerup from field display
                self.field_powerups = [
                    (x, y, pt) for x, y, pt in self.field_powerups if pt != 'erase'
                ]

            # Step 2: Update positions (interpolation)
            for p in self.players.values():
                if not p.alive:
                    continue

                if p.id in state_prev:
                    s0 = state_prev[p.id]
                    sx0, sy0, a0, alive0, turn0 = s0[0], s0[1], s0[2], s0[3], s0[4]
                    holeLeft0 = s0[5] if len(s0) > 5 else -1.0
                    speed0 = s0[6] if len(s0) > 6 else -1.0

                    if p.id in state_next:
                        s1 = state_next[p.id]
                        sx1, sy1, a1, alive1 = s1[0], s1[1], s1[2], s1[3]
                    else:
                        sx1, sy1, a1, alive1 = sx0, sy0, a0, alive0

                    p.prev_x = p.x
                    p.prev_y = p.y

                    # Linear interpolation between samples
                    p.x = sx0 + (sx1 - sx0) * t
                    p.y = sy0 + (sy1 - sy0) * t
                    da = a1 - a0
                    if da > math.pi:
                        da -= 2 * math.pi
                    if da < -math.pi:
                        da += 2 * math.pi
                    p.angle = a0 + da * t
                    # NOTE: Do NOT use turn0 from st.d -- c.turningDirection
                    # is always 0 in the JS extraction. Turning is derived
                    # from tu (turn events) below instead.

                    if not alive0:
                        p.alive = False

                    # v3: Use exact gap state from game engine
                    if holeLeft0 >= 0:
                        # holeLeft > 0 means player is in a gap (distance-based, not frames)
                        # Just use it as a boolean: drawing = not in gap
                        p.drawing = (holeLeft0 <= 0) and (self.frame >= self._spawn_grace_frames)
                    else:
                        # v2 fallback: use hole events + frame-based gap
                        if p.gap_remaining > 0:
                            p.gap_remaining -= 1
                            if p.gap_remaining <= 0:
                                p.drawing = True
                        elif self.frame >= self._spawn_grace_frames:
                            p.drawing = True

            # Step 3: Powerup pickup for speed boosts + erase fallback (v2)
            # v3 doesn't need distance-based erase (handled above), but still
            # needs speed boost tracking for the viewer's speed indicator
            if self.field_powerups:
                remaining = []
                for pup_x, pup_y, pup_type in self.field_powerups:
                    picked = False
                    for p in self.players.values():
                        if not p.alive:
                            continue
                        dist = math.hypot(p.x - pup_x, p.y - pup_y)
                        if dist < self._pickup_radius:
                            if pup_type == 'speed':
                                p.speed_boosts.append(
                                    (self.frame + int(4 * self.fps), 2.0)
                                )
                            elif pup_type == 'erase' and not self._erase_frames:
                                # v2 fallback only (v3 handles erase above)
                                self.trail_owner[:] = 0
                                self.trail_frame[:] = 0
                            picked = True
                            break
                    if not picked:
                        remaining.append((pup_x, pup_y, pup_type))
                self.field_powerups = remaining

            # Step 4: Gap handling for v2 fallback (hole events)
            if not self._erase_frames and self.frame in self.hole_map:
                # Only use hole events for gap in v2 mode (v3 uses holeLeft)
                for cid in self.hole_map[self.frame]:
                    if cid in self.players and self.players[cid].alive:
                        p = self.players[cid]
                        speed_mult = 1.0
                        for ef, m in p.speed_boosts:
                            if ef > self.frame:
                                speed_mult *= m
                        p.gap_remaining = self.gap_duration_frames / speed_mult
                        p.drawing = False

            # Step 4b: Update turning from turn events (tu).
            # The st.d turning field is broken (c.turningDirection always 0),
            # so we derive turning state from the correct tu event stream.
            if self.frame in self.turn_map:
                for pid, direction, sub_ms in self.turn_map[self.frame]:
                    if pid in self.players and self.players[pid].alive:
                        self.players[pid].turning = direction

            # Step 5: Trail drawing
            for p in self.players.values():
                if p.alive and p.drawing:
                    self._draw_trail_pixel(p)

        # === SIMULATION MODE: Reconstruct from events (v1 data, no positions) ===
        else:
            # Collect turn events per player with sub-frame timing
            frame_turn_events = {}
            if self.frame in self.turn_map:
                for pid, direction, sub_ms in self.turn_map[self.frame]:
                    if pid in self.players and self.players[pid].alive:
                        if pid not in frame_turn_events:
                            frame_turn_events[pid] = []
                        frame_turn_events[pid].append((sub_ms, direction))
            for pid in frame_turn_events:
                frame_turn_events[pid].sort(key=lambda e: e[0])

            # Check for gap starts (speed-dependent duration)
            if self.frame in self.hole_map:
                for cid in self.hole_map[self.frame]:
                    if cid in self.players and self.players[cid].alive:
                        p = self.players[cid]
                        speed_mult = 1.0
                        for ef, m in p.speed_boosts:
                            if ef > self.frame:
                                speed_mult *= m
                        p.gap_remaining = self.gap_duration_frames / speed_mult
                        p.drawing = False

            # Move players and draw trails
            for p in self.players.values():
                if not p.alive:
                    continue

                if p.speed_boosts:
                    p.speed_boosts = [
                        (ef, m) for ef, m in p.speed_boosts if ef > self.frame
                    ]
                speed_mult = 1.0
                for _, m in p.speed_boosts:
                    speed_mult *= m
                eff_speed = self.speed_per_frame * speed_mult
                eff_turn_rate = self.turn_rate_per_frame

                p.prev_x = p.x
                p.prev_y = p.y

                events = frame_turn_events.get(p.id)
                if events:
                    prev_frac = 0.0
                    current_dir = p.turning
                    for sub_ms, new_dir in events:
                        frac = max(0.0, min(1.0, sub_ms / self._ms_per_frame))
                        if frac > prev_frac:
                            self._move_player_subframe(
                                p, current_dir, frac - prev_frac,
                                eff_speed, eff_turn_rate)
                        current_dir = new_dir
                        prev_frac = frac
                    if prev_frac < 1.0:
                        self._move_player_subframe(
                            p, current_dir, 1.0 - prev_frac,
                            eff_speed, eff_turn_rate)
                    p.turning = current_dir
                else:
                    self._move_player_subframe(
                        p, p.turning, 1.0, eff_speed, eff_turn_rate)

                p.x = max(float(self.offset_x), min(float(self.offset_x + self.sim_w - 1), p.x))
                p.y = max(float(self.offset_y), min(float(self.offset_y + self.sim_h - 1), p.y))

                if p.gap_remaining > 0:
                    p.gap_remaining -= 1
                    if p.gap_remaining <= 0:
                        p.drawing = True
                elif self.frame >= self._spawn_grace_frames:
                    p.drawing = True

                if p.drawing and p.gap_remaining <= 0:
                    self._draw_trail_pixel(p)

        # Powerup pickup detection (simulation mode only -- position mode handles inline)
        if self.field_powerups and self._pos_data is None:
            remaining = []
            for pup_x, pup_y, pup_type in self.field_powerups:
                picked = False
                for p in self.players.values():
                    if not p.alive:
                        continue
                    dist = math.hypot(p.x - pup_x, p.y - pup_y)
                    if dist < self._pickup_radius:
                        if pup_type == 'speed':
                            p.speed_boosts.append(
                                (self.frame + int(4 * self.fps), 2.0)
                            )
                        elif pup_type == 'erase':
                            self.trail_owner[:] = 0
                            self.trail_frame[:] = 0
                        picked = True
                        break
                if not picked:
                    remaining.append((pup_x, pup_y, pup_type))
            self.field_powerups = remaining

        # Apply deaths
        if self.frame in self.death_map:
            for cid in self.death_map[self.frame]:
                if cid in self.players:
                    self.players[cid].alive = False

        # Generate observations for alive players
        results = {}
        if not skip_obs:
            alive = [p for p in self.players.values() if p.alive]
            if len(alive) > 0:
                for p in alive:
                    obs = self._get_ego_obs(p)
                    action = p.turning + 1
                    results[p.id] = (obs, action)

        self.frame += 1
        return results

    def get_top_players(self, top_n=None):
        """Return set of player IDs ranked by survival (longest alive first).

        Args:
            top_n: Only return top N players. None = all players.

        Returns:
            set of curveIds for the top-N surviving players.
        """
        # Collect death frames per player
        all_pids = {sp[0] for sp in self.spawns}
        death_frame = {}
        for frame, cids in self.death_map.items():
            for cid in cids:
                death_frame[cid] = frame
        # Players not in death_map survived the whole round
        for pid in all_pids:
            if pid not in death_frame:
                death_frame[pid] = self.num_frames + 1  # survived longest

        # Rank by death frame (highest = survived longest)
        ranked = sorted(all_pids, key=lambda p: death_frame[p], reverse=True)
        if top_n is not None and top_n < len(ranked):
            ranked = ranked[:top_n]
        return set(ranked)

    def validate_round(self, immunity_frames=8):
        """Validate a round by checking for impossible events.

        Runs the full simulation and checks at each frame whether any alive
        player's head center pixel overlaps trail pixels (enemy or old self
        trail). If a player overlaps trail but doesn't die, it's a violation.

        Uses center-pixel check only (matching env collision detection).

        Args:
            immunity_frames: Recent self-trail frames that don't count as
                           violations (self-immunity zone, time-based).

        Returns:
            dict with:
                violations: list of (frame, player_id, violation_type) tuples
                total_frames: int
                valid: bool (True if no critical violations)
        """
        violations = []

        for f in range(self.num_frames):
            self.step_frame(skip_obs=True)

            for p in self.players.values():
                if not p.alive:
                    continue
                if f < self._spawn_grace_frames:
                    continue

                ix = int(round(p.x))
                iy = int(round(p.y))
                if ix < 0 or ix >= SIM_SIZE or iy < 0 or iy >= SIM_SIZE:
                    continue

                owner = self.trail_owner[iy, ix]
                if owner == 0:
                    continue

                trail_age = f - self.trail_frame[iy, ix]
                is_self = (owner == p.id + 1)

                if is_self and trail_age < immunity_frames:
                    continue

                # Player head is on trail but alive -- check if death soon
                dies_soon = False
                for df in range(0, 4):
                    if (f + df) in self.death_map and p.id in self.death_map[f + df]:
                        dies_soon = True
                        break

                if not dies_soon:
                    vtype = "self_cross" if is_self else "enemy_cross"
                    violations.append((f, p.id, vtype))

        return {
            "violations": violations,
            "total_frames": self.num_frames,
            "n_violations": len(violations),
            "valid": len(violations) == 0,
        }

    def render_full_round(self, sample_every=1, top_n=None):
        """Render entire round, yielding (obs, action, player_id) triples.

        Args:
            sample_every: Only yield every Nth frame (reduces data size).
            top_n: Only render observations for top-N surviving players.
                   None = all players (legacy behavior).

        Yields:
            (obs: np.array[C,128,128], action: int, player_id: int) triples
            C=4 when gspp=False, C=6 when gspp=True
        """
        keep_pids = self.get_top_players(top_n) if top_n else None

        for f in range(self.num_frames):
            results = self.step_frame()
            if results is None:
                break
            if f % sample_every == 0 and f >= self._spawn_grace_frames:
                for pid, (obs, action) in results.items():
                    if keep_pids is None or pid in keep_pids:
                        yield obs, action, pid


def load_replays(ndjson_path):
    """Load all games from an NDJSON file."""
    games = []
    with open(ndjson_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                games.append(json.loads(line))
    return games


def count_bc_samples(data_dir, sample_every=3, gspp=False):
    """Count total (obs, action) pairs without rendering (approximate)."""
    data_dir = Path(data_dir)
    total = 0

    if gspp:
        files = ["ffa_gspp.ndjson", "1v1_gspp.ndjson",
                 "ffa_none.ndjson", "1v1_none.ndjson"]
    else:
        files = ["ffa_none.ndjson", "1v1_none.ndjson"]

    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        games = load_replays(fpath)
        for game in games:
            for rnd in game["rounds"]:
                nf = rnd["nf"]
                grace = int(game["settings"]["fps"] * 1.5)
                alive_frames = max(0, nf - grace)
                sampled_frames = alive_frames // sample_every
                n_players = len(rnd["sp"])
                total += sampled_frames * n_players
    return total


def prerender_all(data_dir, output_path, sample_every=6, max_rounds=None,
                  gspp=False, top_n=None, file_list=None,
                  chunk_rounds=10):
    """Pre-render replays to memory-mapped .npy files, saving in chunks.

    Renders all rounds, flushing to temporary .npz chunks every chunk_rounds
    to limit RAM usage. Then merges chunks into flat .npy files via memmap
    (O(chunk_size) RAM, not O(total)).

    Output files (BCReplayBuffer-compatible via mmap):
        output_path_obs.npy      -- (N, C, 128, 128) uint8
        output_path_act.npy      -- (N,) uint8
        output_path_round_id.npy -- (N,) int32
        output_path_player_id.npy -- (N,) int16

    Args:
        top_n: Only render top-N surviving players per round.
        chunk_rounds: Save to disk every N rounds to limit RAM usage.
        file_list: Optional list of NDJSON filenames to process.
    """
    data_dir = Path(data_dir)
    n_ch = 6 if gspp else 4

    chunk_obs = []
    chunk_act = []
    chunk_round_id = []
    chunk_player_id = []
    total_rounds = 0
    total_samples = 0
    chunk_idx = 0
    chunk_files = []

    if file_list is not None:
        files = file_list
    elif gspp:
        files = ["ffa_gspp_v2_valid.ndjson", "ffa_none_v2_valid.ndjson"]
    else:
        files = ["ffa_none_v2_valid.ndjson"]

    def flush_chunk():
        nonlocal chunk_obs, chunk_act, chunk_round_id, chunk_player_id, chunk_idx
        if not chunk_obs:
            return
        obs_arr = np.array(chunk_obs, dtype=np.uint8)
        act_arr = np.array(chunk_act, dtype=np.uint8)
        rid_arr = np.array(chunk_round_id, dtype=np.int32)
        pid_arr = np.array(chunk_player_id, dtype=np.int16)

        chunk_path = f"{output_path}_{chunk_idx:03d}"
        np.savez_compressed(chunk_path, obs=obs_arr, act=act_arr,
                            round_id=rid_arr, player_id=pid_arr)
        chunk_files.append(chunk_path + ".npz")
        size_mb = Path(chunk_path + ".npz").stat().st_size / 1e6
        print(f"    Chunk {chunk_idx}: {len(obs_arr):,} samples "
              f"({size_mb:.1f} MB on disk)", flush=True)
        chunk_idx += 1
        chunk_obs = []
        chunk_act = []
        chunk_round_id = []
        chunk_player_id = []

    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        games = load_replays(fpath)
        print(f"  Processing {fname}: {len(games)} games...", flush=True)
        file_rounds = 0

        for game in games:
            settings = game["settings"]
            for rnd in game["rounds"]:
                if rnd["nf"] < settings["fps"] * 3:
                    continue
                if rnd.get("err"):
                    continue
                if max_rounds and total_rounds >= max_rounds:
                    break

                renderer = ReplayRenderer(settings, rnd, gspp=gspp)
                for obs, action, pid in renderer.render_full_round(
                        sample_every=sample_every, top_n=top_n):
                    chunk_obs.append((obs * 255).astype(np.uint8))
                    chunk_act.append(action)
                    chunk_round_id.append(total_rounds)
                    chunk_player_id.append(pid)
                    total_samples += 1

                total_rounds += 1
                file_rounds += 1

                # Flush chunk to disk periodically
                if total_rounds % chunk_rounds == 0:
                    print(f"    {total_rounds} rounds, "
                          f"{total_samples:,} samples total", flush=True)
                    flush_chunk()

            if max_rounds and total_rounds >= max_rounds:
                break

        print(f"    {fname}: {file_rounds} rounds rendered", flush=True)

    # Flush remaining
    flush_chunk()

    if not chunk_files:
        print("No samples rendered!")
        return 0

    # Merge chunks into flat .npy files via memmap (O(chunk_size) RAM)
    print(f"  Merging {len(chunk_files)} chunks into .npy files "
          f"({total_samples:,} total samples)...", flush=True)

    base = str(output_path)
    obs_path = base + "_obs.npy"
    act_path = base + "_act.npy"
    rid_path = base + "_round_id.npy"
    pid_path = base + "_player_id.npy"

    # Create memory-mapped output files
    obs_out = np.lib.format.open_memmap(
        obs_path, mode='w+', dtype=np.uint8,
        shape=(total_samples, n_ch, OBS_SIZE, OBS_SIZE))
    act_out = np.lib.format.open_memmap(
        act_path, mode='w+', dtype=np.uint8,
        shape=(total_samples,))
    rid_out = np.lib.format.open_memmap(
        rid_path, mode='w+', dtype=np.int32,
        shape=(total_samples,))
    pid_out = np.lib.format.open_memmap(
        pid_path, mode='w+', dtype=np.int16,
        shape=(total_samples,))

    offset = 0
    for ci, cf in enumerate(chunk_files):
        d = np.load(cf)
        n = len(d["obs"])
        obs_out[offset:offset + n] = d["obs"]
        act_out[offset:offset + n] = d["act"]
        rid_out[offset:offset + n] = d["round_id"]
        pid_out[offset:offset + n] = d["player_id"]
        d.close()
        offset += n
        print(f"    Merged chunk {ci}/{len(chunk_files)} "
              f"({offset:,}/{total_samples:,})", flush=True)

    # Flush memmap to disk
    del obs_out, act_out, rid_out, pid_out

    obs_size_gb = Path(obs_path).stat().st_size / 1e9
    print(f"Saved {total_samples:,} samples ({n_ch}ch) to {base}_*.npy "
          f"(obs: {obs_size_gb:.1f} GB)", flush=True)

    # Clean up chunk files
    for cf in chunk_files:
        Path(cf).unlink(missing_ok=True)

    return total_samples


# ============================================================
# Subcommand: scrape-leaderboard
# ============================================================

# Scraper constants
_BASE_URL = "https://curvecrash.com"

# Profile-level filters.
# Types we want (profile page shows: FFA, 1v1, Team)
_VALID_TYPES = {"FFA", "1v1"}
# Modes that are definitely good (no replay verification needed)
_GOOD_MODES = {"None", "GS++"}
# Modes that need replay verification (powerupIds hidden on profile)
_TOUR_MODES = {"TOUR", "TOUR FINAL"}
# Combined: all modes we keep as candidates
_VALID_MODES = _GOOD_MODES | _TOUR_MODES

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _make_session(proxy: str | None = None):
    import requests
    s = requests.Session()
    s.headers.update(_HEADERS)
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


def _fetch_with_retry(session, url: str, delay: float, retries: int = 3):
    import requests as _requests
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            time.sleep(delay + random.uniform(0, delay * 0.25))
            return r
        except _requests.RequestException as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [retry {attempt+1}/{retries}] {e} -- waiting {wait:.1f}s")
            time.sleep(wait)
    return None


def _scrape_leaderboard_pages(session, board_type: str,
                              period: str, delay: float,
                              max_pages: int = 2) -> dict[str, int]:
    """Scrape leaderboard pages. Returns {username: elo}.
    period: 'current' or 'record' (all-time)."""
    from bs4 import BeautifulSoup

    players = {}
    for page in range(1, max_pages + 1):
        url = f"{_BASE_URL}/leaderboard/{board_type}/{period}?page={page}"
        label = f"{board_type}/{period}"
        print(f"  Fetching {label} page {page}/{max_pages}...")
        r = _fetch_with_retry(session, url, delay)
        if r is None:
            print(f"  FAILED page {page}, stopping {label}.")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table tbody tr")
        if not rows:
            break

        for row in rows:
            link = row.select_one("a.table-link")
            tds = row.select("td")
            if not link or len(tds) < 2:
                continue

            href = link.get("href", "")
            m = re.search(r"/u/(.+)$", href)
            if not m:
                continue
            name = m.group(1)

            try:
                elo = int(tds[-1].get_text(strip=True))
            except (ValueError, IndexError):
                continue

            # Keep highest ELO if player already seen
            if name not in players or elo > players[name]:
                players[name] = elo

    print(f"  Found {len(players)} players on {label} ({max_pages} pages).")
    return players


def _scrape_player_profile(session, username: str, delay: float) -> list[dict]:
    """Scrape a player's profile page for game history. Returns list of game dicts."""
    from bs4 import BeautifulSoup

    url = f"{_BASE_URL}/u/{username}"
    r = _fetch_with_retry(session, url, delay)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    games = []

    # Game history is the table with 8-column rows containing /game/ links
    # Columns: [date, name(+game link), type, mode, result, elo, delta, replay]
    tables = soup.select("table")
    for table in tables:
        rows = table.select("tbody tr")
        for row in rows:
            tds = row.select("td")
            if len(tds) != 8:
                continue

            # td[1] has the game link: /game/{id}
            game_link = tds[1].select_one("a[href*='/game/']")
            if not game_link:
                continue

            href = game_link.get("href", "")
            gid_match = re.search(r"/game/(\d+)", href)
            if not gid_match:
                continue

            game_id = gid_match.group(1)

            # td[2] = game type (FFA, 1v1, Team)
            game_type = tds[2].get_text(strip=True)
            if game_type == "1 v 1":
                game_type = "1v1"

            # td[3] = mode (None, GS++, GS, All, etc.)
            mode = tds[3].get_text(strip=True)

            games.append({
                "gameId": game_id,
                "type": game_type,
                "mode": mode,
                "player": username,
            })

    return games


def cmd_scrape_leaderboard(args):
    """Scrape CurveCrash leaderboards + player profiles."""
    DATA_DIR.mkdir(exist_ok=True)
    session = _make_session(args.proxy)

    # --- Phase 1: Scrape leaderboards (current + all-time record) ---
    print("=== Phase 1: Scraping leaderboards ===")
    print("  --- Current season ---")
    ffa_cur = _scrape_leaderboard_pages(session, "ffa", "current", args.delay, args.max_pages)
    v1_cur = _scrape_leaderboard_pages(session, "1v1", "current", args.delay, args.max_pages)
    print("  --- All-time record ---")
    ffa_rec = _scrape_leaderboard_pages(session, "ffa", "record", args.delay, args.max_pages)
    v1_rec = _scrape_leaderboard_pages(session, "1v1", "record", args.delay, args.max_pages)

    # Merge: keep highest ELO per player across current + record
    ffa_players: dict[str, int] = {}
    for src in (ffa_cur, ffa_rec):
        for name, elo in src.items():
            if name not in ffa_players or elo > ffa_players[name]:
                ffa_players[name] = elo

    v1_players: dict[str, int] = {}
    for src in (v1_cur, v1_rec):
        for name, elo in src.items():
            if name not in v1_players or elo > v1_players[name]:
                v1_players[name] = elo

    print(f"\n  Merged: {len(ffa_players)} unique FFA players, {len(v1_players)} unique 1v1 players")

    # Build elite set
    elite = {}
    for name, elo in ffa_players.items():
        if elo >= args.min_elo:
            elite[name] = {"ffa_elo": elo, "v1_elo": None}

    for name, elo in v1_players.items():
        if elo >= args.min_elo:
            if name in elite:
                elite[name]["v1_elo"] = elo
            else:
                elite[name] = {"ffa_elo": None, "v1_elo": elo}

    # Add extra players (always included regardless of ELO)
    for name in args.extra_players:
        if name not in elite:
            elite[name] = {"ffa_elo": None, "v1_elo": None, "bonus": True}
            print(f"  + Added bonus player: {name}")

    print(f"\n=== Elite players (ELO >= {args.min_elo}): {len(elite)} ===")

    elite_path = DATA_DIR / "elite_players.json"
    with open(elite_path, "w", encoding="utf-8") as f:
        json.dump(elite, f, indent=2, ensure_ascii=False)
    print(f"Saved to {elite_path}")

    if args.skip_profiles:
        print("Skipping profile scraping (--skip-profiles).")
        return

    # --- Phase 2: Scrape player profiles ---
    print(f"\n=== Phase 2: Scraping {len(elite)} player profiles ===")
    candidate_games: dict[str, dict] = {}

    # Load existing progress if any
    candidates_path = DATA_DIR / "candidate_game_ids.json"
    scraped_profiles_path = DATA_DIR / "scraped_profiles.json"
    scraped_profiles: set[str] = set()

    if scraped_profiles_path.exists():
        scraped_profiles = set(json.loads(scraped_profiles_path.read_text()))
        print(f"  Resuming: {len(scraped_profiles)} profiles already scraped.")

    if candidates_path.exists():
        candidate_games = json.loads(candidates_path.read_text())
        print(f"  Resuming: {len(candidate_games)} candidate games loaded.")

    for i, name in enumerate(elite, 1):
        if name in scraped_profiles:
            continue

        print(f"  [{i}/{len(elite)}] Scraping profile: {name}")
        games = _scrape_player_profile(session, name, args.delay)

        kept = 0
        for g in games:
            # Filter: only FFA/1v1, only None/GS++/TOUR/TOUR FINAL
            if g["type"] not in _VALID_TYPES:
                continue
            if g["mode"] not in _VALID_MODES:
                continue

            gid = g["gameId"]
            kept += 1
            if gid in candidate_games:
                if name not in candidate_games[gid].get("players_seen", []):
                    candidate_games[gid]["players_seen"].append(name)
            else:
                candidate_games[gid] = {
                    "type": g["type"],
                    "mode": g["mode"],
                    "players_seen": [name],
                }
        print(f"    {len(games)} games on profile, {kept} kept (FFA/1v1 + None/GS++/TOUR)")

        scraped_profiles.add(name)

        # Save progress every 10 profiles
        if i % 10 == 0:
            with open(candidates_path, "w", encoding="utf-8") as f:
                json.dump(candidate_games, f, indent=2, ensure_ascii=False)
            with open(scraped_profiles_path, "w", encoding="utf-8") as f:
                json.dump(sorted(scraped_profiles), f, ensure_ascii=False)
            print(f"    Progress saved: {len(candidate_games)} candidate games.")

    # Final save
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidate_games, f, indent=2, ensure_ascii=False)
    with open(scraped_profiles_path, "w", encoding="utf-8") as f:
        json.dump(sorted(scraped_profiles), f, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  Elite players: {len(elite)}")
    print(f"  Candidate games: {len(candidate_games)}")
    print(f"  Saved to {candidates_path}")

    # Stats breakdown
    type_counts = {}
    mode_counts = {}
    for g in candidate_games.values():
        t = g.get("type", "unknown")
        m = g.get("mode", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        mode_counts[m] = mode_counts.get(m, 0) + 1
    print("  By type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")
    print("  By mode:")
    for m, c in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"    {m}: {c}")
    tour_count = sum(c for m, c in mode_counts.items() if m in _TOUR_MODES)
    known_count = sum(c for m, c in mode_counts.items() if m in _GOOD_MODES)
    print(f"  Known good (None/GS++): {known_count}")
    print(f"  TOUR (needs replay check): {tour_count}")


# ============================================================
# Subcommand: scrape-replays
# ============================================================

# JS code injected into the replay page to extract game data.
# Uses Curve.Replay.decodeRound() for reliable binary->action extraction.
# Action toJSON() formats:
#   Turn:         [0, roundNum, frame, roundPlayerId, actionNum, subMillis, turningDirection]
#   Hole:         [1, roundNum, frame, curveId, actionNum]
#   Death:        [2, roundNum, frame, curveId, actionNum, killerId, killerSpeed]
#   PowerupSpawn: [3, roundNum, frame, actionNum, powerupNum, x, y, powerupId]
_EXTRACT_JS = """
async (args) => {
    const gameId = args.gameId;
    const eliteNames = args.eliteNames || [];
    const requireElite = args.requireElite || false;
    const eliteSet = new Set(eliteNames);

    // Fetch replay
    const resp = await fetch('/api/replay/' + gameId);
    if (!resp.ok) return { skip: true, reason: 'http_' + resp.status };

    const b64 = await resp.text();
    if (!b64 || b64.length < 50) return { skip: true, reason: 'empty_replay' };

    // Decode replay
    let replayData;
    try {
        replayData = Curve.Replay.base64ToReplayData(b64);
    } catch (e) {
        return { skip: true, reason: 'decode_error' };
    }

    let game;
    try {
        game = Curve.Replay.replayDataToGame(replayData);
    } catch (e) {
        return { skip: true, reason: 'parse_error' };
    }

    // --- FAST FILTER ---
    const gs = game.gameSettings;
    if (!gs) return { skip: true, reason: 'no_settings' };

    // Game type: gs.gameType is "FFA", "VS", "TEAM"
    const gt = gs.gameType;
    let gameType;
    if (gt === 'FFA') gameType = 'FFA';
    else if (gt === 'VS') gameType = '1v1';
    else return { skip: true, reason: 'type_' + gt };

    // Ranked check
    if (!gs.isRanked) return { skip: true, reason: 'unranked' };

    // Player count sanity
    const nPlayers = game.players ? game.players.length : 0;
    if (gameType === 'FFA' && nPlayers < 3) return { skip: true, reason: 'ffa_' + nPlayers + 'p' };
    if (gameType === '1v1' && nPlayers !== 2) return { skip: true, reason: '1v1_' + nPlayers + 'p' };

    // Powerup mode
    const pids = gs.powerupIds || [];
    let mode;
    if (pids.length === 0) {
        mode = 'none';
    } else if (pids.length === 2 && pids.includes(1) && pids.includes(9)) {
        mode = 'gspp';
    } else {
        return { skip: true, reason: 'powerups_' + JSON.stringify(pids) };
    }

    // Player names (from player.data.username)
    const playerNames = (game.players || []).map(p => p.data ? p.data.username : '');

    // Elite check (scan mode)
    if (requireElite && eliteSet.size > 0) {
        const hasElite = playerNames.some(n => eliteSet.has(n));
        if (!hasElite) return { skip: true, reason: 'no_elite' };
    }

    // --- FULL EXTRACTION via decodeRound ---
    const roundReplays = game.roundReplays || [];
    const rounds = [];

    for (let ri = 0; ri < roundReplays.length; ri++) {
        const rr = roundReplays[ri];
        try {
            const decoded = Curve.Replay.decodeRound(game, rr);
            const allActions = decoded.actions.map(a => a.toJSON());

            // Separate by action type and remap to compact format
            const tu = []; // [frame, playerId, direction, subMillis]
            const de = []; // [frame, whoDied, killedBy]
            const ho = []; // [frame, curveId]
            const pu = []; // [frame, x, y, powerupId]

            for (const a of allActions) {
                switch (a[0]) {
                    case 0: // TURN: [0, rn, frame, pid, anum, sub, dir]
                        tu.push([a[2], a[3], a[6], a[5]]);
                        break;
                    case 1: // HOLE: [1, rn, frame, cid, anum]
                        ho.push([a[2], a[3]]);
                        break;
                    case 2: // DEATH: [2, rn, frame, cid, anum, killerId, killerSpeed]
                        de.push([a[2], a[3], a[5]]);
                        break;
                    case 3: // POWERUP_SPAWN: [3, rn, frame, anum, pnum, x, y, pid]
                        pu.push([a[2], a[5], a[6], a[7]]);
                        break;
                }
            }

            // Spawns: each TeamSpawn has {score, players: [{x, y, angle, score}]}
            // In FFA each team = 1 player, so spawn index = player index
            const sp = [];
            for (let si = 0; si < decoded.spawns.length; si++) {
                const ts = decoded.spawns[si];
                for (const ps of ts.players) {
                    sp.push([si, ps.x, ps.y, ps.angle]);
                }
            }

            rounds.push({
                r: ri,
                nf: rr.numFrames,
                sp: sp,
                tu: tu,
                de: de,
                ho: ho,
                pu: pu
            });
        } catch (e) {
            rounds.push({
                r: ri,
                nf: rr.numFrames,
                sp: [], tu: [], de: [], ho: [], pu: [],
                err: e.message
            });
        }
    }

    // --- PER-FRAME STATE EXTRACTION (v3) ---
    // Extracts exact game engine state: position, angle, alive, turning,
    // holeLeft (gap remaining), speed (including boosts).
    // Also detects erase powerup pickups by tracking fieldPowerups.
    if (args.withPositions) {
        const sampleEvery = args.sampleEvery || 3;
        try {
            const ge = new Curve.GameEngine(game, false);
            const rh = new Curve.ReplayHandler(ge);

            for (let ri = 0; ri < roundReplays.length; ri++) {
                if (rounds[ri].err) continue;

                rh.goToRound(ri);
                const nf = roundReplays[ri].numFrames;
                const curves0 = ge.round.getCurves();
                const pids = curves0.map(c => c.curveId);
                const stFrames = [];
                const eraseFrames = [];  // frames where erase powerup was picked up

                let prevEraseCount = 0;  // track erase items on field

                for (let f = 0; f < nf; f++) {
                    rh.goToFrame(f);

                    // Detect erase pickups: count erase powerups on field
                    const fp = ge.round.state.fieldPowerups || [];
                    let curEraseCount = 0;
                    for (const p of fp) {
                        if (p.powerupId === 9) curEraseCount++;
                    }
                    // If erase count dropped, someone picked up an erase
                    if (curEraseCount < prevEraseCount) {
                        eraseFrames.push(f);
                    }
                    prevEraseCount = curEraseCount;

                    if (f % sampleEvery === 0) {
                        const curves = ge.round.getCurves();
                        // 7 values per player: x, y, angle, alive, turning, holeLeft, speed
                        const row = [];
                        for (const c of curves) {
                            row.push(
                                +(c.state.x.toFixed(1)),
                                +(c.state.y.toFixed(1)),
                                +(c.state.angle.toFixed(3)),
                                c.state.isAlive ? 1 : 0,
                                c.turningDirection || 0,
                                +(c.state.holeLeft.toFixed(1)),
                                +(c.state.speed.toFixed(2))
                            );
                        }
                        stFrames.push([f, row]);
                    }
                }

                rounds[ri].st = {
                    se: sampleEvery, pids: pids, d: stFrames,
                    v: 3,  // version marker
                    ef: eraseFrames  // frames where erase was picked up
                };
            }

            rh.stop();
            ge.destroy();
        } catch(e) {
            console.warn('State extraction failed:', e.message);
        }
    }

    return {
        skip: false,
        data: {
            gameId: gameId,
            mode: mode,
            type: gameType,
            players: (game.players || []).map((p, i) => ({
                id: i,
                name: p.data ? p.data.username : ''
            })),
            settings: {
                fw: gs.fieldWidth,
                fh: gs.fieldHeight,
                fps: gs.fps || 60,
                speed: gs.curveSpeed,
                tr: gs.turnRadius,
                size: gs.curveSize,
                hs: gs.holeSize,
                ranked: gs.isRanked,
                firstTo: gs.firstTo
            },
            rounds: rounds
        }
    };
}
"""


def _get_output_path(game_type: str, mode: str) -> Path:
    """Get the NDJSON output file path for a game type + mode combo."""
    prefix = "ffa" if game_type == "FFA" else "1v1"
    return DATA_DIR / f"{prefix}_{mode}.ndjson"


def _load_scrape_state() -> dict:
    """Load resume state for replay scraping."""
    state = {
        "scraped_ids": set(),
        "scan_last_id": None,
        "stats": {"checked": 0, "ffa_none": 0, "ffa_gspp": 0,
                  "v1_none": 0, "v1_gspp": 0, "skipped": 0, "errors": 0},
    }

    scraped_path = DATA_DIR / "scraped_ids.json"
    if scraped_path.exists():
        ids = json.loads(scraped_path.read_text())
        state["scraped_ids"] = set(ids)

    scan_path = DATA_DIR / "scan_state.json"
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        state["scan_last_id"] = scan.get("last_id")
        state["stats"].update(scan.get("stats", {}))

    return state


def _save_scrape_state(state: dict):
    """Persist resume state for replay scraping."""
    scraped_path = DATA_DIR / "scraped_ids.json"
    with open(scraped_path, "w") as f:
        json.dump(sorted(state["scraped_ids"]), f)

    scan_path = DATA_DIR / "scan_state.json"
    with open(scan_path, "w") as f:
        json.dump({
            "last_id": state.get("scan_last_id"),
            "stats": state["stats"],
        }, f, indent=2)


def _append_game(game_data: dict):
    """Append a game to the appropriate NDJSON file."""
    path = _get_output_path(game_data["type"], game_data["mode"])
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(game_data, ensure_ascii=False, separators=(",", ":")) + "\n")


def _log_msg(msg: str):
    """Print and log a message."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_path = DATA_DIR / "scrape_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


async def _setup_page(browser, proxy: str | None = None):
    """Create a page loaded with Curve.Replay APIs."""
    context = await browser.new_context()
    page = await context.new_page()

    # Navigate to replay page to get Curve.Replay APIs loaded
    _log_msg("Loading replay page to initialize Curve.Replay APIs...")
    await page.goto(f"{_BASE_URL}/replay/v1.1/?game=1", wait_until="networkidle",
                    timeout=30000)

    # Wait for Curve.Replay to be available
    try:
        await page.wait_for_function(
            "() => typeof Curve !== 'undefined' && typeof Curve.Replay !== 'undefined'",
            timeout=15000
        )
        _log_msg("Curve.Replay APIs loaded successfully.")
    except Exception as e:
        _log_msg(f"WARNING: Curve.Replay APIs may not be loaded: {e}")
        # Try loading a known valid game
        await page.goto(f"{_BASE_URL}/replay/v1.1/?game=714000",
                        wait_until="networkidle", timeout=30000)
        await page.wait_for_function(
            "() => typeof Curve !== 'undefined' && typeof Curve.Replay !== 'undefined'",
            timeout=15000
        )
        _log_msg("Curve.Replay APIs loaded on retry.")

    return page


async def _process_game(page, game_id: int, elite_names: list[str] | None,
                        require_elite: bool, with_positions: bool = False,
                        sample_every: int = 3) -> dict | None:
    """Process a single game. Returns game data dict or None if skipped."""
    try:
        result = await page.evaluate(
            _EXTRACT_JS,
            {
                "gameId": game_id,
                "eliteNames": elite_names or [],
                "requireElite": require_elite,
                "withPositions": with_positions,
                "sampleEvery": sample_every,
            }
        )
    except Exception as e:
        return {"error": str(e)}

    if not result:
        return {"error": "null_result"}

    if result.get("skip"):
        return {"skipped": result.get("reason", "unknown")}

    return result.get("data")


async def _run_targeted(page, state: dict, limit: int | None, delay: float):
    """Process games from candidate_game_ids.json."""
    import asyncio

    candidates_path = DATA_DIR / "candidate_game_ids.json"
    if not candidates_path.exists():
        _log_msg("ERROR: data/candidate_game_ids.json not found. Run scrape-leaderboard first.")
        return

    candidates = json.loads(candidates_path.read_text())
    game_ids = [gid for gid in candidates.keys() if gid not in state["scraped_ids"]]

    if limit:
        game_ids = game_ids[:limit]

    total = len(game_ids)
    _log_msg(f"Targeted mode: {total} games to process ({len(state['scraped_ids'])} already done)")

    for i, gid in enumerate(game_ids, 1):
        gid_int = int(gid)
        stats = state["stats"]
        stats["checked"] += 1

        result = await _process_game(page, gid_int, elite_names=None, require_elite=False)

        if result is None:
            stats["errors"] += 1
            _log_msg(f"  [{i}/{total}] Game {gid}: null result")
        elif "error" in result:
            stats["errors"] += 1
            _log_msg(f"  [{i}/{total}] Game {gid}: ERROR {result['error']}")
        elif "skipped" in result:
            stats["skipped"] += 1
            reason = result["skipped"]
            if i % 50 == 0 or "error" in reason:
                _log_msg(f"  [{i}/{total}] Game {gid}: skip ({reason})")
        else:
            # Valid game
            _append_game(result)
            key = f"{'ffa' if result['type'] == 'FFA' else 'v1'}_{result['mode']}"
            stats[key] = stats.get(key, 0) + 1
            nr = len(result.get("rounds", []))
            nt = sum(len(r.get("tu", [])) for r in result.get("rounds", []))
            _log_msg(f"  [{i}/{total}] Game {gid}: {result['type']} {result['mode']} "
                     f"({nr} rounds, {nt} turns)")

        state["scraped_ids"].add(gid)

        # Save state periodically
        if i % 25 == 0:
            _save_scrape_state(state)
            s = stats
            _log_msg(f"  --- Progress: checked={s['checked']}, "
                     f"ffa_none={s.get('ffa_none',0)}, ffa_gspp={s.get('ffa_gspp',0)}, "
                     f"1v1_none={s.get('v1_none',0)}, 1v1_gspp={s.get('v1_gspp',0)}, "
                     f"skip={s['skipped']}, err={s['errors']} ---")

        # Rate limit
        await asyncio.sleep(delay + random.uniform(0, delay * 0.3))

    _save_scrape_state(state)


async def _run_scan(page, state: dict, start_id: int, limit: int | None,
                    delay: float):
    """Backward scan from start_id, filtering for elite player games."""
    import asyncio

    elite_path = DATA_DIR / "elite_players.json"
    if not elite_path.exists():
        _log_msg("ERROR: data/elite_players.json not found. Run scrape-leaderboard first.")
        return

    elite = json.loads(elite_path.read_text())
    elite_names_list = list(elite.keys())
    _log_msg(f"Loaded {len(elite_names_list)} elite players for filtering.")

    # Resume from last scan position
    current_id = start_id
    if state.get("scan_last_id") and state["scan_last_id"] < current_id:
        current_id = state["scan_last_id"]
        _log_msg(f"Resuming scan from game {current_id}")

    processed = 0
    max_games = limit or 999999

    _log_msg(f"Scan mode: starting at game {current_id}, limit={limit or 'none'}")

    consecutive_errors = 0
    max_consecutive_errors = 50

    while processed < max_games and current_id > 0:
        gid_str = str(current_id)
        stats = state["stats"]

        if gid_str in state["scraped_ids"]:
            current_id -= 1
            continue

        stats["checked"] += 1
        processed += 1

        result = await _process_game(page, current_id, elite_names=elite_names_list, require_elite=True)

        if result is None:
            stats["errors"] += 1
            consecutive_errors += 1
        elif "error" in result:
            stats["errors"] += 1
            consecutive_errors += 1
            if consecutive_errors % 10 == 0:
                _log_msg(f"  Game {current_id}: ERROR {result['error']} "
                         f"({consecutive_errors} consecutive errors)")
        elif "skipped" in result:
            stats["skipped"] += 1
            consecutive_errors = 0
        else:
            consecutive_errors = 0
            _append_game(result)
            key = f"{'ffa' if result['type'] == 'FFA' else 'v1'}_{result['mode']}"
            stats[key] = stats.get(key, 0) + 1
            nr = len(result.get("rounds", []))
            nt = sum(len(r.get("tu", [])) for r in result.get("rounds", []))
            _log_msg(f"  Game {current_id}: {result['type']} {result['mode']} "
                     f"({nr} rounds, {nt} turns)")

        state["scraped_ids"].add(gid_str)
        state["scan_last_id"] = current_id

        if consecutive_errors >= max_consecutive_errors:
            _log_msg(f"  WARNING: {max_consecutive_errors} consecutive errors. "
                     f"Skipping ahead by 100.")
            current_id -= 100
            consecutive_errors = 0
            continue

        # Save state periodically
        if processed % 50 == 0:
            _save_scrape_state(state)
            s = stats
            _log_msg(f"  --- Scan at {current_id} | checked={s['checked']}, "
                     f"ffa_none={s.get('ffa_none',0)}, ffa_gspp={s.get('ffa_gspp',0)}, "
                     f"1v1_none={s.get('v1_none',0)}, 1v1_gspp={s.get('v1_gspp',0)}, "
                     f"skip={s['skipped']}, err={s['errors']} ---")

        current_id -= 1
        await asyncio.sleep(delay + random.uniform(0, delay * 0.3))

    _save_scrape_state(state)
    s = stats
    _log_msg(f"\n=== Scan complete ===")
    _log_msg(f"  Checked: {s['checked']}")
    _log_msg(f"  FFA none: {s.get('ffa_none', 0)}, FFA GS++: {s.get('ffa_gspp', 0)}")
    _log_msg(f"  1v1 none: {s.get('v1_none', 0)}, 1v1 GS++: {s.get('v1_gspp', 0)}")
    _log_msg(f"  Skipped: {s['skipped']}, Errors: {s['errors']}")


async def _run_rescrape(page, delay: float, with_positions: bool,
                        sample_every: int):
    """Re-scrape existing games from NDJSON files, outputting v2 data with positions."""
    import asyncio

    # Collect game IDs from existing NDJSON files
    game_ids = []
    for ndjson_file in DATA_DIR.glob("ffa_*.ndjson"):
        if "_v2" in ndjson_file.name:
            continue  # skip existing v2 files
        with open(ndjson_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    game = json.loads(line)
                    gid = game.get("gameId")
                    if gid:
                        game_ids.append(gid)
                except json.JSONDecodeError:
                    continue

    # Deduplicate and sort
    game_ids = sorted(set(game_ids))

    # Skip games already in v2 files
    existing_v2 = set()
    for vf in DATA_DIR.glob("*_v2.ndjson"):
        with open(vf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                    gid = g.get("gameId")
                    if gid:
                        existing_v2.add(gid)
                except json.JSONDecodeError:
                    continue

    game_ids = [gid for gid in game_ids if gid not in existing_v2]
    total = len(game_ids)
    _log_msg(f"Rescrape mode: {total} games to re-process ({len(existing_v2)} already in v2)")

    if total == 0:
        _log_msg("All games already in v2 files. Nothing to do.")
        return

    collected = 0
    errors = 0

    for i, gid in enumerate(game_ids, 1):
        result = await _process_game(
            page, gid, elite_names=None, require_elite=False,
            with_positions=with_positions, sample_every=sample_every,
        )

        if result is None or "error" in result:
            errors += 1
            _log_msg(f"  [{i}/{total}] Game {gid}: ERROR {result}")
        elif "skipped" in result:
            errors += 1
            _log_msg(f"  [{i}/{total}] Game {gid}: SKIP {result['skipped']}")
        else:
            # Write to v2 file
            prefix = "ffa" if result["type"] == "FFA" else "1v1"
            v2_path = DATA_DIR / f"{prefix}_{result['mode']}_v2.ndjson"
            with open(v2_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            collected += 1

            nr = len(result.get("rounds", []))
            has_st = any("st" in r for r in result.get("rounds", []))
            nst = sum(len(r.get("st", {}).get("d", [])) for r in result.get("rounds", []))
            _log_msg(f"  [{i}/{total}] Game {gid}: {result['type']} {result['mode']} "
                     f"({nr} rounds, {nst} state frames, pos={'YES' if has_st else 'NO'})")

        await asyncio.sleep(delay + random.uniform(0, delay * 0.3))

    _log_msg(f"\n=== Rescrape complete ===")
    _log_msg(f"  Collected: {collected}/{total}, Errors: {errors}")


def cmd_scrape_replays(args):
    """Extract CurveCrash replay data via Playwright."""
    import asyncio
    from playwright.async_api import async_playwright

    async def _main():
        DATA_DIR.mkdir(exist_ok=True)

        state = _load_scrape_state()
        _log_msg(f"Starting scrape-replays --mode {args.mode}")
        _log_msg(f"  Resume state: {len(state['scraped_ids'])} games already processed")
        if args.with_positions or args.mode == "rescrape":
            _log_msg(f"  Position extraction: ON (sample_every={args.sample_every})")

        async with async_playwright() as p:
            launch_args = {}
            if args.proxy:
                launch_args["proxy"] = {"server": args.proxy}

            browser = await p.chromium.launch(
                headless=args.headless,
                **launch_args,
            )

            try:
                page = await _setup_page(browser)

                if args.mode == "rescrape":
                    await _run_rescrape(page, args.delay,
                                        with_positions=True,
                                        sample_every=args.sample_every)
                elif args.mode == "targeted":
                    await _run_targeted(page, state, args.limit, args.delay)
                else:
                    await _run_scan(page, state, args.start_id, args.limit, args.delay)

            finally:
                await browser.close()

        _log_msg("Done.")

    asyncio.run(_main())


# ============================================================
# Subcommand: analyze
# ============================================================

def _load_ndjson(path: Path) -> list[dict]:
    """Load games from an NDJSON file. Alias for load_replays with error reporting."""
    return load_replays(str(path))


def _analyze_file(path: Path):
    """Run full analysis on a single NDJSON file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {path.name}")
    print(f"{'='*60}")

    games = _load_ndjson(path)
    if not games:
        print("  No games found.")
        return

    print(f"\n--- Overview ---")
    print(f"  Games: {len(games)}")

    # Basic stats
    total_rounds = 0
    total_turns = 0
    total_deaths = 0
    total_holes = 0
    total_powerups = 0
    total_frames = 0

    turn_dirs = Counter()  # -1, 0, 1
    player_wins = Counter()
    player_games = Counter()
    player_kills = Counter()
    player_deaths_count = Counter()
    player_survival_frames = defaultdict(list)

    rounds_with_holes = 0
    holes_per_round = []
    turns_per_round = []
    deaths_per_round = []
    frames_per_round = []

    corrupt_games = 0
    games_with_errors = 0

    for game in games:
        game_id = game.get("gameId", "?")
        players = {p["id"]: p["name"] for p in game.get("players", [])}
        rounds = game.get("rounds", [])

        if not rounds:
            corrupt_games += 1
            continue

        has_error = False
        for rnd in rounds:
            total_rounds += 1
            nf = rnd.get("nf", 0)
            total_frames += nf
            frames_per_round.append(nf)

            tu = rnd.get("tu", [])
            de = rnd.get("de", [])
            ho = rnd.get("ho", [])
            pu = rnd.get("pu", [])

            total_turns += len(tu)
            total_deaths += len(de)
            total_holes += len(ho)
            total_powerups += len(pu)

            turns_per_round.append(len(tu))
            deaths_per_round.append(len(de))
            holes_per_round.append(len(ho))

            if ho:
                rounds_with_holes += 1

            # Turn direction distribution
            for t in tu:
                if len(t) >= 3:
                    turn_dirs[t[2]] += 1

            # Death tracking
            died_ids = set()
            for d in de:
                if len(d) >= 3:
                    who = d[1]
                    by = d[2]
                    died_ids.add(who)
                    name = players.get(who, f"p{who}")
                    player_deaths_count[name] += 1
                    if by is not None and by != who:
                        killer_name = players.get(by, f"p{by}")
                        player_kills[killer_name] += 1

            # Survival: players who didn't die survived the full round
            for pid, name in players.items():
                player_games[name] += 1
                if pid not in died_ids:
                    player_survival_frames[name].append(nf)

            if rnd.get("err"):
                has_error = True

        if has_error:
            games_with_errors += 1

    print(f"  Rounds: {total_rounds}")
    print(f"  Avg rounds/game: {total_rounds/len(games):.1f}")
    print(f"  Corrupt games (no rounds): {corrupt_games}")
    print(f"  Games with extraction errors: {games_with_errors}")

    # Turn stats
    print(f"\n--- Turn Actions ---")
    print(f"  Total turns: {total_turns}")
    if total_rounds:
        print(f"  Avg turns/round: {total_turns/total_rounds:.1f}")
    left = turn_dirs.get(-1, 0)
    straight = turn_dirs.get(0, 0)
    right = turn_dirs.get(1, 0)
    total_dir = left + straight + right
    if total_dir:
        print(f"  Left: {left} ({100*left/total_dir:.1f}%)")
        print(f"  Straight: {straight} ({100*straight/total_dir:.1f}%)")
        print(f"  Right: {right} ({100*right/total_dir:.1f}%)")

    # Death stats
    print(f"\n--- Deaths ---")
    print(f"  Total deaths: {total_deaths}")
    if total_rounds:
        print(f"  Avg deaths/round: {total_deaths/total_rounds:.1f}")

    # Gap/hole stats
    print(f"\n--- Gap Passes (Hole Events) ---")
    print(f"  Total hole events: {total_holes}")
    if total_rounds:
        print(f"  Rounds with gaps: {rounds_with_holes}/{total_rounds} "
              f"({100*rounds_with_holes/total_rounds:.1f}%)")
        print(f"  Avg gaps/round: {total_holes/total_rounds:.2f}")
    if holes_per_round:
        nonzero = [h for h in holes_per_round if h > 0]
        if nonzero:
            print(f"  Avg gaps/round (rounds with gaps): {sum(nonzero)/len(nonzero):.2f}")
            print(f"  Max gaps in a round: {max(nonzero)}")

    # Powerup stats
    if total_powerups:
        print(f"\n--- Powerups ---")
        print(f"  Total powerup spawns: {total_powerups}")
        if total_rounds:
            print(f"  Avg powerups/round: {total_powerups/total_rounds:.2f}")

    # Round duration
    print(f"\n--- Round Duration ---")
    if frames_per_round:
        avg_frames = sum(frames_per_round) / len(frames_per_round)
        print(f"  Avg frames/round: {avg_frames:.0f}")
        game_fps = games[0].get("settings", {}).get("fps", 60)
        print(f"  Avg duration: {avg_frames/game_fps:.1f}s (at {game_fps} fps)")
        print(f"  Max round: {max(frames_per_round)} frames "
              f"({max(frames_per_round)/game_fps:.1f}s)")

    # Top players
    print(f"\n--- Top Players (by round appearances) ---")
    top = player_games.most_common(20)
    for name, count in top:
        kills = player_kills.get(name, 0)
        deaths = player_deaths_count.get(name, 0)
        kd = kills / deaths if deaths else float("inf")
        print(f"  {name:20s}  rounds={count:4d}  kills={kills:4d}  "
              f"deaths={deaths:4d}  K/D={kd:.2f}")

    # Data quality
    print(f"\n--- Data Quality ---")
    empty_turns = sum(1 for t in turns_per_round if t == 0)
    print(f"  Rounds with 0 turns: {empty_turns}/{total_rounds} "
          f"({100*empty_turns/total_rounds:.1f}%)" if total_rounds else "  No rounds")

    zero_frame_rounds = sum(1 for f in frames_per_round if f == 0)
    print(f"  Rounds with 0 frames: {zero_frame_rounds}/{total_rounds}")

    # Quick game ID range
    game_ids = [g.get("gameId", 0) for g in games]
    if game_ids:
        print(f"\n--- Game ID Range ---")
        print(f"  Min: {min(game_ids)}, Max: {max(game_ids)}")
        print(f"  Span: {max(game_ids) - min(game_ids)} IDs")


def cmd_analyze(args):
    """Analyze scraped CurveCrash replay NDJSON data."""
    # Fix Windows console encoding for Unicode player names
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    paths = []
    if args.all:
        paths = sorted(DATA_DIR.glob("*.ndjson"))
        if not paths:
            print(f"No NDJSON files found in {DATA_DIR}/")
            return
    elif args.files:
        paths = [Path(f) for f in args.files]
    else:
        print("Specify NDJSON files or use --all")
        return

    for path in paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        _analyze_file(path)

    if len(paths) > 1:
        # Cross-file summary
        print(f"\n{'='*60}")
        print(f"Cross-file Summary")
        print(f"{'='*60}")
        total_games = 0
        for path in paths:
            if path.exists():
                count = sum(1 for line in open(path, encoding="utf-8") if line.strip())
                print(f"  {path.name}: {count} games")
                total_games += count
        print(f"  TOTAL: {total_games} games")


# ============================================================
# Subcommand: validate
# ============================================================

def _validate_game_round(settings, round_data, immunity_frames=8, verbose=False):
    """Validate one round, return validation result."""
    renderer = ReplayRenderer(settings, round_data, gspp=True)
    result = renderer.validate_round(immunity_frames=immunity_frames)
    if verbose and result["violations"]:
        for f, pid, vtype in result["violations"][:30]:
            print(f"    frame={f:>5d}  player={pid}  {vtype}")
        if len(result["violations"]) > 30:
            print(f"    ... {len(result['violations']) - 30} more violations")
    return result


def cmd_validate(args):
    """Validate replay rounds against env physics rules."""
    ndjson_path = _resolve_ndjson(args.ndjson)
    if not ndjson_path:
        print("No NDJSON files found")
        return
    print(f"Loading: {ndjson_path}")
    games = load_replays(ndjson_path)

    # Flatten rounds with back-references
    all_rounds = []
    for gi, game in enumerate(games):
        settings = game["settings"]
        for ri, rd in enumerate(game["rounds"]):
            all_rounds.append((gi, ri, settings, rd))
    print(f"  {len(games)} games, {len(all_rounds)} total rounds")

    if not args.all:
        # ---- Single round validation ----
        idx = args.round % len(all_rounds)
        gi, ri, settings, rd = all_rounds[idx]
        n_players = len(rd["sp"])
        n_frames = rd["nf"]
        print(f"\nValidating round #{idx} (game[{gi}] round[{ri}])")
        print(f"  {n_players} players, {n_frames} frames "
              f"({n_frames / settings['fps']:.1f}s)")
        print(f"  Deaths: {len(rd['de'])}, Holes: {len(rd['ho'])}, "
              f"Powerups: {len(rd.get('pu', []))}")

        has_pos = rd.get("st") and "d" in rd.get("st", {})
        print(f"  Position data: {'v' + str(rd['st'].get('v', 2)) if has_pos else 'NONE (sim mode)'}")

        result = _validate_game_round(settings, rd,
                                      immunity_frames=args.immunity,
                                      verbose=True)
        status = "VALID" if result["n_violations"] <= args.threshold else "INVALID"
        print(f"\n  Result: {status}")
        print(f"  Violations: {result['n_violations']} / "
              f"{result['total_frames']} frames")
        if result["violations"]:
            types = {}
            players_hit = set()
            for f, pid, vtype in result["violations"]:
                types[vtype] = types.get(vtype, 0) + 1
                players_hit.add(pid)
            for vtype, count in sorted(types.items()):
                print(f"    {vtype}: {count}")
            print(f"  Players with violations: {sorted(players_hit)}")

            # Show frame distribution
            frames = [f for f, _, _ in result["violations"]]
            print(f"  First violation: frame {min(frames)}, "
                  f"Last: frame {max(frames)}")

    else:
        # ---- Batch validation ----
        print(f"\nValidating all {len(all_rounds)} rounds "
              f"(immunity={args.immunity}, threshold={args.threshold})...")
        t0 = time.time()

        valid_count = 0
        invalid_count = 0
        total_violations = 0
        violation_types = {}
        round_results = []  # (gi, ri, n_violations, valid)
        violation_dist = []  # n_violations per round

        for i, (gi, ri, settings, rd) in enumerate(all_rounds):
            result = _validate_game_round(settings, rd,
                                          immunity_frames=args.immunity)
            is_valid = result["n_violations"] <= args.threshold
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            total_violations += result["n_violations"]
            violation_dist.append(result["n_violations"])
            round_results.append((gi, ri, result["n_violations"], is_valid))

            for _, _, vtype in result["violations"]:
                violation_types[vtype] = violation_types.get(vtype, 0) + 1

            if (i + 1) % 50 == 0 or i == len(all_rounds) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1:>4d}/{len(all_rounds)}] "
                      f"valid={valid_count} invalid={invalid_count} "
                      f"({rate:.1f} rounds/s)")

        elapsed = time.time() - t0
        n = len(all_rounds)
        print(f"\n{'='*50}")
        print(f"  Validation Results")
        print(f"{'='*50}")
        print(f"  Total rounds:  {n}")
        print(f"  Valid:          {valid_count} ({100*valid_count/n:.1f}%)")
        print(f"  Invalid:        {invalid_count} ({100*invalid_count/n:.1f}%)")
        print(f"  Total violations: {total_violations}")

        if violation_types:
            print(f"  Violation types:")
            for vtype, count in sorted(violation_types.items()):
                print(f"    {vtype}: {count}")

        # Violation distribution
        vd = sorted(violation_dist)
        print(f"\n  Violation distribution:")
        print(f"    0 violations:   {vd.count(0)} rounds")
        for bucket in [1, 5, 10, 50, 100]:
            cnt = sum(1 for v in vd if 0 < v <= bucket)
            print(f"    1-{bucket:>3d}:          {cnt} rounds")
        high = sum(1 for v in vd if v > 100)
        if high:
            print(f"    >100:           {high} rounds")

        print(f"\n  Time: {elapsed:.1f}s ({elapsed/n:.2f}s/round)")

        # ---- Filter mode: write cleaned NDJSON ----
        if args.filter:
            # Group valid rounds back by game index
            valid_rounds_per_game = {}
            for gi, ri, n_v, is_valid in round_results:
                if is_valid:
                    if gi not in valid_rounds_per_game:
                        valid_rounds_per_game[gi] = set()
                    valid_rounds_per_game[gi].add(ri)

            out_path = Path(ndjson_path).with_name(
                Path(ndjson_path).stem + "_valid.ndjson"
            )
            kept_games = 0
            kept_rounds = 0
            with open(out_path, "w", encoding="utf-8") as out_f:
                for gi, game in enumerate(games):
                    if gi not in valid_rounds_per_game:
                        continue
                    valid_ri = valid_rounds_per_game[gi]
                    filtered_rounds = [
                        rd for ri, rd in enumerate(game["rounds"])
                        if ri in valid_ri
                    ]
                    if not filtered_rounds:
                        continue
                    game_copy = dict(game)
                    game_copy["rounds"] = filtered_rounds
                    out_f.write(json.dumps(game_copy) + "\n")
                    kept_games += 1
                    kept_rounds += len(filtered_rounds)

            print(f"\n  Filtered output: {out_path}")
            print(f"  Kept: {kept_games} games, {kept_rounds} rounds")


# ============================================================
# Subcommand: render (pre-render observations to .npz)
# ============================================================

def cmd_render(args):
    """Pre-render replay observations to .npy files."""
    output_path = args.output
    if output_path is None:
        output_path = str(DATA_DIR / "bc_data")

    file_list = None
    if args.files:
        file_list = args.files

    print(f"Pre-rendering replays...")
    print(f"  data_dir:     {DATA_DIR}")
    print(f"  output:       {output_path}")
    print(f"  sample_every: {args.sample_every}")
    print(f"  gspp:         {args.gspp}")
    print(f"  top_n:        {args.top_n}")
    print(f"  max_rounds:   {args.max_rounds}")
    print(f"  chunk_rounds: {args.chunk_rounds}")

    total = prerender_all(
        data_dir=DATA_DIR,
        output_path=output_path,
        sample_every=args.sample_every,
        max_rounds=args.max_rounds,
        gspp=args.gspp,
        top_n=args.top_n,
        file_list=file_list,
        chunk_rounds=args.chunk_rounds,
    )
    print(f"\nDone. Total samples: {total:,}")


# ============================================================
# Subcommand: watch (merged watch_replay + watch_validated)
# ============================================================

_PLAYER_COLORS = [
    (255, 255, 255), (255, 80, 80), (80, 180, 255), (80, 255, 80),
    (255, 200, 50), (200, 80, 255), (255, 140, 50), (50, 255, 200),
    (255, 100, 180),
]
_SPEED_COLOR = (255, 165, 0)
_ERASE_COLOR = (0, 200, 255)


def _resolve_ndjson(path_arg):
    """Resolve NDJSON path from arg or auto-discover in DATA_DIR."""
    if path_arg:
        return path_arg
    for pattern in ["ffa_gspp_v2*.ndjson", "ffa_gspp*.ndjson", "*.ndjson"]:
        candidates = list(DATA_DIR.glob(pattern))
        if candidates:
            return str(candidates[0])
    return None


def _find_round_by_validity(games, mode="valid", immunity=8):
    """Find a round matching the requested validity."""
    for gi, game in enumerate(games):
        settings = game["settings"]
        for ri, rd in enumerate(game["rounds"]):
            if rd["nf"] < settings["fps"] * 5:
                continue
            renderer = ReplayRenderer(settings, rd, gspp=True)
            result = renderer.validate_round(immunity_frames=immunity)
            is_valid = result["n_violations"] == 0
            if mode == "valid" and is_valid:
                return gi, ri, settings, rd, result["violations"]
            elif mode == "invalid" and not is_valid:
                return gi, ri, settings, rd, result["violations"]
    return None


def cmd_watch(args):
    """Pygame replay viewer with optional validation overlay."""
    import math
    import pygame

    DISPLAY_SCALE = 2
    DISPLAY_SIZE = SIM_SIZE * DISPLAY_SCALE
    do_validate = args.validate

    ndjson_path = _resolve_ndjson(args.ndjson)
    if not ndjson_path:
        print(f"No NDJSON files found in {DATA_DIR}")
        return
    print(f"Loading: {ndjson_path}")
    games = load_replays(ndjson_path)

    # Flatten rounds
    all_rounds = []
    for gi, game in enumerate(games):
        for ri, rd in enumerate(game["rounds"]):
            all_rounds.append((gi, ri, game["settings"], rd))
    print(f"  {len(games)} games, {len(all_rounds)} rounds")

    # Find the round to display
    violations = []

    if args.valid or args.invalid:
        mode = "valid" if args.valid else "invalid"
        print(f"Searching for a {mode} round...")
        result = _find_round_by_validity(games, mode=mode, immunity=args.immunity)
        if result is None:
            print(f"No {mode} round found!")
            return
        gi, ri, settings, round_data, violations = result
        idx = sum(len(g["rounds"]) for g in games[:gi]) + ri
    elif args.best:
        idx = max(range(len(all_rounds)), key=lambda i: all_rounds[i][3]["nf"])
        gi, ri, settings, round_data = all_rounds[idx]
    elif getattr(args, 'round', None) is not None:
        idx = args.round % len(all_rounds)
        gi, ri, settings, round_data = all_rounds[idx]
    else:
        import random
        idx = random.randint(0, len(all_rounds) - 1)
        gi, ri, settings, round_data = all_rounds[idx]

    # Run validation if needed and not already done by --valid/--invalid path
    if do_validate and not violations:
        violations = ReplayRenderer(settings, round_data, gspp=True).validate_round(
            immunity_frames=args.immunity)["violations"]

    violations_set = {(f, pid) for f, pid, vt in violations}
    n_viol = len(violations)
    n_players = len(round_data["sp"])
    n_frames = round_data["nf"]
    status = "VALID" if n_viol == 0 else f"INVALID ({n_viol} violations)"

    print(f"  Round #{idx}: {n_players}p, {n_frames}f ({n_frames/settings['fps']:.1f}s)")
    if do_validate:
        print(f"  Status: {status}")

    # Create renderer for playback
    renderer = ReplayRenderer(settings, round_data, gspp=True)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
    title = f"Round #{idx} - {n_players}p {n_frames}f"
    if do_validate:
        title += f" - {status}"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    target_fps = int(settings["fps"] * args.speed)

    pid_list = sorted([sp[0] for sp in round_data["sp"]])
    pid_color = {pid: _PLAYER_COLORS[i % len(_PLAYER_COLORS)] for i, pid in enumerate(pid_list)}

    running = True
    paused = False
    frame = 0
    violation_markers = []

    while running and frame < n_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    renderer = ReplayRenderer(settings, round_data, gspp=True)
                    frame = 0
                    violation_markers = []

        if paused:
            clock.tick(30)
            continue

        results = renderer.step_frame()
        current_frame = frame
        frame += 1

        # Record violation marker positions
        if do_validate:
            for p in renderer.players.values():
                if (current_frame, p.id) in violations_set:
                    violation_markers.append((p.x, p.y, current_frame))

        # --- Draw ---
        screen.fill((20, 20, 30))

        # Field boundary
        ox, oy = renderer.offset_x, renderer.offset_y
        sw, sh = renderer.sim_w, renderer.sim_h
        border_rect = pygame.Rect(
            ox * DISPLAY_SCALE, oy * DISPLAY_SCALE,
            sw * DISPLAY_SCALE, sh * DISPLAY_SCALE
        )
        pygame.draw.rect(screen, (60, 60, 80), border_rect, 2)

        # Trails
        trail = renderer.trail_owner
        for pid in pid_list:
            color = pid_color.get(pid, (200, 200, 200))
            mask = (trail == pid + 1)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            for i in range(len(xs)):
                sx = int(xs[i]) * DISPLAY_SCALE
                sy = int(ys[i]) * DISPLAY_SCALE
                screen.set_at((sx, sy), color)
                if DISPLAY_SCALE >= 2:
                    screen.set_at((sx + 1, sy), color)
                    screen.set_at((sx, sy + 1), color)
                    screen.set_at((sx + 1, sy + 1), color)

        # Powerups
        for px, py, ptype in renderer.field_powerups:
            color = _SPEED_COLOR if ptype == 'speed' else _ERASE_COLOR
            pygame.draw.circle(screen, color,
                (int(px * DISPLAY_SCALE), int(py * DISPLAY_SCALE)),
                renderer._powerup_radius * DISPLAY_SCALE, 2)

        # Player heads
        for p in renderer.players.values():
            if not p.alive:
                continue
            color = pid_color.get(p.id, (200, 200, 200))
            cx = int(p.x * DISPLAY_SCALE)
            cy = int(p.y * DISPLAY_SCALE)
            pygame.draw.circle(screen, color, (cx, cy), 4)
            dx = int(math.cos(p.angle) * 10)
            dy = int(math.sin(p.angle) * 10)
            pygame.draw.line(screen, color, (cx, cy), (cx + dx, cy + dy), 2)
            if p.speed_boosts:
                pygame.draw.circle(screen, _SPEED_COLOR, (cx, cy), 10, 2)
            if not p.drawing and current_frame >= renderer._spawn_grace_frames:
                pygame.draw.circle(screen, (100, 100, 100), (cx, cy), 8, 1)

        # Violation markers (red X)
        if do_validate:
            for vx, vy, vf in violation_markers:
                sx = int(vx * DISPLAY_SCALE)
                sy = int(vy * DISPLAY_SCALE)
                age = current_frame - vf
                alpha = max(80, 255 - age * 2)
                size = 8
                pygame.draw.line(screen, (alpha, 0, 0),
                    (sx - size, sy - size), (sx + size, sy + size), 3)
                pygame.draw.line(screen, (alpha, 0, 0),
                    (sx + size, sy - size), (sx - size, sy + size), 3)
                pygame.draw.circle(screen, (alpha, 0, 0), (sx, sy), size + 2, 2)

        # HUD
        alive = sum(1 for p in renderer.players.values() if p.alive)
        time_s = current_frame / settings["fps"]
        hud_parts = [f"F:{current_frame}/{n_frames}", f"T:{time_s:.1f}s",
                     f"Alive:{alive}/{n_players}"]
        if do_validate:
            viol_so_far = sum(1 for _, _, vf in violation_markers if vf <= current_frame)
            hud_parts.append(f"Violations:{viol_so_far}/{n_viol}")
            hud_parts.append("VALID" if n_viol == 0 else "INVALID")
        hud = font.render("  ".join(hud_parts), True, (200, 200, 200))
        screen.blit(hud, (10, 10))
        controls = font.render("[SPACE]=pause [R]=restart [ESC]=quit", True, (120, 120, 120))
        screen.blit(controls, (10, 30))

        pygame.display.flip()
        clock.tick(target_fps)

    # Final screen
    if running:
        end_text = font.render(
            f"ROUND OVER - press any key", True, (255, 255, 100)
        )
        screen.blit(end_text, (DISPLAY_SIZE // 2 - 150, DISPLAY_SIZE // 2))
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            clock.tick(30)

    pygame.quit()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="replay_pipeline",
        description="CurveCrash replay data pipeline - scrape, validate, analyze, render, watch",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- scrape-leaderboard ---
    p_lb = subparsers.add_parser("scrape-leaderboard",
                                 help="Scrape CurveCrash leaderboards + player profiles")
    p_lb.add_argument("--min-elo", type=int, default=1300,
                      help="Minimum ELO for elite players (default: 1300)")
    p_lb.add_argument("--max-pages", type=int, default=2,
                      help="Max leaderboard pages to scrape per board (default: 2, 20 players/page)")
    p_lb.add_argument("--delay", type=float, default=0.2,
                      help="Delay between requests in seconds (default: 0.2)")
    p_lb.add_argument("--proxy", type=str, default=None,
                      help="HTTP/SOCKS proxy URL (e.g. socks5://127.0.0.1:1080)")
    p_lb.add_argument("--skip-profiles", action="store_true",
                      help="Only scrape leaderboards, skip profile pages")
    p_lb.add_argument("--extra-players", nargs="*", default=[],
                      help="Extra player usernames to include (URL-encoded)")

    # --- scrape-replays ---
    p_sr = subparsers.add_parser("scrape-replays",
                                 help="Extract CurveCrash replay data via Playwright")
    p_sr.add_argument("--mode", choices=["targeted", "scan", "rescrape"], default="targeted",
                      help="Scraping mode (default: targeted)")
    p_sr.add_argument("--start-id", type=int, default=714600,
                      help="Starting game ID for scan mode (default: 714600)")
    p_sr.add_argument("--limit", type=int, default=None,
                      help="Max games to process")
    p_sr.add_argument("--delay", type=float, default=0.15,
                      help="Base delay between fetches in seconds (default: 0.15)")
    p_sr.add_argument("--proxy", type=str, default=None,
                      help="HTTP/SOCKS proxy URL (e.g. socks5://127.0.0.1:1080)")
    p_sr.add_argument("--headless", action="store_true", default=False,
                      help="Run browser in headless mode")
    p_sr.add_argument("--with-positions", action="store_true", default=False,
                      help="Extract per-frame positions from game engine (v2)")
    p_sr.add_argument("--sample-every", type=int, default=3,
                      help="Sample state every N frames (default: 3 = 20Hz at 60fps)")

    # --- analyze ---
    p_an = subparsers.add_parser("analyze",
                                 help="Analyze scraped CurveCrash replay NDJSON data")
    p_an.add_argument("files", nargs="*", help="NDJSON files to analyze")
    p_an.add_argument("--all", action="store_true",
                      help="Analyze all NDJSON files in data/")

    # --- validate ---
    p_va = subparsers.add_parser("validate",
                                 help="Validate replay rounds against env physics rules")
    p_va.add_argument("--round", type=int, default=0,
                      help="Round index to validate (single mode)")
    p_va.add_argument("--all", action="store_true", help="Validate all rounds")
    p_va.add_argument("--immunity", type=int, default=8,
                      help="Self-immunity frames (default 8)")
    p_va.add_argument("--ndjson", type=str, default=None)
    p_va.add_argument("--threshold", type=int, default=0,
                      help="Max violations to still consider valid")
    p_va.add_argument("--filter", action="store_true",
                      help="Write filtered NDJSON with only valid rounds")

    # --- render ---
    p_re = subparsers.add_parser("render",
                                 help="Pre-render replay observations to .npy files")
    p_re.add_argument("--output", type=str, default=None,
                      help="Output path prefix (default: data/bc_data)")
    p_re.add_argument("--sample-every", type=int, default=6,
                      help="Sample every N frames (default: 6)")
    p_re.add_argument("--max-rounds", type=int, default=None,
                      help="Max rounds to render")
    p_re.add_argument("--gspp", action="store_true", default=False,
                      help="Render 6ch (GS++ mode with powerup channels)")
    p_re.add_argument("--top-n", type=int, default=None,
                      help="Only render top-N surviving players per round")
    p_re.add_argument("--chunk-rounds", type=int, default=10,
                      help="Flush to disk every N rounds (default: 10)")
    p_re.add_argument("--files", nargs="*", default=None,
                      help="Specific NDJSON filenames to process")

    # --- watch ---
    p_wa = subparsers.add_parser("watch",
                                 help="Pygame replay viewer with optional validation overlay")
    p_wa.add_argument("--round", type=int, default=None, help="Specific round index")
    p_wa.add_argument("--best", action="store_true", help="Pick longest round")
    p_wa.add_argument("--valid", action="store_true", help="Pick first valid round")
    p_wa.add_argument("--invalid", action="store_true", help="Pick first invalid round")
    p_wa.add_argument("--validate", action="store_true",
                      help="Enable validation overlay (red X on violations)")
    p_wa.add_argument("--ndjson", type=str, default=None)
    p_wa.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p_wa.add_argument("--immunity", type=int, default=8, help="Self-immunity frames")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "scrape-leaderboard": cmd_scrape_leaderboard,
        "scrape-replays": cmd_scrape_replays,
        "analyze": cmd_analyze,
        "validate": cmd_validate,
        "render": cmd_render,
        "watch": cmd_watch,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
