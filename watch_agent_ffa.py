"""
Watch trained CurveCrash FFA agent with Pygame rendering.

Auto-detects v4 (CNN-only), v5 (CNN+GRU(64)), or v5.1 (CNN+CBAM+SpatialAttn+GRU(128)).

Usage:
    python watch_agent_ffa.py                                    # AI vs AI (trained)
    python watch_agent_ffa.py --human                            # YOU vs AI opponents
    python watch_agent_ffa.py --human --min-players 2            # 1v1 boss fight
    python watch_agent_ffa.py --checkpoint checkpoints_v5_1_main/agent_final.pt
    python watch_agent_ffa.py --random-opponents                 # trained ego, random opps

Controls: LEFT/RIGHT arrows=steer, R=restart, ESC=quit, SPACE=pause
"""
import argparse
import math
import os

import pygame
import numpy as np
import torch

from curvecrash_env_ffa import (
    CurveCrashFFAEnv, ARENA_SIM, OBS_SIZE, TRAIL_WIDTH,
    ARENA_H_GSPP, ARENA_OFFSET_Y_GSPP,
)
from train_selfplay import (
    Agent, V5Agent, LegacyAgent, GRU_HIDDEN,
    detect_checkpoint_type, get_gru_hidden_size, get_input_channels,
)
from experiments import ImpalaCNNAgent, VoronoiWrapper

DISPLAY_SCALE = 2
DISPLAY_SIZE = ARENA_SIM * DISPLAY_SCALE

# Colors per player slot (ego=white, opponents cycle through these)
PLAYER_COLORS = [
    (255, 255, 255),  # ego: white
    (255, 80, 80),    # red
    (80, 180, 255),   # blue
    (80, 255, 80),    # green
    (255, 200, 50),   # yellow
    (200, 80, 255),   # purple
    (255, 140, 50),   # orange
    (50, 255, 200),   # cyan
    (255, 100, 180),  # pink
    (180, 180, 100),  # olive
    (150, 150, 255),  # lavender
]

ACTION_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def load_powerup_icons(radius_display):
    """Load speed and erase powerup icons from SVG-converted PNGs."""
    size = radius_display * 2
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def load_icon(filename):
        path = os.path.join(script_dir, filename)
        if os.path.exists(path):
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.smoothscale(img, (size, size))
        # Fallback: colored circle
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, (128, 128, 128), (radius_display, radius_display),
                           radius_display)
        return surf

    speed_surf = load_icon("power-up_speeds.png")
    erase_surf = load_icon("power-up_erase.png")
    return speed_surf, erase_surf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint (auto-detects latest if not set)")
    p.add_argument("--human", action="store_true",
                   help="YOU control the ego with arrow keys, AI controls opponents")
    p.add_argument("--random", action="store_true",
                   help="Random ego agent (no model)")
    p.add_argument("--random-opponents", action="store_true",
                   help="Opponents play random instead of using the model")
    p.add_argument("--min-players", type=int, default=2)
    p.add_argument("--max-players", type=int, default=6)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference (use when GPU is busy with training)")
    p.add_argument("--gspp", action="store_true",
                   help="Enable GS++ mode (speed + erase powerups)")
    p.add_argument("--shield-depth", type=int, default=5,
                   help="Safety shield tree-search depth (0=disable, default=5)")
    p.add_argument("--shield-threshold", type=float, default=0.3,
                   help="Override when agent's score < best * threshold (default=0.3)")
    args = p.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect latest checkpoint if not specified
    if args.checkpoint is None:
        import glob, os
        for ckpt_dir in ["checkpoints_v5_1_main", "checkpoints_v5_1_exploiter",
                         "checkpoints_v5_main", "checkpoints_v5_exploiter",
                         "checkpoints_v5", "checkpoints_v4", "checkpoints_v3"]:
            pts = sorted(glob.glob(os.path.join(ckpt_dir, "agent_*.pt")))
            if pts:
                args.checkpoint = pts[-1]
                break
        if args.checkpoint is None:
            args.checkpoint = "checkpoints_v5_1_main/agent_final.pt"

    agent = None
    opp_agent = None
    ckpt_type = None
    gru_hidden = 0

    # Load model
    model_weights = None
    is_impala_ckpt = False
    n_input_ch = 6
    try:
        model_weights = torch.load(args.checkpoint, map_location=device, weights_only=True)
        is_impala_ckpt = any(k.startswith("conv.0.conv.") for k in model_weights)
        if is_impala_ckpt:
            n_input_ch = model_weights["conv.0.conv.weight"].shape[1]
            gru_hidden = model_weights["gru.weight_hh_l0"].shape[1]  # [3*h, h]
            ckpt_type = "impala"
            use_gru = True
            version = f"IMPALA-CNN ({n_input_ch}ch, GRU{gru_hidden})"
        else:
            ckpt_type = detect_checkpoint_type(model_weights)
            gru_hidden = get_gru_hidden_size(model_weights)
            n_input_ch = get_input_channels(model_weights)
            use_gru = ckpt_type in ("v5", "v5_1", "v6")
            version = {"legacy": "v4 (CNN)", "v5": "v5 (GRU64)",
                       "v5_1": "v5.1 (CBAM+GRU128)", "v6": "v6 (GS++ 6ch)"}[ckpt_type]
        print(f"Loaded checkpoint: {args.checkpoint} [{version}]")
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")
        use_gru = False

    def _make_agent(weights):
        # Detect IMPALA-CNN from weight keys
        is_impala = any(k.startswith("conv.0.conv.") for k in weights)
        if is_impala:
            first_conv = weights["conv.0.conv.weight"]
            n_in = first_conv.shape[1]
            gh = weights["gru.weight_hh_l0"].shape[1]  # [3*h, h]
            # Detect channel widths from conv stage weights
            channels = []
            for i in range(10):
                key = f"conv.{i}.conv.weight"
                if key in weights:
                    channels.append(weights[key].shape[0])
            # v10: detect scalar branch
            n_scalar = weights["scalar_fc.0.weight"].shape[1] if "scalar_fc.0.weight" in weights else 0
            a = ImpalaCNNAgent(
                n_input_channels=n_in, gru_hidden=gh,
                channels=tuple(channels),
                use_cbam=any(k.startswith("cbam.") for k in weights),
                n_scalar_inputs=n_scalar,
            ).to(device)
            a.load_state_dict(weights)
            a.eval()
            return a, "impala", gh

        ct = detect_checkpoint_type(weights)
        gh = get_gru_hidden_size(weights)
        n_in = get_input_channels(weights)
        if ct in ("v5_1", "v6"):
            has_sa = any(k.startswith("spatial_attn.") for k in weights)
            a = Agent(use_spatial_attn=has_sa, n_input_channels=n_in).to(device)
        elif ct == "v5":
            a = V5Agent().to(device)
        else:
            a = LegacyAgent().to(device)
        a.load_state_dict(weights)
        a.eval()
        return a, ct, gh

    # v10: detect scalar branch
    use_scalars = model_weights is not None and "scalar_fc.0.weight" in model_weights

    if args.human:
        if model_weights is not None and not args.random_opponents:
            opp_agent, opp_type, opp_gru_h = _make_agent(model_weights)
            print("Mode: HUMAN vs AI opponents")
        else:
            opp_type, opp_gru_h = "legacy", 0
            print("Mode: HUMAN vs RANDOM opponents")
    elif not args.random and model_weights is not None:
        agent, _, _ = _make_agent(model_weights)
        print("Ego: MODEL")
        if not args.random_opponents:
            opp_agent, opp_type, opp_gru_h = _make_agent(model_weights)
            print("Opponents: MODEL")
        else:
            opp_type, opp_gru_h = "legacy", 0
    else:
        opp_type, opp_gru_h = "legacy", 0
        print("Mode: ALL RANDOM")

    ego_mode = "HUMAN" if args.human else ("MODEL" if agent else "RANDOM")
    opp_mode = "MODEL" if opp_agent else "RANDOM"

    # Auto-detect features from checkpoint channel count
    # 7ch = GS++(6) + voronoi(1), 9ch = GS++(6) + voronoi(1) + minimap(2)
    use_voronoi = n_input_ch >= 7
    use_minimap = n_input_ch >= 9
    env_cls = VoronoiWrapper if use_voronoi else CurveCrashFFAEnv
    env = env_cls(
        min_players=args.min_players, max_players=args.max_players,
        gspp=args.gspp,
        bilinear_ds=use_minimap,  # v11 models use bilinear too
        minimap=use_minimap,
    )

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE + 300, DISPLAY_SIZE))
    pygame.display.set_caption("CurveCrash FFA Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    # Load powerup icon surfaces from PNGs (at display resolution)
    # Powerup = 32px diameter (radius 16) in sim at 512 grid, matching 16px at 256 real game
    pup_radius_display = 16 * DISPLAY_SCALE  # 32px sim → display coords
    speed_icon, erase_icon = load_powerup_icons(pup_radius_display)

    obs, info = env.reset(seed=42)
    episode_num = 0
    paused = False
    last_action = 1
    val = 0.0

    # GRU state tracking
    ego_gru_state = None
    if agent and use_gru:
        ego_gru_state = torch.zeros(1, 1, gru_hidden, device=device)
    opp_gru_states = {}  # opp_slot_index -> (1, 1, gru_h) tensor

    def reset_gru_states():
        nonlocal ego_gru_state, opp_gru_states
        if ego_gru_state is not None:
            ego_gru_state = torch.zeros(1, 1, gru_hidden, device=device)
        opp_gru_states.clear()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    reset_gru_states()
                    episode_num += 1
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            screen.blit(pause_text, (DISPLAY_SIZE // 2 - 30, DISPLAY_SIZE // 2))
            pygame.display.flip()
            clock.tick(30)
            continue

        # Ego action
        if args.human:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                last_action = 0
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                last_action = 2
            else:
                last_action = 1
            val = 0.0
        elif agent is not None:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                # v10: compute scalar obs if model uses them
                scalars_t = None
                if use_scalars:
                    s = env.get_scalar_obs(env.ego)
                    scalars_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                if use_gru:
                    done_t = torch.zeros(1, device=device)
                    action, _, _, value, ego_gru_state = agent.get_action_and_value(
                        obs_t, ego_gru_state, done_t, scalars=scalars_t
                    )
                else:
                    action, _, _, value = agent.get_action_and_value(obs_t)
                last_action = action.item()
                val = value.item()
        else:
            last_action = env.action_space.sample()
            val = 0.0

        # Safety shield: macro beam search override
        shield_active = False
        survival_scores = [1.0, 1.0, 1.0]
        if args.shield_depth > 0 and env.ego.alive and not env.ego.spawning:
            scores = env.search_action_scores(env.ego)
            survival_scores = scores  # for display

            best_score = max(scores)
            cur_score = scores[last_action]
            best_action = scores.index(best_score)

            # Override if agent's action scores much lower than best
            # or if agent's action leads to death (score < -500)
            if cur_score < -500 and best_score > -500:
                # Hard: agent's action dies, better option exists
                shield_active = True
                last_action = best_action
            elif (best_score > 0
                    and cur_score < best_score * args.shield_threshold):
                # Soft: agent's action has much less space
                shield_active = True
                last_action = best_action

        # Opponent actions
        opp_actions = None
        if opp_agent is not None:
            opp_obs_list, live_mask = env.get_opponent_observations()
            if opp_obs_list:
                with torch.no_grad():
                    opp_obs_t = torch.tensor(
                        np.stack(opp_obs_list), dtype=torch.float32, device=device
                    )
                    opp_use_gru = opp_type in ("v5", "v5_1", "v6", "impala")
                    if opp_use_gru:
                        # Collect GRU states for live opponents
                        opp_gru_list = []
                        opp_keys = []
                        for j, alive in enumerate(live_mask):
                            if alive:
                                if j not in opp_gru_states:
                                    opp_gru_states[j] = torch.zeros(
                                        1, 1, opp_gru_h, device=device
                                    )
                                opp_gru_list.append(opp_gru_states[j])
                                opp_keys.append(j)
                        opp_gru_batch = torch.cat(opp_gru_list, dim=1)
                        done_zeros = torch.zeros(len(opp_obs_list), device=device)
                        opp_acts, new_opp_gru = opp_agent.get_action_greedy(
                            opp_obs_t, opp_gru_batch, done_zeros
                        )
                        opp_acts = opp_acts.cpu().numpy()
                        for k_idx, key in enumerate(opp_keys):
                            opp_gru_states[key] = new_opp_gru[:, k_idx:k_idx+1, :].clone()
                    else:
                        opp_acts = opp_agent.get_action_greedy(opp_obs_t).cpu().numpy()

                n_opp = len(env.players) - 1
                opp_actions = np.ones(n_opp, dtype=np.int64)
                act_idx = 0
                for j, alive in enumerate(live_mask):
                    if alive:
                        opp_actions[j] = opp_acts[act_idx]
                        act_idx += 1

        obs, reward, terminated, truncated, info = env.step(last_action, opp_actions)

        # === Render ===
        screen.fill((20, 20, 20))

        # Build pixel array with colored trails
        pixel_array = np.zeros((ARENA_SIM, ARENA_SIM, 3), dtype=np.uint8)

        for player in env.players:
            pid = player.id
            mask = env.trail_owner == pid
            color_idx = 0 if pid == env.ego.id else min(pid - 1, len(PLAYER_COLORS) - 1)
            color = PLAYER_COLORS[color_idx]
            pixel_array[mask] = color

        # Walls (rectangular for GS++, square for None)
        w = max(1, int(TRAIL_WIDTH / 2) + 1)
        if env.gspp:
            oy = ARENA_OFFSET_Y_GSPP
            ah = ARENA_H_GSPP
            pixel_array[oy:oy+w, :] = [180, 180, 180]
            pixel_array[oy+ah-w:oy+ah, :] = [180, 180, 180]
            pixel_array[:, :w] = [180, 180, 180]
            pixel_array[:, -w:] = [180, 180, 180]
        else:
            pixel_array[:w, :] = [180, 180, 180]
            pixel_array[-w:, :] = [180, 180, 180]
            pixel_array[:, :w] = [180, 180, 180]
            pixel_array[:, -w:] = [180, 180, 180]

        # Player heads
        for player in env.players:
            if not player.alive:
                continue
            ix, iy = int(round(player.x)), int(round(player.y))
            r = max(2, int(TRAIL_WIDTH / 2) + 1)
            color_idx = 0 if player.id == env.ego.id else min(
                player.id - 1, len(PLAYER_COLORS) - 1
            )
            head_color = (0, 255, 0) if player.id == env.ego.id else PLAYER_COLORS[color_idx]
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        cx, cy = ix + dx, iy + dy
                        if 0 <= cx < ARENA_SIM and 0 <= cy < ARENA_SIM:
                            pixel_array[cy, cx] = head_color

        surf = pygame.surfarray.make_surface(pixel_array.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surf, (DISPLAY_SIZE, DISPLAY_SIZE))

        # Powerup icons BEHIND trails — blit onto scaled surface before screen
        for pup in env.powerups:
            px = int(round(pup.x * DISPLAY_SCALE))
            py = int(round(pup.y * DISPLAY_SCALE))
            icon = speed_icon if pup.ptype == 'speed' else erase_icon
            scaled.blit(icon, (px - pup_radius_display, py - pup_radius_display))

        # Now blit trails on top: re-apply trail pixels over powerup icons
        # by blitting only non-black pixels from the original surface
        trail_surf = pygame.surfarray.make_surface(pixel_array.transpose(1, 0, 2))
        trail_surf.set_colorkey((0, 0, 0))  # black = transparent (background)
        trail_scaled = pygame.transform.scale(trail_surf, (DISPLAY_SIZE, DISPLAY_SIZE))
        scaled.blit(trail_scaled, (0, 0))

        screen.blit(scaled, (0, 0))

        # Direction lines for alive players
        for player in env.players:
            if not player.alive:
                continue
            hx = int(player.x * DISPLAY_SCALE)
            hy = int(player.y * DISPLAY_SCALE)
            dx = math.cos(player.angle) * 15
            dy = math.sin(player.angle) * 15
            color = (0, 255, 0) if player.id == env.ego.id else (100, 100, 100)
            pygame.draw.line(screen, color, (hx, hy),
                             (hx + int(dx), hy + int(dy)), 1)

        # --- Info panel ---
        panel_x = DISPLAY_SIZE + 10
        alive_count = sum(1 for p in env.players if p.alive)
        version_str = {"legacy": "v4 CNN", "v5": "v5 GRU64",
                       "v5_1": "v5.1 CBAM+GRU128",
                       "v6": "v6 GS++"}.get(ckpt_type, "unknown")

        lines = [
            (f"CurveCrash FFA ({version_str})", (0, 255, 0)),
            ("", None),
            (f"Ego: {ego_mode}", (0, 200, 255)),
            (f"Opps: {opp_mode}", (0, 200, 255)),
            (f"Episode: {episode_num}", (200, 200, 200)),
            (f"Players: {len(env.players)}  Alive: {alive_count}", (200, 200, 200)),
            ("", None),
            (f"Ego alive: {env.ego.time_alive:.1f}s", (255, 255, 255)),
            (f"Steps: {env.agent_steps}", (200, 200, 200)),
            (f"Reward: {env.episode_return:.2f}", (200, 200, 200)),
            (f"Action: {ACTION_NAMES[last_action]}", (255, 255, 0)),
            (f"Value: {val:.2f}", (200, 200, 200)),
            ("", None),
        ]

        # Safety shield info
        if args.shield_depth > 0:
            score_str = " ".join(f"{s:.0f}" for s in survival_scores)
            shield_color = (255, 80, 80) if shield_active else (80, 255, 80)
            lines.append((f"Search [{score_str}] L/S/R", shield_color))
            if shield_active:
                lines.append(("  ^ OVERRIDE", (255, 80, 80)))

        # Show GRU hidden state norm
        if ego_gru_state is not None:
            gru_norm = ego_gru_state.norm().item()
            lines.append((f"GRU |h|: {gru_norm:.2f}", (150, 150, 255)))

        # GS++ powerup info
        if env.gspp:
            n_pups = len(env.powerups)
            ego_boosts = len(env.ego.speed_boosts)
            speed_mult = 1.0
            for _, m in env.ego.speed_boosts:
                speed_mult *= m
            lines.append((f"Powerups: {n_pups} on field", (76, 175, 80)))
            if ego_boosts > 0:
                lines.append((f"Speed: {speed_mult:.0f}x ({ego_boosts} active)",
                              (255, 200, 50)))

        if not env.ego.alive:
            lines.insert(1, (">>> EGO DEAD <<<", (255, 0, 0)))
            lines.insert(2, ("Press R to restart", (255, 255, 0)))

        # Player status list
        lines.append(("--- PLAYERS ---", (100, 100, 100)))
        for player in env.players:
            tag = "EGO" if player.id == env.ego.id else f"OPP{player.id-1}"
            status = f"{player.time_alive:.1f}s" if player.alive else "DEAD"
            color_idx = 0 if player.id == env.ego.id else min(
                player.id - 1, len(PLAYER_COLORS) - 1
            )
            lines.append((f"  {tag}: {status}", PLAYER_COLORS[color_idx]))

        for i, (text, color) in enumerate(lines):
            if color:
                rendered = font.render(text, True, color)
                screen.blit(rendered, (panel_x, 10 + i * 18))

        # Obs preview: show self channel (green), enemy channel (red), powerups (blue)
        obs_y = 10 + len(lines) * 18 + 10
        preview_size = 128
        ch_self = obs[0]   # current self+walls
        ch_enemy = obs[1]  # current enemy
        preview_rgb = np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
        preview_rgb[:, :, 1] = (ch_self * 200).astype(np.uint8)   # green = self
        preview_rgb[:, :, 0] = (ch_enemy * 255).astype(np.uint8)  # red = enemy
        # Overlay powerup channels in GS++ mode
        if env.gspp and obs.shape[0] >= 6:
            ch_speed = obs[4]
            ch_erase = obs[5]
            # Speed powerups: bright green overlay
            speed_mask = ch_speed > 0
            preview_rgb[speed_mask, 1] = 255
            # Erase powerups: blue overlay
            erase_mask = ch_erase > 0
            preview_rgb[erase_mask, 2] = 255
        preview_surf = pygame.surfarray.make_surface(preview_rgb.transpose(1, 0, 2))
        preview_scaled = pygame.transform.scale(preview_surf, (preview_size, preview_size))
        label_text = "Obs: G=self R=enemy" + (" B=erase gG=speed" if env.gspp else "")
        label = font.render(label_text, True, (100, 100, 100))
        screen.blit(label, (panel_x, obs_y - 16))
        screen.blit(preview_scaled, (panel_x, obs_y))

        pygame.display.flip()

        if terminated or truncated:
            pygame.time.wait(800)
            obs, info = env.reset()
            reset_gru_states()
            episode_num += 1

        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()
