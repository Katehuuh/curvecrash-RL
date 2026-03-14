"""
Watch trained CurveCrash FFA agent with Pygame rendering.
UI only — model loading via train_selfplay.load_agent(), rendering via env.render().

Usage:
    python watch_agent_ffa.py                                    # AI vs AI
    python watch_agent_ffa.py --human                            # YOU vs AI opponents
    python watch_agent_ffa.py --human --min-players 2            # 1v1
    python watch_agent_ffa.py --checkpoint checkpoints/agent.pt
    python watch_agent_ffa.py --random-opponents                 # trained ego, random opps

Controls: LEFT/RIGHT arrows=steer, R=restart, ESC=quit, SPACE=pause
"""
import argparse
import os

import pygame
import numpy as np
import torch

from curvecrash_env_ffa import CurveCrashFFAEnv, ARENA_SIM, OBS_SIZE
from train_selfplay import load_agent, GRU_HIDDEN
from experiments import VoronoiWrapper

DISPLAY_SCALE = 2
DISPLAY_SIZE = ARENA_SIM * DISPLAY_SCALE
ACTION_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--human", action="store_true")
    p.add_argument("--random", action="store_true")
    p.add_argument("--random-opponents", action="store_true")
    p.add_argument("--min-players", type=int, default=2)
    p.add_argument("--max-players", type=int, default=6)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gspp", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect checkpoint
    if args.checkpoint is None:
        import glob
        for d in ["checkpoints_impala_voronoi_v3", "checkpoints_impala_voronoi", "checkpoints"]:
            pts = sorted(glob.glob(os.path.join(d, "agent_*.pt")))
            if pts:
                args.checkpoint = pts[-1]
                break
        if args.checkpoint is None:
            args.checkpoint = "checkpoints/agent_final.pt"

    # Load model via shared function
    agent, meta = None, {}
    opp_agent, opp_meta = None, {}
    try:
        agent, meta = load_agent(args.checkpoint, device)
        opp_agent, opp_meta = agent, meta
        print(f"Loaded: {args.checkpoint} [{meta['type']}, {meta['n_input_ch']}ch, GRU{meta['gru_hidden']}]")
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")

    if args.human:
        agent = None  # human controls ego
        print("Mode: HUMAN vs AI")
    elif args.random:
        agent = None
        print("Mode: RANDOM")
    if args.random_opponents:
        opp_agent, opp_meta = None, {}

    # Create env
    n_ch = meta.get('n_input_ch', 6)
    use_voronoi = n_ch >= 7
    use_minimap = n_ch >= 9
    env_cls = VoronoiWrapper if use_voronoi else CurveCrashFFAEnv
    env = env_cls(
        min_players=args.min_players, max_players=args.max_players,
        gspp=args.gspp, render_mode="rgb_array",
        bilinear_ds=use_minimap, minimap=use_minimap,
    )

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE + 280, DISPLAY_SIZE))
    pygame.display.set_caption("CurveCrash FFA")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    obs, info = env.reset(seed=42)
    episode_num = 0
    paused = False
    last_action = 1
    val = 0.0

    # GRU states
    gru_h = meta.get('gru_hidden', 0)
    ego_gru = torch.zeros(1, 1, gru_h, device=device) if gru_h > 0 else None
    opp_gru_states = {}

    def reset_states():
        nonlocal ego_gru, opp_gru_states
        if ego_gru is not None:
            ego_gru = torch.zeros(1, 1, gru_h, device=device)
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
                    reset_states()
                    episode_num += 1
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 0)), (DISPLAY_SIZE // 2 - 30, DISPLAY_SIZE // 2))
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
                scalars_t = None
                if meta.get('has_scalars'):
                    s = env.get_scalar_obs(env.ego)
                    scalars_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                if ego_gru is not None:
                    done_t = torch.zeros(1, device=device)
                    action, _, _, value, ego_gru = agent.get_action_and_value(
                        obs_t, ego_gru, done_t, scalars=scalars_t
                    )
                else:
                    action, _, _, value = agent.get_action_and_value(obs_t)
                last_action = action.item()
                val = value.item()
        else:
            last_action = env.action_space.sample()

        # Opponent actions
        opp_actions = None
        if opp_agent is not None:
            opp_obs_list, live_mask = env.get_opponent_observations()
            if opp_obs_list:
                with torch.no_grad():
                    opp_obs_t = torch.tensor(np.stack(opp_obs_list), dtype=torch.float32, device=device)
                    opp_h = opp_meta.get('gru_hidden', 0)
                    if opp_h > 0:
                        opp_gru_list = []
                        opp_keys = []
                        for j, alive in enumerate(live_mask):
                            if alive:
                                if j not in opp_gru_states:
                                    opp_gru_states[j] = torch.zeros(1, 1, opp_h, device=device)
                                opp_gru_list.append(opp_gru_states[j])
                                opp_keys.append(j)
                        opp_gru_batch = torch.cat(opp_gru_list, dim=1)
                        done_zeros = torch.zeros(len(opp_obs_list), device=device)
                        opp_acts, new_gru = opp_agent.get_action_greedy(opp_obs_t, opp_gru_batch, done_zeros)
                        opp_acts = opp_acts.cpu().numpy()
                        for k_idx, key in enumerate(opp_keys):
                            opp_gru_states[key] = new_gru[:, k_idx:k_idx+1, :].clone()
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

        # Render — use env's built-in render()
        screen.fill((20, 20, 20))
        pixel_array = env.render()
        if pixel_array is not None:
            surf = pygame.surfarray.make_surface(pixel_array.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (DISPLAY_SIZE, DISPLAY_SIZE))
            screen.blit(scaled, (0, 0))

        # Info panel
        px = DISPLAY_SIZE + 10
        alive_count = sum(1 for p in env.players if p.alive)
        lines = [
            (f"CurveCrash FFA ({meta.get('type', '?')})", (0, 255, 0)),
            ("", None),
            (f"Ego: {'HUMAN' if args.human else ('MODEL' if agent else 'RANDOM')}", (0, 200, 255)),
            (f"Opps: {'MODEL' if opp_agent else 'RANDOM'}", (0, 200, 255)),
            (f"Episode: {episode_num}  Players: {len(env.players)}  Alive: {alive_count}", (200, 200, 200)),
            ("", None),
            (f"Alive: {env.ego.time_alive:.1f}s  Steps: {env.agent_steps}", (255, 255, 255)),
            (f"Reward: {env.episode_return:.2f}  Value: {val:.2f}", (200, 200, 200)),
            (f"Action: {ACTION_NAMES[last_action]}", (255, 255, 0)),
        ]
        if ego_gru is not None:
            lines.append((f"GRU |h|: {ego_gru.norm().item():.2f}", (150, 150, 255)))
        if env.gspp:
            speed_mult = 1.0
            for _, m in env.ego.speed_boosts:
                speed_mult *= m
            pup_str = f"Powerups: {len(env.powerups)}"
            if speed_mult > 1.01:
                pup_str += f"  Speed: {speed_mult:.0f}x"
            lines.append((pup_str, (76, 175, 80)))
        if not env.ego.alive:
            lines.insert(1, (">>> DEAD <<< Press R", (255, 0, 0)))

        for i, (text, color) in enumerate(lines):
            if color:
                screen.blit(font.render(text, True, color), (px, 10 + i * 18))

        # Obs preview
        obs_y = 10 + len(lines) * 18 + 10
        preview = np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
        preview[:, :, 1] = (obs[0] * 200).astype(np.uint8)
        preview[:, :, 0] = (obs[1] * 255).astype(np.uint8)
        ps = pygame.surfarray.make_surface(preview.transpose(1, 0, 2))
        screen.blit(font.render("Obs: G=self R=enemy", True, (100, 100, 100)), (px, obs_y - 16))
        screen.blit(pygame.transform.scale(ps, (128, 128)), (px, obs_y))

        pygame.display.flip()

        if terminated or truncated:
            pygame.time.wait(800)
            obs, info = env.reset()
            reset_states()
            episode_num += 1

        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()
