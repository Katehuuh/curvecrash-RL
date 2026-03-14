"""
Heavy BC (Behavioral Cloning) training from human replay data.
Trains the agent to imitate ELO 1600-2200 player behavior.

Usage:
    python train_bc.py --checkpoint checkpoints_impala_voronoi_v3/agent_2500608.pt \
        --bc-data data/bc_data --lr 5e-5 --steps 10000 --save-as checkpoints_bc/agent_bc.pt
"""
import argparse
import gc
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from experiments import ImpalaCNNAgent
from train_selfplay import BCReplayBuffer, forward_sequential_bc, GRU_HIDDEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Starting checkpoint (e.g. v8 best)")
    p.add_argument("--bc-data", type=str, required=True,
                   help="Path to BC data directory")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Learning rate for BC training")
    p.add_argument("--steps", type=int, default=10000,
                   help="Number of BC gradient steps")
    p.add_argument("--save-freq", type=int, default=2000,
                   help="Save checkpoint every N steps")
    p.add_argument("--seq-len", type=int, default=128,
                   help="Sequence length for BC chunks")
    p.add_argument("--save-as", type=str, default="checkpoints_bc/agent_bc.pt")
    p.add_argument("--freeze-voronoi", action="store_true",
                   help="Freeze voronoi channel weights in first conv layer")
    p.add_argument("--eval-freq", type=int, default=1000,
                   help="Evaluate every N steps")
    return p.parse_args()


def evaluate_action_dist(model, bc_buffer, device, n_batches=20, target_channels=7):
    """Evaluate action distribution on BC data."""
    model.eval()
    action_counts = np.zeros(3)
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(n_batches):
            chunk_idx = bc_buffer.sample_chunk(128)
            bc_obs = np.ascontiguousarray(bc_buffer.obs[chunk_idx], dtype=np.float32) / 255.0
            bc_act = np.ascontiguousarray(bc_buffer.act[chunk_idx], dtype=np.int64)
            bc_obs_t = torch.from_numpy(bc_obs).to(device)
            bc_act_t = torch.from_numpy(bc_act).to(device)
            h = torch.zeros(1, 1, GRU_HIDDEN, device=device)
            logits, _ = forward_sequential_bc(model, bc_obs_t, h, target_channels=target_channels)
            preds = logits.argmax(dim=1)
            correct += (preds == bc_act_t).sum().item()
            total += len(bc_act_t)
            for a in preds.cpu().numpy():
                action_counts[a] += 1
            del bc_obs, bc_obs_t, bc_act_t, logits, preds

    model.train()
    gc.collect()
    acc = correct / total
    act_pct = action_counts / action_counts.sum() * 100
    return acc, act_pct


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_in = sd['conv.0.conv.weight'].shape[1]
    channels = []
    for ci in range(10):
        key = f'conv.{ci}.conv.weight'
        if key in sd:
            channels.append(sd[key].shape[0])
    agent = ImpalaCNNAgent(
        n_input_channels=n_in, gru_hidden=GRU_HIDDEN,
        channels=tuple(channels),
        use_cbam=any(k.startswith('cbam.') for k in sd),
    ).to(device)
    agent.load_state_dict(sd)
    print(f"Loaded {args.checkpoint} ({n_in}ch, {sum(p.numel() for p in agent.parameters()):,} params)")

    # Optionally freeze voronoi channel weights
    if args.freeze_voronoi and n_in > 6:
        print("Freezing voronoi channel weights in first conv layer")
        # The first conv weight shape is (out_ch, n_in, 3, 3)
        # We want to freeze the slice [:, 6:, :, :] (voronoi channel)
        # Can't selectively freeze, so we'll manually zero the gradients after backward
        freeze_voronoi = True
    else:
        freeze_voronoi = False

    # Load BC data — subset into RAM to avoid mmap OOM
    bc_buffer = BCReplayBuffer(args.bc_data)
    bc_data_channels = bc_buffer.obs.shape[1]
    total_frames = len(bc_buffer.obs)
    total_seqs = len(bc_buffer.sequences)
    print(f"BC data: {total_frames:,} frames, {bc_data_channels}ch, {total_seqs} sequences")

    # Select random subset of sequences that fits in ~4 GB RAM
    max_frames = 40000  # 40K frames * 6ch * 128 * 128 = ~3.8 GB
    np.random.shuffle(bc_buffer.sequences)
    subset_seqs = []
    subset_frames = 0
    for seq in bc_buffer.sequences:
        if subset_frames + len(seq) > max_frames:
            break
        subset_seqs.append(seq)
        subset_frames += len(seq)

    # Gather all indices and load obs into RAM
    all_idx = np.concatenate(subset_seqs)
    print(f"Loading {len(all_idx):,} frames from {len(subset_seqs)} sequences into RAM (~{len(all_idx) * bc_data_channels * 128 * 128 / 1e9:.1f} GB)...")
    subset_obs = np.array(bc_buffer.obs[all_idx])  # copy from mmap to RAM
    subset_act = bc_buffer.act[all_idx].copy()

    # Rebuild sequences with new contiguous indices
    offset = 0
    new_sequences = []
    for seq in subset_seqs:
        new_sequences.append(np.arange(offset, offset + len(seq)))
        offset += len(seq)

    # Replace buffer data with in-RAM subset
    del bc_buffer.obs  # release mmap
    gc.collect()
    bc_buffer.obs = subset_obs
    bc_buffer.act = subset_act
    bc_buffer.sequences = new_sequences
    print(f"Loaded into RAM. {len(subset_obs):,} frames, {len(new_sequences)} sequences")
    print(f"Model expects {n_in}ch — will zero-pad {n_in - bc_data_channels} channels" if n_in > bc_data_channels else "")

    # Optimizer
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"\nBC Training: {args.steps} steps, lr={args.lr}, seq_len={args.seq_len}")

    # Tensorboard
    os.makedirs(os.path.dirname(args.save_as) or '.', exist_ok=True)
    writer = SummaryWriter(f"runs/bc_{int(time.time())}")

    # Evaluate before training
    acc0, dist0 = evaluate_action_dist(agent, bc_buffer, device, target_channels=n_in)
    print(f"\nBefore BC: accuracy={acc0:.1%}, actions=[L={dist0[0]:.1f}% S={dist0[1]:.1f}% R={dist0[2]:.1f}%]")

    # Training loop
    agent.train()
    losses = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        chunk_idx = bc_buffer.sample_chunk(args.seq_len)
        bc_obs = np.ascontiguousarray(bc_buffer.obs[chunk_idx], dtype=np.float32) / 255.0
        bc_act = np.ascontiguousarray(bc_buffer.act[chunk_idx], dtype=np.int64)
        bc_obs_t = torch.from_numpy(bc_obs).to(device)
        bc_act_t = torch.from_numpy(bc_act).to(device)
        h = torch.zeros(1, 1, GRU_HIDDEN, device=device)

        logits, _ = forward_sequential_bc(agent, bc_obs_t, h, target_channels=n_in)
        loss = nn.functional.cross_entropy(logits, bc_act_t)

        optimizer.zero_grad()
        loss.backward()

        # Freeze voronoi channel: zero out gradients for that slice
        if freeze_voronoi:
            for name, param in agent.named_parameters():
                if 'conv.0.conv.weight' in name:
                    if param.grad is not None:
                        param.grad[:, 6:, :, :] = 0.0
                    break

        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        del bc_obs, bc_obs_t, bc_act_t, logits, loss

        if step % 100 == 0:
            writer.add_scalar("bc/loss", loss_val, step)

        if step % 500 == 0 or step == 1:
            avg_loss = np.mean(losses[-500:])
            elapsed = time.time() - t0
            print(f"  step {step:>6d}/{args.steps}: loss={avg_loss:.4f}  ({elapsed:.0f}s)")
            gc.collect()

        if step % args.eval_freq == 0:
            acc, dist = evaluate_action_dist(agent, bc_buffer, device, target_channels=n_in)
            writer.add_scalar("bc/accuracy", acc, step)
            print(f"         eval: accuracy={acc:.1%}, actions=[L={dist[0]:.1f}% S={dist[1]:.1f}% R={dist[2]:.1f}%]")

        if step % args.save_freq == 0:
            ckpt_path = args.save_as.replace(".pt", f"_step{step}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            print(f"         saved: {ckpt_path}")

    # Final evaluation
    acc_f, dist_f = evaluate_action_dist(agent, bc_buffer, device, target_channels=n_in)
    print(f"\nAfter BC:  accuracy={acc_f:.1%}, actions=[L={dist_f[0]:.1f}% S={dist_f[1]:.1f}% R={dist_f[2]:.1f}%]")
    print(f"Before BC: accuracy={acc0:.1%}, actions=[L={dist0[0]:.1f}% S={dist0[1]:.1f}% R={dist0[2]:.1f}%]")

    # Save
    torch.save(agent.state_dict(), args.save_as)
    print(f"\nSaved BC-trained model: {args.save_as}")

    # Weight drift analysis
    drift = 0
    n = 0
    for k in sd:
        if k in agent.state_dict():
            d = (agent.state_dict()[k].cpu().float() - sd[k].cpu().float()).abs().mean().item()
            drift += d * sd[k].numel()
            n += sd[k].numel()
    print(f"Weight drift from original: {drift/n:.6f}")

    writer.close()


if __name__ == "__main__":
    main()
