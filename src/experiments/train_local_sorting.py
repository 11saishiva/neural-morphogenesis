# # src/experiments/train_local_sorting.py
# import os, torch, torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from src.envs.wrappers import SortingEnv
# from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update

# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
# from src.utils.viz import visualize_sequence
# from src.utils.metrics import interfacial_energy, motion_penalty

# # ------------------ hyperparameters ------------------
# H, W = 32, 32 #H, W = 64, 64
# BATCH = 1 #BATCH = 2
# T_STEPS = 32
# PATCH_SIZE = 5
# ACTION_DIM = 3
# GAMMA = 0.99
# LAM = 0.95
# CLIP = 0.2
# EPOCHS = 4
# MINI_BATCH = 2048
# LR = 3e-4
# TOTAL_UPDATES = 30 #TOTAL_UPDATES = 400
# LOG_DIR = "runs/sorting_rl"
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# # -----------------------------------------------------

# def flatten_patches(patches):
#     T, B, N, C, h, w = patches.shape
#     return patches.view(T * B * N, C, h, w)

# def main():
#     os.makedirs("checkpoints", exist_ok=True)
#     writer = SummaryWriter(LOG_DIR)
#     env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')
#     policy = NeuroFuzzyActorCritic(in_ch=4, patch_size=PATCH_SIZE,
#                                feat_dim=128, fuzzy_features=16,
#                                n_mfs=3, n_rules=32,
#                                action_dim=ACTION_DIM).to(DEVICE)
    
#     optimizer = optim.Adam(policy.parameters(), lr=LR)
#     N = H * W

#     for update in range(TOTAL_UPDATES):
#         obs_patches_list, actions_list, logp_list, values_list, rewards_list, dones_list = [], [], [], [], [], []
#         # capture frames for GIF
#         vis_frames = []

#         obs = env.reset(B=BATCH, pA=0.5)
#         patches, coords = obs
#         # with torch.no_grad():
#         #     flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#         #     _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
#         #     values_list.append(vals.view(BATCH, N).mean(1).cpu())

#         # vis_frames.append(env.current_state().detach().cpu())
#         vis_frames.append(env.current_state().detach().cpu())

#         for t in range(T_STEPS):
#             flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#             action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)
#             action_grid = action_flat.reshape(BATCH, N, ACTION_DIM)
#             logp_grid = logp_flat.reshape(BATCH, N)
#             value_grid = value_flat.view(BATCH, N).mean(1)
#             obs2, reward, info = env.step(action_grid)
#             patches, coords = obs2
#             patches = patches.detach()
#             obs_patches_list.append(patches.cpu())
#             actions_list.append(action_grid.detach().cpu())
#             logp_list.append(logp_grid.detach().cpu())
#             values_list.append(value_grid.detach().cpu())
#             rewards_list.append(reward.cpu())
#             dones_list.append(torch.zeros_like(reward.cpu()))
#             vis_frames.append(env.current_state().detach().cpu())

#         # Bootstrap
#         flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#         with torch.no_grad():
#             _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
#             values_list.append(vals.view(BATCH, N).mean(1).cpu())

#         # Tensor assembly
#         T = T_STEPS
#         obs_patches = torch.stack(obs_patches_list)
#         actions = torch.stack(actions_list)
#         logps = torch.stack(logp_list)
#         rewards = torch.stack(rewards_list)
#         dones = torch.stack(dones_list)
#         values = torch.stack(values_list)
#         advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

#         obs_patches_flat = obs_patches.reshape(T * BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#         actions_flat = actions.reshape(T * BATCH * N, ACTION_DIM).to(DEVICE)
#         logps_flat = logps.reshape(T * BATCH * N).to(DEVICE)
#         returns_flat = returns.unsqueeze(2).repeat(1, 1, N).reshape(T * BATCH * N).to(DEVICE)
#         advs_flat = advantages.unsqueeze(2).repeat(1, 1, N).reshape(T * BATCH * N).to(DEVICE)

#         # PPO update
#         ppo_update(policy=policy,optimizer=optimizer,
#            obs_patches=obs_patches_flat,
#            actions=actions_flat,
#            logprobs_old=logps_flat,
#            returns=returns_flat,
#            advantages=advs_flat,
#            clip_eps=CLIP,
#            vf_coef=0.5,
#            ent_coef=0.01,
#            epochs=EPOCHS,
#            batch_size=MINI_BATCH,
#            orth_coef=3e-3,
#            sparsity_coef=5e-3,
#            corr_coef=5e-3)


#         # Diagnostics
#         mean_r = rewards.mean().item()
#         mean_e = interfacial_energy(env.current_state()).mean().item()
#         mean_mpen = motion_penalty(actions_flat.view(T, BATCH, N, ACTION_DIM).to(DEVICE).transpose(1,2)).mean().item()
#         mean_adh = env.current_state()[:, 2].mean().item()

#         writer.add_scalar("reward/avg", mean_r, update)
#         writer.add_scalar("energy/interfacial", mean_e, update)
#         writer.add_scalar("motion/penalty", mean_mpen, update)
#         writer.add_scalar("adhesion/mean", mean_adh, update)

#         print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

#         if update % 50 == 0:
#             ckpt = f"checkpoints/ppo_{update:04d}.pt"
#             torch.save(policy.state_dict(), ckpt)
#             print(f"[ckpt] saved {ckpt}")
#             visualize_sequence(vis_frames, out_path=f"visuals/sorting_{update:04d}.gif", every=2)

#     writer.close()

# if __name__ == "__main__":
#     main()

# src/experiments/train_local_sorting.py
import os
import gc
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.envs.wrappers import SortingEnv
from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update

from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
# visualization disabled for memory stability
# from src.utils.viz import visualize_sequence
from src.utils.metrics import interfacial_energy, motion_penalty

# ------------------ hyperparameters (colab-safe, conservative) ------------------
# grid & batch: keep small to avoid OOM
H, W = 32, 32
BATCH = 1

# rollout length (kept short)
T_STEPS = 8

# patch / model sizes (small)
PATCH_SIZE = 5
ACTION_DIM = 3
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINI_BATCH = 256   # streamed minibatch during update (keeps GPU memory low)
LR = 3e-4
TOTAL_UPDATES = 30
LOG_DIR = "runs/sorting_rl"
# pick device (prefer cuda)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# PyTorch allocation config (help fragmentation; optional but useful)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# fuzzy / feature dims (smaller)
FEAT_DIM = 48
FUZZY_FEATURES = 12
N_MFS = 3
N_RULES = 24
# ------------------------------------------------------------------------------

def flatten_patches(patches):
    T, B, N, C, h, w = patches.shape
    return patches.view(T * B * N, C, h, w)

def safe_ppo_update(policy, optimizer,
                    obs_cpu, actions_cpu, logp_old_cpu, returns_cpu, advantages_cpu,
                    clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
                    epochs=3, batch_size=256):
    """
    Stream CPU-stored data in small minibatches to GPU and call the existing ppo_update.
    This avoids moving the entire rollout onto the GPU at once.
    obs_cpu, actions_cpu, ... are CPU tensors shaped [S, ...]
    """
    S = obs_cpu.shape[0]
    idxs = torch.randperm(S)
    for epoch in range(epochs):
        for start in range(0, S, batch_size):
            batch_idx = idxs[start:start+batch_size]
            # move minibatch to device
            obs_mb = obs_cpu[batch_idx].to(DEVICE, non_blocking=True)
            actions_mb = actions_cpu[batch_idx].to(DEVICE, non_blocking=True)
            logp_mb = logp_old_cpu[batch_idx].to(DEVICE, non_blocking=True)
            returns_mb = returns_cpu[batch_idx].to(DEVICE, non_blocking=True)
            advs_mb = advantages_cpu[batch_idx].to(DEVICE, non_blocking=True)

            # call original ppo_update (it expects tensors on DEVICE)
            logs = ppo_update(policy=policy, optimizer=optimizer,
                              obs_patches=obs_mb,
                              actions=actions_mb,
                              logp_old=logp_mb,
                              returns=returns_mb,
                              advantages=advs_mb,
                              clip_ratio=clip_ratio,
                              value_coef=value_coef,
                              entropy_coef=entropy_coef,
                              epochs=1,      # we already handle epoch loop here
                              batch_size=obs_mb.shape[0])
            # free minibatch device tensors right away
            del obs_mb, actions_mb, logp_mb, returns_mb, advs_mb
            torch.cuda.empty_cache()
    return

def main():
    # ensure deterministic-ish startup (free caches)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')

    # safe / small model
    policy = NeuroFuzzyActorCritic(in_ch=4, patch_size=PATCH_SIZE,
                                   feat_dim=FEAT_DIM,
                                   fuzzy_features=FUZZY_FEATURES,
                                   n_mfs=N_MFS, n_rules=N_RULES,
                                   action_dim=ACTION_DIM).to(DEVICE)

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    N = H * W

    # timing / bench
    iter_start = time.time()
    bench_times = []
    BENCH_UPDATES = min(6, TOTAL_UPDATES)

    for update in range(TOTAL_UPDATES):
        # store rollout data on CPU only (to minimize GPU-residency)
        obs_patches_list_cpu, actions_list_cpu = [], []
        logp_list_cpu, values_list_cpu, rewards_list_cpu, dones_list_cpu = [], [], [], []

        # initial env reset
        obs = env.reset(B=BATCH, pA=0.5)
        patches, coords = obs  # patches initially on DEVICE (env controls placement)

        # rollout loop (inference only -> torch.no_grad)
        for t in range(T_STEPS):
            flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
            with torch.no_grad():
                action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)

            action_grid = action_flat.reshape(BATCH, N, ACTION_DIM).to(DEVICE)
            logp_grid = logp_flat.reshape(BATCH, N).to(DEVICE)
            value_grid = value_flat.view(BATCH, N).mean(1).to(DEVICE)

            obs2, reward, info = env.step(action_grid)
            patches, coords = obs2

            # immediately move stored data to CPU and free GPU references
            obs_patches_list_cpu.append(patches.detach().cpu())
            actions_list_cpu.append(action_grid.detach().cpu())
            logp_list_cpu.append(logp_grid.detach().cpu())
            values_list_cpu.append(value_grid.detach().cpu())
            rewards_list_cpu.append(reward.cpu())
            dones_list_cpu.append(torch.zeros_like(reward.cpu()))

            # free temporary device tensors quickly
            del flat, action_flat, logp_flat, value_flat, action_grid, logp_grid, value_grid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # bootstrap final value (inference)
        flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        with torch.no_grad():
            _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
            values_list_cpu.append(vals.view(BATCH, N).mean(1).cpu())
        del flat, vals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # assemble tensors on CPU
        T = T_STEPS
        obs_patches = torch.stack(obs_patches_list_cpu)   # (T, B, N, C, h, w) CPU
        actions = torch.stack(actions_list_cpu)           # (T, B, N, ACTION_DIM) CPU
        logps = torch.stack(logp_list_cpu)                # (T, B, N) CPU
        rewards = torch.stack(rewards_list_cpu)           # (T, B)
        dones = torch.stack(dones_list_cpu)               # (T, B)
        values = torch.stack(values_list_cpu)             # (T+1, B)

        # compute GAE on CPU (cheap)
        advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

        # flatten & prepare for streamed minibatch update (keep on CPU for now)
        S = T * BATCH * N
        obs_patches_flat_cpu = obs_patches.reshape(S, 4, PATCH_SIZE, PATCH_SIZE).cpu()
        actions_flat_cpu = actions.reshape(S, ACTION_DIM).cpu()
        logps_flat_cpu = logps.reshape(S).cpu()
        returns_flat_cpu = returns.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()
        advs_flat_cpu = advantages.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()

        # safe streamed ppo updates (moves only minibatch to GPU)
        safe_ppo_update(policy=policy, optimizer=optimizer,
                        obs_cpu=obs_patches_flat_cpu,
                        actions_cpu=actions_flat_cpu,
                        logp_old_cpu=logps_flat_cpu,
                        returns_cpu=returns_flat_cpu,
                        advantages_cpu=advs_flat_cpu,
                        clip_ratio=CLIP,
                        value_coef=0.5,
                        entropy_coef=0.01,
                        epochs=EPOCHS,
                        batch_size=MINI_BATCH)

        # diagnostics (compute small metrics)
        mean_r = rewards.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        # compute motion penalty: move minimal tensors to device just for metric
        act_dev = actions_flat_cpu.view(T, BATCH, N, ACTION_DIM).to(DEVICE)
        mean_mpen = motion_penalty(act_dev.transpose(1, 2)).mean().item()
        del act_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean_adh = env.current_state()[:, 2].mean().item()

        writer.add_scalar("reward/avg", mean_r, update)
        writer.add_scalar("energy/interfacial", mean_e, update)
        writer.add_scalar("motion/penalty", mean_mpen, update)
        writer.add_scalar("adhesion/mean", mean_adh, update)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

        # checkpoint occasionally (less frequent I/O)
        if update % 10 == 0:
            ckpt = f"checkpoints/ppo_{update:04d}.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"[ckpt] saved {ckpt}")

        # bench timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bench_times.append(time.time() - iter_start)
        iter_start = time.time()

    writer.close()


if __name__ == "__main__":
    main()
