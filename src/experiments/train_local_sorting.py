# # src/experiments/train_local_sorting.py
# """
# Memory-safe training loop for the local sorting experiment.

# Design goals:
# - Keep rollout tensors on CPU where possible; stream small minibatches to GPU.
# - Be compatible with older/newer ppo_update signatures by trying keyword then positional call.
# - Conservative hyperparams for Colab / small GPU usage.
# """

# import os
# import time
# import gc
# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

# from src.envs.wrappers import SortingEnv
# from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update
# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
# from src.utils.metrics import interfacial_energy, motion_penalty

# # ------------------ hyperparameters (conservative) ------------------
# H, W = 32, 32            # grid size (reduce to lower memory)
# BATCH = 1                # number of parallel envs
# T_STEPS = 8              # rollout length
# PATCH_SIZE = 5           # patch size used by agent (keep matched to env)
# ACTION_DIM = 3

# GAMMA = 0.99
# LAM = 0.95
# CLIP = 0.2

# EPOCHS = 3
# MINI_BATCH = 256         # streamed minibatch (small)
# LR = 3e-4
# TOTAL_UPDATES = 30

# LOG_DIR = "runs/sorting_rl"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Optional PyTorch allocation tweak to reduce fragmentation (helpful on Colab)
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# # Fuzzy / model sizes (kept small)
# FEAT_DIM = 48
# FUZZY_FEATURES = 12
# N_MFS = 3
# N_RULES = 24
# # --------------------------------------------------------------------

# def flatten_patches_tensor(patches):
#     """
#     patches shape used here is expected: (T, B, N, C, h, w) or (B*N, C, h, w) depending call site.
#     This helper assumes patches of shape (T, B, N, C, h, w) and flattens to (T*B*N, C, h, w).
#     """
#     T, B, N, C, h, w = patches.shape
#     return patches.view(T * B * N, C, h, w)


# def safe_ppo_update(policy, optimizer,
#                     obs_cpu, actions_cpu, logp_old_cpu, returns_cpu, advantages_cpu,
#                     clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
#                     epochs=3, batch_size=256):
#     """
#     Stream CPU-stored rollout data in small minibatches to GPU and call ppo_update.
#     This keeps the GPU memory footprint small.

#     obs_cpu: (S, C, h, w)  -- CPU tensor
#     actions_cpu: (S, A)
#     logp_old_cpu: (S,)
#     returns_cpu: (S,)
#     advantages_cpu: (S,)
#     """
#     S = obs_cpu.shape[0]
#     if S == 0:
#         return {}

#     # randomize indices once per epoch set
#     for epoch in range(epochs):
#         perm = torch.randperm(S)
#         for start in range(0, S, batch_size):
#             idx = perm[start:start + batch_size]
#             # move minibatch to device (non_blocking when pinned memory used)
#             obs_mb = obs_cpu[idx].to(DEVICE, non_blocking=True)
#             acts_mb = actions_cpu[idx].to(DEVICE, non_blocking=True)
#             logp_mb = logp_old_cpu[idx].to(DEVICE, non_blocking=True)
#             ret_mb = returns_cpu[idx].to(DEVICE, non_blocking=True)
#             adv_mb = advantages_cpu[idx].to(DEVICE, non_blocking=True)

#             # Two calling styles: try keyword (preferred) then fallback to positional if needed.
#             try:
#                 logs = ppo_update(
#                     policy=policy,
#                     optimizer=optimizer,
#                     obs_patches=obs_mb,
#                     actions=acts_mb,
#                     logprobs_old=logp_mb,
#                     returns=ret_mb,
#                     advantages=adv_mb,
#                     clip_eps=clip_eps,
#                     vf_coef=vf_coef,
#                     ent_coef=ent_coef,
#                     epochs=1,
#                     batch_size=obs_mb.shape[0],
#                 )
#             except TypeError:
#                 # older signature might expect positional args: (policy, optimizer, obs, acts, logp_old, returns, adv, clip, vf_coef, ent_coef)
#                 try:
#                     logs = ppo_update(
#                         policy, optimizer,
#                         obs_mb, acts_mb, logp_mb, ret_mb, adv_mb,
#                         clip_eps, vf_coef, ent_coef
#                     )
#                 except Exception as e:
#                     # If fallback also fails, raise to make user aware
#                     raise RuntimeError("ppo_update call failed (keyword + positional attempts)."
#                                        " Inspect ppo_update signature.") from e

#             # free minibatch device tensors right away
#             del obs_mb, acts_mb, logp_mb, ret_mb, adv_mb
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     return logs


# def main():
#     # housekeeping
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     os.makedirs("checkpoints", exist_ok=True)
#     writer = SummaryWriter(LOG_DIR)

#     # environment (SortingEnv provides local patches when obs_mode='local')
#     env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')

#     # model (small)
#     policy = NeuroFuzzyActorCritic(
#         in_ch=4, patch_size=PATCH_SIZE,
#         feat_dim=FEAT_DIM, fuzzy_features=FUZZY_FEATURES,
#         n_mfs=N_MFS, n_rules=N_RULES,
#         action_dim=ACTION_DIM
#     ).to(DEVICE)

#     optimizer = optim.Adam(policy.parameters(), lr=LR)
#     N = H * W

#     iter_start = time.time()
#     bench_times = []

#     for update in range(TOTAL_UPDATES):
#         # store rollout on CPU lists to avoid holding tensors on GPU
#         obs_patches_cpu = []
#         actions_cpu = []
#         logps_cpu = []
#         values_cpu = []
#         rewards_cpu = []
#         dones_cpu = []

#         # reset env
#         obs = env.reset(B=BATCH, pA=0.5)
#         patches, coords = obs  # expected shape: (B, N, C, h, w) or similar depending wrapper

#         # rollout (inference only)
#         for t in range(T_STEPS):
#             # create flattened patch batch for policy
#             # patches is (B, N, C, h, w) → flatten to (B*N, C, h, w)
#             flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

#             with torch.no_grad():
#                 action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)

#             # reshape back to grid format
#             action_grid = action_flat.reshape(BATCH, N, ACTION_DIM)
#             logp_grid = logp_flat.reshape(BATCH, N)
#             value_grid = value_flat.view(BATCH, N).mean(1)

#             # step environment (env expects action_grid on DEVICE)
#             obs2, reward, info = env.step(action_grid)
#             patches, coords = obs2

#             # Immediately move storage copies to CPU and release GPU tensors
#             obs_patches_cpu.append(patches.detach().cpu())
#             actions_cpu.append(action_grid.detach().cpu())
#             logps_cpu.append(logp_grid.detach().cpu())
#             values_cpu.append(value_grid.detach().cpu())
#             rewards_cpu.append(reward.cpu())
#             dones_cpu.append(torch.zeros_like(reward.cpu()))

#             # free temporaries
#             del flat, action_flat, logp_flat, value_flat, action_grid, logp_grid, value_grid
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         # bootstrap the final value (inference)
#         flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#         with torch.no_grad():
#             _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
#             values_cpu.append(vals.view(BATCH, N).mean(1).cpu())
#         del flat, vals
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         # assemble tensors on CPU
#         T = T_STEPS
#         obs_patches = torch.stack(obs_patches_cpu)   # (T, B, N, C, h, w) - CPU
#         actions = torch.stack(actions_cpu)           # (T, B, N, ACTION_DIM) - CPU
#         logps = torch.stack(logps_cpu)               # (T, B, N) - CPU
#         rewards = torch.stack(rewards_cpu)           # (T, B)
#         dones = torch.stack(dones_cpu)               # (T, B)
#         values = torch.stack(values_cpu)             # (T+1, B)

#         # GAE on CPU
#         advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

#         # flatten to shape (S, ...)
#         S = T * BATCH * N
#         obs_flat_cpu = obs_patches.reshape(S, 4, PATCH_SIZE, PATCH_SIZE).cpu()
#         actions_flat_cpu = actions.reshape(S, ACTION_DIM).cpu()
#         logps_flat_cpu = logps.reshape(S).cpu()
#         returns_flat_cpu = returns.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()
#         advs_flat_cpu = advantages.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()

#         # perform streamed PPO updates (minibatches moved to GPU)
#         safe_ppo_update(
#             policy=policy,
#             optimizer=optimizer,
#             obs_cpu=obs_flat_cpu,
#             actions_cpu=actions_flat_cpu,
#             logp_old_cpu=logps_flat_cpu,
#             returns_cpu=returns_flat_cpu,
#             advantages_cpu=advs_flat_cpu,
#             clip_eps=CLIP,
#             vf_coef=0.5,
#             ent_coef=0.01,
#             epochs=EPOCHS,
#             batch_size=MINI_BATCH
#         )

#         # diagnostics: compute a few metrics (move only tiny tensors to device)
#         mean_r = rewards.mean().item()
#         mean_e = interfacial_energy(env.current_state()).mean().item()

#         # compute motion penalty (move small tensor to device and free quickly)
#         act_dev = actions_flat_cpu.view(T, BATCH, N, ACTION_DIM).to(DEVICE)
#         mean_mpen = motion_penalty(act_dev.transpose(1, 2)).mean().item()
#         del act_dev
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         mean_adh = env.current_state()[:, 2].mean().item()

#         writer.add_scalar("reward/avg", mean_r, update)
#         writer.add_scalar("energy/interfacial", mean_e, update)
#         writer.add_scalar("motion/penalty", mean_mpen, update)
#         writer.add_scalar("adhesion/mean", mean_adh, update)

#         print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

#         # checkpoint occasionally
#         if update % 10 == 0:
#             ckpt = f"checkpoints/ppo_{update:04d}.pt"
#             torch.save(policy.state_dict(), ckpt)
#             print(f"[ckpt] saved {ckpt}")

#         # simple bench timing
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         bench_times.append(time.time() - iter_start)
#         iter_start = time.time()

#     writer.close()


# if __name__ == "__main__":
#     main()

# src/experiments/train_local_sorting.py
import os
import gc
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from src.envs.wrappers import SortingEnv
from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update

from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
# visualization disabled by default (GIF saving still present but can be heavy)
from src.utils.metrics import interfacial_energy, motion_penalty

# ------------------ hyperparameters (colab-safe, conservative) ------------------
H, W = 32, 32
BATCH = 1

T_STEPS = 8

PATCH_SIZE = 5
ACTION_DIM = 3
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
MINI_BATCH = 256   # streamed minibatch during update (keeps GPU memory low)
LR = 3e-4

# RUN LENGTH: change this to 200 updates
TOTAL_UPDATES = 200

# how often to checkpoint + save plots (every CKPT_FREQ updates)
CKPT_FREQ = 50

LOG_DIR = "runs/sorting_rl"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------------------------------------------------------------

# PyTorch fragmentation helper (optional)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# smaller model dims for stability
FEAT_DIM = 48
FUZZY_FEATURES = 12
N_MFS = 3
N_RULES = 24


def flatten_patches(patches):
    T, B, N, C, h, w = patches.shape
    return patches.view(T * B * N, C, h, w)


def safe_ppo_update(policy, optimizer,
                    obs_cpu, actions_cpu, logp_old_cpu, returns_cpu, advantages_cpu,
                    clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
                    epochs=3, batch_size=256):
    """
    Stream CPU-stored data in small minibatches to GPU and call the existing ppo_update.
    obs_cpu, actions_cpu, ... are CPU tensors shaped [S, ...]
    """
    S = obs_cpu.shape[0]
    idxs = torch.randperm(S)
    for epoch in range(epochs):
        for start in range(0, S, batch_size):
            batch_idx = idxs[start:start + batch_size]
            # move minibatch to device
            obs_mb = obs_cpu[batch_idx].to(DEVICE, non_blocking=True)
            actions_mb = actions_cpu[batch_idx].to(DEVICE, non_blocking=True)
            logp_mb = logp_old_cpu[batch_idx].to(DEVICE, non_blocking=True)
            returns_mb = returns_cpu[batch_idx].to(DEVICE, non_blocking=True)
            advs_mb = advantages_cpu[batch_idx].to(DEVICE, non_blocking=True)

            # call ppo_update (uses keywords that match src/agents/ppo.py)
            _ = ppo_update(policy=policy, optimizer=optimizer,
                           obs_patches=obs_mb,
                           actions=actions_mb,
                           logprobs_old=logp_mb,
                           returns=returns_mb,
                           advantages=advs_mb,
                           clip_eps=clip_ratio,
                           vf_coef=value_coef,
                           ent_coef=entropy_coef,
                           epochs=1,
                           batch_size=obs_mb.shape[0])

            # free minibatch device tensors right away
            del obs_mb, actions_mb, logp_mb, returns_mb, advs_mb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return


def plot_metrics(updates, rewards, energies, motion_pens, out_path):
    """
    Plot and save reward/energy/motion_pen vs update index.
    """
    plt.figure(figsize=(8, 5))
    plt.xlabel("update")
    plt.grid(True)
    plt.plot(updates, rewards, label="reward")
    plt.plot(updates, energies, label="interfacial_energy")
    plt.plot(updates, motion_pens, label="motion_penalty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # cleanup and caching
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("visuals/metrics", exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')

    policy = NeuroFuzzyActorCritic(in_ch=4, patch_size=PATCH_SIZE,
                                   feat_dim=FEAT_DIM,
                                   fuzzy_features=FUZZY_FEATURES,
                                   n_mfs=N_MFS, n_rules=N_RULES,
                                   action_dim=ACTION_DIM).to(DEVICE)

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    N = H * W

    # buffers to plot later
    updates_list = []
    rewards_list = []
    energies_list = []
    motion_pen_list = []

    iter_start = time.time()

    for update in range(TOTAL_UPDATES):
        obs_patches_list_cpu, actions_list_cpu = [], []
        logp_list_cpu, values_list_cpu, rewards_list_cpu, dones_list_cpu = [], [], [], []

        obs = env.reset(B=BATCH, pA=0.5)
        patches, coords = obs

        # rollout (inference)
        for t in range(T_STEPS):
            flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
            with torch.no_grad():
                action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)

            action_grid = action_flat.reshape(BATCH, N, ACTION_DIM).to(DEVICE)
            logp_grid = logp_flat.reshape(BATCH, N).to(DEVICE)
            value_grid = value_flat.view(BATCH, N).mean(1).to(DEVICE)

            obs2, reward, info = env.step(action_grid)
            patches, coords = obs2

            # move to CPU ASAP
            obs_patches_list_cpu.append(patches.detach().cpu())
            actions_list_cpu.append(action_grid.detach().cpu())
            logp_list_cpu.append(logp_grid.detach().cpu())
            values_list_cpu.append(value_grid.detach().cpu())
            rewards_list_cpu.append(reward.cpu())
            dones_list_cpu.append(torch.zeros_like(reward.cpu()))

            # free temporary variables
            del flat, action_flat, logp_flat, value_flat, action_grid, logp_grid, value_grid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # bootstrap final value
        flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        with torch.no_grad():
            _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
            values_list_cpu.append(vals.view(BATCH, N).mean(1).cpu())
        del flat, vals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # assemble CPU tensors
        T = T_STEPS
        obs_patches = torch.stack(obs_patches_list_cpu)   # (T, B, N, C, h, w)
        actions = torch.stack(actions_list_cpu)           # (T, B, N, ACTION_DIM)
        logps = torch.stack(logp_list_cpu)                # (T, B, N)
        rewards = torch.stack(rewards_list_cpu)           # (T, B)
        dones = torch.stack(dones_list_cpu)               # (T, B)
        values = torch.stack(values_list_cpu)             # (T+1, B)

        # compute GAE (CPU)
        advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

        # flatten and prepare for streamed update (keep on CPU)
        S = T * BATCH * N
        obs_patches_flat_cpu = obs_patches.reshape(S, 4, PATCH_SIZE, PATCH_SIZE).cpu()
        actions_flat_cpu = actions.reshape(S, ACTION_DIM).cpu()
        logps_flat_cpu = logps.reshape(S).cpu()
        returns_flat_cpu = returns.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()
        advs_flat_cpu = advantages.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()

        # safe streamed ppo updates
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

        # diagnostics
        mean_r = rewards.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        # compute motion penalty: move minimal tensors to device just for metric
        act_dev = actions_flat_cpu.view(T, BATCH, N, ACTION_DIM).to(DEVICE)
        mean_mpen = motion_penalty(act_dev.transpose(1, 2)).mean().item()
        del act_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean_adh = env.current_state()[:, 2].mean().item()

        # logging
        writer.add_scalar("reward/avg", mean_r, update)
        writer.add_scalar("energy/interfacial", mean_e, update)
        writer.add_scalar("motion/penalty", mean_mpen, update)
        writer.add_scalar("adhesion/mean", mean_adh, update)

        # append to arrays for plotting
        updates_list.append(update)
        rewards_list.append(mean_r)
        energies_list.append(mean_e)
        motion_pen_list.append(mean_mpen)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

        # checkpoint + plot every CKPT_FREQ updates
        if (update + 1) % CKPT_FREQ == 0 or update == TOTAL_UPDATES - 1:
            ckpt = f"checkpoints/ppo_{update:04d}.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"[ckpt] saved {ckpt}")

            # save a metrics plot
            plot_path = f"visuals/metrics/metrics_{update:04d}.png"
            plot_metrics(updates_list, rewards_list, energies_list, motion_pen_list, plot_path)
            print(f"[plot] saved {plot_path}")

        # benchmark timing housekeeping
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_time = time.time() - iter_start
        iter_start = time.time()

    writer.close()
    # final plot (full run)
    final_plot = "visuals/metrics/metrics_final.png"
    plot_metrics(updates_list, rewards_list, energies_list, motion_pen_list, final_plot)
    print(f"[plot] final metrics saved to {final_plot}")


if __name__ == "__main__":
    main()
