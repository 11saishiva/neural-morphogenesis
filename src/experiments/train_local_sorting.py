# # ------------------------------------------------------------
# # CLEAN WORKING TRAINING SCRIPT FOR LOCAL SORTING (COLAB SAFE)
# # ------------------------------------------------------------

# import os
# import gc
# import time
# import math
# import numpy as np
# import torch
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import imageio

# from src.envs.wrappers import SortingEnv
# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
# from src.agents.ppo import compute_gae, ppo_update
# from src.utils.metrics import interfacial_energy, motion_penalty

# # ------------------ CONFIG ------------------
# H = 32
# W = 32
# PATCH_SIZE = 5
# ACTION_DIM = 3
# BATCH = 1
# T_STEPS = 32 #T_STEPS = 8
# TOTAL_UPDATES = 1000        # <<< CHANGE HERE

# GAMMA = 0.005 #GAMMA = 0.99
# LAM = 0.95
# EPOCHS = 1 #EPOCHS = 3
# MINI_BATCH = 2048 #MINI_BATCH = 256
# LR = 5e-5 #LR = 3e-4
# CLIP = 0.1

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs("checkpoints", exist_ok=True)
# os.makedirs("visuals", exist_ok=True)

# # --------------------------------------------

# def state_to_img(state):
#     if isinstance(state, torch.Tensor):
#         s = state.detach().cpu().numpy()
#     else:
#         s = np.array(state)

#     if s.ndim == 3:
#         s = s[0]  # (B,N,C) → (N,C)

#     if s.ndim == 2:
#         N, C = s.shape
#         img = s[:, 2 if C > 2 else 0].reshape(H, W)
#     else:
#         img = s.reshape(H, W)

#     img = img - img.min()
#     if img.max() > 0:
#         img = img / img.max()
#     img = (img * 255).astype(np.uint8)
#     img = np.stack([img] * 3, axis=-1)
#     return img

# # --------------------------------------------

# def main():
#     env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode="local")

#     policy = NeuroFuzzyActorCritic(
#         in_ch=4,
#         patch_size=PATCH_SIZE,
#         feat_dim=48,
#         fuzzy_features=12,
#         action_dim=ACTION_DIM
#     ).to(DEVICE)

#     optimz = optim.Adam(policy.parameters(), lr=LR)

#     N = H * W

#     rewards_hist = []
#     energy_hist = []
#     motion_hist = []
#     frames = []

#     for update in range(TOTAL_UPDATES):

#         # Reset environment
#         patches, coords = env.reset(B=BATCH, pA=0.5)

#         # Capture frame
#         try:
#             frames.append(state_to_img(env.current_state()))
#         except:
#             frames.append(np.zeros((H, W, 3), dtype=np.uint8))

#         # Rollout buffers
#         obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
#         rew_buf, done_buf = [], []

#         for t in range(T_STEPS):
#             flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

#             with torch.no_grad():
#                 a, logp, v, _, _ = policy.get_action_and_value(flat)

#             act = a.reshape(BATCH, N, ACTION_DIM)
#             lp = logp.reshape(BATCH, N)
#             val = v.view(BATCH, N).mean(1)

#             (patches, coords), reward, _ = env.step(act)

#             obs_buf.append(patches.cpu())
#             act_buf.append(act.cpu())
#             logp_buf.append(lp.cpu())
#             val_buf.append(val.cpu())
#             rew_buf.append(reward.cpu())
#             done_buf.append(torch.zeros_like(reward.cpu()))

#         # Bootstrap final value
#         with torch.no_grad():
#             _, _, v, _, _ = policy.get_action_and_value(
#                 patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE),
#                 deterministic=True
#             )
#         val_buf.append(v.view(BATCH, N).mean(1).cpu())

#         # Convert
#         T = T_STEPS
#         obs = torch.stack(obs_buf)               # (T,B,N,4,5,5)
#         acts = torch.stack(act_buf)              # (T,B,N,3)
#         logps = torch.stack(logp_buf)            # (T,B,N)
#         rews = torch.stack(rew_buf)              # (T,B)
#         dones = torch.stack(done_buf)            # (T,B)
#         vals = torch.stack(val_buf)              # (T+1,B)

#         adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

#         S = T * BATCH * N
#         obs_f = obs.reshape(S, 4, PATCH_SIZE, PATCH_SIZE)
#         act_f = acts.reshape(S, ACTION_DIM)
#         logp_f = logps.reshape(S)
#         ret_f = ret.unsqueeze(2).repeat(1,1,N).reshape(S)
#         adv_f = adv.unsqueeze(2).repeat(1,1,N).reshape(S)

#         # Update
#         # ppo_update(
#         #     policy,
#         #     optimz,
#         #     obs_f,
#         #     act_f,
#         #     logp_f,
#         #     ret_f,
#         #     adv_f,
#         #     clip_ratio=CLIP,
#         #     epochs=EPOCHS,
#         #     batch_size=MINI_BATCH
#         # )
# # --- normalize advantages (important for stable PPO updates) ---
#         adv_f = adv_f.clone()
#         adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

# # (optional) normalize returns to stabilize value loss — uncomment if you want
# # ret_f = ret_f.clone()
# # ret_f = (ret_f - ret_f.mean()) / (ret_f.std() + 1e-8)

# # make sure optimizer variable name is correct (original had 'optimz' which looks like a typo)
# # if your optimizer variable really is named 'optimz', either rename it to `optimizer` or replace below.
#         ppo_update(
#             policy,
#             optimz,            # << replace with `optimz` only if that's your actual variable name
#             obs_f,
#             act_f,
#             logp_f,
#             ret_f,
#             adv_f,
#             clip_ratio=CLIP,      # keep your global CLIP (0.2) or try 0.1 if updates are aggressive
#             value_coef=0.25,      # reduce value loss weight so actor gradients aren't dominated
#             entropy_coef=0.02,    # slightly larger entropy to encourage exploration
#             epochs=EPOCHS,
#             batch_size=MINI_BATCH,
#         )


        

#         # Diagnostics
#         mean_r = rews.mean().item()
#         mean_e = interfacial_energy(env.current_state()).mean().item()
#         mean_m = motion_penalty(act_f.reshape(T, BATCH, N, ACTION_DIM).to(DEVICE).transpose(1,2)).mean().item()

#         rewards_hist.append(mean_r)
#         energy_hist.append(mean_e)
#         motion_hist.append(mean_m)

#         print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion={mean_m:.4f}")

#         if update % 10 == 0:
#             torch.save(policy.state_dict(), f"checkpoints/ppo_{update:04d}.pt")

#     # ---- Save plot ----
#     plt.figure(figsize=(10,6))
#     plt.plot(rewards_hist, label="reward")
#     plt.plot(energy_hist, label="energy")
#     plt.plot(motion_hist, label="motion penalty")
#     plt.legend()
#     plt.savefig("visuals/training_metrics.png")
#     plt.close()

#     # ---- Save video ----
#     imageio.mimwrite("visuals/sort_perf.mp4", frames, fps=10)

#     print("Training done. Saved plots + video.")

# if __name__ == "__main__":
#     main()


# src/experiments/train_local_sorting.py
# ------------------------------------------------------------
# CLEAN WORKING TRAINING SCRIPT FOR LOCAL SORTING (COLAB SAFE)
# Includes:
#  - advantage normalization
#  - safer PPO hyperparameters
#  - sample-and-save 100 frames evenly across training
#  - metric export (npz) and saved plots/video
#  - small memory housekeeping for Colab
# ------------------------------------------------------------

import os
import gc
import time
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio

from src.envs.wrappers import SortingEnv
from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
from src.agents.ppo import compute_gae, ppo_update
from src.utils.metrics import interfacial_energy, motion_penalty

# ------------------ CONFIG ------------------
H = 32
W = 32
PATCH_SIZE = 5
ACTION_DIM = 3
BATCH = 1
T_STEPS = 32                # rollout length
TOTAL_UPDATES = 1000        # <<< CHANGE HERE if you want fewer

# PPO / optimization hyperparameters (tweak safely)
GAMMA = 0.99                # use 0.99 for GAE stability (original low gamma was atypical)
LAM = 0.95
EPOCHS = 3                  # epochs per update
MINI_BATCH = 256            # minibatch for ppo_update
LR = 3e-4
CLIP = 0.2

# PPO loss weights (safer defaults)
VALUE_COEF = 0.25
ENTROPY_COEF = 0.02

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# output folders
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("visuals", exist_ok=True)
os.makedirs("visuals/frames", exist_ok=True)

# how many frames to sample and save across the whole run
SAMPLE_FRAMES_N = 100

# --------------------------------------------

def state_to_img(state):
    """Convert env.current_state() to HxWx3 uint8 image."""
    if isinstance(state, torch.Tensor):
        s = state.detach().cpu().numpy()
    else:
        s = np.array(state)

    # handle batched state (B, N, C) -> take first batch
    if s.ndim == 3 and s.shape[0] == BATCH:
        s = s[0]

    # expected shapes:
    # (N, C) or (H, W) or (N,)
    if s.ndim == 2:
        N, C = s.shape
        # pick adhesion channel if present else channel 0
        ch = 2 if C > 2 else 0
        try:
            img = s[:, ch].reshape(H, W)
        except Exception:
            # fallback: try transpose
            if s.shape[1] == H * W:
                img = s.T[:, ch].reshape(H, W)
            else:
                img = np.mean(s, axis=-1).reshape(H, W)
    elif s.ndim == 1:
        if s.size == H * W:
            img = s.reshape(H, W)
        else:
            side = int(math.sqrt(s.size))
            if side * side == s.size:
                img = s.reshape(side, side)
            else:
                flat = np.zeros(H * W, dtype=s.dtype)
                length = min(s.size, H * W)
                flat[:length] = s[:length]
                img = flat.reshape(H, W)
    elif s.ndim == 2 and s.shape[0] == H and s.shape[1] == W:
        img = s
    else:
        # last resort: mean across channels then reshape/truncate/pad
        arr = np.array(s, dtype=np.float32)
        if arr.ndim >= 3:
            img = np.mean(arr, axis=-1)
            if img.shape != (H, W):
                try:
                    img = img.reshape(H, W)
                except Exception:
                    img = np.zeros((H, W), dtype=np.float32)
        else:
            img = np.zeros((H, W), dtype=np.float32)

    # normalize to 0-255
    img = img.astype(np.float32)
    img = img - np.nanmin(img)
    mx = np.nanmax(img)
    if mx > 0:
        img = img / mx
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb

# --------------------------------------------

def save_frame_sample(frames_list, total_updates, saved_folder="visuals/frames", n_samples=SAMPLE_FRAMES_N):
    """
    Save exactly n_samples frames evenly across training.
    frames_list contains frames appended in chronological order (one per update).
    """
    os.makedirs(saved_folder, exist_ok=True)
    L = len(frames_list)
    if L == 0:
        return []
    # choose indices evenly
    n = min(n_samples, L)
    indices = [int(round(i * (L - 1) / (n - 1))) if n > 1 else 0 for i in range(n)]
    saved_paths = []
    for idx in indices:
        img = frames_list[idx]
        # ensure uint8 3-channel
        try:
            arr = np.array(img).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            path = os.path.join(saved_folder, f"frame_sample_{idx:04d}.png")
            imageio.imwrite(path, arr)
            saved_paths.append(path)
        except Exception:
            # ignore single failures
            continue
    return saved_paths

# --------------------------------------------

def main():
    # housekeeping
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode="local")

    policy = NeuroFuzzyActorCritic(
        in_ch=4,
        patch_size=PATCH_SIZE,
        feat_dim=48,
        fuzzy_features=12,
        action_dim=ACTION_DIM
    ).to(DEVICE)

    optimizer = optim.Adam(policy.parameters(), lr=LR)

    N = H * W

    rewards_hist = []
    energy_hist = []
    motion_hist = []
    frames = []

    iter_start = time.time()

    for update in range(TOTAL_UPDATES):
        # small GC and cuda free to keep Colab stable
        if update % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Reset environment
        patches, coords = env.reset(B=BATCH, pA=0.5)

        # Capture frame for visualization (store as HxWx3 uint8)
        try:
            frames.append(state_to_img(env.current_state()))
        except Exception:
            frames.append(np.zeros((H, W, 3), dtype=np.uint8))

        # Rollout buffers
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []

        for t in range(T_STEPS):
            flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

            with torch.no_grad():
                a, logp, v, _, _ = policy.get_action_and_value(flat)
        # after with torch.no_grad(): a, logp, v, _, _
        mu_abs = a.abs().mean().item()
        try:
            current_std = policy.logstd.exp().mean().item()
        except Exception:
            current_std = float('nan')
        if update % 1 == 0 and t == 0:
            print(f"[ACT STATS] mu_abs={mu_abs:.4f} std={current_std:.4f}")


            act = a.reshape(BATCH, N, ACTION_DIM)
            lp = logp.reshape(BATCH, N)
            val = v.view(BATCH, N).mean(1)

            (patches, coords), reward, _ = env.step(act)

            obs_buf.append(patches.cpu())
            act_buf.append(act.cpu())
            logp_buf.append(lp.cpu())
            val_buf.append(val.cpu())
            rew_buf.append(reward.cpu())
            done_buf.append(torch.zeros_like(reward.cpu()))

            # free temp tensors
            del flat, a, logp, v, act, lp, val
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Bootstrap final value (deterministic)
        with torch.no_grad():
            final_flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
            _, _, v_final, _, _ = policy.get_action_and_value(final_flat, deterministic=True)
            val_buf.append(v_final.view(BATCH, N).mean(1).cpu())
        try:
            del final_flat, v_final
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to tensors
        T = T_STEPS
        obs = torch.stack(obs_buf)               # (T,B,N,4,5,5)
        acts = torch.stack(act_buf)              # (T,B,N,3)
        logps = torch.stack(logp_buf)            # (T,B,N)
        rews = torch.stack(rew_buf)              # (T,B)
        dones = torch.stack(done_buf)            # (T,B)
        vals = torch.stack(val_buf)              # (T+1,B)

        # compute GAE and returns
        adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

        S = T * BATCH * N
        obs_f = obs.reshape(S, 4, PATCH_SIZE, PATCH_SIZE)
        act_f = acts.reshape(S, ACTION_DIM)
        logp_f = logps.reshape(S)
        ret_f = ret.unsqueeze(2).repeat(1,1,N).reshape(S)
        adv_f = adv.unsqueeze(2).repeat(1,1,N).reshape(S)

        # normalize advantages (important for stable PPO updates)
        adv_f = adv_f.clone()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # (optional) normalize returns to stabilize value loss — uncomment if desired
        # ret_f = ret_f.clone()
        # ret_f = (ret_f - ret_f.mean()) / (ret_f.std() + 1e-8)

        # PPO update
        try:
            logs = ppo_update(
                policy=policy,
                optimizer=optimizer,
                obs_patches=obs_f,
                actions=act_f,
                logp_old=logp_f,
                returns=ret_f,
                advantages=adv_f,
                clip_ratio=CLIP,
                value_coef=VALUE_COEF,
                entropy_coef=ENTROPY_COEF,
                epochs=EPOCHS,
                batch_size=MINI_BATCH,
            )
        except TypeError:
            # fallback for older ppo_update positional signature
            try:
                logs = ppo_update(policy, optimizer, obs_f, act_f, logp_f, ret_f, adv_f, CLIP, VALUE_COEF, ENTROPY_COEF)
            except Exception as e:
                print("[WARN] ppo_update failed with error:", e)
                logs = {}

        # Diagnostics
        mean_r = rews.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        # motion_penalty expects actions shaped appropriately — compute on current rollout actions
        try:
            act_dev = act_f.reshape(T, BATCH, N, ACTION_DIM).to(DEVICE)
            mean_m = motion_penalty(act_dev.transpose(1, 2)).mean().item()
            del act_dev
        except Exception:
            mean_m = float(np.nan)

        rewards_hist.append(mean_r)
        energy_hist.append(mean_e)
        motion_hist.append(mean_m)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion={mean_m:.4f}")

        # checkpoints
        if update % 10 == 0:
            torch.save(policy.state_dict(), f"checkpoints/ppo_{update:04d}.pt")

    # --- end training loop ---

    iter_time = time.time() - iter_start
    print(f"Training finished in {iter_time/60.0:.2f} minutes")

    # ---- Save metrics arrays for later plotting/analysis ----
    np.savez_compressed("visuals/metrics.npz",
                        updates=np.arange(len(rewards_hist)),
                        rewards=np.array(rewards_hist),
                        energies=np.array(energy_hist),
                        motions=np.array(motion_hist))

    # ---- Save plot ----
    plt.figure(figsize=(10,6))
    plt.plot(rewards_hist, label="reward")
    plt.plot(energy_hist, label="energy")
    plt.plot(motion_hist, label="motion penalty")
    plt.legend()
    plt.xlabel("update")
    plt.savefig("visuals/training_metrics.png", dpi=150)
    plt.close()

    # ---- Save sampled frames (exactly SAMPLE_FRAMES_N if available) ----
    sampled_paths = save_frame_sample(frames, TOTAL_UPDATES, saved_folder="visuals/frames", n_samples=SAMPLE_FRAMES_N)
    print(f"Saved {len(sampled_paths)} sampled frames to visuals/frames/")

    # ---- Create video from sampled frames (higher quality) ----
    try:
        if len(sampled_paths) > 0:
            # ensure order
            sampled_paths = sorted(sampled_paths)
            imgs = [imageio.imread(p) for p in sampled_paths]
            imageio.mimwrite("visuals/sort_progress_sampled.mp4", imgs, fps=12)
            print("Saved sampled video → visuals/sort_progress_sampled.mp4")
        else:
            # fallback: use all frames captured (may be many)
            imageio.mimwrite("visuals/sort_perf_allframes.mp4", frames, fps=10)
            print("Saved full-frame video → visuals/sort_perf_allframes.mp4")
    except Exception as e:
        print("[WARN] Video creation failed:", e)

    print("Training done. Saved plots + video + metrics.npz")

if __name__ == "__main__":
    main()
