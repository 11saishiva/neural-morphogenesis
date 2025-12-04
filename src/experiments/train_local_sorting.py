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
#         img = s[:, 2 if C > 2 else 0].contiguous().reshape(H, W)
#     else:
#         img = s.contiguous().reshape(H, W)

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
#             flat = patches.contiguous().reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

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
#         mean_m = motion_penalty(act_f.contiguous().reshape(T, BATCH, N, ACTION_DIM).to(DEVICE).transpose(1,2)).mean().item()

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

# src/experiments/train_local_sorting.py
# ------------------------------------------------------------
# CLEAN WORKING TRAINING SCRIPT FOR LOCAL SORTING (COLAB SAFE)
# - keeps optimizer variable name `optimz` as requested
# - normalizes advantages
# - safer PPO hyperparameters (smaller value_coef, smaller entropy_coef)
# - initial logstd reduced to limit early large actions
# - optional action clamping before env.step
# - action diagnostics and deterministic eval snapshots
# ------------------------------------------------------------
import os
import gc
import time
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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
T_STEPS = 8            # rollout length
TOTAL_UPDATES = 50

GAMMA = 0.99
LAM = 0.95
EPOCHS = 1
MINI_BATCH = 512
LR = 1e-4
CLIP = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_CHECKPOINT_EVERY = 50
DETERMINISTIC_EVAL_EVERY = 100
DETERMINISTIC_EVAL_STEPS = 64
MAX_SAVE_FRAMES = 200

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("visuals", exist_ok=True)
# --------------------------------------------

def state_to_img(state):
    """Converts env.current_state() -> H x W x 3 uint8 image (visualization)."""
    if isinstance(state, torch.Tensor):
        s = state.detach().cpu().numpy()
    else:
        s = np.array(state)

    # prefer first batch if present
    if s.ndim == 4:
        s = s[0]

    # try to produce a single-channel image and scale
    if s.ndim == 3:
        # possible shapes: (C,H,W) or (H,W,C) or (N,C) etc.
        if s.shape[0] <= 6 and s.shape[1] == H and s.shape[2] == W:
            img = np.mean(s, axis=0)
        elif s.shape[-1] == 3 and s.shape[0] == H and s.shape[1] == W:
            img = s
        else:
            # fallback: mean across channels and reshape if needed
            img = np.mean(s, axis=-1)
            if img.size != H * W:
                img = np.resize(img, (H, W))
    elif s.ndim == 2:
        img = s
    elif s.ndim == 1:
        if s.size == H * W:
            img = s.reshape(H, W)
        else:
            img = np.resize(s, (H, W))
    else:
        img = np.zeros((H, W), dtype=np.float32)

    img = np.nan_to_num(img)
    img = img - np.min(img)
    maxv = np.max(img)
    if maxv > 0:
        img = img / maxv
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb

def deterministic_rollout_frames(env, policy, steps=64, clamp_actions=True):
    frames = []
    patches, coords = env.reset(B=1, pA=0.5)
    N_local = env.H * env.W
    for t in range(steps):
        # ensure channels match model expectation (first 4 channels)
        patches_ch4 = patches[:, :, :4, :, :].contiguous()
        flat = patches_ch4.reshape(1 * N_local, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        with torch.no_grad():
            a, logp, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)

        act = a.contiguous().reshape(1, N_local, ACTION_DIM)
        if clamp_actions:
            act = act.clamp(-1.0, 1.0)
        (patches, coords), reward, _ = env.step(act)
        try:
            frames.append(state_to_img(env.current_state()))
        except Exception:
            frames.append(np.zeros((H, W, 3), dtype=np.uint8))
    return frames

def main():
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

    # safer initial logstd
    with torch.no_grad():
        if hasattr(policy, "raw_logstd"):
            policy.raw_logstd.data.fill_(-2.0)
            print(f"[INIT] set policy.raw_logstd -> mean={policy.raw_logstd.data.mean().item():.3f}")
        elif hasattr(policy, "logstd"):
            try:
                print(f"[INIT] policy.logstd mean={policy.logstd.mean().item():.3f}")
            except Exception:
                print("[INIT] policy logstd init read failed")

    optimz = optim.Adam(policy.parameters(), lr=LR)

    N = H * W

    rewards_hist = []
    energy_hist = []
    motion_hist = []
    frames = []
    eval_count = 0

    start_time = time.time()

    for update in range(TOTAL_UPDATES):
        # quick param sanity check
        for p in policy.parameters():
            if not torch.isfinite(p).all():
                print(f"[TRAIN] NaN/Inf found in params at update {update}; aborting.")
                return

        patches, coords = env.reset(B=BATCH, pA=0.5)

        try:
            frames.append(state_to_img(env.current_state()))
        except Exception:
            frames.append(np.zeros((H, W, 3), dtype=np.uint8))
        if len(frames) > MAX_SAVE_FRAMES:
            frames.pop(0)

        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []

        for t in range(T_STEPS):
            # patches shape: (B, N, C, p, p)
            Bp, Np, Cp, ph, pw = patches.shape
            # defensive: pick first 4 channels (model expects 4)
            if Cp < 4:
                raise RuntimeError(f"patch channels <4 ({Cp}) — model expects at least 4")
            patches_ch4 = patches[:, :, :4, :, :].contiguous()
            flat = patches_ch4.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

            with torch.no_grad():
                a, logp, v, _, _ = policy.get_action_and_value(flat)

            if t == 0:
                try:
                    if hasattr(policy, "raw_logstd"):
                        safe_std = F.softplus(policy.raw_logstd) + 1e-6
                        safe_std = torch.clamp(safe_std, min=1e-6, max=20.0)
                        current_std = float(safe_std.mean().item())
                    else:
                        current_std = float(policy.logstd.exp().mean().item())
                except Exception:
                    current_std = float('nan')
                mu_abs = a.abs().mean().item()
                print(f"[UPDATE {update:04d} T{t}] action_mean_abs={mu_abs:.4f} policy_std_mean={current_std:.4f}")

            act = a.contiguous().reshape(BATCH, N, ACTION_DIM)
            act = act.clamp(-1.0, 1.0)

            lp = logp.contiguous().reshape(BATCH, N)
            val = v.contiguous().reshape(BATCH, N).mean(1)

            (patches, coords), reward, _ = env.step(act)

            obs_buf.append(patches.cpu())
            act_buf.append(act.cpu())
            logp_buf.append(lp.cpu())
            val_buf.append(val.cpu())
            rew_buf.append(reward.cpu())
            done_buf.append(torch.zeros_like(reward.cpu()))

        # bootstrap final value
        with torch.no_grad():
            patches_ch4 = patches[:, :, :4, :, :].contiguous()
            flat = patches_ch4.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
            _, _, v, _, _ = policy.get_action_and_value(flat, deterministic=True)
        val_buf.append(v.contiguous().reshape(BATCH, N).mean(1).cpu())

        # stack buffers
        T = T_STEPS
        obs = torch.stack(obs_buf)               # (T,B,N,C,p,p)
        acts = torch.stack(act_buf)
        logps = torch.stack(logp_buf)
        rews = torch.stack(rew_buf)
        dones = torch.stack(done_buf)
        vals = torch.stack(val_buf)

        # Defensive channel selection BEFORE flattening:
        # obs shape: (T, B, N, C, p, p)
        if obs.shape[3] < 4:
            raise RuntimeError(f"Observed patches have <4 channels: {obs.shape}")
        obs = obs[:, :, :, :4, :, :].contiguous()

        # compute GAE & returns
        adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

        # flatten for ppo update
        T_actual, B_actual, N_actual = obs.shape[0], obs.shape[1], obs.shape[2]
        S = T_actual * B_actual * N_actual  # defensive, computed from actual shapes
        obs_f = obs.reshape(S, 4, PATCH_SIZE, PATCH_SIZE)
        act_f = acts.contiguous().reshape(S, ACTION_DIM)
        logp_f = logps.contiguous().reshape(S)
        ret_f = ret.unsqueeze(2).repeat(1,1,N_actual).contiguous().reshape(S)
        adv_f = adv.unsqueeze(2).repeat(1,1,N_actual).contiguous().reshape(S)

        # normalize advantages
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # PPO update
        logs = ppo_update(
            policy,
            optimz,
            obs_f,
            act_f,
            logp_f,
            ret_f,
            adv_f,
            clip_ratio=CLIP,
            value_coef=0.25,
            entropy_coef=0.005,
            epochs=EPOCHS,
            batch_size=MINI_BATCH,
        )

        # Diagnostics
        mean_r = rews.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        try:
            act_dev = act_f.contiguous().reshape(T_actual, BATCH, N_actual, ACTION_DIM).to(DEVICE)
            mean_m = motion_penalty(act_dev.transpose(1, 2)).mean().item()
        except Exception:
            mean_m = act_f.abs().mean().item()

        rewards_hist.append(mean_r)
        energy_hist.append(mean_e)
        motion_hist.append(mean_m)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion={mean_m:.4f}")

        if update % SAVE_CHECKPOINT_EVERY == 0:
            ckpt_path = f"checkpoints/ppo_{update:04d}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

        if (update + 1) % DETERMINISTIC_EVAL_EVERY == 0:
            eval_frames = deterministic_rollout_frames(env, policy, steps=DETERMINISTIC_EVAL_STEPS, clamp_actions=True)
            eval_fn = f"visuals/eval_{eval_count:03d}.mp4"
            try:
                imageio.mimwrite(eval_fn, eval_frames, fps=12)
                print(f"[eval] saved deterministic eval video -> {eval_fn}")
            except Exception as e:
                print("[eval] failed to save eval video:", e)
            eval_count += 1

    # plotting
    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.plot(rewards_hist, label="reward")
    plt.ylabel("reward")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(energy_hist, label="energy")
    plt.ylabel("energy")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(motion_hist, label="motion penalty")
    plt.ylabel("motion")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/training_metrics.png")
    plt.close()

    try:
        if len(frames) == 0:
            print("[viz] no frames recorded.")
        else:
            imageio.mimwrite("visuals/sort_perf.mp4", frames, fps=10)
            print("[viz] saved visuals/sort_perf.mp4")
    except Exception as e:
        print("[viz] failed to write final video:", e)

    total_time = time.time() - start_time
    print(f"Training done ({TOTAL_UPDATES} updates). Time elapsed: {total_time/60:.2f} minutes.")
    print("Saved plots + video to visuals/ and checkpoints/")

if __name__ == "__main__":
    main()
