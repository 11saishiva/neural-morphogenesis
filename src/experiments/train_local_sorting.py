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

# src/experiments/train_local_sorting.py
# ------------------------------------------------------------
# CLEAN WORKING TRAINING SCRIPT FOR LOCAL SORTING (COLAB SAFE)
# ------------------------------------------------------------
# src/experiments/train_local_sorting.py
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
T_STEPS = 8            # rollout length(was 32)
TOTAL_UPDATES = 50    # full run target

GAMMA = 0.99            # environment return discount (set sensible RL default)
LAM = 0.95
EPOCHS = 1
MINI_BATCH = 512 #was 2048 
LR = 1e-4               # safer, smaller lr
CLIP = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_CHECKPOINT_EVERY = 50     # checkpoint cadence
DETERMINISTIC_EVAL_EVERY = 100 # save a deterministic eval rollout every this many updates
DETERMINISTIC_EVAL_STEPS = 64  # steps in a deterministic evaluation rollout
MAX_SAVE_FRAMES = 200          # keep last N frames for final video

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("visuals", exist_ok=True)
# --------------------------------------------

def state_to_img(state):
    """Converts env.current_state() -> H x W x 3 uint8 image (visualization).
    Accepts either a torch.Tensor or numpy array."""
    if isinstance(state, torch.Tensor):
        s = state.detach().cpu().numpy()
    else:
        s = np.array(state)

    # reduce batch dim if present
    if s.ndim == 4 and s.shape[0] > 1:
        s = s[0]

    # Try common shapes and reduce to (H, W)
    img = None
    if s.ndim == 3:
        # (C, H, W) or (H, W, C) or (N, C) where N==H*W
        if s.shape[0] in (H, W) and s.shape[1] in (H, W):
            # likely (H, W, C) or (C, H, W)
            if s.shape[-1] == W and s.shape[-2] == H:
                img = np.mean(s, axis=-1) if s.ndim == 3 else np.mean(s, axis=0)
            else:
                # fallback to channel mean
                img = np.mean(s, axis=0) if s.shape[0] <= 4 else np.mean(s, axis=-1)
        else:
            # possibly (N, C) where N == H*W
            if s.shape[0] == H * W:
                ch = 2 if s.shape[1] > 2 else 0
                img = s[:, ch].reshape(H, W)
            else:
                img = np.mean(s, axis=-1)
    elif s.ndim == 2:
        if s.shape[0] == H and s.shape[1] == W:
            img = s
        elif s.size == H * W:
            img = s.flatten()[:H*W].reshape(H, W)
        else:
            # fallback
            flat = s.flatten()
            tmp = np.zeros(H * W, dtype=flat.dtype)
            tmp[:min(flat.size, H * W)] = flat[:min(flat.size, H * W)]
            img = tmp.reshape(H, W)
    elif s.ndim == 1:
        flat = s
        if flat.size >= H * W:
            img = flat[:H * W].reshape(H, W)
        else:
            tmp = np.zeros(H * W, dtype=flat.dtype)
            tmp[:flat.size] = flat
            img = tmp.reshape(H, W)
    else:
        img = np.full((H, W), float(np.mean(s)), dtype=np.float32)

    img = np.nan_to_num(img)
    img = img - np.min(img)
    maxv = np.max(img)
    if maxv > 0:
        img = img / maxv
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb

def deterministic_rollout_frames(env, policy, steps=64, clamp_actions=True):
    """Run a deterministic rollout (no sampling) and return a list of frames."""
    frames = []
    patches, coords = env.reset(B=1, pA=0.5)
    # infer shapes robustly
    b_p, n_p, c_p, ph, pw = patches.shape
    assert c_p == 4 and ph == PATCH_SIZE and pw == PATCH_SIZE, f"unexpected patch shape: {patches.shape}"
    for t in range(steps):
        flat = patches.contiguous().view(b_p * n_p, c_p, ph, pw).to(DEVICE)
        with torch.no_grad():
            a, logp, v, _, _ = policy.get_action_and_value(flat, deterministic=True)

        act = a.contiguous().view(b_p, n_p, ACTION_DIM)
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

    # env & model
    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode="local")

    policy = NeuroFuzzyActorCritic(
        in_ch=4,
        patch_size=PATCH_SIZE,
        feat_dim=48,
        fuzzy_features=12,
        action_dim=ACTION_DIM
    ).to(DEVICE)

    # safer initial logstd (reduce initial exploration magnitude)
    with torch.no_grad():
        # NeuroFuzzy exposes raw_logstd/logstd differently; attempt safe init
        if hasattr(policy, "raw_logstd"):
            policy.raw_logstd.data.fill_(-2.0)
        elif hasattr(policy, "logstd"):
            try:
                # if property backed by parameter this may fail — ignore if so
                policy.logstd.data.fill_(-2.0)
            except Exception:
                pass

    optimz = optim.Adam(policy.parameters(), lr=LR)

    rewards_hist = []
    energy_hist = []
    motion_hist = []
    frames = []            # rolling store of frames for final video
    eval_count = 0

    start_time = time.time()

    for update in range(TOTAL_UPDATES):
        # Reset environment & get local observation
        patches, coords = env.reset(B=BATCH, pA=0.5)

        # Capture a frame for quick visuals
        try:
            frames.append(state_to_img(env.current_state()))
        except Exception:
            frames.append(np.zeros((H, W, 3), dtype=np.uint8))
        if len(frames) > MAX_SAVE_FRAMES:
            frames.pop(0)

        # Rollout storage (kept on CPU where possible)
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []

        # rollout
        for t in range(T_STEPS):
            # infer actual patch dims and reshape robustly
            b_p, n_p, c_p, ph, pw = patches.shape
            assert c_p == 4 and ph == PATCH_SIZE and pw == PATCH_SIZE, f"unexpected patches shape: {patches.shape}"
            flat = patches.contiguous().view(b_p * n_p, c_p, ph, pw).to(DEVICE)

            with torch.no_grad():
                a, logp, v, _, _ = policy.get_action_and_value(flat)

            # diagnostics: action magnitude & policy std
            if t == 0:
                try:
                    current_std = (policy.logstd.exp().mean().item() if hasattr(policy, "logstd") else float('nan'))
                except Exception:
                    current_std = float('nan')
                mu_abs = a.abs().mean().item()
                print(f"[UPDATE {update:04d} T{t}] action_mean_abs={mu_abs:.4f} policy_std_mean={current_std:.4f}")

            act = a.contiguous().view(b_p, n_p, ACTION_DIM)
            act = act.clamp(-1.0, 1.0)

            lp = logp.contiguous().view(b_p, n_p)
            val = v.contiguous().view(b_p, n_p).mean(1)

            (patches, coords), reward, _ = env.step(act)

            # move observation tensors to CPU for buffering (keep them pinned if needed)
            obs_buf.append(patches.cpu())
            act_buf.append(act.cpu())
            logp_buf.append(lp.cpu())
            val_buf.append(val.cpu())
            rew_buf.append(reward.cpu())
            done_buf.append(torch.zeros_like(reward.cpu()))

        # bootstrap final value (deterministic)
        with torch.no_grad():
            b_p, n_p, c_p, ph, pw = patches.shape
            assert c_p == 4 and ph == PATCH_SIZE and pw == PATCH_SIZE, f"unexpected patches shape: {patches.shape}"
            flat = patches.contiguous().view(b_p * n_p, c_p, ph, pw).to(DEVICE)
            _, _, v, _, _ = policy.get_action_and_value(flat, deterministic=True)
        val_buf.append(v.contiguous().view(b_p, n_p).mean(1).cpu())

        # stack buffers (use actual shapes)
        T = T_STEPS
        obs = torch.stack(obs_buf)               # (T, B_actual, N_actual, C, p, p)
        acts = torch.stack(act_buf)              # (T, B_actual, N_actual, A)
        logps = torch.stack(logp_buf)            # (T, B_actual, N_actual)
        rews = torch.stack(rew_buf)              # (T, B_actual)
        dones = torch.stack(done_buf)            # (T, B_actual)
        vals = torch.stack(val_buf)              # (T+1, B_actual)

        # compute GAE & returns
        adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

        # flatten for ppo update - infer real sizes from obs
        T0, b_obs, n_obs, c_obs, ph, pw = obs.shape
        assert c_obs == 4 and ph == PATCH_SIZE and pw == PATCH_SIZE, f"unexpected obs shape: {obs.shape}"
        S = T0 * b_obs * n_obs

        obs_f = obs.contiguous().view(S, c_obs, ph, pw)
        act_f = acts.contiguous().view(S, ACTION_DIM)
        logp_f = logps.contiguous().view(S)
        ret_f = ret.unsqueeze(2).repeat(1,1,n_obs).contiguous().view(S)
        adv_f = adv.unsqueeze(2).repeat(1,1,n_obs).contiguous().view(S)

        # normalize advantages (important)
        adv_f = adv_f.clone()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        # PPO update with safer hyperparameters
        logs = ppo_update(
            policy,
            optimz,                 # note: your optimizer variable name is `optimz`
            obs_f,
            act_f,
            logp_f,
            ret_f,
            adv_f,
            clip_ratio=CLIP,
            value_coef=0.25,        # reduce value weight a bit
            entropy_coef=0.005,     # smaller entropy -> less random motor babbling
            epochs=EPOCHS,
            batch_size=MINI_BATCH,
        )

        # Diagnostics (compute components cleanly)
        mean_r = rews.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        try:
            # use actual b_obs/n_obs when reshaping
            act_dev = act_f.contiguous().view(T0, b_obs, n_obs, ACTION_DIM).to(DEVICE)
            mean_m = motion_penalty(act_dev.transpose(1, 2)).mean().item()
        except Exception:
            mean_m = act_f.abs().mean().item()

        rewards_hist.append(mean_r)
        energy_hist.append(mean_e)
        motion_hist.append(mean_m)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion={mean_m:.4f}")

        # checkpoint
        if update % SAVE_CHECKPOINT_EVERY == 0:
            ckpt_path = f"checkpoints/ppo_{update:04d}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

        # deterministic eval snapshot (visual check without sampling noise)
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

    # create final short video from saved frames (if any)
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
