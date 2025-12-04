# import os
# import gc
# import time
# import math
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
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
# T_STEPS = 32            # rollout length
# TOTAL_UPDATES = 500

# GAMMA = 0.005
# LAM = 0.95
# EPOCHS = 1
# MINI_BATCH = 512
# LR = 1e-4
# CLIP = 0.1

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAVE_CHECKPOINT_EVERY = 50
# DETERMINISTIC_EVAL_EVERY = 10
# DETERMINISTIC_EVAL_STEPS = 64
# MAX_SAVE_FRAMES = 200

# os.makedirs("checkpoints", exist_ok=True)
# os.makedirs("visuals", exist_ok=True)
# # --------------------------------------------

# def state_to_img(state):
#     """Converts env.current_state() -> H x W x 3 uint8 image (visualization)."""
#     if isinstance(state, torch.Tensor):
#         s = state.detach().cpu().numpy()
#     else:
#         s = np.array(state)

#     # prefer first batch if present
#     if s.ndim == 4:
#         s = s[0]

#     # try to produce a single-channel image and scale
#     if s.ndim == 3:
#         # possible shapes: (C,H,W) or (H,W,C) or (N,C) etc.
#         if s.shape[0] <= 6 and s.shape[1] == H and s.shape[2] == W:
#             img = np.mean(s, axis=0)
#         elif s.shape[-1] == 3 and s.shape[0] == H and s.shape[1] == W:
#             img = s
#         else:
#             # fallback: mean across channels and reshape if needed
#             img = np.mean(s, axis=-1)
#             if img.size != H * W:
#                 img = np.resize(img, (H, W))
#     elif s.ndim == 2:
#         img = s
#     elif s.ndim == 1:
#         if s.size == H * W:
#             img = s.reshape(H, W)
#         else:
#             img = np.resize(s, (H, W))
#     else:
#         img = np.zeros((H, W), dtype=np.float32)

#     img = np.nan_to_num(img)
#     img = img - np.min(img)
#     maxv = np.max(img)
#     if maxv > 0:
#         img = img / maxv
#     img = (img * 255.0).astype(np.uint8)
#     rgb = np.stack([img, img, img], axis=-1)
#     return rgb

# def deterministic_rollout_frames(env, policy, steps=64, clamp_actions=True):
#     frames = []
#     patches, coords = env.reset(B=1, pA=0.5)
#     N_local = env.H * env.W
#     for t in range(steps):
#         # ensure channels match model expectation (first 4 channels)
#         patches_ch4 = patches[:, :, :4, :, :].contiguous()
#         flat = patches_ch4.reshape(1 * N_local, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#         with torch.no_grad():
#             a, logp, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)

#         act = a.contiguous().reshape(1, N_local, ACTION_DIM)
#         if clamp_actions:
#             act = act.clamp(-1.0, 1.0)
#         (patches, coords), reward, _ = env.step(act)
#         try:
#             frames.append(state_to_img(env.current_state()))
#         except Exception:
#             frames.append(np.zeros((H, W, 3), dtype=np.uint8))
#     return frames

# def main():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode="local")

#     policy = NeuroFuzzyActorCritic(
#         in_ch=4,
#         patch_size=PATCH_SIZE,
#         feat_dim=48,
#         fuzzy_features=12,
#         action_dim=ACTION_DIM
#     ).to(DEVICE)

#     # safer initial logstd
#     with torch.no_grad():
#         if hasattr(policy, "raw_logstd"):
#             policy.raw_logstd.data.fill_(-2.0)
#             print(f"[INIT] set policy.raw_logstd -> mean={policy.raw_logstd.data.mean().item():.3f}")
#         elif hasattr(policy, "logstd"):
#             try:
#                 print(f"[INIT] policy.logstd mean={policy.logstd.mean().item():.3f}")
#             except Exception:
#                 print("[INIT] policy logstd init read failed")

#     optimz = optim.Adam(policy.parameters(), lr=LR)

#     N = H * W

#     rewards_hist = []
#     energy_hist = []
#     motion_hist = []
#     frames = []
#     eval_count = 0

#     start_time = time.time()

#     for update in range(TOTAL_UPDATES):
#         # quick param sanity check
#         for p in policy.parameters():
#             if not torch.isfinite(p).all():
#                 print(f"[TRAIN] NaN/Inf found in params at update {update}; aborting.")
#                 return

#         patches, coords = env.reset(B=BATCH, pA=0.5)

#         try:
#             frames.append(state_to_img(env.current_state()))
#         except Exception:
#             frames.append(np.zeros((H, W, 3), dtype=np.uint8))
#         if len(frames) > MAX_SAVE_FRAMES:
#             frames.pop(0)

#         obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
#         rew_buf, done_buf = [], []

#         for t in range(T_STEPS):
#             # patches shape: (B, N, C, p, p)
#             Bp, Np, Cp, ph, pw = patches.shape
#             # defensive: pick first 4 channels (model expects 4)
#             if Cp < 4:
#                 raise RuntimeError(f"patch channels <4 ({Cp}) — model expects at least 4")
#             patches_ch4 = patches[:, :, :4, :, :].contiguous()
#             flat = patches_ch4.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

#             with torch.no_grad():
#                 a, logp, v, _, _ = policy.get_action_and_value(flat)

#             if t == 0:
#                 try:
#                     if hasattr(policy, "raw_logstd"):
#                         safe_std = F.softplus(policy.raw_logstd) + 1e-6
#                         safe_std = torch.clamp(safe_std, min=1e-6, max=20.0)
#                         current_std = float(safe_std.mean().item())
#                     else:
#                         current_std = float(policy.logstd.exp().mean().item())
#                 except Exception:
#                     current_std = float('nan')
#                 mu_abs = a.abs().mean().item()
#                 print(f"[UPDATE {update:04d} T{t}] action_mean_abs={mu_abs:.4f} policy_std_mean={current_std:.4f}")

#             act = a.contiguous().reshape(BATCH, N, ACTION_DIM)
#             act = act.clamp(-1.0, 1.0)

#             lp = logp.contiguous().reshape(BATCH, N)
#             val = v.contiguous().reshape(BATCH, N).mean(1)

#             (patches, coords), reward, _ = env.step(act)

#             obs_buf.append(patches.cpu())
#             act_buf.append(act.cpu())
#             logp_buf.append(lp.cpu())
#             val_buf.append(val.cpu())
#             rew_buf.append(reward.cpu())
#             done_buf.append(torch.zeros_like(reward.cpu()))

#         # bootstrap final value
#         with torch.no_grad():
#             patches_ch4 = patches[:, :, :4, :, :].contiguous()
#             flat = patches_ch4.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#             _, _, v, _, _ = policy.get_action_and_value(flat, deterministic=True)
#         val_buf.append(v.contiguous().reshape(BATCH, N).mean(1).cpu())

#         # stack buffers
#         T = T_STEPS
#         obs = torch.stack(obs_buf)               # (T,B,N,C,p,p)
#         acts = torch.stack(act_buf)
#         logps = torch.stack(logp_buf)
#         rews = torch.stack(rew_buf)
#         dones = torch.stack(done_buf)
#         vals = torch.stack(val_buf)

#         # Defensive channel selection BEFORE flattening:
#         # obs shape: (T, B, N, C, p, p)
#         if obs.shape[3] < 4:
#             raise RuntimeError(f"Observed patches have <4 channels: {obs.shape}")
#         obs = obs[:, :, :, :4, :, :].contiguous()

#         # compute GAE & returns
#         adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

#         # flatten for ppo update
#         T_actual, B_actual, N_actual = obs.shape[0], obs.shape[1], obs.shape[2]
#         S = T_actual * B_actual * N_actual  # defensive, computed from actual shapes
#         obs_f = obs.reshape(S, 4, PATCH_SIZE, PATCH_SIZE)
#         act_f = acts.contiguous().reshape(S, ACTION_DIM)
#         logp_f = logps.contiguous().reshape(S)
#         ret_f = ret.unsqueeze(2).repeat(1,1,N_actual).contiguous().reshape(S)
#         adv_f = adv.unsqueeze(2).repeat(1,1,N_actual).contiguous().reshape(S)

#         # normalize advantages
#         adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

#         # PPO update
#         logs = ppo_update(
#             policy,
#             optimz,
#             obs_f,
#             act_f,
#             logp_f,
#             ret_f,
#             adv_f,
#             clip_ratio=CLIP,
#             value_coef=0.25,
#             entropy_coef=0.005,
#             epochs=EPOCHS,
#             batch_size=MINI_BATCH,
#         )

#         # Diagnostics
#         mean_r = rews.mean().item()
#         mean_e = interfacial_energy(env.current_state()).mean().item()
#         try:
#             act_dev = act_f.contiguous().reshape(T_actual, BATCH, N_actual, ACTION_DIM).to(DEVICE)
#             mean_m = motion_penalty(act_dev.transpose(1, 2)).mean().item()
#         except Exception:
#             mean_m = act_f.abs().mean().item()

#         rewards_hist.append(mean_r)
#         energy_hist.append(mean_e)
#         motion_hist.append(mean_m)
#         sorting_idx = env._sorting_index(env.current_state()).mean().item()

#         print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion={mean_m:.4f} | sort_idx={sorting_idx:.6f}")

#         if update % SAVE_CHECKPOINT_EVERY == 0:
#             ckpt_path = f"checkpoints/ppo_{update:04d}.pt"
#             torch.save(policy.state_dict(), ckpt_path)
#             print(f"[ckpt] saved {ckpt_path}")

#         if (update + 1) % DETERMINISTIC_EVAL_EVERY == 0:
#             eval_frames = deterministic_rollout_frames(env, policy, steps=DETERMINISTIC_EVAL_STEPS, clamp_actions=True)
#             eval_fn = f"visuals/eval_{eval_count:03d}.mp4"
#             try:
#                 imageio.mimwrite(eval_fn, eval_frames, fps=12)
#                 print(f"[eval] saved deterministic eval video -> {eval_fn}")
#             except Exception as e:
#                 print("[eval] failed to save eval video:", e)
#             eval_count += 1

#     # plotting
#     plt.figure(figsize=(10,6))
#     plt.subplot(3,1,1)
#     plt.plot(rewards_hist, label="reward")
#     plt.ylabel("reward")
#     plt.legend()
#     plt.subplot(3,1,2)
#     plt.plot(energy_hist, label="energy")
#     plt.ylabel("energy")
#     plt.legend()
#     plt.subplot(3,1,3)
#     plt.plot(motion_hist, label="motion penalty")
#     plt.ylabel("motion")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("visuals/training_metrics.png")
#     plt.close()

#     try:
#         if len(frames) == 0:
#             print("[viz] no frames recorded.")
#         else:
#             imageio.mimwrite("visuals/sort_perf.mp4", frames, fps=10)
#             print("[viz] saved visuals/sort_perf.mp4")
#     except Exception as e:
#         print("[viz] failed to write final video:", e)

#     total_time = time.time() - start_time
#     print(f"Training done ({TOTAL_UPDATES} updates). Time elapsed: {total_time/60:.2f} minutes.")
#     print("Saved plots + video to visuals/ and checkpoints/")

# if __name__ == "__main__":
#     main()

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

# ------------------ CONFIG (high-impact edits) ------------------
H = 32
W = 32
PATCH_SIZE = 5
ACTION_DIM = 3
BATCH = 1

# Shorter rollouts produce more frequent updates and more stable estimates for small problems.
T_STEPS = 8            # was 32 -> lowered to improve learning signal frequency
TOTAL_UPDATES = 500

# Use standard RL discount (0.99) so episodic returns are meaningful.
GAMMA = 0.99           # changed from 0.005 (was effectively killing returns)
LAM = 0.95
EPOCHS = 3             # small amount of policy epochs per update
MINI_BATCH = 512
LR = 3e-4              # slightly larger learning rate typical for PPO
CLIP = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_CHECKPOINT_EVERY = 50
DETERMINISTIC_EVAL_EVERY = 10
DETERMINISTIC_EVAL_STEPS = 64
MAX_SAVE_FRAMES = 200

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("visuals", exist_ok=True)
# --------------------------------------------

class RewardNormalizer:
    """Simple running mean/std normalizer for scalar rewards (per-batch)."""
    def __init__(self, eps=1e-8, alpha=0.99):
        self.eps = eps
        self.alpha = alpha
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0.0

    def update(self, x):
        # x: numpy scalar or array
        x = float(np.mean(x))
        if self.count == 0:
            self.running_mean = x
            self.running_var = 1.0
            self.count = 1.0
            return 0.0
        # exponential moving estimates
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * x
        # approximate var update with squared deviation (stable enough)
        diff = x - self.running_mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * (diff * diff)
        self.count += 1.0
        return (x - self.running_mean) / (math.sqrt(self.running_var) + self.eps)

reward_normalizer = RewardNormalizer()

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

    # safer initial logstd (handle both possible param names)
    with torch.no_grad():
        if hasattr(policy, "raw_logstd"):
            policy.raw_logstd.data.fill_(-2.0)
            print(f"[INIT] set policy.raw_logstd -> mean={policy.raw_logstd.data.mean().item():.3f}")
        elif hasattr(policy, "logstd"):
            try:
                policy.logstd.data.fill_(-2.0)
                print(f"[INIT] set policy.logstd -> mean={policy.logstd.data.mean().item():.3f}")
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

            (patches, coords), reward, info = env.step(act)

            # reward is (B,)
            # normalize the scalar reward to stabilize GAE
            rew_np = reward.cpu().numpy()
            normed = reward_normalizer.update(rew_np)
            # use normalized reward for learning (but keep original info available)
            rew_buf.append(torch.tensor(rew_np, device="cpu", dtype=torch.float32))
            done_buf.append(torch.zeros_like(reward.cpu()))
            obs_buf.append(patches.cpu())
            act_buf.append(act.cpu())
            logp_buf.append(lp.cpu())
            val_buf.append(val.cpu())

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
        if obs.shape[3] < 4:
            raise RuntimeError(f"Observed patches have <4 channels: {obs.shape}")
        obs = obs[:, :, :, :4, :, :].contiguous()

        # compute GAE & returns; note compute_gae expects rewards and values on same device (cpu here)
        adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

        # flatten for ppo update (compute from actual shapes)
        T_actual, B_actual, N_actual = obs.shape[0], obs.shape[1], obs.shape[2]
        S = T_actual * B_actual * N_actual
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
            entropy_coef=0.01,   # slightly higher entropy to encourage exploration
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
        sorting_idx = env._sorting_index(env.current_state()).mean().item()

        # print normalized reward estimate from normalizer (for debugging)
        print(f"[{update:04d}] reward={mean_r:.6f} (norm estimate={reward_normalizer.running_mean:.6f}/{math.sqrt(reward_normalizer.running_var):.6f}) | "
              f"energy={mean_e:.6f} | motion={mean_m:.6f} | sort_idx={sorting_idx:.6f}")

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
