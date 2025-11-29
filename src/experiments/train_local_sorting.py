# # src/experiments/train_local_sorting.py
# import os
# import gc
# import time
# import math
# import imageio
# import numpy as np
# import torch
# import torch.optim as optim
# import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
# from src.envs.wrappers import SortingEnv
# from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update
# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
# from src.utils.metrics import interfacial_energy, motion_penalty

# # -------------- hyperparams (tweak here) --------------
# H, W = 32, 32
# BATCH = 1
# T_STEPS = 8
# PATCH_SIZE = 5
# ACTION_DIM = 3

# GAMMA = 0.99
# LAM = 0.95
# CLIP = 0.2

# EPOCHS = 3
# MINI_BATCH = 256
# LR = 3e-4
# TOTAL_UPDATES = 1000   # <-- set this to 200/1000 etc.

# LOG_DIR = "runs/sorting_rl"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # environment / model sizes
# FEAT_DIM = 48
# FUZZY_FEATURES = 12
# N_MFS = 3
# N_RULES = 24

# # PyTorch fragmentation helper
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# # ------------------------------------------------------

# def flatten_patches(patches):
#     T, B, N, C, h, w = patches.shape
#     return patches.view(T * B * N, C, h, w)

# # ---------------- safe streamed ppo update ----------------
# def safe_ppo_update(policy, optimizer,
#                     obs_cpu, actions_cpu, logp_old_cpu, returns_cpu, advantages_cpu,
#                     clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
#                     epochs=3, batch_size=256):
#     S = obs_cpu.shape[0]
#     if S == 0:
#         return {}
#     logs = {}
#     for epoch in range(epochs):
#         perm = torch.randperm(S)
#         for start in range(0, S, batch_size):
#             idx = perm[start:start + batch_size]
#             obs_mb = obs_cpu[idx].to(DEVICE, non_blocking=True)
#             acts_mb = actions_cpu[idx].to(DEVICE, non_blocking=True)
#             logp_mb = logp_old_cpu[idx].to(DEVICE, non_blocking=True)
#             ret_mb = returns_cpu[idx].to(DEVICE, non_blocking=True)
#             adv_mb = advantages_cpu[idx].to(DEVICE, non_blocking=True)

#             # try keyword-style call first
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
#                 # fallback to positional older signature
#                 logs = ppo_update(
#                     policy, optimizer,
#                     obs_mb, acts_mb, logp_mb, ret_mb, adv_mb,
#                     clip_eps, vf_coef, ent_coef
#                 )

#             # gradient clipping guard (if optimizer step used gradients)
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)

#             del obs_mb, acts_mb, logp_mb, ret_mb, adv_mb
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#     return logs

# # ---------------- plotting helper ----------------
# def plot_metrics(updates, rewards, energies, motions, out_path="training_metrics.png"):
#     plt.figure(figsize=(10, 6))
#     plt.subplot(3,1,1)
#     plt.plot(updates, rewards, label="reward")
#     plt.ylabel("reward")
#     plt.legend()
#     plt.subplot(3,1,2)
#     plt.plot(updates, energies, label="interfacial energy")
#     plt.ylabel("energy")
#     plt.legend()
#     plt.subplot(3,1,3)
#     plt.plot(updates, motions, label="motion_pen")
#     plt.ylabel("motion_pen")
#     plt.xlabel("update")
#     plt.legend()
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f"[viz] Saved training curve → {out_path}")

# # ---------------- state -> image helper ----------------
# def state_to_image(state_tensor, H=H, W=W):
#     """
#     Convert env.current_state() tensor to HxWx3 uint8 image.
#     Handles a few possible shapes:
#       - (N, C) where N == H*W
#       - (B, N, C) -> take first batch
#       - (N, ) -> reshape to HxW
#     Uses channel index 2 (adhesion) as the main visualization channel if available,
#     otherwise uses channel 0.
#     """
#     if isinstance(state_tensor, torch.Tensor):
#         s = state_tensor.detach().cpu().numpy()
#     else:
#         s = np.array(state_tensor)

#     # if batch present, take first entry
#     if s.ndim == 3 and s.shape[0] == BATCH:
#         # shape (B, N, C) -> pick first batch
#         s = s[0]

#     # now expect s shape (N, C) or (N,)
#     if s.ndim == 2:
#         N, C = s.shape
#         if N != H * W:
#             # try transposed shapes
#             if C == H * W:
#                 s = s.T
#                 N, C = s.shape
#         # choose a reasonable channel to visualize
#         ch_idx = 2 if C > 2 else 0
#         img = s[:, ch_idx]
#         img = img.reshape(H, W)
#     elif s.ndim == 1:
#         # flat vector
#         if s.size != H * W:
#             # fallback: reshape as square if possible
#             side = int(math.sqrt(s.size))
#             if side * side == s.size:
#                 img = s.reshape(side, side)
#             else:
#                 # last resort: pad or truncate to H*W
#                 flat = np.zeros(H * W, dtype=s.dtype)
#                 length = min(s.size, H * W)
#                 flat[:length] = s[:length]
#                 img = flat.reshape(H, W)
#         else:
#             img = s.reshape(H, W)
#     elif s.ndim == 2 and s.shape[0] == H and s.shape[1] == W:
#         img = s
#     else:
#         # unknown format: reduce to grayscale with mean across dims then reshape if possible
#         img = np.mean(s, axis=-1).reshape(H, W)

#     # normalize to 0-255
#     img = np.array(img, dtype=np.float32)
#     lo, hi = np.nanmin(img), np.nanmax(img)
#     if hi - lo < 1e-6:
#         arrn = np.clip(img * 255.0, 0, 255).astype(np.uint8)
#     else:
#         arrn = ((img - lo) / (hi - lo) * 255.0).clip(0,255).astype(np.uint8)

#     rgb = np.stack([arrn, arrn, arrn], axis=-1)
#     return rgb

# # ---------------- video helper ----------------
# def make_video_from_frames(frames, out_path="sort_performance.mp4", fps=12):
#     """
#     frames: list of numpy uint8 HxWx3 images (or torch tensors).
#     We'll normalize to uint8 RGB and write with imageio.
#     """
#     if not frames:
#         raise ValueError("No frames provided to make_video_from_frames().")

#     norm_frames = []
#     for fr in frames:
#         # if raw state tensors were passed, convert
#         if isinstance(fr, torch.Tensor):
#             try:
#                 arr = fr.detach().cpu().numpy()
#             except Exception:
#                 arr = np.array(fr)
#         else:
#             arr = np.array(fr)

#         # If this appears to be a state tensor from env.current_state (N,C), convert via helper
#         # Heuristic: 2D with shape (N, C) where N == H*W or first dim equals BATCH
#         if arr.ndim >= 2 and arr.shape[0] in (H*W, BATCH):
#             try:
#                 arr = state_to_image(arr, H=H, W=W)
#             except Exception:
#                 # fallback to mean projection
#                 arr = np.mean(arr, axis=-1)
#         # If channel-first [C,H,W]
#         if arr.ndim == 3 and arr.shape[0] <= 4:
#             if arr.shape[0] <= 3:
#                 arr = np.transpose(arr, (1,2,0))
#             else:
#                 arr = np.transpose(arr[:3], (1,2,0))

#         # grayscale -> RGB
#         if arr.ndim == 2:
#             arr = np.stack([arr, arr, arr], axis=-1)
#         # single channel last dim HxWx1
#         if arr.ndim == 3 and arr.shape[2] == 1:
#             arr = np.concatenate([arr, arr, arr], axis=2)

#         # convert floats to uint8
#         if np.issubdtype(arr.dtype, np.floating):
#             lo, hi = np.nanmin(arr), np.nanmax(arr)
#             if hi - lo < 1e-6:
#                 arrn = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
#             else:
#                 arrn = ((arr - lo) / (hi - lo) * 255.0).clip(0,255).astype(np.uint8)
#         else:
#             arrn = arr.astype(np.uint8)

#         # ensure 3 channels
#         if arrn.ndim == 2:
#             arrn = np.stack([arrn,arrn,arrn], axis=-1)
#         if arrn.shape[2] == 4:
#             arrn = arrn[:,:,:3]

#         if arrn.ndim != 3 or arrn.shape[2] != 3:
#             raise ValueError("Frame could not be normalized to HxWx3 uint8 image.")

#         norm_frames.append(arrn)

#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     # use imageio's mimwrite (ffmpeg) to create mp4
#     imageio.mimwrite(out_path, norm_frames, fps=fps, macro_block_size=None)
#     print(f"[viz] Saved GIF/MP4: {out_path}")
#     return out_path

# # ---------------- main training loop ----------------
# def main():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     os.makedirs("checkpoints", exist_ok=True)
#     os.makedirs("visuals", exist_ok=True)
#     writer = SummaryWriter(LOG_DIR)

#     env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')

#     # NOTE: some versions of NeuroFuzzyActorCritic do not accept n_mfs/n_rules keyword args.
#     # Use only the common args to avoid TypeError on init.
#     policy = NeuroFuzzyActorCritic(in_ch=4, patch_size=PATCH_SIZE,
#                                    feat_dim=FEAT_DIM,
#                                    fuzzy_features=FUZZY_FEATURES,
#                                    action_dim=ACTION_DIM).to(DEVICE)

#     optimizer = optim.Adam(policy.parameters(), lr=LR)
#     N = H * W

#     # storage for plotting and optional video
#     updates = []
#     rewards_log = []
#     energies_log = []
#     motions_log = []
#     GLOBAL_FRAMES = []

#     iter_start = time.time()

#     for update in range(TOTAL_UPDATES):
#         # a) rollout (keep tensors on CPU where possible)
#         obs_patches_list, actions_list, logp_list, values_list, rewards_list, dones_list = [], [], [], [], [], []

#         obs = env.reset(B=BATCH, pA=0.5)
#         patches, coords = obs

#         # capture a frame for later visualization (convert state -> image)
#         try:
#             state = env.current_state()
#             frame_img = state_to_image(state, H=H, W=W)
#             GLOBAL_FRAMES.append(frame_img)
#         except Exception:
#             # if conversion fails, don't crash training; append a blank frame
#             GLOBAL_FRAMES.append(np.zeros((H, W, 3), dtype=np.uint8))

#         for t in range(T_STEPS):
#             flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#             with torch.no_grad():
#                 action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)

#             action_grid = action_flat.reshape(BATCH, N, ACTION_DIM)
#             logp_grid = logp_flat.reshape(BATCH, N)
#             value_grid = value_flat.view(BATCH, N).mean(1)

#             obs2, reward, info = env.step(action_grid)
#             patches, coords = obs2

#             obs_patches_list.append(patches.detach().cpu())
#             actions_list.append(action_grid.detach().cpu())
#             logp_list.append(logp_grid.detach().cpu())
#             values_list.append(value_grid.detach().cpu())
#             rewards_list.append(reward.cpu())
#             dones_list.append(torch.zeros_like(reward.cpu()))

#             # free temps
#             del flat, action_flat, logp_flat, value_flat, action_grid, logp_grid, value_grid
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         # bootstrap final value
#         flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
#         with torch.no_grad():
#             _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
#             values_list.append(vals.view(BATCH, N).mean(1).cpu())
#         del flat, vals
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         # assemble CPU tensors
#         T = T_STEPS
#         obs_patches = torch.stack(obs_patches_list)  # (T, B, N, C, h, w)
#         actions = torch.stack(actions_list)          # (T, B, N, A)
#         logps = torch.stack(logp_list)               # (T, B, N)
#         rewards = torch.stack(rewards_list)          # (T, B)
#         dones = torch.stack(dones_list)              # (T, B)
#         values = torch.stack(values_list)            # (T+1, B)

#         advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

#         S = T * BATCH * N
#         obs_flat_cpu = obs_patches.reshape(S, 4, PATCH_SIZE, PATCH_SIZE).cpu()
#         actions_flat_cpu = actions.reshape(S, ACTION_DIM).cpu()
#         logps_flat_cpu = logps.reshape(S).cpu()
#         returns_flat_cpu = returns.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()
#         advs_flat_cpu = advantages.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()

#         # streamed updates
#         safe_ppo_update(policy, optimizer,
#                         obs_flat_cpu, actions_flat_cpu, logps_flat_cpu, returns_flat_cpu, advs_flat_cpu,
#                         clip_eps=CLIP, vf_coef=0.5, ent_coef=0.01,
#                         epochs=EPOCHS, batch_size=MINI_BATCH)

#         # diagnostics
#         mean_r = rewards.mean().item()
#         mean_e = interfacial_energy(env.current_state()).mean().item()

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

#         updates.append(update)
#         rewards_log.append(mean_r)
#         energies_log.append(mean_e)
#         motions_log.append(mean_mpen)

#         print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

#         if update % 10 == 0:
#             ckpt = f"checkpoints/ppo_{update:04d}.pt"
#             torch.save(policy.state_dict(), ckpt)
#             print(f"[ckpt] saved {ckpt}")

#         # optional: keep a manageable number of frames for final video
#         if len(GLOBAL_FRAMES) > 500:
#             GLOBAL_FRAMES.pop(0)

#     # after training: produce plots & video
#     plot_metrics(updates, rewards_log, energies_log, motions_log, out_path="visuals/training_metrics.png")

#     # create a short video using sampled frames
#     try:
#         # ensure frames are valid images (H x W x 3 uint8)
#         valid_frames = []
#         for fr in GLOBAL_FRAMES:
#             if isinstance(fr, np.ndarray) and fr.ndim == 3 and fr.shape[2] == 3:
#                 valid_frames.append(fr)
#             else:
#                 try:
#                     valid_frames.append(state_to_image(fr, H=H, W=W))
#                 except Exception:
#                     valid_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
#         make_video_from_frames(valid_frames, out_path="visuals/sort_performance.mp4", fps=12)
#     except Exception as e:
#         print("[viz] Failed to make video:", e)

#     writer.close()

# if __name__ == "__main__":
#     main()


# src/experiments/train_local_sorting.py
import os
import gc
import time
import math
import imageio
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from src.envs.wrappers import SortingEnv
from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update
from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
from src.utils.metrics import interfacial_energy, motion_penalty

# -------------- hyperparams (tweak here) --------------
H, W = 32, 32
BATCH = 1
T_STEPS = 8
PATCH_SIZE = 5
ACTION_DIM = 3

GAMMA = 0.99
LAM = 0.95
CLIP = 0.2

EPOCHS = 3
MINI_BATCH = 256
LR = 3e-4
TOTAL_UPDATES = 1000   # <-- set this to 200/1000 etc.

LOG_DIR = "runs/sorting_rl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# environment / model sizes
FEAT_DIM = 48
FUZZY_FEATURES = 12
N_MFS = 3
N_RULES = 24

# PyTorch fragmentation helper
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ------------------------------------------------------

def flatten_patches(patches):
    T, B, N, C, h, w = patches.shape
    return patches.view(T * B * N, C, h, w)

# ---------------- safe streamed ppo update ----------------
def safe_ppo_update(policy, optimizer,
                    obs_cpu, actions_cpu, logp_old_cpu, returns_cpu, advantages_cpu,
                    clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
                    epochs=3, batch_size=256):
    """
    Streamed wrapper that slices the big CPU tensors into minibatches and calls ppo_update.
    Uses standardized kw names that match ppo_update().
    """
    S = int(obs_cpu.shape[0])
    if S == 0:
        return {}
    logs = {}
    for epoch in range(epochs):
        perm = torch.randperm(S)
        for start in range(0, S, batch_size):
            idx = perm[start:start + batch_size]
            obs_mb = obs_cpu[idx]
            acts_mb = actions_cpu[idx]
            logp_mb = logp_old_cpu[idx]
            ret_mb = returns_cpu[idx]
            adv_mb = advantages_cpu[idx]

            # call the stable ppo_update (it handles device placement)
            minibatch_logs = ppo_update(
                policy=policy,
                optimizer=optimizer,
                obs_patches=obs_mb,
                actions=acts_mb,
                logp_old=logp_mb,
                returns=ret_mb,
                advantages=adv_mb,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                epochs=1,
                batch_size=obs_mb.shape[0],
            )

            # aggregate logs
            for k, v in minibatch_logs.items():
                logs.setdefault(k, 0.0)
                logs[k] += v

            # free local references & clear cuda cache if available
            del obs_mb, acts_mb, logp_mb, ret_mb, adv_mb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # average aggregated logs by number of minibatches (approx)
    return logs

# ---------------- plotting helper ----------------
def plot_metrics(updates, rewards, energies, motions, out_path="visuals/training_metrics.png"):
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.plot(updates, rewards, label="reward")
    plt.ylabel("reward")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(updates, energies, label="interfacial energy")
    plt.ylabel("energy")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(updates, motions, label="motion_pen")
    plt.ylabel("motion_pen")
    plt.xlabel("update")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[viz] Saved training curve → {out_path}")

# ---------------- state -> image helper ----------------
def state_to_image(state_tensor, H=H, W=W):
    """
    Convert env.current_state() tensor to HxWx3 uint8 image.
    Handles a few possible shapes:
      - (N, C) where N == H*W
      - (B, N, C) -> take first batch
      - (N, ) -> reshape to HxW
    Uses channel index 2 (adhesion) as the main visualization channel if available,
    otherwise uses channel 0.
    """
    if isinstance(state_tensor, torch.Tensor):
        s = state_tensor.detach().cpu().numpy()
    else:
        s = np.array(state_tensor)

    # if batch present, take first entry
    if s.ndim == 3 and s.shape[0] == BATCH:
        s = s[0]

    # If it's already HxW image
    if s.ndim == 2 and s.shape[0] == H and s.shape[1] == W:
        img = s
    else:
        # now expect s shape (N, C) or (N,)
        if s.ndim == 2:
            N, C = s.shape
            if N != H * W and C == H * W:
                s = s.T
                N, C = s.shape
            ch_idx = 2 if C > 2 else 0
            img = s[:, ch_idx].reshape(H, W)
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
        else:
            # unknown format: reduce to grayscale with mean across dims then reshape if possible
            img = np.mean(s, axis=-1)
            if img.size == H * W:
                img = img.reshape(H, W)
            else:
                img = np.resize(img, (H, W))

    # normalize to 0-255
    img = np.array(img, dtype=np.float32)
    lo, hi = np.nanmin(img), np.nanmax(img)
    if hi - lo < 1e-6:
        arrn = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        arrn = ((img - lo) / (hi - lo) * 255.0).clip(0,255).astype(np.uint8)

    rgb = np.stack([arrn, arrn, arrn], axis=-1)
    return rgb

# ---------------- video helper ----------------
def make_video_from_frames(frames, out_path="visuals/sort_performance.mp4", fps=12):
    """
    frames: list of numpy uint8 HxWx3 images (or torch tensors).
    We'll normalize to uint8 RGB and write with imageio.
    """
    if not frames:
        raise ValueError("No frames provided to make_video_from_frames().")

    norm_frames = []
    for fr in frames:
        if isinstance(fr, torch.Tensor):
            try:
                arr = fr.detach().cpu().numpy()
            except Exception:
                arr = np.array(fr)
        else:
            arr = np.array(fr)

        # Heuristic: convert env state tensors to images
        if arr.ndim >= 2 and (arr.shape[0] == H * W or arr.shape[0] == BATCH):
            try:
                arr = state_to_image(arr, H=H, W=W)
            except Exception:
                arr = np.mean(arr, axis=-1)

        if arr.ndim == 3 and arr.shape[0] <= 4:
            if arr.shape[0] <= 3:
                arr = np.transpose(arr, (1,2,0))
            else:
                arr = np.transpose(arr[:3], (1,2,0))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=2)

        if np.issubdtype(arr.dtype, np.floating):
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            if hi - lo < 1e-6:
                arrn = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arrn = ((arr - lo) / (hi - lo) * 255.0).clip(0,255).astype(np.uint8)
        else:
            arrn = arr.astype(np.uint8)

        if arrn.ndim == 2:
            arrn = np.stack([arrn,arrn,arrn], axis=-1)
        if arrn.shape[2] == 4:
            arrn = arrn[:,:,:3]

        if arrn.ndim != 3 or arrn.shape[2] != 3:
            # fallback blank frame instead of crashing
            arrn = np.zeros((H, W, 3), dtype=np.uint8)

        norm_frames.append(arrn)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, norm_frames, fps=fps, macro_block_size=None)
    print(f"[viz] Saved GIF/MP4: {out_path}")
    return out_path

# ---------------- main training loop ----------------
def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visuals", exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')

    policy = NeuroFuzzyActorCritic(in_ch=4, patch_size=PATCH_SIZE,
                                   feat_dim=FEAT_DIM,
                                   fuzzy_features=FUZZY_FEATURES,
                                   action_dim=ACTION_DIM).to(DEVICE)

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    N = H * W

    # storage for plotting and optional video
    updates = []
    rewards_log = []
    energies_log = []
    motions_log = []
    GLOBAL_FRAMES = []

    iter_start = time.time()

    for update in range(TOTAL_UPDATES):
        # a) rollout (keep tensors on CPU where possible)
        obs_patches_list, actions_list, logp_list, values_list, rewards_list, dones_list = [], [], [], [], [], []

        obs = env.reset(B=BATCH, pA=0.5)
        patches, coords = obs

        # capture a frame for later visualization (convert state -> image)
        try:
            state = env.current_state()
            frame_img = state_to_image(state, H=H, W=W)
            GLOBAL_FRAMES.append(frame_img)
        except Exception:
            GLOBAL_FRAMES.append(np.zeros((H, W, 3), dtype=np.uint8))

        for t in range(T_STEPS):
            flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
            with torch.no_grad():
                action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)

            action_grid = action_flat.reshape(BATCH, N, ACTION_DIM)
            logp_grid = logp_flat.reshape(BATCH, N)
            value_grid = value_flat.view(BATCH, N).mean(1)

            obs2, reward, info = env.step(action_grid)
            patches, coords = obs2

            obs_patches_list.append(patches.detach().cpu())
            actions_list.append(action_grid.detach().cpu())
            logp_list.append(logp_grid.detach().cpu())
            values_list.append(value_grid.detach().cpu())
            rewards_list.append(reward.cpu())
            dones_list.append(torch.zeros_like(reward.cpu()))

            del flat, action_flat, logp_flat, value_flat, action_grid, logp_grid, value_grid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # bootstrap final value
        flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        with torch.no_grad():
            _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
            values_list.append(vals.view(BATCH, N).mean(1).cpu())
        del flat, vals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # assemble CPU tensors
        T = T_STEPS
        obs_patches = torch.stack(obs_patches_list)  # (T, B, N, C, h, w)
        actions = torch.stack(actions_list)          # (T, B, N, A)
        logps = torch.stack(logp_list)               # (T, B, N)
        rewards = torch.stack(rewards_list)          # (T, B)
        dones = torch.stack(dones_list)              # (T, B)
        values = torch.stack(values_list)            # (T+1, B)

        advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

        S = T * BATCH * N
        obs_flat_cpu = obs_patches.reshape(S, 4, PATCH_SIZE, PATCH_SIZE).cpu()
        actions_flat_cpu = actions.reshape(S, ACTION_DIM).cpu()
        logps_flat_cpu = logps.reshape(S).cpu()
        returns_flat_cpu = returns.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()
        advs_flat_cpu = advantages.unsqueeze(2).repeat(1, 1, N).reshape(S).cpu()

        # streamed updates
        safe_ppo_update(policy, optimizer,
                        obs_flat_cpu, actions_flat_cpu, logps_flat_cpu, returns_flat_cpu, advs_flat_cpu,
                        clip_ratio=CLIP, value_coef=0.5, entropy_coef=0.01,
                        epochs=EPOCHS, batch_size=MINI_BATCH)

        # diagnostics
        mean_r = rewards.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()

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

        updates.append(update)
        rewards_log.append(mean_r)
        energies_log.append(mean_e)
        motions_log.append(mean_mpen)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

        if update % 10 == 0:
            ckpt = f"checkpoints/ppo_{update:04d}.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"[ckpt] saved {ckpt}")

        if len(GLOBAL_FRAMES) > 500:
            GLOBAL_FRAMES.pop(0)

    # after training: produce plots & video
    plot_metrics(updates, rewards_log, energies_log, motions_log, out_path="visuals/training_metrics.png")

    try:
        valid_frames = []
        for fr in GLOBAL_FRAMES:
            if isinstance(fr, np.ndarray) and fr.ndim == 3 and fr.shape[2] == 3:
                valid_frames.append(fr)
            else:
                try:
                    valid_frames.append(state_to_image(fr, H=H, W=W))
                except Exception:
                    valid_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
        make_video_from_frames(valid_frames, out_path="visuals/sort_performance.mp4", fps=12)
    except Exception as e:
        print("[viz] Failed to make video:", e)

    writer.close()

if __name__ == "__main__":
    main()
