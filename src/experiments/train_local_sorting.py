# ------------------------------------------------------------
# CLEAN WORKING TRAINING SCRIPT FOR LOCAL SORTING (COLAB SAFE)
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
T_STEPS = 32 #T_STEPS = 8
TOTAL_UPDATES = 200        # <<< CHANGE HERE

GAMMA = 0.005 #GAMMA = 0.99
LAM = 0.95
EPOCHS = 1 #EPOCHS = 3
MINI_BATCH = 2048 #MINI_BATCH = 256
LR = 5e-5 #LR = 3e-4
CLIP = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

# --------------------------------------------

def state_to_img(state):
    if isinstance(state, torch.Tensor):
        s = state.detach().cpu().numpy()
    else:
        s = np.array(state)

    if s.ndim == 3:
        s = s[0]  # (B,N,C) → (N,C)

    if s.ndim == 2:
        N, C = s.shape
        img = s[:, 2 if C > 2 else 0].reshape(H, W)
    else:
        img = s.reshape(H, W)

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)
    return img

# --------------------------------------------

def main():
    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode="local")

    policy = NeuroFuzzyActorCritic(
        in_ch=4,
        patch_size=PATCH_SIZE,
        feat_dim=48,
        fuzzy_features=12,
        action_dim=ACTION_DIM
    ).to(DEVICE)

    optimz = optim.Adam(policy.parameters(), lr=LR)

    N = H * W

    rewards_hist = []
    energy_hist = []
    motion_hist = []
    frames = []

    for update in range(TOTAL_UPDATES):

        # Reset environment
        patches, coords = env.reset(B=BATCH, pA=0.5)

        # Capture frame
        try:
            frames.append(state_to_img(env.current_state()))
        except:
            frames.append(np.zeros((H, W, 3), dtype=np.uint8))

        # Rollout buffers
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []

        for t in range(T_STEPS):
            flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)

            with torch.no_grad():
                a, logp, v, _, _ = policy.get_action_and_value(flat)

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

        # Bootstrap final value
        with torch.no_grad():
            _, _, v, _, _ = policy.get_action_and_value(
                patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE),
                deterministic=True
            )
        val_buf.append(v.view(BATCH, N).mean(1).cpu())

        # Convert
        T = T_STEPS
        obs = torch.stack(obs_buf)               # (T,B,N,4,5,5)
        acts = torch.stack(act_buf)              # (T,B,N,3)
        logps = torch.stack(logp_buf)            # (T,B,N)
        rews = torch.stack(rew_buf)              # (T,B)
        dones = torch.stack(done_buf)            # (T,B)
        vals = torch.stack(val_buf)              # (T+1,B)

        adv, ret = compute_gae(rews, vals, dones, GAMMA, LAM)

        S = T * BATCH * N
        obs_f = obs.reshape(S, 4, PATCH_SIZE, PATCH_SIZE)
        act_f = acts.reshape(S, ACTION_DIM)
        logp_f = logps.reshape(S)
        ret_f = ret.unsqueeze(2).repeat(1,1,N).reshape(S)
        adv_f = adv.unsqueeze(2).repeat(1,1,N).reshape(S)

        # Update
        # ppo_update(
        #     policy,
        #     optimz,
        #     obs_f,
        #     act_f,
        #     logp_f,
        #     ret_f,
        #     adv_f,
        #     clip_ratio=CLIP,
        #     epochs=EPOCHS,
        #     batch_size=MINI_BATCH
        # )
# --- normalize advantages (important for stable PPO updates) ---
        adv_f = adv_f.clone()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

# (optional) normalize returns to stabilize value loss — uncomment if you want
# ret_f = ret_f.clone()
# ret_f = (ret_f - ret_f.mean()) / (ret_f.std() + 1e-8)

# make sure optimizer variable name is correct (original had 'optimz' which looks like a typo)
# if your optimizer variable really is named 'optimz', either rename it to `optimizer` or replace below.
        ppo_update(
            policy,
            optimz,            # << replace with `optimz` only if that's your actual variable name
            obs_f,
            act_f,
            logp_f,
            ret_f,
            adv_f,
            clip_ratio=CLIP,      # keep your global CLIP (0.2) or try 0.1 if updates are aggressive
            value_coef=0.25,      # reduce value loss weight so actor gradients aren't dominated
            entropy_coef=0.02,    # slightly larger entropy to encourage exploration
            epochs=EPOCHS,
            batch_size=MINI_BATCH,
        )


        

        # Diagnostics
        mean_r = rews.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        mean_m = motion_penalty(act_f.reshape(T, BATCH, N, ACTION_DIM).to(DEVICE).transpose(1,2)).mean().item()

        rewards_hist.append(mean_r)
        energy_hist.append(mean_e)
        motion_hist.append(mean_m)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion={mean_m:.4f}")

        if update % 10 == 0:
            torch.save(policy.state_dict(), f"checkpoints/ppo_{update:04d}.pt")

    # ---- Save plot ----
    plt.figure(figsize=(10,6))
    plt.plot(rewards_hist, label="reward")
    plt.plot(energy_hist, label="energy")
    plt.plot(motion_hist, label="motion penalty")
    plt.legend()
    plt.savefig("visuals/training_metrics.png")
    plt.close()

    # ---- Save video ----
    imageio.mimwrite("visuals/sort_perf.mp4", frames, fps=10)

    print("Training done. Saved plots + video.")

if __name__ == "__main__":
    main()
