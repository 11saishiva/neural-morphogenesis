# """
# sanity_checks.py

# - Requires: your updated SortingEnv available as `from wrappers import SortingEnv`
# - Purpose: print distributions of sort_index after random resets (curriculum on/off),
#   and run a tiny rollout using zero actions for a basic end-to-end smoke test.
# """

# import numpy as np
# import torch
# from statistics import mean
# from pprint import pprint
# # sanity_checks.py (top)
# import os, sys

# # add src/ parent directory to sys.path so package-style imports work
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../src/envs
# SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))   # .../src
# if SRC_ROOT not in sys.path:
#     sys.path.insert(0, SRC_ROOT)


# # adjust import path/name as needed
# # from yourpackage.wrappers import SortingEnv
# from envs.wrappers import SortingEnv

# torch.set_num_threads(1)

# def reset_stats(env, trials=500, use_curriculum=None, B=1):
#     vals = []
#     for _ in range(trials):
#         obs = env.reset(B=B, pA=0.5, use_curriculum=use_curriculum)
#         # safe retrieval: compute sorting index from the internal state
#         with torch.no_grad():
#             sidx = env._sorting_index(env.state).cpu().numpy()  # shape (B,)
#         vals.extend(sidx.tolist())
#     vals = np.array(vals, dtype=float)
#     return {
#         "count": len(vals),
#         "mean": float(vals.mean()),
#         "std": float(vals.std()),
#         "min": float(vals.min()),
#         "25%": float(np.percentile(vals, 25)),
#         "50%": float(np.percentile(vals, 50)),
#         "75%": float(np.percentile(vals, 75)),
#         "max": float(vals.max()),
#     }

# def small_rollout(env, steps=10, B=2):
#     """
#     Tiny rollout using zeros as actions with shape (B, N, 1).
#     NOTE: If your DCA expects a different action-channel count, change last dim.
#     This rollout purpose: ensure step() runs end-to-end and inspect reward shape.
#     """
#     obs = env.reset(B=B, pA=0.5)
#     H, W = env.H, env.W
#     N = H * W
#     # action channels = 1 is a safe guess for a smoke test; change if needed.
#     actions = torch.zeros(B, N, 1, device=env.device)
#     rewards = []
#     infos = []
#     for t in range(steps):
#         obs, r, info = env.step(actions)
#         # r is a tensor (B,), info holds tensors moved to CPU in your code
#         rewards.append(r.cpu().numpy())
#         infos.append({k: v.numpy() if hasattr(v, "numpy") else v for k, v in info.items()})
#     rewards = np.stack(rewards, axis=0)  # (steps, B)
#     result = {
#         "rewards_mean_per_step": list(rewards.mean(axis=1)),
#         "rewards_std_per_step": list(rewards.std(axis=1)),
#         "final_sort_index": info["sort_index"],
#     }
#     return result, infos

# if __name__ == "__main__":
#     env = SortingEnv(H=64, W=64, device='cpu', obs_mode='local', steps_per_action=6)

#     print("== RESET STATS WITHOUT CURRICULUM ==")
#     stats_no_cur = reset_stats(env, trials=500, use_curriculum=False, B=1)
#     pprint(stats_no_cur)

#     print("\n== RESET STATS WITH CURRICULUM (default prob forced on) ==")
#     stats_cur = reset_stats(env, trials=500, use_curriculum=True, B=1)
#     pprint(stats_cur)

#     print("\n== SMALL ROLLOUT (zero actions) ==")
#     rollout_result, rollout_infos = small_rollout(env, steps=12, B=2)
#     pprint(rollout_result)

#     # You can inspect a sample info dict:
#     print("\nSample info keys from last step (keys):")
#     print(list(rollout_infos[-1].keys()))
#     print("\nSample metrics (interfacial_energy, motion_penalty, sort_index) from last step:")
#     print("interfacial_energy:", rollout_infos[-1]["interfacial_energy"])
#     print("motion_penalty:", rollout_infos[-1]["motion_penalty"])
#     print("sort_index:", rollout_infos[-1]["sort_index"])


import torch
import csv
from wrappers import SortingEnv   # ensure wrappers.py is in same folder or import path

# ---------------------------------------
# CONFIG
# ---------------------------------------
DEVICE = "cpu"          # or "cuda"
BATCH = 4               # 4 environments to test smoothing behaviour
STEPS = 200             # 200 diagnostic steps
LOCAL = True            # True: local mode, False: global mode
H = 64
W = 64

# ---------------------------------------
# Instantiate environment
# ---------------------------------------
env = SortingEnv(
    H=H,
    W=W,
    device=DEVICE,
    steps_per_action=2,     # small for debugging
    obs_mode="local" if LOCAL else "global",
)

obs = env.reset(B=BATCH)

# ---------------------------------------
# Prepare CSV log
# ---------------------------------------
csv_file = open("sanity_log.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "step",
    "raw_sort_idx_mean",
    "sort_ma20",
    "sort_ma50",
    "norm_pos_delta_mean",
    "reward_mean",
    "interfacial_energy",
    "motion_penalty"
])

# ---------------------------------------
# Main diagnostic loop
# ---------------------------------------
for step in range(1, STEPS + 1):

    # Zero-actions + tiny noise so dynamics move without collapsing
    if LOCAL:
        actions = torch.randn(BATCH, H*W, 3) * 0.01
    else:
        actions = torch.randn(BATCH, 3, H, W) * 0.01

    obs, reward, info = env.step(actions)

    # Extract scalars
    raw = info["raw_sort_index"].mean()
    ma20 = info["raw_sort_ma20"].mean()
    ma50 = info["raw_sort_ma50"].mean()
    npd = info["norm_pos_delta"].mean()
    rmean = info["reward_components"]["reward_mean"]
    e = info["interfacial_energy"].mean()
    m = info["motion_penalty"].mean()

    # Write to CSV
    writer.writerow([
        step,
        float(raw),
        float(ma20),
        float(ma50),
        float(npd),
        float(rmean),
        float(e),
        float(m)
    ])

    # Print occasionally
    if step % 10 == 0:
        print(f"[CHECK] step={step} raw={raw:.5f} ma20={ma20:.5f} npd={npd:.5e} reward={rmean:.5f}")

csv_file.close()
print("Saved sanity_log.csv")
