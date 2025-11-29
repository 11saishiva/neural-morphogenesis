# scripts/diagnose_training.py
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

LOGFILE = sys.argv[1] if len(sys.argv) > 1 else "train_log.txt"
OUTDIR = "diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

# regex to match lines like: [0000] reward=-0.1500 | energy=0.0271 | motion_pen=4.6233
pattern = re.compile(r"\[\s*(\d+)\s*\].*?reward=([-\d\.eE]+)\s*\|\s*energy=([-\d\.eE]+)\s*\|\s*motion_pen=([-\d\.eE]+)")

updates, rewards, energy, motion = [], [], [], []
with open(LOGFILE, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            updates.append(int(m.group(1)))
            rewards.append(float(m.group(2)))
            energy.append(float(m.group(3)))
            motion.append(float(m.group(4)))

if not updates:
    print("No matching log lines found in", LOGFILE)
    sys.exit(1)

df = pd.DataFrame({"update": updates, "reward": rewards, "energy": energy, "motion": motion})
df = df.sort_values("update").reset_index(drop=True)
df.to_csv(os.path.join(OUTDIR, "training_metrics.csv"), index=False)
print("Saved CSV:", os.path.join(OUTDIR, "training_metrics.csv"))

# moving averages
def moving_avg(x, k=9):
    if len(x) < k: return np.array(x)
    return np.convolve(x, np.ones(k)/k, mode="same")

df["reward_ma"] = moving_avg(df["reward"].values, k=11)
df["energy_ma"] = moving_avg(df["energy"].values, k=11)
df["motion_ma"] = moving_avg(df["motion"].values, k=11)

# plots
plt.figure(figsize=(10,6))
plt.plot(df["update"], df["reward"], alpha=0.3, label="reward")
plt.plot(df["update"], df["reward_ma"], linewidth=2, label="reward (MA)")
plt.xlabel("update"); plt.ylabel("reward"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUTDIR, "reward.png"), dpi=150); plt.close()

plt.figure(figsize=(10,6))
plt.plot(df["update"], df["energy"], alpha=0.3, label="energy")
plt.plot(df["update"], df["energy_ma"], linewidth=2, label="energy (MA)")
plt.xlabel("update"); plt.ylabel("interfacial energy"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUTDIR, "energy.png"), dpi=150); plt.close()

plt.figure(figsize=(10,6))
plt.plot(df["update"], df["motion"], alpha=0.3, label="motion_pen")
plt.plot(df["update"], df["motion_ma"], linewidth=2, label="motion_pen (MA)")
plt.xlabel("update"); plt.ylabel("motion_pen"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUTDIR, "motion.png"), dpi=150); plt.close()

# correlations and slopes
def stats(x,y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    corr, p_corr = pearsonr(x,y)
    return {"slope": slope, "lin_r": r_value, "lin_p": p_value, "pearson_r": corr, "pearson_p": p_corr}

res_reward = stats(df["update"].values, df["reward"].values)
res_energy = stats(df["update"].values, df["energy"].values)
res_motion = stats(df["update"].values, df["motion"].values)
res_rew_vs_energy = stats(df["energy"].values, df["reward"].values)
res_rew_vs_motion = stats(df["motion"].values, df["reward"].values)

with open(os.path.join(OUTDIR, "diagnostics.txt"), "w") as f:
    f.write("Trend slopes (update -> metric):\n")
    f.write(f" reward slope: {res_reward['slope']:.6f}, lin_r: {res_reward['lin_r']:.4f}, p: {res_reward['lin_p']:.3e}\n")
    f.write(f" energy slope: {res_energy['slope']:.6f}, lin_r: {res_energy['lin_r']:.4f}, p: {res_energy['lin_p']:.3e}\n")
    f.write(f" motion slope: {res_motion['slope']:.6f}, lin_r: {res_motion['lin_r']:.4f}, p: {res_motion['lin_p']:.3e}\n\n")
    f.write("Reward vs energy correlation:\n")
    f.write(f" pearson_r: {res_rew_vs_energy['pearson_r']:.4f}, p: {res_rew_vs_energy['pearson_p']:.3e}\n")
    f.write("Reward vs motion correlation:\n")
    f.write(f" pearson_r: {res_rew_vs_motion['pearson_r']:.4f}, p: {res_rew_vs_motion['pearson_p']:.3e}\n")
print("Saved plots + diagnostics in", OUTDIR)
