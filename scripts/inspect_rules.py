# scripts/inspect_rules.py
import os
import torch
from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
from src.envs.wrappers import SortingEnv
from src.utils.interpret_rules import (
    print_readable_rules,
    plot_membership_functions,
    correlate_fuzzy_features,
    save_rule_report,
)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("[inspect] device:", device)

    # instantiate policy to match what you trained
    policy = NeuroFuzzyActorCritic(
        in_ch=4, patch_size=5, feat_dim=128,
        fuzzy_features=16, n_mfs=3, n_rules=32, action_dim=3
    ).to(device)

    # try to load the checkpoint if it exists (best-effort)
    ckpt_paths = ["checkpoints/ppo_0000.pt", "checkpoints/ppo_0050.pt", "checkpoints/ppo_0100.pt"]
    loaded = False
    for p in ckpt_paths:
        if os.path.exists(p):
            try:
                policy.load_state_dict(torch.load(p, map_location=device), strict=False)
                print(f"[inspect] Loaded checkpoint: {p}")
                loaded = True
                break
            except Exception as e:
                print(f"[inspect] Failed to load {p}: {e}")
    if not loaded:
        print("[inspect] No checkpoint loaded — using randomly initialized policy (still fine for testing)")

    # create environment for correlation/patch sampling (obs_mode local)
    env = SortingEnv(H=64, W=64, device=device, obs_mode='local')

    # 1) Print top rules (text)
    print("\n--- Top readable rules ---\n")
    print_readable_rules(policy, feature_names=None, mf_names=['Low','Mid','High'], topk=12)

    # 2) Plot membership functions for first few features
    print("\n--- plotting membership functions for features 0..4 ---\n")
    for fi in range(5):
        plot_membership_functions(policy, feature_idx=fi, outdir="visuals/rule_plots")

    # 3) Correlate fuzzy features with interpretable metrics (this will sample patches from the env)
    print("\n--- correlating fuzzy features with interpretable metrics (this may take a little while) ---\n")
    corr, f_names, metric_names = correlate_fuzzy_features(policy, env, n_samples=1024, batch=4, outdir="visuals/rule_plots")
    print("[inspect] Correlation matrix shape:", corr.shape)
    print("Fuzzy features (rows):", f_names)
    print("Interpretable metrics (cols):", metric_names)

    # 4) Save a small human-readable report (membership images + correlation heatmap + printed rules)
    outdir = "visuals/rule_report"
    print(f"\n--- saving full rule report to {outdir} ---\n")
    save_rule_report(policy, env, outdir=outdir, topk=12)
    print("[inspect] Done. Inspect visuals/rule_plots and visuals/rule_report folders.")

if __name__ == "__main__":
    main()
