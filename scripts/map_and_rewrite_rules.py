import os
import torch
import numpy as np

from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
from src.envs.wrappers import SortingEnv
from src.utils.interpret_rules import extract_rule_data, correlate_fuzzy_features

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[map] Loading policy…")
    policy = NeuroFuzzyActorCritic(
        in_ch=4, patch_size=5,
        feat_dim=128, fuzzy_features=16,
        n_mfs=3, n_rules=32, 
        action_dim=3
    ).to(device)

    # Load best checkpoint
    ckpt = "checkpoints/ppo_0000.pt"
    policy.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    policy.eval()

    print("[map] Creating env for correlation sampling…")
    env = SortingEnv(H=64, W=64, device=device, obs_mode="local")

    print("[map] Computing fuzzy feature correlations with interpretable metrics…")
    corr, fuzzy_names, metric_names = correlate_fuzzy_features(
        policy, env, 
        n_samples=1024, batch=4,
        outdir=None  # skip saving, we just want the matrix
    )

    corr = np.array(corr)

    print("\n=== FUZZY FEATURE → METRIC MAPPING ===")
    feature_map = {}
    for i in range(corr.shape[0]):
        idx = np.argmax(np.abs(corr[i]))
        metric = metric_names[idx]
        feature_map[i] = metric
        print(f"F{i:02d} → {metric} (corr={corr[i][idx]:.3f})")

    print("\n=== REWRITING RULES IN PLAIN ENGLISH ===")

    # Load rule masks and consequents
    masks, consequents, scales = extract_rule_data(policy)

    mf_names = ["Low", "Mid", "High"]

    for ri in range(12):
        print(f"\nRule {ri}: scale={scales[ri]:.3f}")
        conditions = []
        for fi in range(masks.shape[1]):
            mf = masks[ri, fi].argmax()
            metric = feature_map[fi]
            linguistic = mf_names[mf]
            conditions.append(f"{metric} is {linguistic}")

        cons = consequents[ri]
        dadh = cons[0]; vx = cons[1]; vy = cons[2]

        print("IF", " AND ".join(conditions))
        print(f"THEN Δadh={dadh:.3f}, move=({vx:.3f}, {vy:.3f})")

        # Biological meaning helper:
        bio = []
        if dadh > 0:
            bio.append("increase adhesion")
        else:
            bio.append("decrease adhesion")

        # movement vector interpretation:
        move_str = ""
        if abs(vx) > 0.03 or abs(vy) > 0.03:
            move_str = " and move "
            if vy < -0.03: move_str += "up "
            if vy > 0.03: move_str += "down "
            if vx < -0.03: move_str += "left "
            if vx > 0.03: move_str += "right "

        print("Biology:", "Cell will " + bio[0] + move_str)

    print("\n[map] Done. You now have interpretable fuzzy rules.")
    
if __name__ == "__main__":
    main()
