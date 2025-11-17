# src/utils/interpret_rules.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

"""
Utilities to inspect NeuroFuzzyActorCritic rules and map them to interpretable local metrics.

Functions:
- extract_rule_data(policy) -> (rule_masks, consequents, scales)
- print_readable_rules(policy, feature_names=None, mf_names=None, topk=10)
- plot_membership_functions(policy, feature_idx, outdir)
- correlate_fuzzy_features(policy, env, n_samples=2000, batch=4, outdir=None)
- save_rule_report(policy, env, outdir)
Notes:
- policy must be an instance of NeuroFuzzyActorCritic from src/agents/neuro_fuzzy.py
- env is optional for correlation analyses; if provided, it should be SortingEnv and in obs_mode='local'
"""

def extract_rule_data(policy):
    """Return CPU tensors: (rule_masks (r,f,m), consequents (r,action_dim), scales (r,))"""
    masks, consequents, scales = policy.get_rule_info()
    return (
        masks.detach().cpu().numpy(),
        consequents.detach().cpu().numpy(),
        scales.detach().cpu().numpy()
    )

def _mf_label(m_idx):
    return f"MF{m_idx}"

def _feature_label(i, feature_names):
    return feature_names[i] if feature_names is not None else f"F{i}"

def print_readable_rules(policy, feature_names=None, mf_names=None, topk=20):
    """
    Print the top `topk` rules sorted by rule_scale (importance).
    For each rule, show the highest-weighted MF per feature and the consequent vector.
    """
    masks, consequents, scales = extract_rule_data(policy)
    r, f, m = masks.shape
    order = np.argsort(-scales)[:topk]
    print(f"Total rules: {r}. Showing top {len(order)} by scale.\n")
    for idx in order:
        print(f"RULE {idx}  | scale={scales[idx]:.4f}")
        mask = masks[idx]  # (f,m)
        for fi in range(f):
            mf_idx = int(mask[fi].argmax())
            mf_name = mf_names[mf_idx] if mf_names is not None else _mf_label(mf_idx)
            feat_name = _feature_label(fi, feature_names)
            strength = mask[fi, mf_idx]
            print(f"  IF {feat_name} IS {mf_name}  (strength={strength:.3f})")
        cons = consequents[idx]
        cons_str = ", ".join([f"{c:.3f}" for c in cons])
        print(f"  -> Consequent action (Δadh, v_x, v_y): [{cons_str}]\n")

def plot_membership_functions(policy, feature_idx, outdir="visuals/rules", x_range=(-1.5, 1.5), npoints=400):
    """
    Plot the Gaussian membership functions for a selected fuzzy feature index.
    Saves plots to outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    # extract centers & sigmas from policy.mf
    centers = policy.mf.centers.detach().cpu().numpy()  # (n_features, n_mfs)
    sigs = policy.mf.log_sigmas.detach().exp().cpu().numpy()
    n_mfs = centers.shape[1]
    x = np.linspace(x_range[0], x_range[1], npoints)
    fig, ax = plt.subplots(1,1,figsize=(6,3))
    for k in range(n_mfs):
        c = centers[feature_idx, k]
        s = sigs[feature_idx, k]
        y = np.exp(-0.5 * ((x - c)**2) / (s**2 + 1e-8))
        ax.plot(x, y, label=f"MF{k} (c={c:.2f}, σ={s:.2f})")
    ax.set_title(f"Membership functions (feature {feature_idx})")
    ax.legend()
    out = os.path.join(outdir, f"mf_feature_{feature_idx}.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[interpret] Saved membership plot: {out}")
    return out

def _compute_interpretable_metrics_from_patches(patches):
    """
    Given patches (B, N, 4, 5, 5) in torch, compute simple interpretable measurements per patch:
    - adhesion_mean (scalar)
    - morphogen_mean (scalar)
    - type_A_frac (mean of channel 0)
    - type_B_frac (mean of channel 1)
    - local_mismatch = |typeA - typeB| mean
    Returns (M, K) numpy array of metrics where M = B*N
    """
    if isinstance(patches, torch.Tensor):
        arr = patches.detach().cpu().numpy()
    else:
        arr = np.asarray(patches)
    B, N, C, h, w = arr.shape
    arr = arr.reshape(B*N, C, h, w)
    adhesion = arr[:,2].mean(axis=(1,2))
    morph = arr[:,3].mean(axis=(1,2))
    typeA = arr[:,0].mean(axis=(1,2))
    typeB = arr[:,1].mean(axis=(1,2))
    mismatch = np.abs(typeA - typeB)
    crowding = (typeA + typeB)  # density proxy
    metrics = np.stack([adhesion, morph, typeA, typeB, mismatch, crowding], axis=1)
    names = ["adhesion_mean", "morph_mean", "typeA_frac", "typeB_frac", "mismatch", "crowding"]
    return metrics, names

def correlate_fuzzy_features(policy, env=None, n_samples=2000, batch=4, outdir=None):
    """
    Draw patches either from env (if provided) or randomly sample, compute:
    - fuzzy_proj outputs (projected features before membership)
    - interpretable metrics from patches
    Then compute Pearson correlations between fuzzy_features and interpretable metrics.
    Saves a small heatmap to outdir if provided and returns the correlation matrix and labels.
    """
    policy.eval()
    device = next(policy.parameters()).device
    collected_patches = []
    total = 0
    if env is None:
        raise ValueError("env is required for correlation analysis (we need realistic patches).")
    # sample episodes
    while total < n_samples:
        patches, coords = env.reset(B=batch)
        # patches shape (B, N, 4, 5, 5)
        collected_patches.append(patches.cpu())
        total += batch * patches.shape[1]
    patches_all = torch.cat(collected_patches, dim=0)  # (M, N, 4,5,5) -> may exceed n_samples; we'll slice later
    # flatten to M*N patches
    Btot, N, C, h, w = patches_all.shape
    all_patches = patches_all.reshape(Btot * N, C, h, w)
    M = all_patches.shape[0]
    sel = min(n_samples, M)
    sel_idx = np.random.choice(M, sel, replace=False)
    sel_patches = all_patches[sel_idx].to(device)  # (sel,4,5,5)
    # compute fuzzy_proj values
    with torch.no_grad():
        feat = policy.perception(sel_patches)            # (sel, feat_dim)
        fuzzy_in = torch.tanh(policy.fuzzy_proj(feat))  # (sel, fuzzy_features)
        fuzzy_np = fuzzy_in.detach().cpu().numpy()
    # compute interpretable metrics
    metrics, metric_names = _compute_interpretable_metrics_from_patches(sel_patches.cpu().unsqueeze(0).numpy())
    metrics = metrics[:sel]  # ensure shape
    # correlation
    corr = np.corrcoef(fuzzy_np.T, metrics.T)  # block matrix
    fdim = fuzzy_np.shape[1]
    corr_ff = corr[:fdim, :fdim]
    corr_fm = corr[:fdim, fdim:fdim+metrics.shape[1]]
    # Save heatmap if outdir
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        plt.figure(figsize=(8,6))
        plt.imshow(corr_fm, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(metrics.shape[1]), metric_names, rotation=45)
        plt.yticks(range(fdim), [f"f{ii}" for ii in range(fdim)])
        plt.title("Correlation: fuzzy features vs interpretable metrics")
        fname = os.path.join(outdir, "corr_fuzzy_metrics.png")
        plt.tight_layout()
        plt.savefig(fname); plt.close()
        print(f"[interpret] Saved correlation heatmap: {fname}")
    return corr_fm, [f"f{ii}" for ii in range(fdim)], metric_names

def save_rule_report(policy, env, outdir="visuals/rules_report", topk=20):
    """
    Generate a small report: readable top rules, membership plots for first few features,
    and correlation heatmap between fuzzy features and interpretable metrics (requires env).
    """
    os.makedirs(outdir, exist_ok=True)
    print("[interpret] Extracting rule data...")
    print_readable_rules(policy, topk=topk)
    # membership plots for first min(8, fuzzy_features)
    fcount = policy.mf.centers.shape[0]
    for i in range(min(8, fcount)):
        plot_membership_functions(policy, i, outdir=outdir)
    try:
        corr, fnames, metric_names = correlate_fuzzy_features(policy, env, n_samples=1024, batch=4, outdir=outdir)
    except Exception as e:
        print("[interpret] correlation step failed:", e)
        corr = None
    print(f"[interpret] Saved report to {outdir}")
    return outdir
