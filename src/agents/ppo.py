# # src/agents/ppo.py
# import math
# import torch
# import torch.nn.functional as F
# from torch.distributions import Normal
# from torch.utils.data import DataLoader, TensorDataset


# # ------------------------------------------------------------
# # PatchActorCritic (legacy alias for compatibility)
# # ------------------------------------------------------------
# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic

# class PatchActorCritic(NeuroFuzzyActorCritic):
#     """
#     Backwards-compatible alias. The trainer imports PatchActorCritic,
#     so we subclass NeuroFuzzyActorCritic directly.
#     """
#     pass

# # import fuzzy regularizers & policy class
# from src.agents.neuro_fuzzy import (
#     orthogonality_loss,
#     sparsity_loss,
#     correlation_loss,
#     NeuroFuzzyActorCritic,
# )


# # ============================================================
# # compute_gae (compatible with train_local_sorting.py shapes)
# # rewards: (T, B)
# # values:  (T+1, B)
# # dones:   (T, B)
# # ============================================================
# def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
#     T, B = rewards.shape
#     advantages = torch.zeros(T, B, device=rewards.device)
#     last_gae = torch.zeros(B, device=rewards.device)
#     for t in reversed(range(T)):
#         mask = 1.0 - dones[t]
#         delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
#         last_gae = delta + gamma * lam * mask * last_gae
#         advantages[t] = last_gae
#     returns = advantages + values[:-1]
#     return advantages.detach(), returns.detach()


# # ============================================================
# # ppo_update
# # Matches the exact keyword args your train script passes.
# # Includes fuzzy regularizers (orthogonality, sparsity, correlation, auxiliary).
# # ============================================================
# def ppo_update(
#     policy,
#     optimizer,
#     obs_patches,
#     actions,
#     logprobs_old,
#     returns,
#     advantages,
#     clip_eps=0.2,
#     vf_coef=0.5,
#     ent_coef=0.01,
#     epochs=4,
#     batch_size=2048,
#     orth_coef=3e-3,
#     sparsity_coef=5e-3,
#     corr_coef=5e-3,
#     aux_coef=1e-2,
#     device=None,
# ):
#     """
#     obs_patches: (N, C, H, W)  flattened across T, B, Ncells
#     actions:     (N, A)
#     logprobs_old:(N,)
#     returns:     (N,)
#     advantages:  (N,)
#     """

#     if device is None:
#         device = next(policy.parameters()).device

#     # move to device
#     obs_patches = obs_patches.to(device)
#     actions = actions.to(device)
#     logprobs_old = logprobs_old.to(device)
#     returns = returns.to(device)
#     advantages = advantages.to(device)

#     N = obs_patches.shape[0]
#     batch_size = min(batch_size, N)

#     dataset = TensorDataset(obs_patches, actions, logprobs_old, returns, advantages)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     logs = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
#             "sparse": 0.0, "orth": 0.0, "corr": 0.0, "aux": 0.0, "total": 0.0}
#     iters = 0

#     for epoch in range(epochs):
#         for batch in loader:
#             obs_b, act_b, logp_old_b, ret_b, adv_b = batch
#             obs_b = obs_b.to(device)
#             act_b = act_b.to(device)
#             logp_old_b = logp_old_b.to(device)
#             ret_b = ret_b.to(device)
#             adv_b = adv_b.to(device)

#             # forward through policy
#             mu, value_pred, fuzzy_feats = policy(obs_b)  # mu: (B,A), value_pred: (B,1)
#             std = policy.logstd.exp().unsqueeze(0).to(device)
#             dist = Normal(mu, std)

#             # logprobs and entropy
#             logp = dist.log_prob(act_b).sum(-1)
#             entropy = dist.entropy().sum(-1).mean()

#             # ratio
#             ratio = torch.exp(logp - logp_old_b)
#             surr1 = ratio * adv_b
#             surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
#             actor_loss = -torch.min(surr1, surr2).mean()

#             # value loss
#             value_loss = vf_coef * F.mse_loss(value_pred.squeeze(-1), ret_b)

#             # fuzzy regularizers (A1 selected)
#             sparse_l = sparsity_loss(policy.fuzzy.rule_masks)
#             orth_l = orthogonality_loss(fuzzy_feats)
#             # correlation: compute interpretable metrics from the corresponding patches
#             # obs_b has shape (B, C, H, W), so we can call patches_to_metrics
#             try:
#                 metrics = policy.patches_to_metrics(obs_b)
#             except Exception:
#                 # fallback: zeros
#                 metrics = torch.zeros((obs_b.size(0), 4), device=device)
#             corr_l = correlation_loss(fuzzy_feats, metrics)
#             # auxiliary prediction loss
#             pred_metrics = policy.aux_head(fuzzy_feats)
#             aux_l = F.mse_loss(pred_metrics, metrics)

#             reg_loss = (sparsity_coef * sparse_l) + (orth_coef * orth_l) + (corr_coef * corr_l) + (aux_coef * aux_l)

#             # total loss
#             loss = actor_loss + value_loss + (-ent_coef * entropy) + reg_loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # logs accumulation
#             logs["actor_loss"] += actor_loss.item()
#             logs["value_loss"] += value_loss.item()
#             logs["entropy"] += entropy.item()
#             logs["sparse"] += sparse_l.item()
#             logs["orth"] += orth_l.item()
#             logs["corr"] += corr_l.item()
#             logs["aux"] += aux_l.item()
#             logs["total"] += loss.item()
#             iters += 1

#     # average over iterations
#     if iters > 0:
#         for k in logs:
#             logs[k] /= iters

#     return logs


# src/agents/ppo.py
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

# policy class import (compatible)
from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic

# legacy name — trainer still imports PatchActorCritic
class PatchActorCritic(NeuroFuzzyActorCritic):
    pass


# ============================================================
# GAE — Matches train_local_sorting exactly
# ============================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T,B)
    values:  (T+1,B)
    dones:   (T,B)
    """
    T, B = rewards.shape
    advantages = torch.zeros(T, B, device=rewards.device)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:-1]
    return advantages.detach(), returns.detach()


# ============================================================
# PPO UPDATE — FINAL VERSION
# Fully compatible with safe_ppo_update()
# ============================================================
def ppo_update(
    policy,
    optimizer,
    obs_patches,     # (batch, C, H, W)
    actions,         # (batch, A)
    logp_old,        # (batch,)
    returns,         # (batch,)
    advantages,      # (batch,)
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=1,
    batch_size=None,
):
    """
    This is the final stable PPO update.

    ✔ exact argument names your script expects  
    ✔ no fuzzy regularizers  
    ✔ stable gradient clipping  
    ✔ supports per-minibatch update  
    """

    device = next(policy.parameters()).device

    # Move everything to the correct device
    obs_patches = obs_patches.to(device)
    actions = actions.to(device)
    logp_old = logp_old.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    N = obs_patches.shape[0]
    batch_size = batch_size if batch_size is not None else N

    dataset = TensorDataset(obs_patches, actions, logp_old, returns, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # logging
    logs = {
        "actor_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }
    steps = 0

    for epoch in range(epochs):
        for obs_b, act_b, logp_old_b, ret_b, adv_b in loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            logp_old_b = logp_old_b.to(device)
            ret_b = ret_b.to(device)
            adv_b = adv_b.to(device)

            # forward through policy
            mu, value_pred, fuzzy_feats = policy(obs_b)
            std = policy.logstd.exp().unsqueeze(0)
            dist = Normal(mu, std)

            # new logprob
            logp = dist.log_prob(act_b).sum(dim=-1)

            # compute ratio
            ratio = torch.exp(logp - logp_old_b)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

            actor_loss = -torch.min(ratio * adv_b, clipped_ratio * adv_b).mean()
            value_loss = value_coef * F.mse_loss(value_pred.squeeze(-1), ret_b)
            entropy = dist.entropy().sum(-1).mean()

            # final loss
            loss = actor_loss + value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            # logging
            logs["actor_loss"] += actor_loss.item()
            logs["value_loss"] += value_loss.item()
            logs["entropy"] += entropy.item()
            logs["total_loss"] += loss.item()
            steps += 1

    # average logs
    for k in logs:
        logs[k] /= max(steps, 1)

    return logs
