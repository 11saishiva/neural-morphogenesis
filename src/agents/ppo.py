# import torch
# import torch.nn.functional as F
# from torch.distributions import Normal
# from torch.utils.data import DataLoader, TensorDataset

# # Policy class import (keeps backward compatibility)
# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic

# # Legacy alias kept for imports that expect PatchActorCritic
# class PatchActorCritic(NeuroFuzzyActorCritic):
#     pass


# def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
#     """
#     rewards: (T,B)
#     values:  (T+1,B)
#     dones:   (T,B)
#     """
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


# def ppo_update(
#     policy,
#     optimizer,
#     obs_patches,     # (batch, C, H, W)
#     actions,         # (batch, A)
#     logp_old,        # (batch,)
#     returns,         # (batch,)
#     advantages,      # (batch,)
#     clip_ratio=0.2,
#     value_coef=0.5,
#     entropy_coef=0.01,
#     epochs=1,
#     batch_size=None,
# ):
#     """
#     Stable PPO update with safe contiguous minibatches.

#     - Moves tensors to device once and ensures contiguous minibatches
#     - Normalizes advantages per-minibatch
#     - Uses gradient clipping
#     """
#     device = next(policy.parameters()).device

#     # Move the whole dataset to device once (keeps loader simple)
#     obs_patches = obs_patches.to(device)
#     actions = actions.to(device)
#     logp_old = logp_old.to(device)
#     returns = returns.to(device)
#     advantages = advantages.to(device)

#     # Normalization already performed by caller, but keep a safeguard:
#     adv_mean = advantages.mean()
#     adv_std = advantages.std(unbiased=False) + 1e-8
#     advantages = (advantages - adv_mean) / adv_std

#     N = obs_patches.shape[0]
#     batch_size = batch_size if batch_size is not None else N

#     dataset = TensorDataset(obs_patches, actions, logp_old, returns, advantages)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     logs = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
#     steps = 0

#     for epoch in range(epochs):
#         for obs_b, act_b, logp_old_b, ret_b, adv_b in loader:
#             # Move minibatch to device and ensure contiguous layout
#             obs_b = obs_b.to(device, non_blocking=True).contiguous()
#             act_b = act_b.to(device, non_blocking=True)
#             logp_old_b = logp_old_b.to(device, non_blocking=True)
#             ret_b = ret_b.to(device, non_blocking=True)
#             adv_b = adv_b.to(device, non_blocking=True)

#             # Re-normalize per-minibatch (stable training)
#             adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

#             # Forward through policy (policy.forward returns mu, value, fuzzy_feats)
#             mu, value_pred, _ = policy(obs_b)

#             # Get numeric std from policy safely (policy exposes logstd as tensor-like)
#             # policy.logstd should be a tensor (not Parameter) or property returning log(std)
#             try:
#                 std = (policy.logstd.exp()).unsqueeze(0).expand_as(mu).to(device)
#             except Exception:
#                 # fallback: if policy has raw_logstd param
#                 try:
#                     std = (F.softplus(policy.raw_logstd) + 1e-6).unsqueeze(0).expand_as(mu).to(device)
#                 except Exception:
#                     # final fallback: small constant std
#                     std = torch.full_like(mu, 0.1, device=device)

#             # clamp std for numerical stability
#             std = torch.clamp(std, min=1e-6, max=10.0)

#             dist = Normal(mu, std)

#             # new logprob (sum over action dims)
#             logp = dist.log_prob(act_b).sum(dim=-1)

#             # ratio and clipped surrogate
#             ratio = torch.exp(logp - logp_old_b)
#             clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
#             actor_loss = -torch.min(ratio * adv_b, clipped_ratio * adv_b).mean()

#             # value loss
#             value_loss = value_coef * F.mse_loss(value_pred.squeeze(-1), ret_b)

#             # entropy (encourage exploration)
#             entropy = dist.entropy().sum(-1).mean()

#             loss = actor_loss + value_loss - entropy_coef * entropy
#             backup_state = {k: v.clone() for k, v in policy.state_dict().items()}


#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
#             optimizer.step()
#             with torch.no_grad():
#                 if hasattr(policy, "raw_logstd"):
#                 # prevent extreme std values
#                 policy.raw_logstd.clamp_(-10.0, 2.0)

# # detect NaNs in params; revert if found
#             nan_found = False
#             for p in policy.parameters():
#                 if torch.isnan(p).any():
#                     nan_found = True
#                     break

#             if nan_found:
#                 policy.load_state_dict(backup_state)
#                 optimizer.zero_grad()
#     # Optional: reduce lr to be conservative
#                 for g in optimizer.param_groups:
#                     g['lr'] = max(g.get('lr', 1e-6) * 0.5, 1e-8)
#                 print("[ppo_update] NaN detected in parameters after optimizer.step(); reverted parameters and reduced LR.")
#     # skip logging this minibatch's stats (we reverted)
#                 continue

#             logs["actor_loss"] += actor_loss.item()
#             logs["value_loss"] += value_loss.item()
#             logs["entropy"] += entropy.item()
#             logs["total_loss"] += loss.item()
#             steps += 1

#     # average logs
#     if steps > 0:
#         for k in logs:
#             logs[k] /= steps
#     return logs

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
    Stable PPO update.

    - normalizes advantages per minibatch
    - ensures std shape matches mu
    - performs gradient clipping
    - reverts step and reduces LR if NaNs appear
    """
    device = next(policy.parameters()).device

    # Move everything to the correct device
    obs_patches = obs_patches.to(device)
    actions = actions.to(device)
    logp_old = logp_old.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    # normalize advantages globally (helpful, but minibatch normalize again later)
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

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
            # move to device + ensure contiguous memory layout to avoid as_strided/view autograd issues
            obs_b = obs_b.to(device, non_blocking=True).contiguous()
            act_b = act_b.to(device, non_blocking=True)
            logp_old_b = logp_old_b.to(device, non_blocking=True)
            ret_b = ret_b.to(device, non_blocking=True)
            adv_b = adv_b.to(device, non_blocking=True)

            # normalize advantages for this minibatch (stable)
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

            # forward through policy
            mu, value_pred, fuzzy_feats = policy(obs_b)
            # ensure std shape matches mu for broadcasting
            # Use policy.raw_logstd if available (safe transform); fallback to policy.logstd
            if hasattr(policy, "raw_logstd"):
                std_scalar = F.softplus(policy.raw_logstd) + 1e-6
                std = std_scalar.unsqueeze(0).expand_as(mu).to(device)
            else:
                std = policy.logstd.exp().unsqueeze(0).expand_as(mu).to(device)

            # clamp std for numerical stability
            std = torch.clamp(std, min=1e-6, max=1e3)
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

            # ---------------- safer optimizer step ----------------
            # backup params in case step produces NaNs
            backup_state = {k: v.clone() for k, v in policy.state_dict().items()}

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            # clamp raw_logstd to a sane range and keep it finite
            with torch.no_grad():
                if hasattr(policy, "raw_logstd"):
                    policy.raw_logstd.clamp_(-10.0, 2.0)

            # detect NaNs or infs in parameters; revert & reduce LR if found
            nan_or_inf = False
            for p in policy.parameters():
                # .isfinite checks both NaN and Inf
                if not torch.isfinite(p).all():
                    nan_or_inf = True
                    break

            if nan_or_inf:
                # revert
                policy.load_state_dict(backup_state)
                optimizer.zero_grad()
                # reduce lr conservatively
                for g in optimizer.param_groups:
                    old = g.get("lr", 1e-6)
                    g["lr"] = max(old * 0.5, 1e-8)
                print("[ppo_update] NaN/Inf detected in params after optimizer.step(); reverted params and reduced LR.")
                # skip logging this minibatch's stats
                continue
            # ------------------------------------------------------

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
