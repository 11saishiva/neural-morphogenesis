# # src/agents/ppo.py
# import torch
# import torch.nn.functional as F
# from torch.distributions import Normal
# from torch.utils.data import DataLoader, TensorDataset

# # policy class import (compatible)
# from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic

# # legacy name — trainer still imports PatchActorCritic
# class PatchActorCritic(NeuroFuzzyActorCritic):
#     pass


# # ============================================================
# # GAE — Matches train_local_sorting exactly
# # ============================================================
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
#         delta = rewards[t] + gamma * values[t+1] * mask - values[t]
#         last_gae = delta + gamma * lam * mask * last_gae
#         advantages[t] = last_gae

#     returns = advantages + values[:-1]
#     return advantages.detach(), returns.detach()


# # ============================================================
# # PPO UPDATE — FINAL VERSION
# # Fully compatible with safe_ppo_update()
# # ============================================================
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
#     This is the final stable PPO update.

#     ✔ exact argument names your script expects  
#     ✔ no fuzzy regularizers  
#     ✔ stable gradient clipping  
#     ✔ supports per-minibatch update  
#     """

#     device = next(policy.parameters()).device

#     # Move everything to the correct device
#     obs_patches = obs_patches.to(device)
#     actions = actions.to(device)
#     logp_old = logp_old.to(device)
#     returns = returns.to(device)
#     advantages = advantages.to(device)

#     N = obs_patches.shape[0]
#     batch_size = batch_size if batch_size is not None else N

#     dataset = TensorDataset(obs_patches, actions, logp_old, returns, advantages)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # logging
#     logs = {
#         "actor_loss": 0.0,
#         "value_loss": 0.0,
#         "entropy": 0.0,
#         "total_loss": 0.0,
#     }
#     steps = 0

#     for epoch in range(epochs):
#         for obs_b, act_b, logp_old_b, ret_b, adv_b in loader:
#             obs_b = obs_b.to(device)
#             act_b = act_b.to(device)
#             logp_old_b = logp_old_b.to(device)
#             ret_b = ret_b.to(device)
#             adv_b = adv_b.to(device)

#             # forward through policy
#             mu, value_pred, fuzzy_feats = policy(obs_b)
#             std = policy.logstd.exp().unsqueeze(0)
#             dist = Normal(mu, std)

#             # new logprob
#             logp = dist.log_prob(act_b).sum(dim=-1)

#             # compute ratio
#             ratio = torch.exp(logp - logp_old_b)
#             clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

#             actor_loss = -torch.min(ratio * adv_b, clipped_ratio * adv_b).mean()
#             value_loss = value_coef * F.mse_loss(value_pred.squeeze(-1), ret_b)
#             entropy = dist.entropy().sum(-1).mean()

#             # final loss
#             loss = actor_loss + value_loss - entropy_coef * entropy

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
#             optimizer.step()

#             # logging
#             logs["actor_loss"] += actor_loss.item()
#             logs["value_loss"] += value_loss.item()
#             logs["entropy"] += entropy.item()
#             logs["total_loss"] += loss.item()
#             steps += 1

#     # average logs
#     for k in logs:
#         logs[k] /= max(steps, 1)

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
    Stable PPO update.

    - normalizes advantages per minibatch
    - ensures std shape matches mu
    - performs gradient clipping
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

            # normalize advantages for this minibatch
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

            # forward through policy
            mu, value_pred, fuzzy_feats = policy(obs_b)
            # ensure std shape matches mu for broadcasting
            std = policy.logstd.exp().unsqueeze(0).expand_as(mu).to(device)
            # small clamp on std for numerical stability
            std = torch.clamp(std, min=1e-3, max=10.0)
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
