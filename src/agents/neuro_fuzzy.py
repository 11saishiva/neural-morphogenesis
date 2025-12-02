# # src/agents/neuro_fuzzy.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal

# # ---------------------------------------------------------------------
# # Helper regularizers
# # ---------------------------------------------------------------------

# def orthogonality_loss(fuzzy_feats):
#     # fuzzy_feats : (B, F)
#     if fuzzy_feats.dim() != 2:
#         fuzzy_feats = fuzzy_feats.view(fuzzy_feats.size(0), -1)
#     gram = fuzzy_feats @ fuzzy_feats.t()  # (B,B)
#     I = torch.eye(gram.size(0), device=gram.device)
#     return ((gram - I) ** 2).mean()

# def sparsity_loss(rule_masks):
#     return torch.mean(torch.abs(rule_masks))

# def correlation_loss(fuzzy_feats, metrics):
#     if fuzzy_feats.shape[0] != metrics.shape[0]:
#         min_b = min(fuzzy_feats.shape[0], metrics.shape[0])
#         fuzzy_feats = fuzzy_feats[:min_b]
#         metrics = metrics[:min_b]
#     corr = torch.corrcoef(torch.cat([fuzzy_feats, metrics], dim=1).T)
#     return torch.mean(torch.abs(corr))


# # ---------------------------------------------------------------------
# # Metrics extraction from patches for corr_loss / aux head
# # ---------------------------------------------------------------------
# def patches_to_metrics_default(patches):
#     """
#     patches: (B, C, H, W)
#     Returns simple interpretable metrics: (B,4)
#     """
#     B = patches.size(0)
#     m1 = patches.mean(dim=[1, 2, 3])
#     m2 = patches.var(dim=[1, 2, 3])
#     m3 = patches.abs().sum(dim=[1, 2, 3]) / (patches.size(1)*patches.size(2)*patches.size(3))
#     m4 = patches[:, 0].mean(dim=[1, 2])
#     return torch.stack([m1, m2, m3, m4], dim=1)


# # ---------------------------------------------------------------------
# # FINAL VERSION — compatible with your trainer
# # ---------------------------------------------------------------------
# class NeuroFuzzyActorCritic(nn.Module):
#     """
#     Fully compatible with train_local_sorting.py and ppo_update().
#     Accepts:
#         in_ch
#         patch_size
#         feat_dim
#         fuzzy_features
#         n_mfs
#         n_rules
#         action_dim
#     """
#     def __init__(self, in_ch=4, patch_size=5,
#                  feat_dim=48, fuzzy_features=12,
#                  n_mfs=3, n_rules=24,
#                  action_dim=3):
#         super().__init__()

#         self.in_ch = in_ch
#         self.patch_size = patch_size
#         self.feat_dim = feat_dim
#         self.fuzzy_features = fuzzy_features
#         self.n_mfs = n_mfs
#         self.n_rules = n_rules
#         self.action_dim = action_dim

#         # ------------------------------
#         # Encoder: small CNN
#         # ------------------------------
#         self.encoder_conv = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         flat_dim = 32 * patch_size * patch_size
#         self.encoder_fc = nn.Linear(flat_dim, feat_dim)

#         # ------------------------------
#         # Fuzzy memberships per feature
#         # ------------------------------
#         self.mf_centers = nn.Parameter(torch.randn(fuzzy_features, n_mfs))
#         self.mf_scales = nn.Parameter(torch.ones(fuzzy_features, n_mfs))

#         # ------------------------------
#         # Rule mask matrix
#         # ------------------------------
#         self.rule_masks = nn.Parameter(
#             torch.randn(n_rules, fuzzy_features)
#         )

#         # ------------------------------
#         # Heads
#         # ------------------------------
#         self.actor_head = nn.Linear(n_rules, action_dim)
#         self.value_head = nn.Linear(n_rules, 1)

#         # aux head predicts metrics (4-dim)
#         self.aux_head = nn.Linear(n_rules, 4)

#         # log std for Gaussian policy
#         self.logstd = nn.Parameter(torch.zeros(action_dim))

#         # metric extractor
#         self.patches_to_metrics = patches_to_metrics_default

#     # ------------------------------------------------------------------
#     # Encoding
#     # ------------------------------------------------------------------
#     def encode(self, patches):
#         h = self.encoder_conv(patches)
#         return self.encoder_fc(h)

#     # ------------------------------------------------------------------
#     # Fuzzy forward pass
#     # ------------------------------------------------------------------
#     def fuzzy_forward(self, feats):
#         """
#         feats : (B, feat_dim)
#         Returns:
#             fuzzy_rule_output: (B, n_rules)
#             fuzzy_features:    (B, fuzzy_features)
#         """
#         # reduce to fuzzy_features (linear projection)
#         f = feats[:, :self.fuzzy_features]

#         # compute membership values
#         # f: (B,F) -> expand to (B,F,n_mfs)
#         f_exp = f.unsqueeze(-1)
#         centers = self.mf_centers.unsqueeze(0)
#         scales = F.softplus(self.mf_scales).unsqueeze(0)

#         mvals = torch.exp(-((f_exp - centers) ** 2) / (2 * scales ** 2))  # (B,F,n_mfs)
#         mvals = mvals.mean(dim=-1)                                        # (B,F)

#         # rule activation
#         rule_act = torch.matmul(mvals, self.rule_masks.t())               # (B, n_rules)
#         rule_act = torch.relu(rule_act)

#         return rule_act, mvals

#     # ------------------------------------------------------------------
#     # Main forward
#     # ------------------------------------------------------------------
#     def forward(self, patches):
#         feats = self.encode(patches)
#         rule_out, fuzzy_feats = self.fuzzy_forward(feats)

#         mu = self.actor_head(rule_out)
#         value = self.value_head(rule_out)
#         return mu, value, fuzzy_feats

#     # ------------------------------------------------------------------
#     # get action & value
#     # ------------------------------------------------------------------
#     def get_action_and_value(self, patches, deterministic=False):
#         mu, value, fuzzy_feats = self.forward(patches)
#         std = self.logstd.exp().unsqueeze(0).expand_as(mu)
#         dist = Normal(mu, std)

#         if deterministic:
#             action = mu
#         else:
#             action = dist.rsample()

#         logp = dist.log_prob(action).sum(-1)
#         return action, logp, value, mu, std

# src/agents/neuro_fuzzy.py
"""
Neuro-fuzzy actor-critic (safe, simple, and compatible with the trainer)

Design goals:
- Avoid in-place ops and non-contiguous views that break autograd.
- Forward API matches trainer and ppo.update:
    * policy(obs_patches) -> (mu, value, fuzzy_feats)
      - mu: (B, action_dim)
      - value: (B, 1)
      - fuzzy_feats: (B, fuzzy_features) (useful for logging/regularizers)
- get_action_and_value(patches, deterministic=False) -> (action, logp, value, mu, std)
  Shapes:
    - action: (B, action_dim)
    - logp: (B,)
    - value: (B, 1)
    - mu: (B, action_dim)
    - std: (B, action_dim)
- No external dependencies beyond torch.
- Includes small helper losses (orthogonality, sparsity, correlation) you can optionally call.
"""

from typing import Callable, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------
# Helper regularizers
# ---------------------------
def orthogonality_loss(fuzzy_feats: torch.Tensor) -> torch.Tensor:
    """
    Encourages orthogonal features across the batch.
    fuzzy_feats: (B, F)
    Output: scalar
    """
    if fuzzy_feats.dim() != 2:
        fuzzy_feats = fuzzy_feats.view(fuzzy_feats.size(0), -1)
    # small-batch safe: compute gram on transposed features (F x F)
    # then encourage identity scaled by batch size.
    gram = fuzzy_feats.t() @ fuzzy_feats  # (F,F)
    I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    # normalize by squared norm to keep scale-invariant behavior
    gram = gram / (fuzzy_feats.size(0) + 1e-6)
    return ((gram - I) ** 2).mean()


def sparsity_loss(rule_masks: torch.Tensor) -> torch.Tensor:
    """L1 penalty on rule masks"""
    return torch.mean(torch.abs(rule_masks))


def correlation_loss(fuzzy_feats: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
    """
    Simple correlation loss between fuzzy features and metrics.
    fuzzy_feats: (B, F)
    metrics: (B, M)
    """
    if fuzzy_feats.shape[0] != metrics.shape[0]:
        min_b = min(fuzzy_feats.shape[0], metrics.shape[0])
        fuzzy_feats = fuzzy_feats[:min_b]
        metrics = metrics[:min_b]
    # standardize
    def standardize(x):
        x = x - x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        std = std.clamp(min=1e-6)
        return x / std

    f_s = standardize(fuzzy_feats)
    m_s = standardize(metrics)
    corr = (f_s.t() @ m_s) / (f_s.size(0) - 1.0 + 1e-6)  # (F, M)
    return torch.mean(torch.abs(corr))


# ---------------------------
# Patch -> metrics extractor
# ---------------------------
def patches_to_metrics_default(patches: torch.Tensor) -> torch.Tensor:
    """
    patches: (B, C, H, W)
    Returns (B, 4) simple metrics:
      - mean over all channels and pixels
      - var over all channels and pixels
      - mean absolute value
      - mean of channel 0 (useful as a representative channel)
    """
    B = patches.size(0)
    m_mean = patches.mean(dim=[1, 2, 3])
    m_var = patches.var(dim=[1, 2, 3])
    m_absmean = patches.abs().mean(dim=[1, 2, 3])
    ch0_mean = patches[:, 0].mean(dim=[1, 2])
    return torch.stack([m_mean, m_var, m_absmean, ch0_mean], dim=1)


# ---------------------------
# Neuro-Fuzzy Actor-Critic
# ---------------------------
class NeuroFuzzyActorCritic(nn.Module):
    """
    Neuro-fuzzy actor-critic network.

    Parameters:
      in_ch: input channels per patch (e.g. 4)
      patch_size: patch height/width (e.g. 5)
      feat_dim: dimension of CNN features
      fuzzy_features: number of features used for fuzzy memberships
      n_mfs: number of membership functions per fuzzy feature (unused in complex way here)
      n_rules: number of fuzzy rules (this controls final hidden dim for heads)
      action_dim: dimensionality of continuous action
    """

    def __init__(
        self,
        in_ch: int = 4,
        patch_size: int = 5,
        feat_dim: int = 48,
        fuzzy_features: int = 12,
        n_mfs: int = 3,
        n_rules: int = 24,
        action_dim: int = 3,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.fuzzy_features = fuzzy_features
        self.n_mfs = n_mfs
        self.n_rules = n_rules
        self.action_dim = action_dim

        # --------------------
        # Encoder: small safe CNN
        # --------------------
        # avoid in-place ReLU; keep outputs contiguous by using Flatten -> Linear
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Flatten(),
        )

        flat_dim = 32 * patch_size * patch_size
        self.encoder_fc = nn.Sequential(
            nn.Linear(flat_dim, feat_dim),
            nn.ReLU(inplace=False),
        )

        # --------------------
        # Fuzzy membership params
        # Use a simple Gaussian membership parameterization.
        # centers: (fuzzy_features, n_mfs)
        # scales: positive scales (fuzzy_features, n_mfs)
        # --------------------
        self.mf_centers = nn.Parameter(torch.randn(fuzzy_features, n_mfs) * 0.1)
        # initialize scales to small positive values, parameterize in unconstrained space
        self._mf_logscales = nn.Parameter(torch.zeros(fuzzy_features, n_mfs) + math.log(0.5))

        # --------------------
        # Rule mask matrix: map fuzzy_features -> n_rules
        # Keep rule_masks as unconstrained params and apply bounded activation in forward
        # --------------------
        self.rule_masks = nn.Parameter(torch.randn(n_rules, fuzzy_features) * 0.1)

        # --------------------
        # Heads
        # --------------------
        self.actor_head = nn.Linear(n_rules, action_dim)
        self.value_head = nn.Linear(n_rules, 1)
        self.aux_head = nn.Linear(n_rules, 4)  # predicts the patch-level metrics if needed

        # log-std parameter for gaussian policy (unconstrained)
        # we parameterize as raw_logstd and convert via softplus to ensure positivity.
        self.raw_logstd = nn.Parameter(torch.zeros(action_dim) - 2.0)

        # metric extractor (function)
        self.patches_to_metrics: Callable[[torch.Tensor], torch.Tensor] = patches_to_metrics_default

        # small weight init tweak
        self._init_weights()

    def _init_weights(self):
        # orthogonal-ish for linear layers helps stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in + 1e-6)
                    nn.init.uniform_(m.bias, -bound, bound)

    # --------------------
    # Encode patches -> feature vector
    # patches: (B, C, H, W)
    # returns: (B, feat_dim)
    # --------------------
    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        # ensure contiguous memory layout to avoid as_strided/backward errors
        if not patches.is_contiguous():
            patches = patches.contiguous()
        h = self.encoder_conv(patches)  # (B, flat_dim_enc)
        # encoder_fc expects contiguous input
        if not h.is_contiguous():
            h = h.contiguous()
        feats = self.encoder_fc(h)
        return feats

    # --------------------
    # Fuzzy forward: produce rule activations and fuzzy features
    # feats: (B, feat_dim)
    # returns:
    #   rule_act: (B, n_rules)
    #   fuzzy_feats: (B, fuzzy_features)
    # --------------------
    def fuzzy_forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ensure dims
        B = feats.size(0)
        # reduce to fuzzy_features via linear slice (stable and simple)
        if feats.size(1) < self.fuzzy_features:
            # pad with zeros if encoder smaller than requested fuzzy_features
            pad = feats.new_zeros((B, self.fuzzy_features - feats.size(1)))
            f = torch.cat([feats, pad], dim=1)[:, : self.fuzzy_features]
        else:
            f = feats[:, : self.fuzzy_features]

        # compute Gaussian memberships:
        # f: (B, F) -> (B, F, 1)
        f_exp = f.unsqueeze(-1)  # (B, F, 1)
        centers = self.mf_centers.unsqueeze(0)  # (1, F, n_mfs)
        scales = (self._mf_logscales.exp()).unsqueeze(0)  # positive scales, (1, F, n_mfs)
        # compute squared distance / (2*scale^2)
        # Broadcasting shapes: (B, F, n_mfs)
        diff = f_exp - centers
        denom = 2.0 * (scales ** 2 + 1e-9)
        mvals = torch.exp(- (diff ** 2) / denom)  # (B, F, n_mfs)

        # aggregate membership functions per feature into a single feature:
        # Use softmax-weighted combination across mfs to keep gradients stable
        # weights: (B, F, n_mfs)
        weights = F.softmax(mvals, dim=-1)
        # fuzzy feature per (B, F) is weighted sum of centers by weights (interpretable)
        # centers: (1, F, n_mfs)
        fuzzy_feats = (weights * centers).sum(dim=-1)  # (B, F)

        # compute rule activations: linear map then nonlinearity
        # rule_masks: (n_rules, F)
        rule_raw = fuzzy_feats @ self.rule_masks.t()  # (B, n_rules)
        # apply a smooth nonlinearity (softplus) to keep activations positive and differentiable
        rule_act = F.softplus(rule_raw)

        return rule_act, fuzzy_feats

    # --------------------
    # forward: patches -> mu, value, fuzzy_feats
    # patches: (B, C, H, W)
    # returns:
    #   mu: (B, action_dim)
    #   value: (B, 1)
    #   fuzzy_feats: (B, fuzzy_features)
    # --------------------
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.encode(patches)                     # (B, feat_dim)
        rule_out, fuzzy_feats = self.fuzzy_forward(feats)  # (B, n_rules), (B, fuzzy_features)

        # heads operate on rule_out: (B, n_rules)
        mu = self.actor_head(rule_out)        # (B, action_dim)
        value = self.value_head(rule_out)     # (B, 1)

        # ensure outputs are contiguous for downstream operations
        if not mu.is_contiguous():
            mu = mu.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()
        if not fuzzy_feats.is_contiguous():
            fuzzy_feats = fuzzy_feats.contiguous()

        return mu, value, fuzzy_feats

    # --------------------
    # produce action and value (used by trainer)
    # patches: (B, C, H, W)
    # deterministic: if True, return mu as action (no sampling)
    # returns: action, logp, value, mu, std
    # - action: (B, action_dim)
    # - logp: (B,)
    # - value: (B, 1)
    # - mu: (B, action_dim)
    # - std: (B, action_dim)
    # --------------------
    def get_action_and_value(self, patches: torch.Tensor, deterministic: bool = False):
        mu, value, fuzzy_feats = self.forward(patches)  # mu: (B, A)
        # convert raw_logstd to positive std with softplus (stable)
        std_scalar = F.softplus(self.raw_logstd) + 1e-6  # (A,)
        # expand to (B, A)
        std = std_scalar.unsqueeze(0).expand(mu.size(0), -1).contiguous()

        dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            # use reparameterized sample for stable gradients where needed
            action = dist.rsample()

        # log_prob per sample (sum over action dims)
        # sum returns shape (B,)
        logp = dist.log_prob(action).sum(dim=-1)

        # make sure types and contiguity are safe
        if not action.is_contiguous():
            action = action.contiguous()
        if not logp.is_contiguous():
            logp = logp.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()

        return action, logp, value, mu, std

    # --------------------
    # optional: predict metrics from patches via aux head
    # this is useful for regularizers / monitoring
    # --------------------
    def predict_aux_metrics(self, patches: torch.Tensor) -> torch.Tensor:
        feats = self.encode(patches)
        rule_out, _ = self.fuzzy_forward(feats)
        aux = self.aux_head(rule_out)
        return aux

    # --------------------
    # convenient getter for policy std (for logging)
    # --------------------
    def policy_std(self) -> torch.Tensor:
        return (F.softplus(self.raw_logstd) + 1e-6).detach()


        logp = dist.log_prob(action).sum(-1)
        action = torch.tanh(action)
        return action, logp, value, mu, std
