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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------
# Helper regularizers
# ---------------------------------------------------------------------

def orthogonality_loss(fuzzy_feats):
    # fuzzy_feats : (B, F)
    if fuzzy_feats.dim() != 2:
        fuzzy_feats = fuzzy_feats.view(fuzzy_feats.size(0), -1)
    gram = fuzzy_feats @ fuzzy_feats.t()  # (B,B)
    I = torch.eye(gram.size(0), device=gram.device)
    return ((gram - I) ** 2).mean()

def sparsity_loss(rule_masks):
    return torch.mean(torch.abs(rule_masks))

def correlation_loss(fuzzy_feats, metrics):
    if fuzzy_feats.shape[0] != metrics.shape[0]:
        min_b = min(fuzzy_feats.shape[0], metrics.shape[0])
        fuzzy_feats = fuzzy_feats[:min_b]
        metrics = metrics[:min_b]
    corr = torch.corrcoef(torch.cat([fuzzy_feats, metrics], dim=1).T)
    return torch.mean(torch.abs(corr))


# ---------------------------------------------------------------------
# Metrics extraction from patches for corr_loss / aux head
# ---------------------------------------------------------------------
def patches_to_metrics_default(patches):
    """
    patches: (B, C, H, W)
    Returns simple interpretable metrics: (B,4)
    """
    B = patches.size(0)
    m1 = patches.mean(dim=[1, 2, 3])
    m2 = patches.var(dim=[1, 2, 3])
    m3 = patches.abs().sum(dim=[1, 2, 3]) / (patches.size(1)*patches.size(2)*patches.size(3))
    m4 = patches[:, 0].mean(dim=[1, 2])
    return torch.stack([m1, m2, m3, m4], dim=1)


# ---------------------------------------------------------------------
# FINAL VERSION — compatible with your trainer
# ---------------------------------------------------------------------
class NeuroFuzzyActorCritic(nn.Module):
    """
    Fully compatible with train_local_sorting.py and ppo_update().
    Accepts:
        in_ch
        patch_size
        feat_dim
        fuzzy_features
        n_mfs
        n_rules
        action_dim
    """
    def __init__(self, in_ch=4, patch_size=5,
                 feat_dim=48, fuzzy_features=12,
                 n_mfs=3, n_rules=24,
                 action_dim=3):
        super().__init__()

        self.in_ch = in_ch
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.fuzzy_features = fuzzy_features
        self.n_mfs = n_mfs
        self.n_rules = n_rules
        self.action_dim = action_dim

        # ------------------------------
        # Encoder: small CNN
        # ------------------------------
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        flat_dim = 32 * patch_size * patch_size
        self.encoder_fc = nn.Linear(flat_dim, feat_dim)

        # ------------------------------
        # Fuzzy memberships per feature
        # ------------------------------
        self.mf_centers = nn.Parameter(torch.randn(fuzzy_features, n_mfs))
        self.mf_scales = nn.Parameter(torch.ones(fuzzy_features, n_mfs))

        # ------------------------------
        # Rule mask matrix
        # ------------------------------
        self.rule_masks = nn.Parameter(
            torch.randn(n_rules, fuzzy_features)
        )

        # ------------------------------
        # Heads
        # ------------------------------
        self.actor_head = nn.Linear(n_rules, action_dim)
        self.value_head = nn.Linear(n_rules, 1)

        # aux head predicts metrics (4-dim)
        self.aux_head = nn.Linear(n_rules, 4)

        # log std for Gaussian policy (clamp in forward uses)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

        # metric extractor
        self.patches_to_metrics = patches_to_metrics_default

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, patches):
        h = self.encoder_conv(patches)
        return self.encoder_fc(h)

    # ------------------------------------------------------------------
    # Fuzzy forward pass
    # ------------------------------------------------------------------
    def fuzzy_forward(self, feats):
        """
        feats : (B, feat_dim)
        Returns:
            fuzzy_rule_output: (B, n_rules)
            fuzzy_features:    (B, fuzzy_features)
        """
        # reduce to fuzzy_features (linear projection)
        f = feats[:, :self.fuzzy_features]

        # compute membership values
        # f: (B,F) -> expand to (B,F,n_mfs)
        f_exp = f.unsqueeze(-1)
        centers = self.mf_centers.unsqueeze(0)
        scales = F.softplus(self.mf_scales).unsqueeze(0)

        mvals = torch.exp(-((f_exp - centers) ** 2) / (2 * scales ** 2))  # (B,F,n_mfs)
        mvals = mvals.mean(dim=-1)                                        # (B,F)

        # rule activation
        rule_act = torch.matmul(mvals, self.rule_masks.t())               # (B, n_rules)
        rule_act = torch.relu(rule_act)

        return rule_act, mvals

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(self, patches):
        feats = self.encode(patches)
        rule_out, fuzzy_feats = self.fuzzy_forward(feats)

        mu = self.actor_head(rule_out)
        value = self.value_head(rule_out)
        return mu, value, fuzzy_feats

    # ------------------------------------------------------------------
    # get action & value
    # ------------------------------------------------------------------
    def get_action_and_value(self, patches, deterministic=False):
        mu, value, fuzzy_feats = self.forward(patches)
        # make std match mu shape for broadcasting
        std = self.logstd.exp().unsqueeze(0).expand_as(mu)
        # clamp std for numerical stability
        std = torch.clamp(std, min=1e-3, max=10.0)
        dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, mu, std
