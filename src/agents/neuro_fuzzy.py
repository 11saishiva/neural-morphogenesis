# # src/agents/neuro_fuzzy.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal


# # ------------------------------------------------------------
# # Utilities (exported so ppo.py can import these)
# # ------------------------------------------------------------
# def orthogonality_loss(features):
#     # features: (B, F)
#     Fnorm = features - features.mean(0)
#     cov = (Fnorm.T @ Fnorm) / (Fnorm.size(0) - 1)
#     I = torch.eye(cov.size(0), device=cov.device)
#     return ((cov - I) ** 2).mean()


# def sparsity_loss(rule_masks):
#     # rule_masks: (F, MF) in our design
#     return rule_masks.abs().mean()


# def correlation_loss(features, metrics):
#     # features: (B,F)
#     # metrics:  (B,K)
#     B, Fdim = features.shape
#     _, Kdim = metrics.shape

#     feats = (features - features.mean(0)) / (features.std(0) + 1e-6)
#     mets = (metrics - metrics.mean(0)) / (metrics.std(0) + 1e-6)

#     corr = torch.zeros(Fdim, Kdim, device=features.device)
#     for i in range(Fdim):
#         for j in range(Kdim):
#             corr[i, j] = (feats[:, i] * mets[:, j]).mean()
#     return corr.abs().mean()


# # ------------------------------------------------------------
# # Neuro-Fuzzy Layer
# # ------------------------------------------------------------
# class FuzzyLayer(nn.Module):
#     def __init__(self, feat_dim=16, mf_per_feat=3, freeze_dims=None):
#         super().__init__()
#         self.feat_dim = feat_dim
#         self.mf_per_feat = mf_per_feat

#         # membership centers and scales (learnable)
#         self.mf_centers = nn.Parameter(torch.randn(feat_dim, mf_per_feat) * 0.1)
#         self.mf_scales = nn.Parameter(torch.ones(feat_dim, mf_per_feat) * 0.5)

#         # rule masks per-feature (F, MF)
#         self.rule_masks = nn.Parameter(torch.randn(feat_dim, mf_per_feat) * 0.1)

#         # consequent vectors per-feature: map fuzzy weight to Δadh, v_x, v_y (F, 3)
#         self.consequents = nn.Parameter(torch.randn(feat_dim, 3) * 0.1)

#         # dims to freeze (parameter indices)
#         self.freeze_dims = list(freeze_dims) if freeze_dims is not None else []

#     def freeze_selected_dims(self):
#         # Set requires_grad False for parameters corresponding to freeze_dims
#         with torch.no_grad():
#             for d in self.freeze_dims:
#                 if 0 <= d < self.feat_dim:
#                     self.mf_centers[d].requires_grad = False
#                     self.mf_scales[d].requires_grad = False
#                     self.rule_masks[d].requires_grad = False
#                     self.consequents[d].requires_grad = False

#     def forward(self, x):
#         """
#         x: (B, F) - raw feature projection coming from encoder
#         returns:
#            f_feat: (B, F) - per-feature aggregated membership score
#            action_raw: (B, 3) - aggregated consequent action
#         """
#         B, Fdim = x.shape
#         x_exp = x.unsqueeze(-1)  # (B,F,1)
#         centers = self.mf_centers.unsqueeze(0)  # (1,F,MF)
#         scales = (self.mf_scales ** 2).unsqueeze(0) + 1e-6  # (1,F,MF)

#         mf = torch.exp(-((x_exp - centers) ** 2) / (2 * scales))  # (B,F,MF)

#         masked = mf * self.rule_masks.unsqueeze(0)  # (B,F,MF)
#         f_feat = masked.sum(-1)  # (B,F)

#         # produce action vector by weighted consequents
#         action_raw = torch.matmul(f_feat, self.consequents)  # (B,3)

#         # freeze important dims (parameter-level)
#         self.freeze_selected_dims()

#         return f_feat, action_raw


# # ------------------------------------------------------------
# # Main Actor–Critic with fuzzy block
# # ------------------------------------------------------------
# class NeuroFuzzyActorCritic(nn.Module):
#     """
#     Backwards-compatible constructor:
#        NeuroFuzzyActorCritic(in_ch=4, patch_size=5, feat_dim=16, action_dim=3,
#                              fuzzy_features=None, n_mfs=None, n_rules=None, freeze_dims=[...])
#     """

#     def __init__(
#         self,
#         in_ch=4,
#         patch_size=5,
#         feat_dim=16,
#         action_dim=3,
#         fuzzy_features=None,  # alias for feat_dim
#         n_mfs=None,  # alias for mf_per_feat
#         n_rules=None,  # ignored (kept for API compat)
#         freeze_dims=None,
#     ):
#         super().__init__()

#         # -----------------------------
#         # Backwards compatibility
#         # -----------------------------
#         if fuzzy_features is not None:
#             feat_dim = int(fuzzy_features)

#         mf_per_feat = int(n_mfs) if n_mfs is not None else 3

#         # n_rules is ignored because rules are implicit (feat_dim * mf_per_feat),
#         # but accept it for API compatibility.
#         _ = n_rules  # no-op

#         # default freeze dims if not provided
#         if freeze_dims is None:
#             freeze_dims = [1, 3, 4, 7, 11, 12, 13, 14]

#         # map compatibility names to local names used previously
#         C = int(in_ch)
#         S = int(patch_size)
#         Fdim = int(feat_dim)
#         A = int(action_dim)

#         # CNN encoder
#         self.encoder_conv = nn.Sequential(
#             nn.Conv2d(C, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#         )
#         self.flatten = nn.Linear(32 * S * S, Fdim)

#         # fuzzy layer
#         self.fuzzy = FuzzyLayer(feat_dim=Fdim, mf_per_feat=mf_per_feat, freeze_dims=freeze_dims)

#         # actor MLP → mean action (includes concatenated fuzzy action)
#         self.actor = nn.Sequential(
#             nn.Linear(Fdim + 3, 64),  # fuzzy_actions is fixed to 3 dims (Δadh, vx, vy)
#             nn.ReLU(),
#             nn.Linear(64, A),
#         )

#         # critic MLP
#         self.critic = nn.Sequential(nn.Linear(Fdim, 64), nn.ReLU(), nn.Linear(64, 1))

#         # auxiliary disentanglement head (predict interpretable metrics)
#         self.aux_head = nn.Sequential(nn.Linear(Fdim, 64), nn.ReLU(), nn.Linear(64, 4))

#         # action log-std parameter for stochastic policy compatibility
#         self.logstd = nn.Parameter(torch.zeros(A) - 1.0)

#         # bookkeeping
#         self.feat_dim = Fdim
#         self.action_dim = A
#         self.patch_size = S
#         self.in_ch = C
#         self.mf_per_feat = mf_per_feat

#     # -------------------------
#     def patches_to_metrics(self, patches):
#         # patches: (B,4,H,W)
#         # compute interpretable metrics for correlation / aux loss
#         A = patches[:, 0].reshape(patches.size(0), -1).mean(1)
#         B = patches[:, 1].reshape(patches.size(0), -1).mean(1)
#         adh = patches[:, 2].reshape(patches.size(0), -1).mean(1)
#         crowd = A + B
#         return torch.stack([adh, A, B, crowd], dim=1)

#     # -------------------------
#     def encode(self, patches):
#         h = self.encoder_conv(patches)
#         h = h.reshape(h.size(0), -1)
#         f = self.flatten(h)
#         return f

#     # -------------------------
#     def forward(self, patches):
#         """
#         Standard forward used by training and inference.
#         Returns:
#            mu: (B, A) action mean
#            value: (B,1)
#            fuzzy_feats: (B, F)
#         """
#         f = self.encode(patches)  # (B, Fdim)
#         fuzzy_feats, fuzzy_actions = self.fuzzy(f)  # (B, F), (B, 3)
#         actor_input = torch.cat([f, fuzzy_actions], dim=1)
#         mu = self.actor(actor_input)
#         value = self.critic(f)
#         return mu, value, fuzzy_feats

#     # -------------------------
#     def act(self, patches, deterministic=False):
#         """
#         Convenience: returns sampled action (or mean if deterministic).
#         """
#         mu, _, _ = self.forward(patches)
#         std = self.logstd.exp().unsqueeze(0)
#         dist = Normal(mu, std)
#         if deterministic:
#             return mu
#         return dist.rsample()

#     # -------------------------
#     def get_action_and_value(self, patches, deterministic=False):
#         """
#         Backwards-compatible method used in your scripts.
#         Returns:
#            action, logprob, value, mean, logstd
#         """
#         mu, value, fuzzy_feats = self.forward(patches)
#         std = self.logstd.exp().unsqueeze(0)
#         dist = Normal(mu, std)
#         if deterministic:
#             action = mu
#         else:
#             action = dist.rsample()
#         logprob = dist.log_prob(action).sum(dim=-1)
#         return action, logprob, value, mu, self.logstd

#     # -------------------------
#     def get_fuzzy_activations(self, patches):
#         """
#         Returns the encoder projection (the input to the fuzzy layer).
#         This is useful for disentanglement regularizers.
#         Shape: (B, feat_dim)
#         """
#         with torch.no_grad():
#             f = self.encode(patches)
#         return f

#     # -------------------------
#     def get_rule_info(self):
#         """
#         Produce a rule-like summary compatible with interpretation utilities.
#         We synthesize 'rules' by treating each feature as a rule:
#            masks: (R=F, F, MF) -- rule i strongly references feature i's MF distribution
#            consequents: (R=F, action_dim) -- per-feature consequent
#            scales: (R=F,) -- ones
#         """
#         F = self.fuzzy.feat_dim
#         MF = self.fuzzy.mf_per_feat
#         device = self.fuzzy.rule_masks.device

#         masks = torch.zeros((F, F, MF), device=device)
#         center_masks = F.softmax(self.fuzzy.rule_masks, dim=-1).detach()  # (F, MF)
#         for r in range(F):
#             masks[r, r, :] = center_masks[r, :]

#         consequents = self.fuzzy.consequents.detach().clone()  # (F, 3)
#         scales = torch.ones(F, device=device)

#         return masks.cpu(), consequents.cpu(), scales.cpu()

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

        # log std for Gaussian policy
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
        std = self.logstd.exp().unsqueeze(0).expand_as(mu)
        dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = dist.rsample()

        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, mu, std
