# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from src.utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

# class RunningMeanStd:
#     def __init__(self, eps=1e-4, device="cpu"):
#         self.mean = torch.zeros(1, device=device)
#         self.var = torch.ones(1, device=device)
#         self.count = eps

#     def update(self, x):
#         batch_mean = x.mean()
#         batch_var = x.var(unbiased=False)
#         batch_count = x.numel()

#         delta = batch_mean - self.mean
#         tot_count = self.count + batch_count

#         new_mean = self.mean + delta * batch_count / tot_count
#         m_a = self.var * self.count
#         m_b = batch_var * batch_count
#         M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count

#         self.mean = new_mean
#         self.var = M2 / tot_count
#         self.count = tot_count

#     def normalize(self, x):
#         return (x - self.mean) / torch.sqrt(self.var + 1e-8)

# class SortingEnv:
#     """
#     Sorting environment with HYBRID observation:
#     local patches + broadcasted global purity channel.
#     Fully compatible with train_local_sorting.py (unchanged).
#     """

#     def __init__(
#         self,
#         H=64,
#         W=64,
#         device="cpu",
#         gamma_motion=0.002,
#         steps_per_action=1,
#         obs_mode="local",
#     ):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode

#         # dynamics
#         self.dca = DCA().to(self.device)
#         self.state = None
#         self.reward_rms = RunningMeanStd(device=self.device)

#         # reward weights
#         self.purity_delta_weight = 1000.0
#         self.purity_anchor_weight = 0.05
#         self.energy_weight = 1.0
#         self.motion_weight = gamma_motion

#         # bookkeeping
#         self.prev_purity = None
#         self._env_step = 0

#     # ------------------------------------------------------------------
#     # helpers
#     # ------------------------------------------------------------------
#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device)
#         x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         return x

#     def _sorting_index(self, state):
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1, 2])
#         right = A[:, :, mid:].mean(dim=[1, 2])
#         return torch.abs(left - right)

#     # ------------------------------------------------------------------
#     # reset (stochastic per episode)
#     # ------------------------------------------------------------------
#     def reset(self, B=1, pA=0.5):
#         self._env_step = 0

#         # low-frequency spatial noise
#         noise = torch.randn(B, 1, self.H, self.W, device=self.device)
#         noise = F.avg_pool2d(noise, kernel_size=9, stride=1, padding=4)
#         noise = torch.tanh(noise)

#         # random orientation bias
#         if torch.rand(1).item() < 0.5:
#             bias = torch.linspace(-1, 1, self.W, device=self.device)
#             bias = bias.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         else:
#             bias = torch.linspace(-1, 1, self.H, device=self.device)
#             bias = bias.view(1, 1, self.H, 1).repeat(B, 1, 1, self.W)

#         logits = 0.8 * noise + 0.6 * bias
#         probA = torch.sigmoid(logits)

#         types = torch.cat([probA, 1.0 - probA], dim=1)
#         types = F.softmax(types, dim=1)

#         adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
#         morphogen = self._make_morphogen(B)
#         center = torch.ones(B, 1, self.H, self.W, device=self.device)

#         self.state = torch.cat(
#             [types, adhesion, morphogen, center], dim=1
#         ).detach()

#         with torch.no_grad():
#             self.prev_purity = self._sorting_index(self.state)
#             self.initial_purity = self.prev_purity.clone()

#         return self._get_observation()

#     # ------------------------------------------------------------------
#     # HYBRID OBSERVATION
#     # ------------------------------------------------------------------
#     def _get_observation(self):
#         if self.obs_mode != "local":
#             return self.state.clone()

#         patches, coords = extract_local_patches(self.state, patch_size=5)
#         # patches: (B, N, C, P, P)

#         with torch.no_grad():
#             purity = self._sorting_index(self.state)  # (B,)
#             purity = purity.view(-1, 1, 1, 1, 1)

#         B, N, _, P, _ = patches.shape
#         purity_channel = purity.expand(B, N, 1, P, P)

#         patches = torch.cat([patches, purity_channel], dim=2)
#         return patches, coords

#     # ------------------------------------------------------------------
#     # step
#     # ------------------------------------------------------------------
#     def step(self, actions):
#         B = self.state.shape[0]
#         self._env_step += 1

#         # reshape actions if local
#         if self.obs_mode == "local":
#             actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

#         actions = actions.to(self.device)

#         with torch.no_grad():
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             self.state = s.detach()

#             # --- purity ---
#             purity = self._sorting_index(self.state)

#             # baseline-relative shaping (CORRECT)
#             purity_gain = purity - self.initial_purity

#             # --- penalties (YOU MUST COMPUTE THESE) ---
#             energy = interfacial_energy(self.state)
#             motion = motion_penalty(actions)

#             # --- reward ---
#             reward = (
#                 5.0 * purity_gain
#                 - self.energy_weight * energy
#                 - self.motion_weight * motion
#             )

#             if self._env_step % 10 == 0:
#                 print(
#                     f"[ENV] step={self._env_step} "
#                     f"purity={purity.mean():.4e} "
#                     f"gain={purity_gain.mean():+.4e} "
#                     f"reward={reward.mean().item():+.4f}",
#                     flush=True,
#                 )

#             info = {
#                 "purity": purity.detach().cpu(),
#                 "purity_gain": purity_gain.detach().cpu(),
#                 "energy": energy.detach().cpu(),
#                 "motion": motion.detach().cpu(),
#             }

#         return self._get_observation(), reward, info

#     # ------------------------------------------------------------------
#     def current_state(self):
#         return self.state.clone()
import torch
import torch.nn.functional as F

from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from src.utils.metrics import extract_local_patches


# ---------------------------------------------------------------------
# Mixing metric
# ---------------------------------------------------------------------
def local_interface_mixing(state):
    """
    Measures A/B interface density.
    state: (B, C, H, W)
    """
    A = state[:, TYPE_A]  # (B, H, W)

    dx = torch.abs(A[:, :, 1:] - A[:, :, :-1])
    dy = torch.abs(A[:, 1:, :] - A[:, :-1, :])

    return dx.mean(dim=(1, 2)) + dy.mean(dim=(1, 2))


# ---------------------------------------------------------------------
# Sorting Environment
# ---------------------------------------------------------------------
class SortingEnv:
    """
    Morphogenetic sorting environment
    (fully compatible with train_local_sorting.py)
    """

    def __init__(
        self,
        H=64,
        W=64,
        device="cpu",
        gamma_motion=0.01,
        steps_per_action=1,
        obs_mode="local",
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode

        # dynamics
        self.dca = DCA().to(self.device)
        self.state = None

        # -------------------------------
        # Reward weights (FIXED)
        # -------------------------------
        self.mixing_weight = 1000.0        # amplify real signal
        self.energy_weight = 0.01          # very weak penalty
        self.motion_weight = gamma_motion  # weak regularizer
        self.persistence_weight = 0.05     # meaningful bonus

        # bookkeeping
        self.prev_mixing = None
        self.sorted_steps = None
        self._env_step = 0

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------
    def reset(self, B=1, pA=0.5):
        self._env_step = 0

        noise = torch.randn(B, 1, self.H, self.W, device=self.device)
        noise = F.avg_pool2d(noise, kernel_size=9, stride=1, padding=4)
        noise = torch.tanh(noise)

        morphogen = torch.linspace(0, 1, self.W, device=self.device)
        morphogen = morphogen.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)

        logits = 0.7 * noise + 0.6 * (morphogen - 0.5)
        probA = torch.sigmoid(logits)

        types = torch.cat([probA, 1.0 - probA], dim=1)
        types = F.softmax(types, dim=1)

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        center = torch.ones(B, 1, self.H, self.W, device=self.device)

        self.state = torch.cat(
            [types, adhesion, morphogen, center], dim=1
        ).detach()

        self.prev_mixing = local_interface_mixing(self.state)
        self.sorted_steps = torch.zeros_like(self.prev_mixing)

        return self._get_obs()

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------
    def step(self, actions):
        B = self.state.shape[0]
        self._env_step += 1

        if self.obs_mode in ["local", "hybrid"]:
            actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

        actions = actions.to(self.device)

        with torch.no_grad():
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            self.state = s.detach()

            # -----------------------------
            # Metrics
            # -----------------------------
            mixing = local_interface_mixing(self.state)
            delta_mixing = self.prev_mixing - mixing
            self.prev_mixing = mixing.clone()

            threshold = 9e-4
            is_sorted = (mixing < threshold).float()
            self.sorted_steps = is_sorted * (self.sorted_steps + 1)

            persistence_bonus = self.persistence_weight * self.sorted_steps

            energy = (self.state[:, ADH] ** 2).mean(dim=(1, 2))
            motion = actions.abs().mean(dim=(1, 2, 3))

            # -----------------------------
            # FIXED reward
            # -----------------------------
            reward = (
                self.mixing_weight * delta_mixing
                + persistence_bonus
                - self.energy_weight * energy
                - self.motion_weight * motion
            )

            # optional but safe
            reward = torch.clamp(reward, -1.0, 1.0)

            if self._env_step % 10 == 0:
                print(
                    f"[ENV] step={self._env_step} "
                    f"mixing={mixing.mean():.4e} "
                    f"gain={delta_mixing.mean():+.4e} "
                    f"reward={reward.mean():+.4f}",
                    flush=True,
                )

            info = {
                "mixing": mixing.detach().cpu(),
                "delta_mixing": delta_mixing.detach().cpu(),
                "sorted_steps": self.sorted_steps.detach().cpu(),
                "energy": energy.detach().cpu(),
                "motion": motion.detach().cpu(),
            }

        return self._get_obs(), reward, info

    # -----------------------------------------------------------------
    # Legacy metric (required)
    # -----------------------------------------------------------------
    def _sorting_index(self, state):
        A = state[:, TYPE_A]
        mid = A.shape[-1] // 2
        left = A[:, :, :mid].mean(dim=(1, 2))
        right = A[:, :, mid:].mean(dim=(1, 2))
        return torch.abs(left - right)

    # -----------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------
    def _get_obs(self):
        if self.obs_mode == "local":
            return extract_local_patches(self.state, patch_size=5)

        if self.obs_mode == "hybrid":
            patches, coords = extract_local_patches(self.state, patch_size=5)
            global_mean = self.state.mean(dim=(2, 3), keepdim=True)
            return (patches, coords, global_mean)

        return self.state.clone()

    def current_state(self):
        return self.state.clone()
