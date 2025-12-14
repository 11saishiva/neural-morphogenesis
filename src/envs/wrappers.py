# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from src.utils.metrics import (
#     interfacial_energy,
#     motion_penalty,
#     extract_local_patches,
# )


# class SortingEnv:
#     """
#     Sorting environment with stochastic-but-structured initialization.
#     Option 3: stochastic per episode, non-trivial task.
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

#         # reward weights (SAFE INITIAL VALUES)
#         self.purity_delta_weight = 50.0     # << FIX 1: sane scale
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
#     # reset (Option 3: stochastic per episode)
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

#         # purity baseline
#         with torch.no_grad():
#             self.prev_purity = self._sorting_index(self.state)

#         if self.obs_mode == "local":
#             return extract_local_patches(self.state, patch_size=5)
#         else:
#             return self.state.clone()

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

#             # purity
#             purity = self._sorting_index(self.state)
#             delta_purity = purity - self.prev_purity
#             self.prev_purity = purity.clone()

#             # penalties
#             energy = interfacial_energy(self.state)
#             motion = motion_penalty(actions)

#             # -----------------------------
#             # reward (SAFE + CLAMPED)
#             # -----------------------------
#             purity_term = torch.clamp(
#                 self.purity_delta_weight * delta_purity,
#                 -1.0,
#                 1.0,
#             )

#             reward = (
#                 purity_term
#                 + self.purity_anchor_weight * purity
#                 - self.energy_weight * energy
#                 - self.motion_weight * motion
#             )

#             # curriculum: gentle decay, never below 50
#             self.purity_delta_weight = max(
#                 50.0,
#                 self.purity_delta_weight * 0.9995
#             )

#             if self._env_step % 10 == 0:
#                 print(
#                     f"[ENV] step={self._env_step} "
#                     f"purity={purity.mean():.4f} "
#                     f"Δpurity={delta_purity.mean():+.4e} "
#                     f"reward={reward.mean().item():.4f}",
#                     flush=True,
#                 )

#             info = {
#                 "purity": purity.detach().cpu(),
#                 "delta_purity": delta_purity.detach().cpu(),
#                 "energy": energy.detach().cpu(),
#                 "motion": motion.detach().cpu(),
#             }

#         if self.obs_mode == "local":
#             obs = extract_local_patches(self.state, patch_size=5)
#         else:
#             obs = self.state.clone()

#         return obs, reward, info

#     # ------------------------------------------------------------------
#     def current_state(self):
#         return self.state.clone()

import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from src.utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


class SortingEnv:
    """
    Sorting environment with HYBRID observation:
    local patches + broadcasted global purity channel.
    Fully compatible with train_local_sorting.py (unchanged).
    """

    def __init__(
        self,
        H=64,
        W=64,
        device="cpu",
        gamma_motion=0.002,
        steps_per_action=1,
        obs_mode="local",
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode

        # dynamics
        self.dca = DCA().to(self.device)
        self.state = None

        # reward weights
        self.purity_delta_weight = 1000.0
        self.purity_anchor_weight = 0.05
        self.energy_weight = 1.0
        self.motion_weight = gamma_motion

        # bookkeeping
        self.prev_purity = None
        self._env_step = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device)
        x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        return x

    def _sorting_index(self, state):
        A = state[:, TYPE_A]  # (B,H,W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])
        return torch.abs(left - right)

    # ------------------------------------------------------------------
    # reset (stochastic per episode)
    # ------------------------------------------------------------------
    def reset(self, B=1, pA=0.5):
        self._env_step = 0

        # low-frequency spatial noise
        noise = torch.randn(B, 1, self.H, self.W, device=self.device)
        noise = F.avg_pool2d(noise, kernel_size=9, stride=1, padding=4)
        noise = torch.tanh(noise)

        # random orientation bias
        if torch.rand(1).item() < 0.5:
            bias = torch.linspace(-1, 1, self.W, device=self.device)
            bias = bias.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        else:
            bias = torch.linspace(-1, 1, self.H, device=self.device)
            bias = bias.view(1, 1, self.H, 1).repeat(B, 1, 1, self.W)

        logits = 0.8 * noise + 0.6 * bias
        probA = torch.sigmoid(logits)

        types = torch.cat([probA, 1.0 - probA], dim=1)
        types = F.softmax(types, dim=1)

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        morphogen = self._make_morphogen(B)
        center = torch.ones(B, 1, self.H, self.W, device=self.device)

        self.state = torch.cat(
            [types, adhesion, morphogen, center], dim=1
        ).detach()

        with torch.no_grad():
            self.prev_purity = self._sorting_index(self.state)

        return self._get_observation()

    # ------------------------------------------------------------------
    # HYBRID OBSERVATION
    # ------------------------------------------------------------------
    def _get_observation(self):
        if self.obs_mode != "local":
            return self.state.clone()

        patches, coords = extract_local_patches(self.state, patch_size=5)
        # patches: (B, N, C, P, P)

        with torch.no_grad():
            purity = self._sorting_index(self.state)  # (B,)
            purity = purity.view(-1, 1, 1, 1, 1)

        B, N, _, P, _ = patches.shape
        purity_channel = purity.expand(B, N, 1, P, P)

        patches = torch.cat([patches, purity_channel], dim=2)
        return patches, coords

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, actions):
        B = self.state.shape[0]
        self._env_step += 1

        # reshape actions if local
        if self.obs_mode == "local":
            actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

        actions = actions.to(self.device)

        with torch.no_grad():
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            self.state = s.detach()

            purity = self._sorting_index(self.state)
            delta_purity = purity - self.prev_purity
            self.prev_purity = purity.clone()

            energy = interfacial_energy(self.state)
            motion = motion_penalty(actions)

            reward = (
                self.purity_delta_weight * delta_purity
                + self.purity_anchor_weight * purity
                - self.energy_weight * energy
                - self.motion_weight * motion
            )

            # curriculum decay
            self.purity_delta_weight = max(
                50.0, self.purity_delta_weight * 0.9995
            )

            if self._env_step % 10 == 0:
                print(
                    f"[ENV] step={self._env_step} "
                    f"purity={purity.mean():.4e} "
                    f"Δpurity={delta_purity.mean():+.3e} "
                    f"reward={reward.mean():.4f}",
                    flush=True,
                )

            info = {
                "purity": purity.cpu(),
                "delta_purity": delta_purity.cpu(),
                "energy": energy.cpu(),
                "motion": motion.cpu(),
            }

        return self._get_observation(), reward, info

    # ------------------------------------------------------------------
    def current_state(self):
        return self.state.clone()
