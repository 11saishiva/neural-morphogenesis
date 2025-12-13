# import torch
# import torch.nn.functional as F

# from .dca import DCA, TYPE_A, TYPE_B
# from src.utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


# class SortingEnv:
#     """
#     Stable sorting environment with non-trivial initialization.
#     """

#     def __init__(
#         self,
#         H=64,
#         W=64,
#         device="cpu",
#         gamma_motion=0.01,
#         steps_per_action=1,
#         obs_mode="local",
#     ):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode

#         # dynamics
#         self.dca = DCA().to(self.device)

#         # reward weights (unchanged)
#         self.sort_weight = 300.0
#         self.sort_bonus = 3.0
#         self.energy_weight = 1.0
#         self.motion_weight = gamma_motion

#         self.SORT_AMPLIFY = 80.0

#         # trackers
#         self._last_sort_idx = None
#         self._sort_ema = None
#         self._pos_delta_rms = None
#         self._norm_pos_delta_ema = None
#         self._env_step = 0

#         self.state = None

#     # ------------------------------------------------------------------
#     # helpers (THESE WERE MISSING — now restored)
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
#     # reset (Option 1: non-trivial task)
#     # ------------------------------------------------------------------
#     def reset(self, B=1, pA=0.5):
#         device = self.device

#         # create spatial pattern (NO batch here)
#         yy, xx = torch.meshgrid(
#             torch.linspace(0, 1, self.H, device=self.device),
#             torch.linspace(0, 1, self.W, device=self.device),
#             indexing="ij"
#         )

#         freq = torch.randint(2, 5, (1,), device=self.device).item()
#         phase = torch.rand(1, device=self.device) * 2 * torch.pi

#         pattern = torch.sin(freq * torch.pi * xx + phase) * torch.sin(
#             freq * torch.pi * yy + phase
#         )                      # (H, W)

#         # expand to batch
#         pattern = pattern.unsqueeze(0).repeat(B, 1, 1)   # (B, H, W)
#         pattern = pattern.unsqueeze(1)                   # (B, 1, H, W)
#         noise = 0.05 * torch.randn_like(pattern)
#         logits_A = pattern + noise
#         logits_B = -pattern + noise

#         types = torch.cat([logits_A, logits_B], dim=1)
#         types = F.softmax(types, dim=1)

#         adhesion = 0.4 + 0.2 * torch.rand(B, 1, self.H, self.W, device=device)
#         morphogen = self._make_morphogen(B)
#         center = torch.ones(B, 1, self.H, self.W, device=device)

#         self.state = torch.cat([types, adhesion, morphogen, center], dim=1).detach()

#         # trackers
#         self._sort_ema = torch.zeros(B, device=device)
#         self._pos_delta_rms = torch.ones(B, device=device)
#         self._norm_pos_delta_ema = torch.zeros(B, device=device)

#         with torch.no_grad():
#             raw = self._sorting_index(self.state)
#             self._last_sort_idx = raw * self.SORT_AMPLIFY

#         self._env_step = 0
#         return self.get_observation()

#     # ------------------------------------------------------------------
#     def get_observation(self):
#         if self.obs_mode == "global":
#             return self.state.clone()
#         patches, coords = extract_local_patches(self.state, patch_size=5)
#         return patches, coords

#     # ------------------------------------------------------------------
#     def step(self, actions):
#         B = self.state.shape[0]
#         self._env_step += 1

#         if self.obs_mode == "local":
#             actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

#         with torch.no_grad():
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             self.state = s.detach()

#             e = interfacial_energy(self.state)
#             m = motion_penalty(actions)

#             raw = self._sorting_index(self.state)
#             sort_idx = raw * self.SORT_AMPLIFY
#             delta = sort_idx - self._last_sort_idx
#             self._last_sort_idx = sort_idx.clone()

#             pos = torch.relu(delta)

#             reward = (
#                 self.sort_weight * pos
#                 + self.sort_bonus * raw
#                 - self.energy_weight * e
#                 - self.motion_weight * m
#             )
#             reward = reward / self.steps_per_action

#         info = {
#             "raw_sort_index": raw.cpu(),
#             "sort_index": sort_idx.cpu(),
#             "interfacial_energy": e.cpu(),
#             "motion_penalty": m.cpu(),
#         }

#         return self.get_observation(), reward.detach(), info

#     # ------------------------------------------------------------------
#     def current_state(self):
#         return self.state.clone()

import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from src.utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


class SortingEnv:
    """
    Sorting environment with stochastic-but-structured initialization.
    Compatible with train_local_sorting.py (unchanged).
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
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode

        # dynamics
        self.dca = DCA().to(self.device)
        self.state = None

        # reward weights
        self.purity_delta_weight = 1.0
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
    # reset (OPTION 3: stochastic per episode, non-trivial)
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

        # purity baseline
        with torch.no_grad():
            self.prev_purity = self._sorting_index(self.state)

        if self.obs_mode == "local":
            return extract_local_patches(self.state, patch_size=5)
        else:
            return self.state.clone()

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

            # purity
            purity = self._sorting_index(self.state)
            delta_purity = purity - self.prev_purity
            self.prev_purity = purity.clone()

            # penalties
            energy = interfacial_energy(self.state)
            motion = motion_penalty(actions)

            # reward (Option 3)
            reward = (
                self.purity_delta_weight * delta_purity
                + self.purity_anchor_weight * purity
                - self.energy_weight * energy
                - self.motion_weight * motion
            )

            reward = reward.mean()

            if self._env_step % 10 == 0:
                print(
                    f"[ENV] step={self._env_step} "
                    f"purity={purity.mean():.4f} "
                    f"Δpurity={delta_purity.mean():+.4e} "
                    f"reward={reward.item():.4f}",
                    flush=True,
                )

            info = {
                "purity": purity.detach().cpu(),
                "delta_purity": delta_purity.detach().cpu(),
                "energy": energy.detach().cpu(),
                "motion": motion.detach().cpu(),
            }

        if self.obs_mode == "local":
            obs = extract_local_patches(self.state, patch_size=5)
        else:
            obs = self.state.clone()

        return obs, reward, info

    # ------------------------------------------------------------------
    def current_state(self):
        return self.state.clone()
