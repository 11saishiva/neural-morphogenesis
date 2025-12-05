# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

# class SortingEnv:
#     """
#     Cell-sorting environment with 'global' or 'local' observation modes.
#     Local observations: (B, N, C, p, p) where N = H*W.
#     """

#     def __init__(self, H=64, W=64, device='cpu', gamma_motion=0.1,
#                  steps_per_action=6, obs_mode='local'):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode
#         self.dca = DCA().to(self.device)
#         self.state = None

#         # --- Tuned reward shaping coefficients (DECISIVE CHANGE) ---
#         # Increased incentives for sorting while keeping safeguards.
#         self.sort_weight = 1500.0      # raised from 800.0 -> stronger reward for sustained delta
#         self.sort_bonus = 200.0        # raised from 40.0  -> stronger linear incentive for sort_idx
#         self.energy_weight = 2.0
#         self.motion_weight = 0.08      # reduced from 0.12 -> allow more movement/exploration
#         self.reward_clip = 10.0

#         # Per-term clipping (helps avoid single-step spikes)
#         self.term_clip = 3.0  # tightened from 5.0 to keep per-term contributions bounded

#         # EMA smoothing for delta_sort — keeps directional signal
#         self.sort_ema_alpha = 0.2
#         self._sort_ema = None
#         self._last_sort_idx = None

#         # Running RMS normalizer for pos_delta — faster adaptation but safe init
#         self.pos_delta_rms_alpha = 0.10  # increased from 0.05 -> adapt more quickly to signal
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6

#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device).view(1,1,1,self.W).repeat(B,1,self.H,1)
#         return x

#     def reset(self, B=1, pA=0.5):
#         types = torch.rand(B, 2, self.H, self.W, device=self.device)
#         types = F.softmax(types, dim=1)
#         types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
#         types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
#         types = F.softmax(types, dim=1)

#         adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
#         morphogen = self._make_morphogen(B)
#         center = torch.ones(B, 1, self.H, self.W, device=self.device)

#         state = torch.cat([types, adhesion, morphogen, center], dim=1)
#         self.state = state.detach().clone()

#         # initialize per-batch bookkeeping
#         B_actual = state.shape[0]
#         self._sort_ema = torch.zeros(B_actual, device=self.device)
#         with torch.no_grad():
#             cur_sort = self._sorting_index(self.state).detach()
#             self._last_sort_idx = cur_sort.clone()

#         # initialize RMS normalizer to a safe value (1.0)
#         self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

#         return self.get_observation()

#     def get_observation(self):
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         elif self.obs_mode == 'local':
#             patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
#             return patches, coords
#         else:
#             raise ValueError("obs_mode must be 'global' or 'local'")

#     def _sorting_index(self, state):
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1,2])
#         right = A[:, :, mid:].mean(dim=[1,2])
#         return torch.abs(left - right)

#     def step(self, actions):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]

#         if self.obs_mode == 'local':
#             N = self.H * self.W
#             if actions.dim() != 3 or actions.shape[1] != N:
#                 raise ValueError(f"Expected local actions shape (B, N, A). Got {actions.shape}")
#             actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

#         with torch.no_grad():
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             self.state = s.detach().clone()

#             e = interfacial_energy(self.state).detach()      # (B,)
#             mpen = motion_penalty(actions.detach()).detach() # (B,)
#             sort_idx = self._sorting_index(self.state).detach()

#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             self._last_sort_idx = sort_idx.detach().clone()

#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)

#             alpha = float(self.sort_ema_alpha)
#             self._sort_ema = (1.0 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

#             pos_delta = torch.relu(self._sort_ema)

#             # Running RMS normalization
#             if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
#                 self._pos_delta_rms = torch.ones_like(pos_delta) * 1.0

#             beta = float(self.pos_delta_rms_alpha)
#             sq = (pos_delta ** 2)
#             self._pos_delta_rms = (1.0 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
#             running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

#             norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)

#             # Reward terms with the new stronger incentives
#             sort_term = self.sort_weight * norm_pos_delta
#             bonus_term = self.sort_bonus * sort_idx
#             energy_term = - (self.energy_weight * e)
#             motion_term = - (self.motion_weight * mpen)

#             # Clip per-term
#             sort_term = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             info = {
#                 "interfacial_energy": e.cpu(),
#                 "motion_penalty": mpen.cpu(),
#                 "sort_index": sort_idx.cpu(),
#                 "delta_sort_index": delta_sort.cpu(),
#                 "smoothed_delta": self._sort_ema.cpu(),
#                 "pos_delta": pos_delta.cpu(),
#                 "pos_delta_rms": self._pos_delta_rms.cpu(),
#                 "running_scale": running_scale.cpu(),
#                 "norm_pos_delta": norm_pos_delta.cpu(),
#                 "sort_term": sort_term.cpu(),
#                 "bonus_term": bonus_term.cpu(),
#                 "energy_term": energy_term.cpu(),
#                 "motion_term": motion_term.cpu(),
#             }

#         return self.get_observation(), reward.detach(), info

#     def current_state(self):
#         return self.state.detach().clone()
# src/envs/wrappers.py

# # Use absolute imports so `PYTHONPATH=src` works reliably.
# from envs.dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from utils.metrics import interfacial_energy, motion_penalty, extract_local_patches
import torch
from typing import Tuple
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


class SortingEnv:
    """
    Cell-sorting environment with 'global' or 'local' observation modes.

    Local observations: (B, N, C, p, p) where N = H*W.
    """

    def __init__(
        self,
        H: int = 64,
        W: int = 64,
        device: str = "cpu",
        gamma_motion: float = 0.1,
        steps_per_action: int = 1,  # lowered debug default (was 6)
        obs_mode: str = "local",
        debug: bool = False,
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = int(steps_per_action)
        self.obs_mode = obs_mode
        self.debug = bool(debug)

        # instantiate DCA on the correct device
        self.dca = DCA().to(self.device)
        self.state = None

        # --- Tuned reward shaping coefficients (DECISIVE CHANGE) ---
        self.sort_weight = 1500.0
        self.sort_bonus = 200.0
        self.energy_weight = 2.0
        self.motion_weight = 0.08
        self.reward_clip = 10.0

        # Per-term clipping
        self.term_clip = 3.0

        # EMA smoothing for delta_sort
        self.sort_ema_alpha = 0.2
        self._sort_ema = None
        self._last_sort_idx = None

        # Running RMS normalizer for pos_delta
        self.pos_delta_rms_alpha = 0.10
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

    def _make_morphogen(self, B: int) -> torch.Tensor:
        # shape (B,1,H,W)
        x = torch.linspace(0, 1, self.W, device=self.device).view(1, 1, 1, self.W)
        x = x.repeat(B, 1, self.H, 1)
        return x

    def reset(self, B: int = 1, pA: float = 0.5):
        # initialize types (B,2,H,W)
        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)
        types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
        types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
        types = F.softmax(types, dim=1)

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        morphogen = self._make_morphogen(B)
        center = torch.ones(B, 1, self.H, self.W, device=self.device)

        state = torch.cat([types, adhesion, morphogen, center], dim=1)
        # keep a detached clone as canonical state
        self.state = state.detach().clone()

        # initialize per-batch bookkeeping
        B_actual = self.state.shape[0]
        self._sort_ema = torch.zeros(B_actual, device=self.device)
        with torch.no_grad():
            cur_sort = self._sorting_index(self.state).detach()
            self._last_sort_idx = cur_sort.clone()

        # initialize RMS normalizer to a safe value (1.0)
        self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

        if self.debug:
            print(
                f"[env.reset] B={B_actual}, state.shape={self.state.shape}, "
                f"initial_sort={self._last_sort_idx.cpu().tolist()}"
            )

        return self.get_observation()

    def get_observation(self):
        if self.obs_mode == "global":
            return self.state.detach().clone()
        elif self.obs_mode == "local":
            patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
            return patches, coords
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    def _sorting_index(self, state: torch.Tensor) -> torch.Tensor:
        # A = state[:, TYPE_A] with shape (B,H,W)
        A = state[:, TYPE_A]
        mid = self.W // 2
        # left and right are (B,)
        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])
        return torch.abs(left - right)

    def step(self, actions: torch.Tensor) -> Tuple[object, torch.Tensor, dict]:
        """
        actions: either
          - global: (B, A, H, W) already in env layout, or
          - local: (B, N, A) where N = H*W (old format from your agent)
        Returns (observation, reward_tensor(B,), info_dict)
        """

        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        # ensure actions on our device
        actions = actions.to(self.device)

        B = self.state.shape[0]

        # handle local actions shape -> reshape to (B, A, H, W)
        if self.obs_mode == "local":
            N = self.H * self.W
            if actions.dim() == 3 and actions.shape[1] == N:
                # expected (B, N, A) -> transpose to (B, A, N)
                actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)
            elif actions.dim() == 4:
                # already a (B, A, H, W)
                pass
            else:
                raise ValueError(f"Expected local actions shape (B, N, A) or (B, A, H, W). Got {actions.shape}")

        # main env step: run DCA for steps_per_action iterations
        with torch.no_grad():
            s = self.state
            # guard: if someone accidentally sets huge steps_per_action, cap logging
            sp = max(1, int(self.steps_per_action))
            for i in range(sp):
                # use single-step DCA calls to keep control and allow prints when debug
                if self.debug and (i % 10 == 0):
                    print(f"[env.step] dca step {i+1}/{sp}")
                s = self.dca(s, actions, steps=1)
            # commit state
            self.state = s.detach().clone()

            # compute diagnostics
            e = interfacial_energy(self.state).detach()      # (B,)
            mpen = motion_penalty(actions.detach()).detach() # (B,)
            sort_idx = self._sorting_index(self.state).detach()

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            # keep last sort index
            self._last_sort_idx = sort_idx.detach().clone()

            # init sort_ema if needed
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            alpha = float(self.sort_ema_alpha)
            self._sort_ema = (1.0 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

            pos_delta = torch.relu(self._sort_ema)

            # Running RMS normalization
            if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
                self._pos_delta_rms = torch.ones_like(pos_delta) * 1.0

            beta = float(self.pos_delta_rms_alpha)
            sq = (pos_delta ** 2)
            self._pos_delta_rms = (1.0 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
            running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

            norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)

            # Reward terms
            sort_term = self.sort_weight * norm_pos_delta
            bonus_term = self.sort_bonus * sort_idx
            energy_term = - (self.energy_weight * e)
            motion_term = - (self.motion_weight * mpen)

            # Clip each term (per batch element)
            sort_term = torch.clamp(sort_term, -self.term_clip, self.term_clip)
            bonus_term = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
            energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
            motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

            reward = sort_term + bonus_term + energy_term + motion_term
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            # prepare info dictionary with cpu tensors (safe for printing/serialization)
            info = {
                "interfacial_energy": e.cpu(),
                "motion_penalty": mpen.cpu(),
                "sort_index": sort_idx.cpu(),
                "delta_sort_index": delta_sort.cpu(),
                "smoothed_delta": self._sort_ema.cpu(),
                "pos_delta": pos_delta.cpu(),
                "pos_delta_rms": self._pos_delta_rms.cpu(),
                "running_scale": running_scale.cpu(),
                "norm_pos_delta": norm_pos_delta.cpu(),
                "sort_term": sort_term.cpu(),
                "bonus_term": bonus_term.cpu(),
                "energy_term": energy_term.cpu(),
                "motion_term": motion_term.cpu(),
            }

            if self.debug:
                # print compact diagnostics
                print(
                    f"[env.step.debug] sort_idx mean={sort_idx.mean().item():.6f}, "
                    f"reward mean={reward.mean().item():.6f}, energy mean={e.mean().item():.6f}"
                )

        return self.get_observation(), reward.detach(), info

    def current_state(self):
        return self.state.detach().clone()


# Quick smoke demo if you execute this module directly (won't run when imported).
if __name__ == "__main__":
    # Only run demo if you explicitly want; toggle with this flag to avoid accidental long runs.
    debug_demo = True
    if debug_demo:
        print("wrappers.py demo: creating small env and doing one step")
        env = SortingEnv(H=32, W=32, device="cpu", steps_per_action=1, obs_mode="local", debug=True)
        obs = env.reset(B=2)
        # create a random action matching local obs format: (B, N, A). We'll assume A=1 channel
        N = env.H * env.W
        # Make a dummy action of zeros that will not cause big motion
        dummy_actions = torch.zeros(2, N, 1, dtype=torch.float32)
        obs2, reward, info = env.step(dummy_actions)
        print("demo done:", reward, info["sort_index"])
