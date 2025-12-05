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

import random
import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

class SortingEnv:
    """
    Cell-sorting environment with 'global' or 'local' observation modes.
    Local observations: (B, N, C, p, p) where N = H*W.

    Modifications included:
    - left/right-biased curriculum reset method (reset_with_mild_cluster)
    - curriculum probability to apply the curriculum on reset
    - increased sort_bonus for clearer gradient signal once sort_idx > noise
    """

    def __init__(self, H=64, W=64, device='cpu', gamma_motion=0.1,
                 steps_per_action=6, obs_mode='local'):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode
        self.dca = DCA().to(self.device)
        self.state = None

        # --- Tuned reward shaping coefficients (DECISIVE CHANGE) ---
        # Increased incentives for sorting while keeping safeguards.
        self.sort_weight = 1500.0      # kept from your tuned value
        self.sort_bonus = 400.0        # raised from 200.0 -> stronger linear incentive for sort_idx
        self.energy_weight = 2.0
        self.motion_weight = 0.08      # reduced from 0.12 -> allow more movement/exploration
        self.reward_clip = 10.0

        # Per-term clipping (helps avoid single-step spikes)
        self.term_clip = 3.0  # tightened from 5.0 to keep per-term contributions bounded

        # EMA smoothing for delta_sort — keeps directional signal
        self.sort_ema_alpha = 0.2
        self._sort_ema = None
        self._last_sort_idx = None

        # Running RMS normalizer for pos_delta — faster adaptation but safe init
        self.pos_delta_rms_alpha = 0.10  # increased from 0.05 -> adapt more quickly to signal
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

        # Curriculum behavior: with some probability, use a mild left/right bias
        # to explicitly align initial states with the sort_idx metric.
        self.curriculum_prob = 0.20  # 20% of resets use the curriculum by default
        self.curriculum_left_bias = 0.12  # amount to bias TYPE_A to the left (mild)

    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device).view(1,1,1,self.W).repeat(B,1,self.H,1)
        return x

    def reset_with_mild_cluster(self, B=1, pA=0.5, left_bias=None):
        """
        Curriculum version: gently bias TYPE_A to the left half and TYPE_B to the right.
        This directly aligns with the sort_idx definition (difference between left/right means).
        left_bias: magnitude of the mild bias. If None uses self.curriculum_left_bias.
        """
        if left_bias is None:
            left_bias = float(self.curriculum_left_bias)

        # Base random typing then nudged towards pA / (1-pA)
        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)
        types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
        types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
        types = F.softmax(types, dim=1)

        # Create left/right masks
        mid = self.W // 2
        left_mask = torch.zeros(self.H, self.W, device=self.device)
        right_mask = torch.zeros_like(left_mask)
        left_mask[:, :mid] = 1.0
        right_mask[:, mid:] = 1.0

        # Broadcast masks to batch
        left_mask = left_mask.unsqueeze(0)   # (1,H,W)
        right_mask = right_mask.unsqueeze(0) # (1,H,W)

        # Slight left/right bias — keep small so it's "mild"
        # Add bias to TYPE_A on left half, add same bias to TYPE_B on right half.
        types[:, TYPE_A] = types[:, TYPE_A] + left_bias * left_mask
        types[:, TYPE_B] = types[:, TYPE_B] + left_bias * right_mask

        # Renormalize the two-type channels across the type axis
        s = types[:, TYPE_A] + types[:, TYPE_B]
        # avoid division by zero
        s = s + 1e-12
        types[:, TYPE_A] = types[:, TYPE_A] / s
        types[:, TYPE_B] = types[:, TYPE_B] / s

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        morphogen = self._make_morphogen(B)
        center = torch.ones(B, 1, self.H, self.W, device=self.device)

        state = torch.cat([types, adhesion, morphogen, center], dim=1)
        return state.detach().clone()

    def reset(self, B=1, pA=0.5, use_curriculum=None):
        """
        Reset environment. By default uses curriculum with probability self.curriculum_prob.
        Pass use_curriculum=True/False to force behavior.
        """
        if use_curriculum is None:
            use_curriculum = (random.random() < float(self.curriculum_prob))

        if use_curriculum:
            state = self.reset_with_mild_cluster(B=B, pA=pA)
        else:
            types = torch.rand(B, 2, self.H, self.W, device=self.device)
            types = F.softmax(types, dim=1)
            types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
            types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
            types = F.softmax(types, dim=1)

            adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
            morphogen = self._make_morphogen(B)
            center = torch.ones(B, 1, self.H, self.W, device=self.device)

            state = torch.cat([types, adhesion, morphogen, center], dim=1)

        self.state = state.detach().clone()

        # initialize per-batch bookkeeping
        B_actual = state.shape[0]
        self._sort_ema = torch.zeros(B_actual, device=self.device)
        with torch.no_grad():
            cur_sort = self._sorting_index(self.state).detach()
            self._last_sort_idx = cur_sort.clone()

        # initialize RMS normalizer to a safe value (1.0)
        self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

        return self.get_observation()

    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        elif self.obs_mode == 'local':
            patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
            return patches, coords
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    def _sorting_index(self, state):
        A = state[:, TYPE_A]  # (B,H,W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1,2])
        right = A[:, :, mid:].mean(dim=[1,2])
        return torch.abs(left - right)

    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]

        if self.obs_mode == 'local':
            N = self.H * self.W
            if actions.dim() != 3 or actions.shape[1] != N:
                raise ValueError(f"Expected local actions shape (B, N, A). Got {actions.shape}")
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

        with torch.no_grad():
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            self.state = s.detach().clone()

            e = interfacial_energy(self.state).detach()      # (B,)
            mpen = motion_penalty(actions.detach()).detach() # (B,)
            sort_idx = self._sorting_index(self.state).detach()

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

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

            # Reward terms with the new stronger incentives
            sort_term = self.sort_weight * norm_pos_delta
            bonus_term = self.sort_bonus * sort_idx
            energy_term = - (self.energy_weight * e)
            motion_term = - (self.motion_weight * mpen)

            # Clip per-term
            sort_term = torch.clamp(sort_term, -self.term_clip, self.term_clip)
            bonus_term = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
            energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
            motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

            reward = sort_term + bonus_term + energy_term + motion_term
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

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

        return self.get_observation(), reward.detach(), info

    def current_state(self):
        return self.state.detach().clone()
