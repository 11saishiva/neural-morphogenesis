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

import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

class SortingEnv:
    """
    Cell-sorting environment with 'global' or 'local' observation modes.

    Changes included:
    - curriculum support in reset(use_curriculum=...)
    - flexible local action shape handling (accepts (B,N) or (B,N,1))
    - deterministic seeding support
    - 'done' return (currently False always; API-compatible)
    - progress-to-goal metric and helper setter
    - robust info dictionary with CPU numpy values for easy logging
    """

    def __init__(self, H=64, W=64, device='cpu', gamma_motion=0.1,
                 steps_per_action=6, obs_mode='local',
                 curriculum_prob=0.0, curriculum_bias=0.12,
                 seed=None, goal_sort_idx=0.03):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode
        self.dca = DCA().to(self.device)
        self.state = None

        # Curriculum / goal
        self.curriculum_prob = float(curriculum_prob)
        self.curriculum_bias = float(curriculum_bias)
        self.goal_sort_idx = float(goal_sort_idx)

        # RNG for deterministic resets if requested
        self._seed = None
        if seed is not None:
            self.seed(seed)

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

    def seed(self, seed):
        """Set deterministic seed for torch (best-effort)."""
        self._seed = int(seed)
        torch.manual_seed(self._seed)

    def set_goal_sort_idx(self, goal):
        """Set a target sort index considered 'complete' for progress calculation."""
        self.goal_sort_idx = float(goal)

    def progress_percent(self, sort_idx):
        """Return percent to goal in [0,1] for a tensor of shape (B,)."""
        if self.goal_sort_idx <= 0:
            return torch.zeros_like(sort_idx)
        pct = (sort_idx / self.goal_sort_idx).clamp(min=0.0, max=1.0)
        return pct

    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device).view(1,1,1,self.W).repeat(B,1,self.H,1)
        return x

    def reset(self, B=1, pA=0.5, use_curriculum=None):
        """
        Reset the environment.
        - B: batch size
        - pA: base left-type fraction (float between 0 and 1)
        - use_curriculum: None (use curriculum_prob), True, or False
        """
        # Decide curriculum application
        if use_curriculum is None:
            apply_curriculum = (torch.rand(B) < self.curriculum_prob).to(torch.bool)
        else:
            apply_curriculum = torch.ones(B, dtype=torch.bool) if bool(use_curriculum) else torch.zeros(B, dtype=torch.bool)

        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)

        # compute per-sample pA with curriculum bias applied for samples where apply_curriculum is True
        pA = float(pA)
        # apply curriculum_bias where requested
        pA_vec = torch.full((B,), pA, device=self.device, dtype=torch.float32)
        if apply_curriculum.any():
            # bump towards left (TYPE_A) by curriculum_bias, clip to [0,1]
            pA_vec[apply_curriculum] = torch.clamp(pA_vec[apply_curriculum] + self.curriculum_bias, 0.0, 1.0)

        # expand pA_vec to match shapes and apply similar mixing to types channels
        # safe broadcast: multiply first channel, complement to second
        types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA_vec.view(B,1,1)
        types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1.0 - pA_vec).view(B,1,1)
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

        # Return observation
        return self.get_observation()

    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        elif self.obs_mode == 'local':
            patches_coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
            # extract_local_patches may return (patches, coords) or patches only; handle both
            if isinstance(patches_coords, tuple) or isinstance(patches_coords, list):
                patches, coords = patches_coords
                return patches, coords
            else:
                return patches_coords, None
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    def _sorting_index(self, state):
        # A: (B, H, W) after indexing channel TYPE_A
        A = state[:, TYPE_A]
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1,2])
        right = A[:, :, mid:].mean(dim=[1,2])
        return torch.abs(left - right)

    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]

        # Flexible action handling for 'local' observation mode
        if self.obs_mode == 'local':
            N = self.H * self.W
            # Accept both (B, N) and (B, N, A)
            if actions.dim() == 2 and actions.shape[0] == B and actions.shape[1] == N:
                actions = actions.unsqueeze(-1)  # (B, N, 1)
            if actions.dim() != 3 or actions.shape[1] != N:
                raise ValueError(f"Expected local actions shape (B, N) or (B, N, A). Got {tuple(actions.shape)}.")
            # Now convert to the internal (B, A, H, W) layout by transposing and reshaping
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

        # run dynamics for steps_per_action
        with torch.no_grad():
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            self.state = s.detach().clone()

            # compute diagnostics
            e = interfacial_energy(self.state).detach()      # (B,)
            mpen = motion_penalty(actions.detach()).detach() # (B,)
            sort_idx = self._sorting_index(self.state).detach()

            # delta_sort (compute before we update last)
            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

            # EMA smoothing
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

            # Reward computation
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

            # Done flag (API-compatible); not implementing episode termination now => all False
            done = torch.zeros(B, dtype=torch.bool, device=self.device)

            # percent to goal metric
            percent_to_goal = self.progress_percent(sort_idx)

            # Prepare info with cpu numpy values for easy logging
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
                "percent_to_goal": percent_to_goal.cpu(),
                "goal_sort_idx": torch.tensor(self.goal_sort_idx),
            }

        # Return: observation, reward (B,), done (B,) and info. Keep reward as float tensor.
        return self.get_observation(), reward.detach(), done, info

    def current_state(self):
        return self.state.detach().clone()
