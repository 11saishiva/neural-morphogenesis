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
#                  steps_per_action=1, obs_mode='local'):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode
#         self.dca = DCA().to(self.device)
#         self.state = None

#                              # Reward shaping coefficients (kept safe)
#         self.sort_weight = 1e4      # as applied earlier
#         self.sort_bonus = 800.0     # small bump from 500 -> rewards steady configurations
#         self.energy_weight = 2.0
#         self.motion_weight = 0.6
#         self.reward_clip = 10.0

#         # EMA smoothing for delta_sort — tiny alpha to keep signal directional
#         self.sort_ema_alpha = 0.2   # 0.1 is small but effective
#         self._sort_ema = None        # will be initialized in reset per-batch
#         self._last_sort_idx = None   # store last sort_idx for delta computation

#         # track last sort index per-batch so we can reward progress
#         self._last_sort_idx = None

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
#         # detach to prevent accidental autograd from environment state
#         self.state = state.detach().clone()
#                 # initialize per-batch bookkeeping for sorting EMA and last sort index
#         B_actual = state.shape[0]
#         self._sort_ema = torch.zeros(B_actual, device=self.device)
#         # set last_sort_idx to current sorting index so first delta is zero
#         self._last_sort_idx = self._sorting_index(self.state).detach().clone()


#         # initialize last_sort_idx to the current sorting index (detached)
#         with torch.no_grad():
#             cur_sort = self._sorting_index(self.state).detach()
#             # store on device but we'll use detached cpu tensors for rewards/infos
#             self._last_sort_idx = cur_sort.clone()

#         return self.get_observation()

#     def get_observation(self):
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         elif self.obs_mode == 'local':
#             # extract_local_patches expects a (B,C,H,W) tensor and returns (B,N,C,p,p)
#             patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
#             return patches, coords
#         else:
#             raise ValueError("obs_mode must be 'global' or 'local'")

#     def _sorting_index(self, state):
#         """
#         Sorting index: |mean(A_left) - mean(A_right)| (per-batch)
#         state: (B,C,H,W)
#         returns: (B,)
#         """
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1,2])
#         right = A[:, :, mid:].mean(dim=[1,2])
#         return torch.abs(left - right)


#     def step(self, actions):
#         """
#         actions:
#           - local mode: (B, N, A)  -> reshaped to (B, A, H, W)
#           - global mode: (B, A, H, W)
#         Returns (observation, reward (detached), info dict with cpu tensors)
#         """
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]

#         if self.obs_mode == 'local':
#             N = self.H * self.W
#             if actions.dim() != 3 or actions.shape[1] != N:
#                 raise ValueError(f"Expected local actions shape (B, N, A). Got {actions.shape}")
#             # reshape to (B, A, H, W)
#             actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

#         # run dynamics in no_grad to avoid building graph
#         with torch.no_grad():
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             # detach snapshot after dynamics
#             self.state = s.detach().clone()

#             # compute metrics using detached tensors
#             e = interfacial_energy(self.state).detach()      # (B,)
#             mpen = motion_penalty(actions.detach()).detach() # (B,)
#                         # compute current sorting index (B,)
#             sort_idx = self._sorting_index(self.state).detach()

#             # compute per-batch raw delta (current - last)
#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             # update last_sort_idx for next step
#             self._last_sort_idx = sort_idx.detach().clone()

#             # initialize EMA if not present (safety)
#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)

#             # EMA update (keeps signal directional and smooth)
#             alpha = float(self.sort_ema_alpha)
#             # move EMA to same device and dtype
#             self._sort_ema = (1.0 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

#             # positive-only smoothed delta
#             pos_delta = torch.relu(self._sort_ema)

#             # reward: strong encouragement for sustained positive progress + small bonus for current sort idx
#             reward = (self.sort_weight * pos_delta) + (self.sort_bonus * sort_idx) \
#                      - (self.energy_weight * e) - (self.motion_weight * mpen)

#             # clip for stability
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)


#             info = {
#                 "interfacial_energy": e.cpu(),
#                 "motion_penalty": mpen.cpu(),
#                 "sort_index": sort_idx.cpu(),
#                 "delta_sort_index": delta_sort.cpu()
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
    Local observations: (B, N, C, p, p) where N = H*W.
    """

    def __init__(self, H=64, W=64, device='cpu', gamma_motion=0.1,
                 steps_per_action=1, obs_mode='local'):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode
        self.dca = DCA().to(self.device)
        self.state = None

        # --- Tuned reward shaping coefficients (DECISIVE CHANGE) ---
        # Increased incentives for sorting while keeping safeguards.
        self.sort_weight = 1500.0      # raised from 800.0 -> stronger reward for sustained delta
        self.sort_bonus = 200.0        # raised from 40.0  -> stronger linear incentive for sort_idx
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

    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device).view(1,1,1,self.W).repeat(B,1,self.H,1)
        return x

    def reset(self, B=1, pA=0.5):
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
