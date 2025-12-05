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

        # Reward shaping coefficients (kept safe)
        self.sort_weight = 1e4      # as applied earlier (kept as a hyperparam)
        self.sort_bonus = 800.0     # linear bonus, keep as requested (not sqrt)
        self.energy_weight = 2.0
        self.motion_weight = 0.6
        self.reward_clip = 10.0

        # EMA smoothing for delta_sort — tiny alpha to keep signal directional
        self.sort_ema_alpha = 0.2   # 0.2 is small but effective
        self._sort_ema = None       # will be initialized in reset per-batch
        self._last_sort_idx = None  # store last sort_idx for delta computation

        # Running scale for normalization of pos_delta (safer normalization)
        self._pos_delta_scale = None

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
        # detach to prevent accidental autograd from environment state
        self.state = state.detach().clone()

        # initialize per-batch bookkeeping for sorting EMA and last sort index
        B_actual = state.shape[0]
        self._sort_ema = torch.zeros(B_actual, device=self.device)
        # set last_sort_idx to current sorting index so first delta is zero
        with torch.no_grad():
            cur_sort = self._sorting_index(self.state).detach()
            self._last_sort_idx = cur_sort.clone()

        # initialize pos-delta running scale (for normalization)
        self._pos_delta_scale = torch.zeros(B_actual, device=self.device)

        return self.get_observation()

    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        elif self.obs_mode == 'local':
            # extract_local_patches expects a (B,C,H,W) tensor and returns (B,N,C,p,p)
            patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
            return patches, coords
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    def _sorting_index(self, state):
        """
        Sorting index: |mean(A_left) - mean(A_right)| (per-batch)
        state: (B,C,H,W)
        returns: (B,)
        """
        A = state[:, TYPE_A]  # (B,H,W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1,2])
        right = A[:, :, mid:].mean(dim=[1,2])
        return torch.abs(left - right)


    def step(self, actions):
        """
        actions:
          - local mode: (B, N, A)  -> reshaped to (B, A, H, W)
          - global mode: (B, A, H, W)
        Returns (observation, reward (detached), info dict with cpu tensors)
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]

        if self.obs_mode == 'local':
            N = self.H * self.W
            if actions.dim() != 3 or actions.shape[1] != N:
                raise ValueError(f"Expected local actions shape (B, N, A). Got {actions.shape}")
            # reshape to (B, A, H, W)
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

        # run dynamics in no_grad to avoid building graph
        with torch.no_grad():
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            # detach snapshot after dynamics
            self.state = s.detach().clone()

            # compute metrics using detached tensors
            e = interfacial_energy(self.state).detach()      # (B,)
            mpen = motion_penalty(actions.detach()).detach() # (B,)
            # compute current sorting index (B,)
            sort_idx = self._sorting_index(self.state).detach()

            # compute per-batch raw delta (current - last)
            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            # update last_sort_idx for next step
            self._last_sort_idx = sort_idx.detach().clone()

            # initialize EMA if not present (safety)
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            # EMA update (keeps signal directional and smooth) for delta_sort
            alpha = float(self.sort_ema_alpha)
            self._sort_ema = (1.0 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

            # positive-only smoothed delta
            pos_delta = torch.relu(self._sort_ema)

            # -------------------------------
            # Safer normalization for pos_delta:
            # - faster EMA for scale (beta)
            # - larger eps floor
            # - clamp normalized progress to [0, 1.0]
            # - slightly smaller effective sort_weight at use time
            # -------------------------------
                        # -------------------------------
            # Safer RMS normalization for pos_delta:
            # - EMA of squares (RMS) for scale
            # - slower beta, larger eps floor
            # - clamp normalized progress to [0, 1.0]
            # - reduce effective sort_weight
            # -------------------------------
            eps = 1e-2                      # larger safety floor
            beta = 0.02                     # slower update for scale (less reactive)
            
            # ensure _pos_delta_scale is present and on correct device
            # Now _pos_delta_scale stores EMA of squared pos_delta (for RMS)
            if self._pos_delta_scale is None or self._pos_delta_scale.shape[0] != pos_delta.shape[0]:
                self._pos_delta_scale = torch.zeros_like(pos_delta, device=pos_delta.device)

            # update running EMA of squared pos_delta
            sq_pd = pos_delta * pos_delta
            self._pos_delta_scale = (1.0 - beta) * self._pos_delta_scale.to(sq_pd.device) + beta * sq_pd

            # RMS scale + eps floor
            scale = torch.sqrt(self._pos_delta_scale.clamp(min=0.0)) + eps

            # normalized positive delta, clipped to [0, 1]
            norm_pos = pos_delta / scale
            norm_pos = torch.clamp(norm_pos, 0.0, 1.0)

            # more conservative effective_sort_weight to prevent domination
            effective_sort_weight = float(self.sort_weight) * 0.2

            # reward: normalized progress + linear sort bonus - penalties
            reward = (effective_sort_weight * norm_pos) + (self.sort_bonus * sort_idx) \
                     - (self.energy_weight * e) - (self.motion_weight * mpen)

            # clip for stability
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            info = {
                "interfacial_energy": e.cpu(),
                "motion_penalty": mpen.cpu(),
                "sort_index": sort_idx.cpu(),
                "delta_sort_index": delta_sort.cpu(),
                "pos_delta_smoothed": self._sort_ema.cpu(),
                "pos_delta_scale": self._pos_delta_scale.cpu()
            }

        return self.get_observation(), reward.detach(), info

    def current_state(self):
        """
        Return a detached copy of the current state for logging/inspection.
        """
        if self.state is None:
            return None
        return self.state.detach().clone()
