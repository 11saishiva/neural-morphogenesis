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

#         # --- Reward shaping coefficients: conservative research defaults ---
#         # These balance learning signal and stability for research runs.
#         self.sort_weight   = 800.0     # conservative: earlier working magnitude
#         self.sort_bonus    = 40.0      # conservative bonus for sort index
#         self.energy_weight = 1.0       # energy penalty
#         self.motion_weight = 0.03      # penalize motion at a reasonable level(was 0.12 before)
#         self.reward_clip   = 5.0       # overall clipping to avoid spikes

#         # Per-term clipping
#         self.term_clip = 5.0

#         # EMA smoothing for delta_sort (keeps directional signal)
#         self.sort_ema_alpha = 0.2
#         self._sort_ema = None
#         self._last_sort_idx = None

#         # Running RMS normalizer for pos_delta — safe init
#         self.pos_delta_rms_alpha = 0.05
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6


#         # Amount to amplify the raw sorting index by (critical to provide usable signal)
#         self.SORT_AMPLIFY = 1.0 #changed SORT_AMPLIFY from 1000 to 1

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
#             # compute raw sort index then store amplified last_sort
#             raw_sort = self._sorting_index(self.state).detach()
#             amp_sort = raw_sort * self.SORT_AMPLIFY
#             self._last_sort_idx = amp_sort.clone()

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

#             # compute raw sort index and amplified version for use in deltas/EMA
#             raw_sort_idx = self._sorting_index(self.state).detach()
#             sort_idx = raw_sort_idx * self.SORT_AMPLIFY  # AMPLIFIED (critical)
#             # keep a copy of raw sort in info as well for debugging

#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             # store last amplified sort idx
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

#             # Reward terms with the stronger incentives but with relaxed clipping
#             sort_term = self.sort_weight * norm_pos_delta
#             bonus_term = self.sort_bonus * (raw_sort_idx)  # raw sort index is a sensible linear bonus
#             energy_term = - (self.energy_weight * e)
#             motion_term = - (self.motion_weight * mpen)

#             # Clip per-term (looser now to allow amplified sort signal)
#             sort_term = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             # normalize by number of DCA steps applied per agent action
#             reward = reward / float(self.steps_per_action)

#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             info = {
#                 "interfacial_energy": e.cpu(),
#                 "motion_penalty": mpen.cpu(),
#                 "raw_sort_index": raw_sort_idx.cpu(),
#                 "sort_index": sort_idx.cpu(),             # amplified sort index (used in delta)
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
    Local observations: (B, N, C, p, p) where N = H*W.
    """

    def __init__(
        self, H=64, W=64, device='cpu', gamma_motion=0.1,
        steps_per_action=6, obs_mode='local'
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode

        self.dca = DCA().to(self.device)
        self.state = None

        # --- Reward shaping coefficients: “research-clean” defaults ---
        self.sort_weight   = 1600.0
        self.sort_bonus    = 40.0
        self.energy_weight = 1.0
        self.motion_weight = 0.03
        self.reward_clip   = 5.0
        self.term_clip     = 5.0

        # EMA smoothing of delta_sort
        self.sort_ema_alpha = 0.2
        self._sort_ema = None
        self._last_sort_idx = None

        # RMS normalizer for pos_delta
        self.pos_delta_rms_alpha = 0.05
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

        # Critical: scaling factor for sorting index
        self.SORT_AMPLIFY = 5000.0   # your current research-default


    # ---------------------------------------------------------
    # Helper: create a morphogen band
    # ---------------------------------------------------------
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device)
        x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        return x


    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------
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

        B_actual = state.shape[0]

        # initialize EMA of delta-sort
        self._sort_ema = torch.zeros(B_actual, device=self.device)

        with torch.no_grad():
            raw = self._sorting_index(self.state).detach()
            amp = raw * self.SORT_AMPLIFY
            self._last_sort_idx = amp.clone()

        # RMS normalizer initialised to 1
        self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

        return self.get_observation()


    # ---------------------------------------------------------
    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        elif self.obs_mode == 'local':
            patches, coords = extract_local_patches(
                self.state.detach().clone(), patch_size=5
            )
            return patches, coords
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")


    # ---------------------------------------------------------
    # Sorting index = difference of mean A on left vs right half
    # ---------------------------------------------------------
    def _sorting_index(self, state):
        A = state[:, TYPE_A]  # (B,H,W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])
        return torch.abs(left - right)


    # ---------------------------------------------------------
    # MAIN STEP FUNCTION
    # ---------------------------------------------------------
    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]

        # reshape local actions
        if self.obs_mode == 'local':
            N = self.H * self.W
            if actions.dim() != 3 or actions.shape[1] != N:
                raise ValueError(f"Expected (B, N, A). Got {actions.shape}")
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

        with torch.no_grad():
            # -------------------
            # Apply DCA for k steps
            # -------------------
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)

            self.state = s.detach().clone()

            # Compute energy + motion
            e = interfacial_energy(self.state).detach()           # (B,)
            mpen = motion_penalty(actions.detach()).detach()      # (B,)

            # Sorting index
            raw_sort_idx = self._sorting_index(self.state).detach()
            sort_idx = raw_sort_idx * self.SORT_AMPLIFY

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

            # EMA of delta-sort
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            α = float(self.sort_ema_alpha)
            self._sort_ema = (1 - α) * self._sort_ema.to(sort_idx.device) + α * delta_sort

            pos_delta = torch.relu(self._sort_ema)

            # RMS norm
            if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
                self._pos_delta_rms = torch.ones_like(pos_delta)

            β = float(self.pos_delta_rms_alpha)
            sq = pos_delta ** 2
            self._pos_delta_rms = (1 - β) * self._pos_delta_rms.to(pos_delta.device) + β * sq
            running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

            norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)

            # -----------------------------------------------------
            # REWARD TERMS (unchanged, but now we will add diagnostics)
            # -----------------------------------------------------
            sort_term  = self.sort_weight * norm_pos_delta
            bonus_term = self.sort_bonus * raw_sort_idx
            energy_term = -self.energy_weight * e
            motion_term = -self.motion_weight * mpen

            # Clip per-term
            sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
            bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
            energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
            motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

            reward = sort_term + bonus_term + energy_term + motion_term
            reward = reward / float(self.steps_per_action)
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            # -----------------------------------------------------
            # NEW DIAGNOSTICS (Action A)
            # -----------------------------------------------------
            def _scalar(x):
                return float(x.detach().cpu().mean())

            reward_components = {
                "reward_mean": _scalar(reward),
                "sort_term_mean": _scalar(sort_term),
                "bonus_term_mean": _scalar(bonus_term),
                "energy_term_mean": _scalar(energy_term),
                "motion_term_mean": _scalar(motion_term),
                "raw_sort_idx_mean": _scalar(raw_sort_idx),
                "sort_idx_mean": _scalar(sort_idx),
                "pos_delta_mean": _scalar(pos_delta),
                "interfacial_energy_mean": _scalar(e),
                "motion_penalty_mean": _scalar(mpen),
                "running_scale_mean": _scalar(running_scale),
                "steps_per_action": float(self.steps_per_action),
            }

            # Full tensor info for debugging
            info = {
                "interfacial_energy": e.cpu(),
                "motion_penalty": mpen.cpu(),
                "raw_sort_index": raw_sort_idx.cpu(),
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
                "reward_components": reward_components,   # NEW
            }

        return self.get_observation(), reward.detach(), info


    # ---------------------------------------------------------
    def current_state(self):
        return self.state.detach().clone()
