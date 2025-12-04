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
        self.sort_weight = 1e4      # as applied earlier
        self.sort_bonus = 800.0     # small bump from 500 -> rewards steady configurations
        self.energy_weight = 2.0
        self.motion_weight = 0.6
        self.reward_clip = 10.0

        # EMA smoothing for delta_sort — tiny alpha to keep signal directional
        self.sort_ema_alpha = 0.2   # 0.1 is small but effective
        self._sort_ema = None        # will be initialized in reset per-batch
        self._last_sort_idx = None   # store last sort_idx for delta computation

        # track last sort index per-batch so we can reward progress
        self._last_sort_idx = None

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
        self._last_sort_idx = self._sorting_index(self.state).detach().clone()


        # initialize last_sort_idx to the current sorting index (detached)
        with torch.no_grad():
            cur_sort = self._sorting_index(self.state).detach()
            # store on device but we'll use detached cpu tensors for rewards/infos
            self._last_sort_idx = cur_sort.clone()

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

    # ----------------------
    # Experiment toggles (tweak these on the env instance)
    # - keep defaults so baseline behavior remains intact
    # ----------------------
    # If True, reward uses sort_idx directly (stable) instead of noisy delta-based positive EMA only.
        use_sort_idx_in_reward = getattr(self, "use_sort_idx_in_reward", False)

    # If True, apply EMA smoothing to delta_sort before using it.
        enable_ema = getattr(self, "enable_sort_ema", False)
        ema_alpha = float(getattr(self, "sort_ema_alpha", 0.05))  # small by default

    # If True, disable motion penalty term entirely (ablation).
        disable_motion_penalty = getattr(self, "disable_motion_penalty", False)

    # If True, normalize each term (running mean/std) before weighted combination.
        normalize_terms = getattr(self, "normalize_reward_terms", True)
        norm_momentum = float(getattr(self, "norm_momentum", 0.99))
        norm_eps = 1e-6

    # Optionally clip actions to prevent explosion (experiment).
        if getattr(self, "clip_actions", False):
            clip_val = float(getattr(self, "action_clip", 0.5))
            actions = actions.clamp(-clip_val, clip_val)

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

        # ----------------------
        # EMA smoothing (optional) - on delta_sort
        # ----------------------
            if enable_ema:
            # initialize EMA if not present or batch size changed
                if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                    self._sort_ema = torch.zeros_like(sort_idx, device=sort_idx.device, dtype=sort_idx.dtype)
            # ensure same device/dtype
                self._sort_ema = ((1.0 - ema_alpha) * self._sort_ema.to(sort_idx.device) +
                                  ema_alpha * delta_sort)
                smoothed_delta = self._sort_ema
            else:
            # make sure _sort_ema exists for diagnostics (but keep zeros if disabled)
                if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                    self._sort_ema = torch.zeros_like(sort_idx, device=sort_idx.device, dtype=sort_idx.dtype)
                smoothed_delta = delta_sort

        # positive-only smoothed delta (for reward when using delta)
            pos_delta = torch.relu(smoothed_delta)

        # ----------------------
        # Running normalization (per-term running mean/std)
        # ----------------------
        # initialize running stats containers if absent
            if not hasattr(self, "_reward_running"):
            # stored as dict of (mean, var) tensors on CPU by default
                self._reward_running = {
                    "sort_idx": {"mean": torch.zeros(1), "var": torch.ones(1)},
                    "delta":    {"mean": torch.zeros(1), "var": torch.ones(1)},
                    "energy":   {"mean": torch.zeros(1), "var": torch.ones(1)},
                    "motion":   {"mean": torch.zeros(1), "var": torch.ones(1)}
                }

        def update_running(name, x):
            # x: (B,) tensor
            st = self._reward_running[name]
            # keep stats on same device temporarily, convert back to CPU storage
            mean = st["mean"].to(x.device)
            var = st["var"].to(x.device)
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            # exponential moving avg update
            mean = norm_momentum * mean + (1.0 - norm_momentum) * batch_mean
            var = norm_momentum * var + (1.0 - norm_momentum) * batch_var
            # store back (keep CPU tensors to avoid GPU memory growth across episodes)
            st["mean"] = mean.detach().cpu()
            st["var"] = var.detach().cpu()
            return mean, var

        # compute (and update) running stats
        if normalize_terms:
            sort_mean, sort_var = update_running("sort_idx", sort_idx)
            delta_mean, delta_var = update_running("delta", delta_sort)
            energy_mean, energy_var = update_running("energy", e)
            motion_mean, motion_var = update_running("motion", mpen)
            # compute per-sample normalized values
            norm_sort = (sort_idx - sort_mean.to(sort_idx.device)) / (torch.sqrt(sort_var.to(sort_idx.device)) + norm_eps)
            norm_delta = (delta_sort - delta_mean.to(delta_sort.device)) / (torch.sqrt(delta_var.to(delta_sort.device)) + norm_eps)
            norm_energy = (e - energy_mean.to(e.device)) / (torch.sqrt(energy_var.to(e.device)) + norm_eps)
            norm_motion = (mpen - motion_mean.to(mpen.device)) / (torch.sqrt(motion_var.to(mpen.device)) + norm_eps)
        else:
            # no normalization: use raw tensors
            norm_sort = sort_idx
            norm_delta = delta_sort
            norm_energy = e
            norm_motion = mpen

        # ----------------------
        # Compose reward (configurable)
        # ----------------------
        # weights (kept from env config)
        sort_weight = float(getattr(self, "sort_weight", 1.0))
        sort_bonus  = float(getattr(self, "sort_bonus", 0.0))
        energy_weight = float(getattr(self, "energy_weight", 1.0))
        motion_weight = 0.0 if disable_motion_penalty else float(getattr(self, "motion_weight", 1.0))

        if use_sort_idx_in_reward:
            # reward primarily on current (normalized) sort index + small encouragement for positive delta
            small_delta_scale = float(getattr(self, "small_delta_scale", 0.1))
            reward_signal = sort_weight * norm_sort + sort_bonus * norm_sort
            # add small immediate positive delta (if available), use relu to keep positive-only drive
            reward_signal = reward_signal + small_delta_scale * torch.relu(norm_delta)
        else:
            # reward primarily on sustained positive delta (smoothed) + small bonus for current sort index
            delta_scale = float(getattr(self, "delta_scale", 1.0))
            reward_signal = delta_scale * torch.relu(pos_delta) + (sort_bonus * norm_sort)

        # final composition subtracting costs
        reward = reward_signal - (energy_weight * norm_energy) - (motion_weight * norm_motion)

        # clip for numerical stability
        reward = torch.clamp(reward, -float(self.reward_clip), float(self.reward_clip))

        # ----------------------
        # package info (include both raw and normalized for diagnostics)
        # ----------------------
        info = {
            "interfacial_energy": e.cpu(),
            "motion_penalty": mpen.cpu(),
            "sort_index": sort_idx.cpu(),
            "delta_sort_index": delta_sort.cpu(),
            "smoothed_delta": smoothed_delta.cpu(),
            "pos_delta": pos_delta.cpu(),
            # normalized terms (helpful for debugging/plots)
            "norm_sort_index": norm_sort.cpu(),
            "norm_delta_sort_index": norm_delta.cpu(),
            "norm_interfacial_energy": norm_energy.cpu(),
            "norm_motion_penalty": norm_motion.cpu(),
            # running stats snapshot (CPU scalars)
            "running_stats": {
                "sort_mean": self._reward_running["sort_idx"]["mean"].clone(),
                "sort_var":  self._reward_running["sort_idx"]["var"].clone(),
                "delta_mean": self._reward_running["delta"]["mean"].clone(),
                "delta_var":  self._reward_running["delta"]["var"].clone(),
                "energy_mean": self._reward_running["energy"]["mean"].clone(),
                "energy_var":  self._reward_running["energy"]["var"].clone(),
                "motion_mean": self._reward_running["motion"]["mean"].clone(),
                "motion_var":  self._reward_running["motion"]["var"].clone(),
            }
        }

        # detach reward for RL algorithm (return CPU tensors in info)
        return_obs = self._get_observation()  # keep your existing observation builder
        return return_obs, reward.detach(), info




    def current_state(self):
        return self.state.detach().clone()
