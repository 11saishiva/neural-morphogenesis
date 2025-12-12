# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


# class SortingEnv:
#     """
#     Cell-sorting environment with 'global' or 'local' observation modes.
#     Local observations: (B, N, C, p, p) where N = H*W.

#     Notes:
#     - Actions expected:
#       * obs_mode == 'local'  -> actions shape (B, N, A) where A == 3 (adh, vx, vy)
#       * obs_mode == 'global' -> actions shape (B, 3, H, W)
#     """

#     def __init__(
#         self, H=64, W=64, device='cpu', gamma_motion=0.1,
#         steps_per_action=6, obs_mode='local'
#     ):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         assert obs_mode in ('local', 'global')
#         self.obs_mode = obs_mode

#         # Dynamics model
#         self.dca = DCA().to(self.device)
#         self.state = None

#         # --- Reward shaping coefficients (final tuned defaults) ---
#         self.sort_weight   = 200.0    # main multiplier for sort-related reward (kept moderate)
#         self.sort_bonus    = 2.0
#         self.energy_weight = 1.0
#         self.motion_weight = 0.03
#         self.reward_clip   = 50.0
#         self.term_clip     = 50.0

#         # EMA smoothing of delta_sort
#         self.sort_ema_alpha = 0.4
#         self._sort_ema = None
#         self._last_sort_idx = None

#         # RMS normalizer for pos_delta
#         self.pos_delta_rms_alpha = 0.20
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6

#         # Scaling factor for sorting index (kept moderate)
#         self.SORT_AMPLIFY = 200.0

#         # Diagnostics
#         self._raw_sort_ma20 = None
#         self._raw_sort_ma50 = None
#         self._env_step = 0

#     # -------------------------
#     # Helpers
#     # -------------------------
#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device)
#         x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         return x

#     def reset(self, B=1, pA=0.5):
#         # initialize types with soft probabilities biased by pA
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

#         B_actual = state.shape[0]

#         # initialize EMA and RMS trackers
#         self._sort_ema = torch.zeros(B_actual, device=self.device)
#         self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

#         # initial sorting index baseline
#         with torch.no_grad():
#             raw = self._sorting_index(self.state).detach()
#             self._last_sort_idx = (raw * self.SORT_AMPLIFY).clone()
#             # initialize moving averages
#             self._raw_sort_ma20 = raw.clone()
#             self._raw_sort_ma50 = raw.clone()

#         self._env_step = 0
#         return self.get_observation()

#     def get_observation(self):
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         else:
#             patches, coords = extract_local_patches(
#                 self.state.detach().clone(), patch_size=5
#             )
#             return patches, coords

#     def _sorting_index(self, state):
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1, 2])
#         right = A[:, :, mid:].mean(dim=[1, 2])
#         return torch.abs(left - right)

#     # -------------------------
#     # Step
#     # -------------------------
#     def step(self, actions):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]
#         self._env_step += 1

#         # Validate & reshape actions
#         if self.obs_mode == 'local':
#             # actions must be (B, N, A) where N = H*W
#             if not (actions.dim() == 3 and actions.shape[0] == B):
#                 raise ValueError(f"For local mode expected actions (B, N, A); got {actions.shape}")
#             N = self.H * self.W
#             if actions.shape[1] != N:
#                 raise ValueError(f"Local actions second dim must be H*W={N}; got {actions.shape[1]}")
#             # transpose to (B, A, H, W) for DCA
#             actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)
#         else:
#             # global mode: expect (B, 3, H, W)
#             if not (actions.dim() == 4 and actions.shape == (B, 3, self.H, self.W)):
#                 raise ValueError(f"For global mode expected (B,3,H,W). Got {actions.shape}")

#         with torch.no_grad():
#             # Apply dynamics (DCA) for k steps
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             self.state = s.detach().clone()

#             # energy and motion
#             e = interfacial_energy(self.state).detach()           # (B,)
#             mpen = motion_penalty(actions.detach()).detach()      # (B,)

#             # sorting index and amplify
#             raw_sort_idx = self._sorting_index(self.state).detach()  # tensor (B,)
#             sort_idx = raw_sort_idx * self.SORT_AMPLIFY

#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             self._last_sort_idx = sort_idx.detach().clone()

#             # EMA of delta-sort
#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)

#             alpha = float(self.sort_ema_alpha)
#             self._sort_ema = (1 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

#             pos_delta = torch.relu(self._sort_ema)  # tensor (B,)

#             # RMS normalization of pos_delta (running variance-like)
#             if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
#                 self._pos_delta_rms = torch.ones_like(pos_delta)

#             beta = float(self.pos_delta_rms_alpha)
#             sq = pos_delta ** 2
#             self._pos_delta_rms = (1 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
#             running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

#             norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)  # tensor (B,)

#             # -----------------------------
#             # moving averages (diagnostics)
#             # -----------------------------
#             if (self._raw_sort_ma20 is None) or (self._raw_sort_ma20.shape[0] != raw_sort_idx.shape[0]):
#                 self._raw_sort_ma20 = raw_sort_idx.clone()
#                 self._raw_sort_ma50 = raw_sort_idx.clone()
#             else:
#                 alpha20 = 2.0 / (20.0 + 1.0)
#                 alpha50 = 2.0 / (50.0 + 1.0)
#                 self._raw_sort_ma20 = (1.0 - alpha20) * self._raw_sort_ma20 + alpha20 * raw_sort_idx
#                 self._raw_sort_ma50 = (1.0 - alpha50) * self._raw_sort_ma50 + alpha50 * raw_sort_idx

#             # -----------------------------
#             # reward composition
#             # -----------------------------
#             sort_term   = self.sort_weight * norm_pos_delta
#             bonus_term  = self.sort_bonus * raw_sort_idx
#             energy_term = -self.energy_weight * e
#             motion_term = -self.motion_weight * mpen

#             # clip per-term (element-wise)
#             sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             reward = reward / float(self.steps_per_action)
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             # -----------------------------
#             # scalar helpers that accept both tensors and floats
#             # -----------------------------
#             def _scalar(x):
#                 # Accepts: tensor, numpy scalar, python float
#                 if isinstance(x, torch.Tensor):
#                     return float(x.detach().cpu().mean())
#                 try:
#                     return float(x)
#                 except Exception:
#                     # fallback safe conversion
#                     return float(torch.as_tensor(x).cpu().mean())

#             # compute scalars (always produce python floats)
#             raw_mean = _scalar(raw_sort_idx)
#             ma20_mean = _scalar(self._raw_sort_ma20)
#             ma50_mean = _scalar(self._raw_sort_ma50)
#             reward_mean = _scalar(reward)
#             sort_term_mean = _scalar(sort_term)
#             bonus_term_mean = _scalar(bonus_term)
#             energy_term_mean = _scalar(energy_term)
#             motion_term_mean = _scalar(motion_term)
#             pos_delta_mean = _scalar(pos_delta)
#             norm_pos_delta_mean = _scalar(norm_pos_delta)
#             running_scale_mean = _scalar(running_scale)
#             interfacial_energy_mean = _scalar(e)
#             motion_penalty_mean = _scalar(mpen)

#             # periodic minimal logging
#             if (self._env_step % 10) == 0:
#                 print(
#                     f"[SORT-MA] step={self._env_step} "
#                     f"raw={raw_mean:.6e} ma20={ma20_mean:.6e} ma50={ma50_mean:.6e} "
#                     f"reward_mean={reward_mean:.6f} norm_pos_delta={norm_pos_delta_mean:.6e}",
#                     flush=True,
#                 )

#             reward_components = {
#                 "reward_mean": reward_mean,
#                 "sort_term_mean": sort_term_mean,
#                 "bonus_term_mean": bonus_term_mean,
#                 "energy_term_mean": energy_term_mean,
#                 "motion_term_mean": motion_term_mean,
#                 "raw_sort_idx_mean": raw_mean,
#                 "sort_idx_mean": _scalar(sort_idx),
#                 "pos_delta_mean": pos_delta_mean,
#                 "norm_pos_delta_mean": norm_pos_delta_mean,
#                 "interfacial_energy_mean": interfacial_energy_mean,
#                 "motion_penalty_mean": motion_penalty_mean,
#                 "running_scale_mean": running_scale_mean,
#                 "raw_sort_ma20_mean": ma20_mean,
#                 "raw_sort_ma50_mean": ma50_mean,
#                 "steps_per_action": float(self.steps_per_action),
#             }

#             # Full tensor info for debugging (returned as CPU tensors)
#             info = {
#                 "interfacial_energy": e.cpu(),
#                 "motion_penalty": mpen.cpu(),
#                 "raw_sort_index": raw_sort_idx.cpu(),
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
#                 "raw_sort_ma20": self._raw_sort_ma20.cpu(),
#                 "raw_sort_ma50": self._raw_sort_ma50.cpu(),
#                 "reward_components": reward_components,
#             }

#         # Return (obs, reward tensor, info dict)
#         return self.get_observation(), reward.detach(), info

#     # ---------------------------------------------------------
#     def current_state(self):
#         return self.state.detach().clone()


import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


class SortingEnv:
    """
    Stable, production-ready Sorting environment.
    Obs modes:
      - 'local'  -> get_observation() returns (patches, coords)
      - 'global' -> get_observation() returns full state (B,C,H,W)

    Actions expected:
      - local: (B, N, A) where N = H*W, A = 3 (delta_adh, vx, vy)
      - global: (B, 3, H, W)
    """

    def __init__(
        self, H=64, W=64, device='cpu', gamma_motion=0.1,
        steps_per_action=6, obs_mode='local'
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        assert obs_mode in ('local', 'global')
        self.obs_mode = obs_mode

        # Dynamics model
        self.dca = DCA().to(self.device)
        self.state = None

        # --- Reward shaping (tuned for stability) ---
        # Moderated amplifications to avoid spikes seen in logs
        self.sort_weight   = 150.0    # main multiplier for normalized sorting signal
        self.sort_bonus    = 2.0
        self.energy_weight = 1.0
        self.motion_weight = 0.03

        # clipping thresholds (reduced from previous 50 -> 20)
        self.term_clip     = 20.0
        self.reward_clip   = 20.0

        # EMA smoothing of delta_sort (slower than instantaneous, but reactive)
        self.sort_ema_alpha = 0.20
        self._sort_ema = None
        self._last_sort_idx = None

        # RMS normalizer for pos_delta
        # Use smaller alpha so running variance is robust to spikes
        self.pos_delta_rms_alpha = 0.05
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

        # Critical scaling factor for sorting index (reduced)
        self.SORT_AMPLIFY = 100.0

        # Small lower floor for running_scale to avoid divide-by-tiny
        self._running_scale_floor = 1e-4

        # Small EMA for norm_pos_delta so it can't vanish instantly
        self._norm_pos_delta_ema = None
        self._norm_pos_delta_ema_alpha = 0.5

        # Diagnostics
        self._raw_sort_ma20 = None
        self._raw_sort_ma50 = None
        self._env_step = 0

    # -------------------------
    # Helpers
    # -------------------------
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device)
        x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        return x

    def reset(self, B=1, pA=0.5):
        # initialize types with soft probabilities biased by pA
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

        # initialize EMA and RMS trackers
        self._sort_ema = torch.zeros(B_actual, device=self.device)
        self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0
        self._norm_pos_delta_ema = torch.zeros(B_actual, device=self.device)

        # initial sorting index baseline
        with torch.no_grad():
            raw = self._sorting_index(self.state).detach()
            self._last_sort_idx = (raw * self.SORT_AMPLIFY).clone()
            # initialize moving averages for diagnostics
            self._raw_sort_ma20 = raw.clone()
            self._raw_sort_ma50 = raw.clone()

        self._env_step = 0
        return self.get_observation()

    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        else:
            patches, coords = extract_local_patches(
                self.state.detach().clone(), patch_size=5
            )
            return patches, coords

    def _sorting_index(self, state):
        A = state[:, TYPE_A]  # (B,H,W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])
        return torch.abs(left - right)

    # -------------------------
    # Step
    # -------------------------
    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]
        self._env_step += 1

        # Validate & reshape actions
        if self.obs_mode == 'local':
            if not (actions.dim() == 3 and actions.shape[0] == B):
                raise ValueError(f"For local mode expected actions (B, N, A); got {actions.shape}")
            N = self.H * self.W
            if actions.shape[1] != N:
                raise ValueError(f"Local actions second dim must be H*W={N}; got {actions.shape[1]}")
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)
        else:
            if not (actions.dim() == 4 and actions.shape == (B, 3, self.H, self.W)):
                raise ValueError(f"For global mode expected (B,3,H,W). Got {actions.shape}")

        with torch.no_grad():
            # Apply dynamics (DCA) for k steps
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            self.state = s.detach().clone()

            # energy and motion
            e = interfacial_energy(self.state).detach()           # (B,)
            mpen = motion_penalty(actions.detach()).detach()      # (B,)

            # sorting index and amplify
            raw_sort_idx = self._sorting_index(self.state).detach()  # (B,)
            sort_idx = raw_sort_idx * self.SORT_AMPLIFY

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

            # EMA of delta-sort
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            alpha = float(self.sort_ema_alpha)
            self._sort_ema = (1 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

            pos_delta = torch.relu(self._sort_ema)  # (B,)

            # RMS normalization of pos_delta (running variance-like)
            if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
                self._pos_delta_rms = torch.ones_like(pos_delta)

            beta = float(self.pos_delta_rms_alpha)
            sq = pos_delta ** 2
            self._pos_delta_rms = (1 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
            running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

            # enforce a sensible minimum floor (prevents huge norm when running_scale tiny)
            running_scale = torch.max(running_scale, torch.full_like(running_scale, self._running_scale_floor))

            norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)  # (B,)

            # small EMA on norm_pos_delta so it doesn't instantly become exactly zero
            if self._norm_pos_delta_ema is None or self._norm_pos_delta_ema.shape[0] != norm_pos_delta.shape[0]:
                self._norm_pos_delta_ema = torch.zeros_like(norm_pos_delta)
            nna = float(self._norm_pos_delta_ema_alpha)
            self._norm_pos_delta_ema = (1 - nna) * self._norm_pos_delta_ema.to(norm_pos_delta.device) + nna * norm_pos_delta
            # use the EMA value as the normalized sort signal (more stable)
            stable_norm_pos_delta = self._norm_pos_delta_ema

            # -----------------------------
            # moving averages (diagnostics)
            # -----------------------------
            if (self._raw_sort_ma20 is None) or (self._raw_sort_ma20.shape[0] != raw_sort_idx.shape[0]):
                self._raw_sort_ma20 = raw_sort_idx.clone()
                self._raw_sort_ma50 = raw_sort_idx.clone()
            else:
                alpha20 = 2.0 / (20.0 + 1.0)
                alpha50 = 2.0 / (50.0 + 1.0)
                self._raw_sort_ma20 = (1.0 - alpha20) * self._raw_sort_ma20 + alpha20 * raw_sort_idx
                self._raw_sort_ma50 = (1.0 - alpha50) * self._raw_sort_ma50 + alpha50 * raw_sort_idx

            # -----------------------------
            # reward composition
            # -----------------------------
            # use stable_norm_pos_delta so we avoid spuriously zero sort_term
            sort_term   = self.sort_weight * stable_norm_pos_delta
            bonus_term  = self.sort_bonus * raw_sort_idx
            energy_term = -self.energy_weight * e
            motion_term = -self.motion_weight * mpen

            # clip per-term (element-wise)
            sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
            bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
            energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
            motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

            reward = sort_term + bonus_term + energy_term + motion_term
            reward = reward / float(self.steps_per_action)
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            # -----------------------------
            # scalar helpers that accept both tensors and floats
            # -----------------------------
            def _scalar(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().mean())
                try:
                    return float(x)
                except Exception:
                    return float(torch.as_tensor(x).cpu().mean())

            # compute scalars (always produce python floats)
            raw_mean = _scalar(raw_sort_idx)
            ma20_mean = _scalar(self._raw_sort_ma20)
            ma50_mean = _scalar(self._raw_sort_ma50)
            reward_mean = _scalar(reward)
            sort_term_mean = _scalar(sort_term)
            bonus_term_mean = _scalar(bonus_term)
            energy_term_mean = _scalar(energy_term)
            motion_term_mean = _scalar(motion_term)
            pos_delta_mean = _scalar(pos_delta)
            norm_pos_delta_mean = _scalar(stable_norm_pos_delta)
            running_scale_mean = _scalar(running_scale)
            interfacial_energy_mean = _scalar(e)
            motion_penalty_mean = _scalar(mpen)

            # periodic minimal logging
            if (self._env_step % 10) == 0:
                print(
                    f"[SORT-MA] step={self._env_step} raw={raw_mean:.6e} "
                    f"ma20={ma20_mean:.6e} ma50={ma50_mean:.6e} reward_mean={reward_mean:.6f} "
                    f"norm_pos_delta={norm_pos_delta_mean:.6e}",
                    flush=True,
                )

            reward_components = {
                "reward_mean": reward_mean,
                "sort_term_mean": sort_term_mean,
                "bonus_term_mean": bonus_term_mean,
                "energy_term_mean": energy_term_mean,
                "motion_term_mean": motion_term_mean,
                "raw_sort_idx_mean": raw_mean,
                "sort_idx_mean": _scalar(sort_idx),
                "pos_delta_mean": pos_delta_mean,
                "norm_pos_delta_mean": norm_pos_delta_mean,
                "interfacial_energy_mean": interfacial_energy_mean,
                "motion_penalty_mean": motion_penalty_mean,
                "running_scale_mean": running_scale_mean,
                "raw_sort_ma20_mean": ma20_mean,
                "raw_sort_ma50_mean": ma50_mean,
                "steps_per_action": float(self.steps_per_action),
            }

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
                "norm_pos_delta": stable_norm_pos_delta.cpu(),
                "sort_term": sort_term.cpu(),
                "bonus_term": bonus_term.cpu(),
                "energy_term": energy_term.cpu(),
                "motion_term": motion_term.cpu(),
                "raw_sort_ma20": self._raw_sort_ma20.cpu(),
                "raw_sort_ma50": self._raw_sort_ma50.cpu(),
                "reward_components": reward_components,
            }

        return self.get_observation(), reward.detach(), info

    # ---------------------------------------------------------
    def current_state(self):
        return self.state.detach().clone()
