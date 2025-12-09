# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


# class SortingEnv:
#     """
#     Cell-sorting environment with 'global' or 'local' observation modes.
#     Local observations: (B, N, C, p, p) where N = H*W.
#     """

#     def __init__(
#         self, H=64, W=64, device='cpu', gamma_motion=0.1,
#         steps_per_action=6, obs_mode='local'
#     ):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode

#         self.dca = DCA().to(self.device)
#         self.state = None

#         # --- Reward shaping coefficients: “research-clean” defaults ---
#         self.sort_weight   = 1600.0
#         self.sort_bonus    = 40.0
#         self.energy_weight = 1.0
#         self.motion_weight = 0.03
#         self.reward_clip   = 50.0
#         self.term_clip     = 50.0

#         # EMA smoothing of delta_sort
#         self.sort_ema_alpha = 0.2
#         self._sort_ema = None
#         self._last_sort_idx = None

#         # RMS normalizer for pos_delta
#         self.pos_delta_rms_alpha = 0.05
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6

#         # Critical: scaling factor for sorting index
#         self.SORT_AMPLIFY = 5000.0   # your current research-default


#     # ---------------------------------------------------------
#     # Helper: create a morphogen band
#     # ---------------------------------------------------------
#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device)
#         x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         return x


#     # ---------------------------------------------------------
#     # Reset environment
#     # ---------------------------------------------------------
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

#         B_actual = state.shape[0]

#         # initialize EMA of delta-sort
#         self._sort_ema = torch.zeros(B_actual, device=self.device)

#         with torch.no_grad():
#             raw = self._sorting_index(self.state).detach()
#             amp = raw * self.SORT_AMPLIFY
#             self._last_sort_idx = amp.clone()

#         # RMS normalizer initialised to 1
#         self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

#         return self.get_observation()


#     # ---------------------------------------------------------
#     def get_observation(self):
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         elif self.obs_mode == 'local':
#             patches, coords = extract_local_patches(
#                 self.state.detach().clone(), patch_size=5
#             )
#             return patches, coords
#         else:
#             raise ValueError("obs_mode must be 'global' or 'local'")


#     # ---------------------------------------------------------
#     # Sorting index = difference of mean A on left vs right half
#     # ---------------------------------------------------------
#     def _sorting_index(self, state):
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1, 2])
#         right = A[:, :, mid:].mean(dim=[1, 2])
#         return torch.abs(left - right)


#     # ---------------------------------------------------------
#     # MAIN STEP FUNCTION
#     # ---------------------------------------------------------
#     def step(self, actions):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]

#         # reshape local actions
#         if self.obs_mode == 'local':
#             N = self.H * self.W
#             if actions.dim() != 3 or actions.shape[1] != N:
#                 raise ValueError(f"Expected (B, N, A). Got {actions.shape}")
#             actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

#         with torch.no_grad():
#             # -------------------
#             # Apply DCA for k steps
#             # -------------------
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)

#             self.state = s.detach().clone()

#             # Compute energy + motion
#             e = interfacial_energy(self.state).detach()           # (B,)
#             mpen = motion_penalty(actions.detach()).detach()      # (B,)

#             # Sorting index
#             raw_sort_idx = self._sorting_index(self.state).detach()
#             sort_idx = raw_sort_idx * self.SORT_AMPLIFY

#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             self._last_sort_idx = sort_idx.detach().clone()

#             # EMA of delta-sort
#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)

#             α = float(self.sort_ema_alpha)
#             self._sort_ema = (1 - α) * self._sort_ema.to(sort_idx.device) + α * delta_sort

#             pos_delta = torch.relu(self._sort_ema)

#             # RMS norm
#             if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
#                 self._pos_delta_rms = torch.ones_like(pos_delta)

#             β = float(self.pos_delta_rms_alpha)
#             sq = pos_delta ** 2
#             self._pos_delta_rms = (1 - β) * self._pos_delta_rms.to(pos_delta.device) + β * sq
#             running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

#             norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)

#             # -----------------------------------------------------
#             # REWARD TERMS (unchanged, but now we will add diagnostics)
#             # -----------------------------------------------------
#             sort_term  = self.sort_weight * norm_pos_delta
#             bonus_term = self.sort_bonus * raw_sort_idx
#             energy_term = -self.energy_weight * e
#             motion_term = -self.motion_weight * mpen

#             # Clip per-term
#             sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             reward = reward / float(self.steps_per_action)
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             # -----------------------------------------------------
#             # NEW DIAGNOSTICS (Action A)
#             # -----------------------------------------------------
#             def _scalar(x):
#                 return float(x.detach().cpu().mean())

#             reward_components = {
#                 "reward_mean": _scalar(reward),
#                 "sort_term_mean": _scalar(sort_term),
#                 "bonus_term_mean": _scalar(bonus_term),
#                 "energy_term_mean": _scalar(energy_term),
#                 "motion_term_mean": _scalar(motion_term),
#                 "raw_sort_idx_mean": _scalar(raw_sort_idx),
#                 "sort_idx_mean": _scalar(sort_idx),
#                 "pos_delta_mean": _scalar(pos_delta),
#                 "interfacial_energy_mean": _scalar(e),
#                 "motion_penalty_mean": _scalar(mpen),
#                 "running_scale_mean": _scalar(running_scale),
#                 "steps_per_action": float(self.steps_per_action),
#             }

#             # Full tensor info for debugging
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
#                 "reward_components": reward_components,   # NEW
#             }

#         return self.get_observation(), reward.detach(), info


#     # ---------------------------------------------------------
#     def current_state(self):
#         return self.state.detach().clone()

# wrappers.py
"""
Reward / wrapper utilities including integrated sort-term handling.

Drop this file into your repo as the main wrappers module (replace or merge
with your existing wrappers.py). This file contains the safe, inlined logic
for computing an RL 'sort' contribution and applying it to rewards --
no extra files needed.

Key public function:
    apply_sort_term_to_reward(reward, raw_sort_index, cfg, debug=False)
        -> returns (new_reward, diagnostics_dict)

Cfg is a simple dict controlling behavior (see DEFAULT_CFG below).
"""

from typing import Any, Dict, Optional, Union
import math
import sys

# Optional torch support: if present, we accept tensors and convert safely.
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

Number = Union[float, int]


# ----------------------
# Default configuration
# ----------------------
DEFAULT_CFG: Dict[str, Any] = {
    # Multiply raw_sort_index by this (was often large, e.g. 5000)
    "SORT_AMPLIFY": 5000.0,
    # Running scale (EMA or other scalar applied afterwards). Avoid set to 0.
    "running_scale": 1.0,
    # If set (float>0), will soft- or hard-clip the applied sort to this magnitude.
    "term_clip": 0.05,
    # If set (float >= 0), any abs(pre_scaled) < zero_gate_threshold will be zeroed.
    # Set to None to disable gating.
    "zero_gate_threshold": 1e-4,
    # When gating is disabled this can help (visible test) to ensure reward composition
    # accepts additions. Set to None normally.
    "visible_test_value": None,
    # If True use soft clipping; otherwise use hard clipping.
    "use_soft_clip": True,
    # If True, bypass gating and zeroing behavior (useful for debug/run tests).
    "bypass_gating": False,
    # Avoid running_scale becoming exactly zero (min clamp).
    "min_running_scale": 1e-8,
}


# ----------------------
# Internal helpers
# ----------------------
def _to_float(x: Union[Number, "torch.Tensor"]) -> float:
    """Convert numeric or torch tensor to plain Python float safely."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        # detach + cpu + item is safe for 0-dim tensors
        try:
            return float(x.detach().cpu().item())
        except Exception:
            # fall back to convert via float()
            return float(x.detach().cpu().numpy().item())
    return float(x)


def _is_tensor(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


# ----------------------
# Core: compute applied sort
# ----------------------
def compute_applied_sort(
    raw_sort_index: Union[Number, "torch.Tensor"],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Compute the applied sort term from raw_sort_index using cfg.
    Returns diagnostics dict:
        {
           "raw_sort_index": float,
           "pre_scaled": float,  # raw * SORT_AMPLIFY
           "scaled_by_running": float,
           "applied_sort": float,
           "gating_reason": Optional[str],
           "clip_mode": Optional[str],
           "clip_denom": Optional[float],
           ... (echoed cfg used)
        }
    """
    cfg_use = dict(DEFAULT_CFG)
    if cfg:
        cfg_use.update(cfg)

    # Extract config with safe casts
    ampl = float(cfg_use.get("SORT_AMPLIFY", 1.0))
    running_scale = float(cfg_use.get("running_scale", 1.0))
    term_clip = cfg_use.get("term_clip", None)
    zero_gate_threshold = cfg_use.get("zero_gate_threshold", None)
    visible_test_value = cfg_use.get("visible_test_value", None)
    use_soft_clip = bool(cfg_use.get("use_soft_clip", True))
    bypass_gating = bool(cfg_use.get("bypass_gating", False))
    min_running_scale = float(cfg_use.get("min_running_scale", 1e-8))

    # Promote running_scale away from zero to avoid multiplied-to-zero cases
    running_scale = max(running_scale, min_running_scale)

    raw = _to_float(raw_sort_index)
    pre_scaled = raw * ampl
    scaled_by_running = pre_scaled * running_scale

    diagnostics: Dict[str, Any] = {
        "raw_sort_index": raw,
        "SORT_AMPLIFY": ampl,
        "running_scale_used": running_scale,
        "pre_scaled": pre_scaled,
        "scaled_by_running": scaled_by_running,
        "term_clip_config": term_clip,
        "zero_gate_threshold_config": zero_gate_threshold,
        "use_soft_clip": use_soft_clip,
        "bypass_gating": bypass_gating,
    }

    # Start with scaled value
    applied = scaled_by_running
    gating_reason: Optional[str] = None

    # Apply gating (zeroing) if configured and not bypassed
    if (not bypass_gating) and (zero_gate_threshold is not None):
        try:
            zt = float(zero_gate_threshold)
            if abs(pre_scaled) < zt:
                gating_reason = (
                    f"zeroed_by_gate: abs(pre_scaled)({abs(pre_scaled)}) < zero_gate_threshold({zt})"
                )
                applied = 0.0
        except Exception:
            # Ignore mis-configured gate
            gating_reason = "zero_gate_threshold_misconfigured"

    # If a clip is set and applied is non-zero, apply clip (soft or hard)
    clip_mode = None
    clip_denom = None
    if (term_clip is not None) and (applied != 0.0):
        try:
            clip_val = float(term_clip)
            if clip_val <= 0.0:
                # treat non-positive as disable
                pass
            else:
                if use_soft_clip:
                    # soft clip: applied / (1 + |applied|/clip_val)
                    denom = 1.0 + (abs(applied) / clip_val)
                    clip_denom = denom
                    applied = applied / denom
                    clip_mode = "soft"
                else:
                    # hard clip
                    applied = max(min(applied, clip_val), -clip_val)
                    clip_mode = "hard"
        except Exception:
            clip_mode = "clip_misconfigured"

    # Visible test override (force a non-zero for integration testing)
    if visible_test_value is not None:
        applied = float(visible_test_value)
        diagnostics["visible_test_value_used"] = float(visible_test_value)

    diagnostics.update({
        "gating_reason": gating_reason,
        "applied_sort": applied,
        "clip_mode": clip_mode,
        "clip_denom": clip_denom,
    })

    if debug:
        _print_debug(diagnostics)

    return diagnostics


# ----------------------
# Public API: apply sort term to reward
# ----------------------
def apply_sort_term_to_reward(
    reward: Union[Number, "torch.Tensor"],
    raw_sort_index: Union[Number, "torch.Tensor"],
    cfg: Optional[Dict[str, Any]] = None,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Apply the computed sort term into the reward value.

    Returns a dict:
        {
            "new_reward": float,
            "reward_before": float,
            "applied_sort": float,
            "diagnostics": { ... }  # see compute_applied_sort
        }

    Use this function in your reward composition path.
    """
    # Compute diagnostics + applied_sort
    diag = compute_applied_sort(raw_sort_index, cfg, debug=debug)

    applied_sort = float(diag["applied_sort"])
    reward_before = _to_float(reward)
    new_reward = reward_before + applied_sort

    result = {
        "new_reward": new_reward,
        "reward_before": reward_before,
        "applied_sort": applied_sort,
        "diagnostics": diag,
    }

    if debug:
        print(f"[apply_sort_term_to_reward] reward_before={reward_before} applied_sort={applied_sort} new_reward={new_reward}")

    return result


# ----------------------
# Small debug printer
# ----------------------
def _print_debug(d: Dict[str, Any]) -> None:
    print("---- SORT TERM DIAGNOSTICS ----")
    print(f" raw_sort_index: {d.get('raw_sort_index')}")
    print(f" pre_scaled: {d.get('pre_scaled')}")
    print(f" scaled_by_running: {d.get('scaled_by_running')}")
    if d.get("gating_reason"):
        print(" GATING:", d.get("gating_reason"))
    if d.get("clip_mode"):
        print(" CLIP MODE:", d.get("clip_mode"), "clip_denom:", d.get("clip_denom"))
    if d.get("visible_test_value_used") is not None:
        print(" VISIBLE TEST VALUE USED:", d.get("visible_test_value_used"))
    print(" applied_sort:", d.get("applied_sort"))
    print("---- END DIAGNOSTICS ----\n")


# ----------------------
# Example usage & quick tests (run standalone)
# ----------------------
if __name__ == "__main__":
    # Quick sanity checks you can run locally:
    print("wrappers.py: quick self-test of sort-term logic\n")

    cfg = dict(DEFAULT_CFG)
    # Example: default behavior (gating enabled) for a very tiny raw_sort_index
    tiny_raw = 1.4662742614746094e-05
    out = apply_sort_term_to_reward(0.0, tiny_raw, cfg, debug=True)
    print("default cfg result:", out)

    # Bypass gating to test pipeline
    cfg_bypass = dict(cfg)
    cfg_bypass["bypass_gating"] = True
    out_bypass = apply_sort_term_to_reward(0.0, tiny_raw, cfg_bypass, debug=True)
    print("bypass gating result:", out_bypass)

    # Visible test: force a non-zero to confirm reward compositions accept additions
    cfg_visible = dict(cfg)
    cfg_visible["visible_test_value"] = 0.01
    out_visible = apply_sort_term_to_reward(0.0, tiny_raw, cfg_visible, debug=True)
    print("visible test result:", out_visible)

    if _HAS_TORCH:
        # torch path test
        import torch as th
        raw_t = th.tensor(tiny_raw, dtype=th.float32)
        out_t = apply_sort_term_to_reward(th.tensor(0.0), raw_t, cfg_bypass, debug=True)
        print("torch tensor path result:", out_t)

    print("self-test complete.")
