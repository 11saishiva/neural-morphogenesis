# import torch
# import torch.nn.functional as F

# def extract_local_patches(state: torch.Tensor, patch_size: int = 5):
#     """
#     Safe patch extractor (no as_strided).
#     Input:
#         state: (B, C, H, W)
#     Output:
#         patches: (B, H*W, C, patch_size, patch_size)  -- contiguous
#         coords:  (B, H*W, 2)                           -- int coords (y,x)
#     """
#     if state.dim() != 4:
#         raise ValueError("state must be (B,C,H,W)")

#     B, C, H, W = state.shape
#     p = patch_size
#     pad = p // 2

#     # pad to allow extracting patches at borders
#     padded = F.pad(state, (pad, pad, pad, pad), mode="reflect")  # (B, C, H+2pad, W+2pad)

#     patches_list = []
#     coords_list = []

#     # iterate spatially (H small enough typically)
#     for y in range(H):
#         for x in range(W):
#             # slice returns contiguous only if we clone; clone -> contiguous copy
#             patch = padded[:, :, y:(y+p), x:(x+p)].clone().contiguous()  # (B, C, p, p)
#             patches_list.append(patch)
#             coords_list.append(torch.tensor([y, x], device=state.device, dtype=torch.long))

#     # stack patches: list length = N (=H*W), each element (B,C,p,p)
#     patches_stack = torch.stack(patches_list, dim=0)        # (N, B, C, p, p)
#     patches_stack = patches_stack.permute(1, 0, 2, 3, 4)   # (B, N, C, p, p)
#     patches_stack = patches_stack.contiguous()

#     coords = torch.stack(coords_list, dim=0)               # (N, 2)
#     coords = coords.unsqueeze(0).repeat(B, 1, 1)           # (B, N, 2)

#     return patches_stack, coords


# def interfacial_energy(state: torch.Tensor, axis_pairs=None):
#     """
#     Simple interfacial energy estimator.
#     Input:
#         state: (B, C, H, W) where channels include TYPE_A and TYPE_B probabilities
#     Output:
#         energy: (B,) per-batch scalar energy (mean magnitude of gradients between types)
#     """
#     if state.dim() != 4:
#         raise ValueError("state must be (B,C,H,W)")

#     B, C, H, W = state.shape

#     # If two or more channels, difference of first two channels
#     if C >= 2:
#         # field shape: (B, H, W)
#         field = state[:, 0] - state[:, 1]
#     else:
#         field = state[:, 0]

#     # compute absolute finite differences along x and y
#     # dx: differences along width (x)
#     if W > 1:
#         dx = torch.abs(field[:, :, 1:] - field[:, :, :-1])   # (B, H, W-1)
#         gx = dx.mean(dim=[1, 2])
#     else:
#         gx = torch.zeros(B, device=state.device, dtype=state.dtype)

#     # dy: differences along height (y)
#     if H > 1:
#         dy = torch.abs(field[:, 1:, :] - field[:, :-1, :])   # (B, H-1, W)
#         gy = dy.mean(dim=[1, 2])
#     else:
#         gy = torch.zeros(B, device=state.device, dtype=state.dtype)

#     energy = 0.5 * (gx + gy)
#     return energy


# def motion_penalty(actions: torch.Tensor, kind: str = "l2"):
#     """
#     Motion penalty for actions.
#     Accepts:
#       - (B, A, H, W)  (global)
#       - (B, N, A)     (flattened per-cell actions)
#       - (B, A, N)     (variant)
#     Returns:
#       - penalty: (B,) per-batch scalar (mean squared magnitude)
#     """
#     if not isinstance(actions, torch.Tensor):
#         actions = torch.as_tensor(actions)

#     # ensure float on same device if possible
#     try:
#         actions = actions.float()
#     except Exception:
#         actions = actions.to(dtype=torch.float32)

#     # Case 1: (B, A, H, W)
#     if actions.dim() == 4:
#         # per-batch mean squared action
#         mag = actions.pow(2).mean(dim=[1, 2, 3])
#         return mag

#     # Case 2: (B, N, A) or (B, A, N)
#     if actions.dim() == 3:
#         B, d1, d2 = actions.shape
#         # If last dim small (e.g. 3), treat as (B, N, A)
#         if d2 <= 8:
#             mag = actions.pow(2).mean(dim=[1, 2])  # (B,)
#             return mag
#         # else, maybe (B, A, N)
#         mag = actions.pow(2).mean(dim=[1, 2])
#         return mag

#     # Case 3: (B, K) or other -> mean over all but batch dim
#     if actions.dim() == 2:
#         mag = actions.pow(2).mean(dim=1)
#         return mag

#     # Fallback: global mean scalar -> return per-batch repeated value if possible
#     mag_scalar = actions.pow(2).mean()
#     if actions.dim() == 0:
#         return mag_scalar.view(1)
#     B_try = actions.shape[0] if actions.dim() > 0 else 1
#     return mag_scalar.view(1).repeat(B_try)

import torch
import torch.nn.functional as F

def extract_local_patches(state: torch.Tensor, patch_size: int = 5):
    """
    Safe patch extractor (no as_strided).
    Input:
        state: (B, C, H, W)
    Output:
        patches: (B, H*W, C, patch_size, patch_size)  -- contiguous
        coords:  (B, H*W, 2)                           -- int coords (y,x)
    """
    if state.dim() != 4:
        raise ValueError("state must be (B,C,H,W)")

    B, C, H, W = state.shape
    p = int(patch_size)
    pad = p // 2

    # pad to allow extracting patches at borders
    padded = F.pad(state, (pad, pad, pad, pad), mode="reflect")  # (B, C, H+2pad, W+2pad)

    patches_list = []
    coords_list = []

    # iterate spatially; H and W are expected to be modest (e.g. 32..128)
    for y in range(H):
        for x in range(W):
            patch = padded[:, :, y:(y + p), x:(x + p)].clone().contiguous()  # (B, C, p, p)
            patches_list.append(patch)
            coords_list.append(torch.tensor([y, x], device=state.device, dtype=torch.long))

    # stack patches: (N, B, C, p, p) -> permute -> (B, N, C, p, p)
    patches_stack = torch.stack(patches_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()
    coords = torch.stack(coords_list, dim=0)               # (N, 2)
    coords = coords.unsqueeze(0).repeat(B, 1, 1)           # (B, N, 2)

    return patches_stack, coords


def interfacial_energy(state: torch.Tensor, axis_pairs=None):
    """
    Robust interfacial energy estimator.

    Input:
      state: (B, C, H, W) where channels include type probabilities (e.g. TYPE_A, TYPE_B)
    Output:
      energy: (B,) per-batch scalar energy (mean magnitude of gradients between types)
    Notes:
      - Handles edge cases H==1 or W==1 safely.
      - Returns float tensor on same device/dtype as input.
    """
    if state.dim() != 4:
        raise ValueError("state must be (B,C,H,W)")

    B, C, H, W = state.shape
    device = state.device
    dtype = state.dtype

    # derive the scalar field: difference of first two channels if present
    if C >= 2:
        # shape -> (B, H, W)
        field = state[:, 0] - state[:, 1]
    else:
        field = state[:, 0]

    # dx: differences along width axis (x)
    if W > 1:
        dx = torch.abs(field[:, :, 1:] - field[:, :, :-1])   # (B, H, W-1)
        gx = dx.mean(dim=[1, 2])  # (B,)
    else:
        gx = torch.zeros(B, device=device, dtype=dtype)

    # dy: differences along height axis (y)
    if H > 1:
        dy = torch.abs(field[:, 1:, :] - field[:, :-1, :])   # (B, H-1, W)
        gy = dy.mean(dim=[1, 2])  # (B,)
    else:
        gy = torch.zeros(B, device=device, dtype=dtype)

    energy = 0.5 * (gx + gy)
    return energy


def motion_penalty(actions: torch.Tensor, kind: str = "l2"):
    """
    Motion penalty for actions.
    Accepts:
        - (B, A, H, W)  global actions
        - (B, N, A)     flattened per-cell actions
        - (B, A, N)     variant
        - (B, K)        fallback
    Returns:
        - penalty: (B,) per-batch scalar (mean squared magnitude)
    """
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)

    # ensure float dtype
    actions = actions.float()

    # Case: global (B, A, H, W)
    if actions.dim() == 4:
        return actions.pow(2).mean(dim=[1, 2, 3])

    # Case: (B, N, A) or (B, A, N)
    if actions.dim() == 3:
        B = actions.shape[0]
        d1, d2 = actions.shape[1], actions.shape[2]
        # heuristics: if last dim small (<=8) treat as (B, N, A)
        if d2 <= 8:
            return actions.pow(2).mean(dim=[1, 2])
        return actions.pow(2).mean(dim=[1, 2])

    # Case: (B, K)
    if actions.dim() == 2:
        return actions.pow(2).mean(dim=1)

    # Scalar or unknown -> return scalar repeated per-batch if possible
    mag = actions.pow(2).mean()
    if actions.dim() == 0:
        return mag.view(1)
    # try derive a batch dim, else return single-value tensor
    B_try = actions.shape[0] if actions.dim() > 0 else 1
    return mag.view(1).repeat(B_try)
