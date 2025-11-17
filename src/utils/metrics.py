# src/utils/metrics.py
import torch
import torch.nn.functional as F

TYPE_A = 0
TYPE_B = 1
ADH    = 2
MORPH  = 3
CENTER = 4

def interfacial_energy(state):
    """
    Differentiable estimate of interfacial energy:
    Sum over neighbors of A-B mismatch, weighted by local adhesion.
    Lower is better (less A|B boundary).
    """
    types = F.softmax(state[:, :2], dim=1)  # ensure valid probs
    a = types[:, TYPE_A:TYPE_A+1]
    b = types[:, TYPE_B:TYPE_B+1]
    adhesion = state[:, ADH:ADH+1]

    # 4-neighborhood kernel
    k4 = state.new_tensor([[0., 1., 0.],
                           [1., 0., 1.],
                           [0., 1., 0.]]).view(1,1,3,3)

    # neighbor averages
    a_nb = F.conv2d(a, k4, padding=1) / 4.0
    b_nb = F.conv2d(b, k4, padding=1) / 4.0

    mismatch = (a * b_nb) + (b * a_nb)  # high if neighbors differ
    # weight by adhesion (higher adhesion penalizes heterotypic contacts more)
    energy = (adhesion * mismatch).mean(dim=(1,2,3))  # per-batch scalar
    return energy  # (B,)

def motion_penalty(actions):
    """
    Quadratic penalty for velocities to discourage unnecessary motion.
    """
    vx = actions[:, 1:2]
    vy = actions[:, 2:3]
    pen = (vx * vx + vy * vy).mean(dim=(1,2,3))
    return pen  # (B,)
# src/utils/metrics.py  (append at end)

def extract_local_patches(state, patch_size=5):
    """
    Extracts local 5x5x4 patches centered on every cell (excluding center_mask).
    state: (B,5,H,W)
    returns: patches (B, N, C', patch_size, patch_size), where N = H*W, C' = 4
             centers: tensor of shape (B,N,2) with (y,x) coords
    """
    B, C, H, W = state.shape
    assert C >= 4
    pad = patch_size // 2
    # remove center_mask channel for observation
    obs_field = state[:, :4]
    # unfold extracts all patches
    patches = torch.nn.functional.unfold(obs_field, kernel_size=patch_size, padding=pad)
    # shape: (B, 4*patch_size*patch_size, H*W)
    patches = patches.transpose(1, 2)  # (B, N, 4*ps*ps)
    patches = patches.view(B, H * W, 4, patch_size, patch_size)

    # build centers grid (y,x)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=state.device),
        torch.arange(W, device=state.device),
        indexing="ij"
    )
    coords = torch.stack((ys.flatten(), xs.flatten()), dim=-1).unsqueeze(0).repeat(B, 1, 1)
    return patches, coords