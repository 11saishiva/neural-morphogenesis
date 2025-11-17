# from src.envs.wrappers import SortingEnv
from src.envs.wrappers import SortingEnv
import torch
env = SortingEnv(obs_mode='local', device='cuda' if torch.cuda.is_available() else 'cpu')
obs = env.reset(B=1)
patches, coords = obs
print("patches:", patches.shape)  # (1, H*W, 4, 5, 5)
print("coords:", coords.shape)    # (1, H*W, 2)
