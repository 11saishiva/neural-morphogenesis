# src/experiments/train_local_sorting.py
import os, torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.envs.wrappers import SortingEnv
from src.agents.ppo import PatchActorCritic, compute_gae, ppo_update


from src.agents.neuro_fuzzy import NeuroFuzzyActorCritic
from src.utils.viz import visualize_sequence
from src.utils.metrics import interfacial_energy, motion_penalty

# ------------------ hyperparameters ------------------
H, W = 64, 64
BATCH = 2
T_STEPS = 32
PATCH_SIZE = 5
ACTION_DIM = 3
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 4
MINI_BATCH = 2048
LR = 3e-4
TOTAL_UPDATES = 30 #TOTAL_UPDATES = 400
LOG_DIR = "runs/sorting_rl"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------

def flatten_patches(patches):
    T, B, N, C, h, w = patches.shape
    return patches.view(T * B * N, C, h, w)

def main():
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    env = SortingEnv(H=H, W=W, device=DEVICE, gamma_motion=0.01, steps_per_action=1, obs_mode='local')
    policy = NeuroFuzzyActorCritic(in_ch=4, patch_size=PATCH_SIZE,
                               feat_dim=128, fuzzy_features=16,
                               n_mfs=3, n_rules=32,
                               action_dim=ACTION_DIM).to(DEVICE)
    
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    N = H * W

    for update in range(TOTAL_UPDATES):
        obs_patches_list, actions_list, logp_list, values_list, rewards_list, dones_list = [], [], [], [], [], []
        # capture frames for GIF
        vis_frames = []

        obs = env.reset(B=BATCH, pA=0.5)
        patches, coords = obs
        # with torch.no_grad():
        #     flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        #     _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
        #     values_list.append(vals.view(BATCH, N).mean(1).cpu())

        # vis_frames.append(env.current_state().detach().cpu())
        vis_frames.append(env.current_state().detach().cpu())

        for t in range(T_STEPS):
            flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
            action_flat, logp_flat, value_flat, _, _ = policy.get_action_and_value(flat)
            action_grid = action_flat.reshape(BATCH, N, ACTION_DIM)
            logp_grid = logp_flat.reshape(BATCH, N)
            value_grid = value_flat.view(BATCH, N).mean(1)
            obs2, reward, info = env.step(action_grid)
            patches, coords = obs2
            patches = patches.detach()
            obs_patches_list.append(patches.cpu())
            actions_list.append(action_grid.detach().cpu())
            logp_list.append(logp_grid.detach().cpu())
            values_list.append(value_grid.detach().cpu())
            rewards_list.append(reward.cpu())
            dones_list.append(torch.zeros_like(reward.cpu()))
            vis_frames.append(env.current_state().detach().cpu())

        # Bootstrap
        flat = patches.reshape(BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        with torch.no_grad():
            _, _, vals, _, _ = policy.get_action_and_value(flat, deterministic=True)
            values_list.append(vals.view(BATCH, N).mean(1).cpu())

        # Tensor assembly
        T = T_STEPS
        obs_patches = torch.stack(obs_patches_list)
        actions = torch.stack(actions_list)
        logps = torch.stack(logp_list)
        rewards = torch.stack(rewards_list)
        dones = torch.stack(dones_list)
        values = torch.stack(values_list)
        advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM)

        obs_patches_flat = obs_patches.reshape(T * BATCH * N, 4, PATCH_SIZE, PATCH_SIZE).to(DEVICE)
        actions_flat = actions.reshape(T * BATCH * N, ACTION_DIM).to(DEVICE)
        logps_flat = logps.reshape(T * BATCH * N).to(DEVICE)
        returns_flat = returns.unsqueeze(2).repeat(1, 1, N).reshape(T * BATCH * N).to(DEVICE)
        advs_flat = advantages.unsqueeze(2).repeat(1, 1, N).reshape(T * BATCH * N).to(DEVICE)

        # PPO update
        ppo_update(policy=policy,optimizer=optimizer,
           obs_patches=obs_patches_flat,
           actions=actions_flat,
           logprobs_old=logps_flat,
           returns=returns_flat,
           advantages=advs_flat,
           clip_eps=CLIP,
           vf_coef=0.5,
           ent_coef=0.01,
           epochs=EPOCHS,
           batch_size=MINI_BATCH,
           orth_coef=3e-3,
           sparsity_coef=5e-3,
           corr_coef=5e-3)


        # Diagnostics
        mean_r = rewards.mean().item()
        mean_e = interfacial_energy(env.current_state()).mean().item()
        mean_mpen = motion_penalty(actions_flat.view(T, BATCH, N, ACTION_DIM).to(DEVICE).transpose(1,2)).mean().item()
        mean_adh = env.current_state()[:, 2].mean().item()

        writer.add_scalar("reward/avg", mean_r, update)
        writer.add_scalar("energy/interfacial", mean_e, update)
        writer.add_scalar("motion/penalty", mean_mpen, update)
        writer.add_scalar("adhesion/mean", mean_adh, update)

        print(f"[{update:04d}] reward={mean_r:.4f} | energy={mean_e:.4f} | motion_pen={mean_mpen:.4f}")

        if update % 50 == 0:
            ckpt = f"checkpoints/ppo_{update:04d}.pt"
            torch.save(policy.state_dict(), ckpt)
            print(f"[ckpt] saved {ckpt}")
            visualize_sequence(vis_frames, out_path=f"visuals/sorting_{update:04d}.gif", every=2)

    writer.close()

if __name__ == "__main__":
    main()
