# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
import json
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from core.isaac_gym_multi_env import IsaacSimGo2MultiEnv
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    record_every: int = 50000
    """how often to record videos, will record at around this many episodes"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Environment kwargs - allows passing additional parameters to the environment
    env_kwargs: dict = None
    """additional keyword arguments to pass to the environment constructor"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    task_id: str = ""
    """the task id encoding for kwargs (computed in runtime)"""


# def make_env(env_id, idx, capture_video, run_name, gamma, experiment_dir=None, env_kwargs=None, record_every=50000, name_prefix="rl-video"):
#     def thunk():
#         env = IsaacSimGo2MultiEnv(
#             num_envs=1,
#             env_spacing=3.0,
#             safety_margin=0.1,
#             use_relative_control=False,
#             relative_scale=0.1,
#         )
#         return env

#     return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def main(args: Args):
    """Main training function that can be called with args."""

    # Set computed values
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Create experiment directory
    experiment_dir = f"runs/{args.exp_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define run name 
    #TODO: Change to env_id or name when we have a new env
    run_name = f"Go2MultiEnv__{args.seed}__{int(time.time())}"

    # Initialize wandb if tracking is enabled
    if args.track:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(env_id, i, args.capture_video, run_name, args.gamma, experiment_dir, env_kwargs, record_every=int(args.record_every/args.num_envs)) for i in range(args.num_envs)]
    # )
    
    # Prepare environment kwargs, using constructor defaults for missing values
    env_kwargs = {"num_envs": args.num_envs}  # Always override num_envs from args
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)
    
    envs = IsaacSimGo2MultiEnv(**env_kwargs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    global_ep_counter = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Check if we have episode information with the _episode boolean array
            if "_episode" in infos:
                episode_flags = infos["_episode"]
                # Convert to numpy arrays for vectorized operations
                episode_flags_array = episode_flags
                episode_rewards = infos["episode"]['r']
                episode_lengths = infos["episode"]['l']
                
                # Count how many episodes completed
                num_completed_episodes = np.sum(episode_flags_array)
                global_ep_counter += num_completed_episodes
                
                # Multiply arrays by boolean flags and sum up
                total_reward = np.sum(episode_rewards * episode_flags_array)
                total_length = np.sum(episode_lengths * episode_flags_array)
                
                # Calculate averages
                avg_episodic_reward = total_reward / num_completed_episodes
                avg_episodic_length = total_length / num_completed_episodes
                
                # Log episode metrics to wandb if tracking is enabled
                if args.track:
                    wandb.log({
                        "charts/episodic_acc_reward": avg_episodic_reward,
                        "charts/episodic_acc_length": avg_episodic_length,
                        "charts/num_episodes": global_ep_counter,
                        "global_step": global_step
                    }, step=global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Calculate SPS (Steps Per Second)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)

        # Create metrics dictionary for wandb logging
        if args.track:
            metrics_dict = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": sps,
                "global_step": global_step
            }
            
            # Log weight histograms every 100000 steps
            if global_step % 100000 == 0:
                for name, param in agent.named_parameters():
                    metrics_dict[f"weights/{name}"] = wandb.Histogram(param.clone().cpu().numpy())
            
            # Log all metrics at once
            wandb.log(metrics_dict, step=global_step)

    if args.save_model:
        model_path = f"{experiment_dir}/{run_name}/model.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # Save args as a config.yml in the experiment_dir
        config_save_path = os.path.join(experiment_dir, f"{run_name}/config.yml")
        # Convert dataclass to dict, then dump as YAML
        try:
            import yaml
            with open(config_save_path, "w") as f:
                yaml.safe_dump(vars(args), f, default_flow_style=False, sort_keys=False)
            print(f"Config saved to {config_save_path}")
        except ImportError:
            print("PyYAML is not installed. Skipping config save.")
        # from ppo_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}",
        #     Model=Agent,
        #     device=device,
        #     gamma=args.gamma,
        #     experiment_dir=experiment_dir,  # Pass experiment_dir to evaluation
        # )
        
        # # Log evaluation results to wandb
        # if args.track:
        #     # Log mean and std of episodic returns
        #     mean_return = np.mean(episodic_returns)
        #     std_return = np.std(episodic_returns)
        #     wandb.log({
        #         "eval/episodic_return_mean": mean_return,
        #         "eval/episodic_return_std": std_return,
        #         "global_step": global_step
        #     }, step=global_step)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"{experiment_dir}", f"videos/{run_name}-eval")

    envs.close()