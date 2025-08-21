"""
PPO adaptado para IsaacSim + CleanRL

Baseado no ppo_continuous_action_isaacgym.py do CleanRL, mas adaptado
para usar com IsaacSim e o ambiente multi-robô existente.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import traceback

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Importar ambiente adaptado
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cleanrl_isaacsim.envs.multi_env_wrapper import make_env
from cleanrl_isaacsim.envs.wrappers import apply_standard_wrappers
from cleanrl_isaacsim.utils.wandb_utils import log_robot_metrics, log_isaac_sim_metrics


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
    wandb_project_name: str = "isaacsim"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Go2MultiEnv"
    """the id of the environment"""
    total_timesteps: int = 10000000  # Reduzido para testes iniciais
    """total timesteps of the experiments"""
    learning_rate: float = 0.0026
    """the learning rate of the optimizer"""
    num_envs: int = 16  # Número de robôs em paralelo
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 2
    """coefficient of the value function"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    reward_scaler: float = 1
    """the scale factor applied to the reward during training"""
    
    # IsaacSim specific arguments
    spacing: float = 3.0
    """spacing between robots in meters"""
    safety_margin: float = 0.1
    """safety margin for joint limits"""
    use_relative_control: bool = False
    """use relative control instead of absolute"""
    relative_scale: float = 0.1
    """scale for relative control"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize neural network layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    PPO Agent com arquitetura adaptada para IsaacSim.
    
    Baseado no agent do CleanRL IsaacGym, mas com dimensões ajustadas
    para o ambiente GO2 multi-robô.
    """

    def __init__(self, envs):
        super().__init__()
        
        # Obter dimensões dos espaços
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        
        print(f"[Agent] Observation dim: {obs_dim}")
        print(f"[Agent] Action dim: {action_dim}")
        
        # Critic network - estima valor do estado
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        # Actor network - política (média das ações)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        
        # Log std das ações (parâmetro aprendível)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Get state value from critic."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Get action and value for given observation.
        
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value from critic
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train(args: Args):
    """
    Função principal de treinamento PPO para IsaacSim.
    """
    # Calcular parâmetros derivados
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Configurar tracking (WandB)
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Configurar tensorboard
    writer = SummaryWriter(f"experiments/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Configurar seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"[PPO] Using device: {device}")

    # Criar ambiente
    print("[PPO] Creating environment...")
    try:
        envs = make_env(
            num_envs=args.num_envs,
            spacing=args.spacing,
            safety_margin=args.safety_margin,
            use_relative_control=args.use_relative_control,
            relative_scale=args.relative_scale
        )
        
        # Aplicar wrappers padrão
        envs = apply_standard_wrappers(
            envs, 
            reward_scale=args.reward_scaler,
            action_noise_std=0.0  # Sem ruído durante treinamento inicial
        )
        
        print(f"[PPO] Environment created successfully with {args.num_envs} robots")
        print(f"[PPO] Observation space: {envs.single_observation_space}")
        print(f"[PPO] Action space: {envs.single_action_space}")
        
    except Exception as e:
        print(f"[PPO] Error creating environment: {e}")
        raise

    # Verificar espaços
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Criar agente
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Configurar storage para rollouts
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)

    # Iniciar treinamento
    global_step = 0
    start_time = time.time()
    
    print("[PPO] Starting training...")
    
    try:
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
        
        print(f"[PPO] Initial observation shape: {next_obs.shape}")
        print(f"[PPO] Starting main training loop with {args.num_iterations} iterations...")
        
        for iteration in range(1, args.num_iterations + 1):
            # Annealing learning rate
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Rollout phase
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Ação do agente
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Step no ambiente
                step_result = envs.step(action.cpu().numpy())
                
                # Unpack step result safely
                if len(step_result) != 5:
                    raise ValueError(f"env.step() returned {len(step_result)} items, expected 5: (obs, rewards, terminations, truncations, infos)")
                
                next_obs, reward, terminations, truncations, infos = step_result
                
                # Convert to proper arrays
                next_obs = np.asarray(next_obs, dtype=np.float32)
                reward = np.asarray(reward, dtype=np.float32)
                terminations = np.asarray(terminations, dtype=bool)
                truncations = np.asarray(truncations, dtype=bool)
                
                # Validate shapes
                assert next_obs.shape[0] == args.num_envs, f"obs shape[0] {next_obs.shape[0]} != num_envs {args.num_envs}"
                assert reward.shape[0] == args.num_envs, f"reward shape[0] {reward.shape[0]} != num_envs {args.num_envs}"
                assert terminations.shape[0] == args.num_envs, f"terminations shape[0] {terminations.shape[0]} != num_envs {args.num_envs}"
                assert truncations.shape[0] == args.num_envs, f"truncations shape[0] {truncations.shape[0]} != num_envs {args.num_envs}"
                
                # CRITICAL: Avoid array boolean context
                next_done = np.logical_or(terminations, truncations)
                
                # Convert to tensors safely
                rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
                next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                next_done = torch.tensor(next_done, dtype=torch.float32).to(device)

                # Log episódios completos e métricas dos robôs
                episode_rewards = []
                episode_lengths = []
                
                if isinstance(infos, list):
                    for i, info in enumerate(infos):
                        if isinstance(info, dict) and "episode" in info:
                            episodic_return = info["episode"]["r"]
                            episodic_length = info["episode"]["l"]
                            episode_rewards.append(episodic_return)
                            episode_lengths.append(episodic_length)
                            
                            print(f"global_step={global_step}, robot_{i}, episodic_return={episodic_return:.2f}, length={episodic_length}")
                            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                            
                            # Log métricas específicas por robô
                            writer.add_scalar(f"robots/robot_{i}_return", episodic_return, global_step)
                            writer.add_scalar(f"robots/robot_{i}_length", episodic_length, global_step)
                
                # Log métricas agregadas dos robôs se houver episódios completos
                if len(episode_rewards) > 0 and args.track:
                    try:
                        import wandb
                        log_robot_metrics(
                            global_step,
                            np.array(episode_rewards),
                            np.array(episode_lengths)
                        )
                    except Exception as e:
                        print(f"[WARN] Could not log robot metrics to WandB: {e}")
                        
                elif not isinstance(infos, list):
                    # Debug: print what we actually got
                    print(f"[DEBUG] infos type: {type(infos)}, content: {infos}")

            # Calcular vantagens (GAE)
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

            # Flatten para treinamento
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Update phase
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # KL divergence approximation
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

                    # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"Iteration {iteration}/{args.num_iterations}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)
        
        # Log métricas específicas do IsaacSim
        if args.track and iteration % 10 == 0:  # A cada 10 iterações
            try:
                import wandb
                log_isaac_sim_metrics(
                    global_step,
                    physics_fps=sps,  # Aproximação
                    render_fps=sps,   # Aproximação
                    simulation_time=time.time() - start_time,
                    num_active_robots=args.num_envs,
                    collision_count=0,  # Placeholder
                    fallen_robots=0     # Placeholder
                )
                
                # Log métricas adicionais para WandB
                wandb.log({
                    "training/iteration": iteration,
                    "training/progress": iteration / args.num_iterations,
                    "training/timesteps_total": global_step,
                    "training/time_elapsed": time.time() - start_time,
                    "performance/steps_per_second": sps,
                    "performance/episodes_per_hour": sps * 3600 / 1000,  # Estimativa
                }, step=global_step)
                
            except Exception as e:
                print(f"[WARN] Could not log Isaac Sim metrics to WandB: {e}")
            
    except KeyboardInterrupt:
        print("[PPO] Training interrupted by user")
    except Exception as e:
        print(f"[PPO] Training error: {e}")
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        print("[PPO] Cleaning up...")
        if hasattr(envs, 'close'):
            envs.close()
        writer.close()
        print("[PPO] Training finished")


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args) 