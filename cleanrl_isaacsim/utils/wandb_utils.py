"""
Ferramentas para integração com Weights & Biases (wandb).

Este módulo fornece utilitários para configurar e usar WandB com
treinamento de RL no IsaacSim, incluindo métricas customizadas.
"""

import wandb
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import io
import PIL.Image


def init_wandb(
    args: Any, 
    run_name: str,
    project_name: str = "cleanrl-isaacsim",
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    Inicializa wandb para tracking de experimentos.
    
    Args:
        args: Argumentos de configuração do experimento
        run_name: Nome único do run
        project_name: Nome do projeto no wandb
        entity: Entidade (usuário/time) no wandb
        tags: Tags para categorizar o experimento
    """
    config = vars(args) if hasattr(args, '__dict__') else args
    
    # Tags padrão para experimentos IsaacSim
    default_tags = ["isaacSim", "multi-robot", "go2", "ppo"]
    if tags:
        default_tags.extend(tags)
    
    wandb.init(
        project=project_name,
        entity=entity,
        name=run_name,
        config=config,
        tags=default_tags,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    print(f"[WandB] Initialized run: {run_name}")
    print(f"[WandB] Project: {project_name}")
    print(f"[WandB] Dashboard: {wandb.run.url}")


def log_metrics(step: int, metrics: Dict[str, Any], prefix: str = "") -> None:
    """
    Loga métricas no wandb.
    
    Args:
        step: Step atual do treinamento
        metrics: Dicionário com métricas para logar
        prefix: Prefixo para as métricas (ex: "train/", "eval/")
    """
    if not wandb.run:
        print("Warning: WandB not initialized. Call init_wandb() first.")
        return
    
    # Preparar métricas com prefixo
    log_dict = {}
    for key, value in metrics.items():
        # Converter tensors para valores escalares
        if torch.is_tensor(value):
            value = value.item() if value.numel() == 1 else value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        
        log_key = f"{prefix}{key}" if prefix else key
        log_dict[log_key] = value
    
    wandb.log(log_dict, step=step)


def log_robot_metrics(
    step: int, 
    robot_rewards: np.ndarray,
    robot_episode_lengths: np.ndarray,
    robot_states: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Loga métricas específicas dos robôs.
    
    Args:
        step: Step atual do treinamento
        robot_rewards: Array de recompensas por robô
        robot_episode_lengths: Array de comprimentos de episódio por robô
        robot_states: Estados opcionais dos robôs (posições, velocidades, etc.)
    """
    if not wandb.run:
        return
    
    # Estatísticas básicas das recompensas
    metrics = {
        "robots/reward_mean": np.mean(robot_rewards),
        "robots/reward_std": np.std(robot_rewards),
        "robots/reward_min": np.min(robot_rewards),
        "robots/reward_max": np.max(robot_rewards),
        "robots/episode_length_mean": np.mean(robot_episode_lengths),
        "robots/episode_length_std": np.std(robot_episode_lengths),
    }
    
    # Adicionar estados dos robôs se fornecidos
    if robot_states:
        for state_name, state_values in robot_states.items():
            if isinstance(state_values, np.ndarray):
                metrics[f"robots/{state_name}_mean"] = np.mean(state_values)
                metrics[f"robots/{state_name}_std"] = np.std(state_values)
    
    wandb.log(metrics, step=step)


def log_training_progress(
    step: int,
    policy_loss: float,
    value_loss: float,
    entropy_loss: float,
    learning_rate: float,
    kl_divergence: float,
    clipfrac: float,
    sps: int
) -> None:
    """
    Loga métricas de progresso do treinamento.
    
    Args:
        step: Step atual do treinamento
        policy_loss: Loss da política
        value_loss: Loss da função valor
        entropy_loss: Loss de entropia
        learning_rate: Taxa de aprendizado atual
        kl_divergence: Divergência KL
        clipfrac: Fração de valores clippados
        sps: Steps por segundo
    """
    metrics = {
        "train/policy_loss": policy_loss,
        "train/value_loss": value_loss,
        "train/entropy_loss": entropy_loss,
        "train/learning_rate": learning_rate,
        "train/kl_divergence": kl_divergence,
        "train/clipfrac": clipfrac,
        "performance/sps": sps,
    }
    
    log_metrics(step, metrics)


def log_histogram(step: int, name: str, values: np.ndarray) -> None:
    """
    Loga histograma de valores.
    
    Args:
        step: Step atual
        name: Nome do histograma
        values: Valores para criar histograma
    """
    if not wandb.run:
        return
    
    wandb.log({name: wandb.Histogram(values)}, step=step)


def log_video(step: int, name: str, video_frames: np.ndarray) -> None:
    """
    Loga vídeo no wandb.
    
    Args:
        step: Step atual
        name: Nome do vídeo
        video_frames: Array de frames (T, H, W, C) ou (T, C, H, W)
    """
    if not wandb.run:
        return
    
    # Garantir formato correto (T, H, W, C)
    if video_frames.ndim == 4 and video_frames.shape[1] == 3:
        # Converter de (T, C, H, W) para (T, H, W, C)
        video_frames = video_frames.transpose(0, 2, 3, 1)
    
    wandb.log({name: wandb.Video(video_frames, fps=30)}, step=step)


def log_custom_plot(step: int, name: str, x_data: np.ndarray, y_data: np.ndarray, 
                   xlabel: str = "X", ylabel: str = "Y", title: str = "") -> None:
    """
    Cria e loga plot customizado.
    
    Args:
        step: Step atual
        name: Nome do plot
        x_data: Dados do eixo X
        y_data: Dados do eixo Y
        xlabel: Label do eixo X
        ylabel: Label do eixo Y
        title: Título do plot
    """
    if not wandb.run:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or name)
    plt.grid(True)
    
    # Converter plot para imagem
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Logar como imagem
    image = PIL.Image.open(buf)
    wandb.log({name: wandb.Image(image)}, step=step)
    
    plt.close()
    buf.close()


def log_isaac_sim_metrics(
    step: int,
    physics_fps: float,
    render_fps: float,
    simulation_time: float,
    num_active_robots: int,
    collision_count: int = 0,
    fallen_robots: int = 0
) -> None:
    """
    Loga métricas específicas do IsaacSim.
    
    Args:
        step: Step atual
        physics_fps: FPS da simulação física
        render_fps: FPS da renderização
        simulation_time: Tempo de simulação
        num_active_robots: Número de robôs ativos
        collision_count: Número de colisões detectadas
        fallen_robots: Número de robôs que caíram
    """
    metrics = {
        "isaac_sim/physics_fps": physics_fps,
        "isaac_sim/render_fps": render_fps,
        "isaac_sim/simulation_time": simulation_time,
        "isaac_sim/active_robots": num_active_robots,
        "isaac_sim/collisions": collision_count,
        "isaac_sim/fallen_robots": fallen_robots,
        "isaac_sim/success_rate": (num_active_robots - fallen_robots) / max(num_active_robots, 1)
    }
    
    log_metrics(step, metrics)


def create_experiment_summary(
    final_reward: float,
    total_timesteps: int,
    training_time: float,
    best_reward: float,
    convergence_step: Optional[int] = None
) -> None:
    """
    Cria sumário final do experimento.
    
    Args:
        final_reward: Recompensa final média
        total_timesteps: Total de timesteps treinados
        training_time: Tempo total de treinamento (segundos)
        best_reward: Melhor recompensa obtida
        convergence_step: Step onde o modelo convergiu (opcional)
    """
    if not wandb.run:
        return
    
    summary = {
        "experiment/final_reward": final_reward,
        "experiment/best_reward": best_reward,
        "experiment/total_timesteps": total_timesteps,
        "experiment/training_time_hours": training_time / 3600,
        "experiment/sps_average": total_timesteps / training_time,
    }
    
    if convergence_step:
        summary["experiment/convergence_step"] = convergence_step
        summary["experiment/steps_to_convergence"] = convergence_step
    
    # Atualizar sumário do run
    for key, value in summary.items():
        wandb.run.summary[key] = value
    
    print("[WandB] Experiment summary logged:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def finish_wandb() -> None:
    """Finaliza sessão do wandb."""
    if wandb.run:
        wandb.finish()
        print("[WandB] Session finished")


class WandBCallback:
    """
    Callback para logging automático durante treinamento.
    
    Use este callback para logar métricas automaticamente
    durante o loop de treinamento.
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
    
    def on_step(self, metrics: Dict[str, Any]) -> None:
        """
        Chamado a cada step do treinamento.
        
        Args:
            metrics: Métricas do step atual
        """
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            log_metrics(self.step_count, metrics)
    
    def on_episode_end(self, episode_metrics: Dict[str, Any]) -> None:
        """
        Chamado ao final de cada episódio.
        
        Args:
            episode_metrics: Métricas do episódio
        """
        log_metrics(self.step_count, episode_metrics, prefix="episode/")
    
    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """
        Chamado ao final do treinamento.
        
        Args:
            final_metrics: Métricas finais
        """
        create_experiment_summary(**final_metrics)
        finish_wandb() 