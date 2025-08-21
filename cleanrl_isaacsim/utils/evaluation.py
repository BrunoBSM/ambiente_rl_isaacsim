"""
Utilitários para avaliação de modelos treinados.

Este módulo fornece funções para avaliar agentes treinados
em ambientes IsaacSim, incluindo métricas de performance.
"""

import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
import os
import sys

# Importar dependências locais
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cleanrl_isaacsim.envs.multi_env_wrapper import make_env
from cleanrl_isaacsim.envs.wrappers import apply_standard_wrappers


def evaluate_model(
    model: torch.nn.Module,
    env_config: Dict[str, Any],
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    device: torch.device = torch.device("cpu"),
    render: bool = False,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Avalia um modelo treinado em um ambiente.
    
    Args:
        model: Modelo PyTorch treinado
        env_config: Configuração do ambiente
        num_episodes: Número de episódios para avaliação
        max_steps_per_episode: Máximo de passos por episódio
        device: Device para execução do modelo
        render: Se deve renderizar durante avaliação
        verbose: Se deve imprimir progresso
        
    Returns:
        Dicionário com métricas de avaliação
    """
    model.eval()
    
    # Criar ambiente
    env = make_env(**env_config)
    env = apply_standard_wrappers(env, reward_scale=1.0, action_noise_std=0.0)
    
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    
    if verbose:
        print(f"[Evaluation] Starting evaluation with {num_episodes} episodes...")
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_success = False
            
            for step in range(max_steps_per_episode):
                # Converter observação para tensor
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).to(device)
                else:
                    obs_tensor = obs.to(device)
                
                # Obter ação do modelo
                with torch.no_grad():
                    if hasattr(model, 'get_action_and_value'):
                        action, _, _, _ = model.get_action_and_value(obs_tensor)
                    else:
                        # Assumir que é só um actor
                        action = model(obs_tensor)
                    
                    action = action.cpu().numpy()
                
                # Executar ação
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += np.mean(reward) if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Verificar condições de término
                if isinstance(terminated, np.ndarray):
                    if np.any(terminated) or np.any(truncated):
                        break
                else:
                    if terminated or truncated:
                        break
                
                # Verificar sucesso (pode ser customizado)
                if isinstance(info, list):
                    for inf in info:
                        if isinstance(inf, dict) and inf.get('success', False):
                            episode_success = True
                            break
                elif isinstance(info, dict) and info.get('success', False):
                    episode_success = True
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_rates.append(1.0 if episode_success else 0.0)
            
            if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
                print(f"[Evaluation] Episode {episode + 1}/{num_episodes}: "
                      f"Reward={episode_reward:.2f}, Length={episode_length}")
    
    finally:
        env.close()
    
    # Calcular métricas
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'success_rate': np.mean(success_rates),
        'total_episodes': len(episode_rewards)
    }
    
    if verbose:
        print("\n[Evaluation] Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return metrics


def load_model_for_evaluation(
    model_path: str,
    model_class: torch.nn.Module,
    env_config: Dict[str, Any],
    device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """
    Carrega modelo salvo para avaliação.
    
    Args:
        model_path: Caminho para o arquivo do modelo
        model_class: Classe do modelo
        env_config: Configuração do ambiente (para inicializar modelo)
        device: Device para carregar o modelo
        
    Returns:
        Modelo carregado
    """
    # Criar ambiente temporário para obter dimensões
    temp_env = make_env(**env_config)
    
    # Criar instância do modelo
    model = model_class(temp_env).to(device)
    
    # Carregar estado salvo
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    temp_env.close()
    
    return model


def benchmark_model_performance(
    model: torch.nn.Module,
    env_config: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
    benchmark_steps: int = 1000
) -> Dict[str, float]:
    """
    Faz benchmark de performance computacional do modelo.
    
    Args:
        model: Modelo para benchmark
        env_config: Configuração do ambiente
        device: Device para execução
        benchmark_steps: Número de steps para benchmark
        
    Returns:
        Métricas de performance computacional
    """
    model.eval()
    
    # Criar ambiente
    env = make_env(**env_config)
    obs, _ = env.reset()
    
    # Preparar observação
    if isinstance(obs, np.ndarray):
        obs_tensor = torch.FloatTensor(obs).to(device)
    else:
        obs_tensor = obs.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            if hasattr(model, 'get_action_and_value'):
                model.get_action_and_value(obs_tensor)
            else:
                model(obs_tensor)
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(benchmark_steps):
            if hasattr(model, 'get_action_and_value'):
                action, _, _, _ = model.get_action_and_value(obs_tensor)
            else:
                action = model(obs_tensor)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = benchmark_steps / total_time
    
    env.close()
    
    return {
        'inference_fps': fps,
        'avg_inference_time_ms': (total_time / benchmark_steps) * 1000,
        'total_benchmark_time': total_time,
        'benchmark_steps': benchmark_steps
    }


def compare_models(
    model_paths: List[str],
    model_classes: List[torch.nn.Module],
    env_config: Dict[str, Any],
    model_names: Optional[List[str]] = None,
    num_episodes: int = 10,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Dict[str, float]]:
    """
    Compara múltiplos modelos em termos de performance.
    
    Args:
        model_paths: Lista de caminhos para modelos
        model_classes: Lista de classes dos modelos
        env_config: Configuração do ambiente
        model_names: Nomes opcionais para os modelos
        num_episodes: Número de episódios para avaliação
        device: Device para execução
        
    Returns:
        Dicionário com métricas de cada modelo
    """
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(model_paths))]
    
    results = {}
    
    for i, (path, model_class, name) in enumerate(zip(model_paths, model_classes, model_names)):
        print(f"\n[Comparison] Evaluating {name}...")
        
        # Carregar modelo
        model = load_model_for_evaluation(path, model_class, env_config, device)
        
        # Avaliar
        eval_metrics = evaluate_model(
            model, env_config, num_episodes=num_episodes, 
            device=device, verbose=False
        )
        
        # Benchmark performance
        perf_metrics = benchmark_model_performance(model, env_config, device)
        
        # Combinar métricas
        results[name] = {**eval_metrics, **perf_metrics}
    
    # Imprimir comparação
    print("\n[Comparison] Results Summary:")
    print(f"{'Model':<15} {'Mean Reward':<12} {'Success Rate':<12} {'Inference FPS':<12}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['mean_reward']:<12.2f} "
              f"{metrics['success_rate']:<12.2f} {metrics['inference_fps']:<12.1f}")
    
    return results


def save_evaluation_results(
    results: Dict[str, Any],
    save_path: str,
    experiment_name: str = "evaluation"
) -> None:
    """
    Salva resultados de avaliação em arquivo.
    
    Args:
        results: Resultados da avaliação
        save_path: Caminho para salvar
        experiment_name: Nome do experimento
    """
    import json
    import datetime
    
    # Adicionar metadados
    results_with_meta = {
        'experiment_name': experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'results': results
    }
    
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salvar como JSON
    with open(save_path, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    print(f"[Evaluation] Results saved to: {save_path}") 