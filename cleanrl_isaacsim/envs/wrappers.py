"""
Wrappers adicionais para ambientes IsaacSim

Este módulo contém wrappers para normalização, estatísticas e outras
funcionalidades úteis para treinamento de RL com IsaacSim.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple
import torch


class TensorToNumpyWrapper(gym.Wrapper):
    """
    Wrapper para converter tensors PyTorch para numpy arrays.
    
    Útil quando o ambiente IsaacSim retorna tensors GPU e o CleanRL
    espera numpy arrays.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def _convert_tensor(self, data):
        """Converte tensor para numpy array se necessário."""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        elif isinstance(data, (list, tuple)):
            return [self._convert_tensor(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_tensor(value) for key, value in data.items()}
        return data
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_tensor(obs), self._convert_tensor(info)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._convert_tensor(obs),
            self._convert_tensor(reward),
            self._convert_tensor(terminated),
            self._convert_tensor(truncated),
            self._convert_tensor(info)
        )


class RewardScalingWrapper(gym.Wrapper):
    """
    Wrapper para escalar recompensas por um fator constante.
    
    Útil para ajustar a magnitude das recompensas para melhor
    performance de treinamento.
    """
    
    def __init__(self, env, scale_factor: float = 1.0):
        super().__init__(env)
        self.scale_factor = scale_factor
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale_factor
        return obs, scaled_reward, terminated, truncated, info


class ActionNoiseWrapper(gym.Wrapper):
    """
    Wrapper para adicionar ruído às ações durante treinamento.
    
    Útil para melhorar a robustez do agente treinado.
    """
    
    def __init__(self, env, noise_std: float = 0.1, training_mode: bool = True):
        super().__init__(env)
        self.noise_std = noise_std
        self.training_mode = training_mode
    
    def step(self, action):
        if self.training_mode and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return self.env.step(action)
    
    def set_training_mode(self, training: bool):
        """Define se o wrapper está em modo de treinamento."""
        self.training_mode = training


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Wrapper para coletar estatísticas detalhadas de episódios.
    
    Coleta informações como:
    - Tempo total de episódio
    - Recompensa acumulada
    - Número de passos
    - Estatísticas customizadas do IsaacSim
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_stats = {}
        self.reset_stats()
    
    def reset_stats(self):
        """Reset das estatísticas do episódio."""
        self.episode_stats = {
            'episode_reward': 0.0,
            'episode_length': 0,
            'episode_start_time': None,
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_stats()
        import time
        self.episode_stats['episode_start_time'] = time.time()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Atualizar estatísticas
        self.episode_stats['episode_reward'] += reward
        self.episode_stats['episode_length'] += 1
        
        # Se episódio terminou, adicionar estatísticas ao info
        # FIXED: Use np.any() to handle arrays properly
        if np.any(terminated) or np.any(truncated):
            import time
            episode_time = time.time() - self.episode_stats['episode_start_time']
            
            # FIXED: info is a list of dicts (one per environment), not a single dict
            if isinstance(info, list):
                for i, env_info in enumerate(info):
                    if isinstance(env_info, dict):
                        if 'episode_stats' not in env_info:
                            env_info['episode_stats'] = {}
                        
                        env_info['episode_stats'].update({
                            'total_reward': self.episode_stats['episode_reward'],
                            'episode_length': self.episode_stats['episode_length'],
                            'episode_time': episode_time,
                            'average_reward_per_step': self.episode_stats['episode_reward'] / max(1, self.episode_stats['episode_length'])
                        })
            else:
                # Fallback for single-env case (shouldn't happen in multi-env)
                if 'episode_stats' not in info:
                    info['episode_stats'] = {}
                
                info['episode_stats'].update({
                    'total_reward': self.episode_stats['episode_reward'],
                    'episode_length': self.episode_stats['episode_length'],
                    'episode_time': episode_time,
                    'average_reward_per_step': self.episode_stats['episode_reward'] / max(1, self.episode_stats['episode_length'])
                })
        
        return obs, reward, terminated, truncated, info


def apply_standard_wrappers(env, reward_scale: float = 1.0, action_noise_std: float = 0.0):
    """
    Aplica conjunto padrão de wrappers para ambientes IsaacSim.
    
    Args:
        env: Ambiente base
        reward_scale: Fator de escala para recompensas
        action_noise_std: Desvio padrão do ruído nas ações
        
    Returns:
        Ambiente com wrappers aplicados
    """
    # Converter tensors para numpy
    env = TensorToNumpyWrapper(env)
    
    # Escalar recompensas se necessário
    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale_factor=reward_scale)
    
    # Adicionar ruído nas ações se especificado
    if action_noise_std > 0:
        env = ActionNoiseWrapper(env, noise_std=action_noise_std)
    
    # Coletar estatísticas de episódio
    env = EpisodeStatsWrapper(env)
    
    return env 