"""
MultiEnvWrapper para adaptar ambiente multi-robô IsaacSim para CleanRL

Este wrapper converte o ambiente multi-robô existente (core/isaac_gym_multi_env.py)
para a API esperada pelo CleanRL, mantendo compatibilidade com gym.vector.VectorEnv.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import os

# Adicionar o diretório core ao path para importar o ambiente existente
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

try:
    from isaac_gym_multi_env import IsaacSimGo2MultiEnv
except ImportError as e:
    print(f"Warning: Could not import IsaacSimGo2MultiEnv: {e}")
    IsaacSimGo2MultiEnv = None


class MultiEnvWrapper(gym.vector.VectorEnv):
    """
    Wrapper para adaptar seu ambiente multi-robô atual
    (core/isaac_gym_multi_env.py) para a API esperada pelo CleanRL.
    
    Este wrapper:
    1. Converte observações de formato batch (num_envs, obs_dim) para lista
    2. Adapta retornos de step() para formato CleanRL esperado
    3. Gerencia episódios individuais por robô
    4. Mantém compatibilidade com tensors GPU do IsaacSim
    
    Args:
        num_envs: Número de ambientes paralelos (robôs)
        spacing: Espaçamento entre robôs em metros
        safety_margin: Margem de segurança para limites das juntas
        use_relative_control: Usar controle relativo ou absoluto
        relative_scale: Escala para controle relativo
    """

    def __init__(
        self,
        num_envs: int = 4,
        spacing: float = 3.0,
        safety_margin: float = 0.1,
        use_relative_control: bool = False,
        relative_scale: float = 0.1,
        **kwargs
    ):
        if IsaacSimGo2MultiEnv is None:
            raise ImportError("Could not import IsaacSimGo2MultiEnv. Make sure core/ is accessible.")
        
        # Criar o ambiente multi-robô existente
        self.env = IsaacSimGo2MultiEnv(
            num_envs=num_envs,
            spacing=spacing,
            safety_margin=safety_margin,
            use_relative_control=use_relative_control,
            relative_scale=relative_scale
        )
        
        self.num_envs = num_envs
        
        # Adaptar espaços de ação e observação para formato single-env
        # CleanRL espera (obs_dim,) por ambiente, não (num_envs, obs_dim)
        single_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.obs_dim,),  # obs_dim por robô individual
            dtype=np.float32
        )
        
        single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.n_joints,),  # n_joints por robô individual
            dtype=np.float32
        )
        
        # Inicializar VectorEnv com espaços adaptados
        super().__init__(num_envs, single_obs_space, single_action_space)
        
        # Promover atributos para evitar warnings de deprecação
        self.single_observation_space = single_obs_space
        self.single_action_space = single_action_space
        
        # Estados para tracking de episódios individuais
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        
        print(f"[MultiEnvWrapper] Initialized with {num_envs} environments")
        print(f"[MultiEnvWrapper] Single obs space: {single_obs_space.shape}")
        print(f"[MultiEnvWrapper] Single action space: {single_action_space.shape}")

    def reset(
        self, 
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments and return observations in CleanRL format.
        
        Returns:
            observations: Array of shape (num_envs, obs_dim)
            infos: List of info dicts, one per environment
        """
        # Reset do ambiente multi-robô existente
        # Seu ambiente retorna (num_envs, obs_dim)
        batch_obs, batch_info = self.env.reset()
        
        # Reset episode tracking
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        
        # Converter para formato esperado pelo CleanRL
        # CleanRL espera np.ndarray de shape (num_envs, obs_dim)
        observations = np.array(batch_obs, dtype=np.float32)
        
        # Criar lista de infos (um dict por ambiente)
        infos = []
        for i in range(self.num_envs):
            info = {}
            if isinstance(batch_info, dict):
                # Se batch_info é dict, distribuir as chaves
                for key, value in batch_info.items():
                    try:
                        if hasattr(value, '__len__') and len(value) == self.num_envs:
                            info[key] = value[i]
                        else:
                            info[key] = value
                    except (TypeError, IndexError):
                        info[key] = value
            infos.append(info)
        
        return observations, infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of shape (num_envs, action_dim)
            
        Returns:
            observations: Array of shape (num_envs, obs_dim)
            rewards: Array of shape (num_envs,)
            terminations: Array of shape (num_envs,) - episode ended due to task completion
            truncations: Array of shape (num_envs,) - episode ended due to time limit
            infos: List of info dicts, one per environment
        """
        # Garantir que actions está no formato correto
        if actions.shape != (self.num_envs, self.env.n_joints):
            raise ValueError(f"Expected actions shape {(self.num_envs, self.env.n_joints)}, got {actions.shape}")
        
        # Step no ambiente multi-robô existente
        # Seu ambiente espera (num_envs, n_joints) e retorna formato similar
        batch_obs, batch_rewards, batch_dones, batch_truncated, batch_info = self.env.step(actions)
        
        # Conversão robusta dos tipos retornados
        
        # Converter para arrays numpy com shapes corretas
        observations = np.asarray(batch_obs, dtype=np.float32)
        rewards = np.asarray(batch_rewards, dtype=np.float32)
        
        # CleanRL distingue entre terminations (tarefa concluída) e truncations (limite de tempo)
        # Por enquanto, tratamos todos os dones como terminations
        terminations = np.asarray(batch_dones, dtype=bool)
        
        # Tratar truncations de forma mais robusta
        if batch_truncated is not None:
            truncations = np.asarray(batch_truncated, dtype=bool)
        else:
            truncations = np.zeros(self.num_envs, dtype=bool)
            
        # Garantir shapes corretos
        assert observations.shape[0] == self.num_envs, f"Obs shape mismatch: {observations.shape}"
        assert rewards.shape == (self.num_envs,), f"Rewards shape mismatch: {rewards.shape}"
        assert terminations.shape == (self.num_envs,), f"Terminations shape mismatch: {terminations.shape}"
        assert truncations.shape == (self.num_envs,), f"Truncations shape mismatch: {truncations.shape}"
        
        # Atualizar tracking de episódios
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        # Criar lista de infos com estatísticas de episódio
        infos = []
        for i in range(self.num_envs):
            info = {}
            
            # Adicionar info do ambiente original
            if isinstance(batch_info, dict):
                for key, value in batch_info.items():
                    try:
                        if hasattr(value, '__len__') and len(value) == self.num_envs:
                            info[key] = value[i]
                        else:
                            info[key] = value
                    except (TypeError, IndexError):
                        # Se valor não é indexável, apenas copiar
                        info[key] = value
            
            # Adicionar estatísticas de episódio quando termina
            # Usar .item() para garantir que é um valor escalar
            term_i = terminations[i].item() if hasattr(terminations[i], 'item') else bool(terminations[i])
            trunc_i = truncations[i].item() if hasattr(truncations[i], 'item') else bool(truncations[i])
            
            if term_i or trunc_i:
                info["episode"] = {
                    "r": float(self.episode_returns[i]),
                    "l": int(self.episode_lengths[i])
                }
                # Reset tracking para este ambiente
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0
            
            infos.append(info)
        
        return observations, rewards, terminations, truncations, infos

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return []

    def render(self, mode: str = "human"):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None

    @property
    def unwrapped(self):
        """Return the original environment."""
        return self.env


def make_env(
    num_envs: int = 4,
    spacing: float = 3.0,
    safety_margin: float = 0.1,
    use_relative_control: bool = False,
    relative_scale: float = 0.1,
    **kwargs
) -> MultiEnvWrapper:
    """
    Factory function para criar o ambiente wrapped.
    
    Args:
        num_envs: Número de ambientes paralelos
        spacing: Espaçamento entre robôs
        safety_margin: Margem de segurança para juntas
        use_relative_control: Usar controle relativo
        relative_scale: Escala para controle relativo
        
    Returns:
        MultiEnvWrapper configurado
    """
    return MultiEnvWrapper(
        num_envs=num_envs,
        spacing=spacing,
        safety_margin=safety_margin,
        use_relative_control=use_relative_control,
        relative_scale=relative_scale,
        **kwargs
    ) 