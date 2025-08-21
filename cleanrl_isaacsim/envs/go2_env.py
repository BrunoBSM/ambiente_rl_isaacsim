"""
Go2 Single Robot Environment (Placeholder)

Ambiente para um único robô GO2, útil para testes individuais ou
para usar com gym.vector.SyncVectorEnv no futuro.

Por enquanto, este é um placeholder. A implementação principal
está no MultiEnvWrapper que usa o ambiente multi-robô existente.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


class Go2SingleEnv(gym.Env):
    """
    Ambiente Gymnasium para um único robô GO2 no Isaac Sim.
    
    Este é um placeholder para futuras implementações.
    Por enquanto, use MultiEnvWrapper que adapta o ambiente multi-robô existente.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # TODO: Implementar ambiente single-robot
        # Por enquanto, levanta NotImplementedError
        raise NotImplementedError(
            "Go2SingleEnv não implementado ainda. "
            "Use MultiEnvWrapper do cleanrl_isaacsim.envs.multi_env_wrapper"
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        raise NotImplementedError
    
    def step(self, action: np.ndarray):
        raise NotImplementedError
    
    def close(self):
        pass 