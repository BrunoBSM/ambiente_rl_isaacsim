"""
Utilitários para algoritmos de RL.

Este módulo contém funções auxiliares para implementação
de algoritmos de reinforcement learning.
"""

import random
import numpy as np
import torch
from typing import Optional, Dict, Any, List
import os


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Define seed para reprodutibilidade.
    
    Args:
        seed: Valor do seed
        deterministic: Se deve usar operações determinísticas
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[Utils] Seed set to {seed} (deterministic={deterministic})")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Conta o número de parâmetros treináveis em um modelo.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Número de parâmetros treináveis
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Obtém o device apropriado para treinamento.
    
    Args:
        force_cpu: Se deve forçar uso de CPU
        
    Returns:
        Device para usar
    """
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Utils] Using GPU: {torch.cuda.get_device_name()}")
        print(f"[Utils] GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        print("[Utils] Using CPU")
    
    return device


def save_model(
    model: torch.nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Salva modelo e opcionalmente optimizer.
    
    Args:
        model: Modelo para salvar
        save_path: Caminho para salvar
        optimizer: Optimizer opcional para salvar
        step: Step atual do treinamento
        metadata: Metadados adicionais
    """
    # Criar diretório se não existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Preparar dados para salvar
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if optimizer is not None:
        save_data['optimizer_state_dict'] = optimizer.state_dict()
        save_data['optimizer_class'] = optimizer.__class__.__name__
    
    if step is not None:
        save_data['step'] = step
    
    if metadata is not None:
        save_data['metadata'] = metadata
    
    # Salvar
    torch.save(save_data, save_path)
    print(f"[Utils] Model saved to: {save_path}")


def load_model(
    model: torch.nn.Module,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Carrega modelo e opcionalmente optimizer.
    
    Args:
        model: Modelo para carregar estado
        load_path: Caminho para carregar
        optimizer: Optimizer opcional para carregar estado
        device: Device para mapear tensors
        
    Returns:
        Dicionário com informações carregadas
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Carregar dados
    checkpoint = torch.load(load_path, map_location=device)
    
    # Carregar estado do modelo
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Carregar estado do optimizer se disponível
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[Utils] Model loaded from: {load_path}")
    
    return {
        'step': checkpoint.get('step', 0),
        'metadata': checkpoint.get('metadata', {}),
        'model_class': checkpoint.get('model_class', 'Unknown'),
        'optimizer_class': checkpoint.get('optimizer_class', 'Unknown')
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> torch.Tensor:
    """
    Calcula Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Tensor de recompensas (T, N)
        values: Tensor de valores (T, N)
        dones: Tensor de terminações (T, N)
        next_value: Valor do próximo estado (1, N)
        gamma: Fator de desconto
        gae_lambda: Parâmetro lambda do GAE
        
    Returns:
        Tensor de vantagens (T, N)
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    
    return advantages


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza vantagens para ter média 0 e desvio padrão 1.
    
    Args:
        advantages: Tensor de vantagens
        eps: Valor pequeno para evitar divisão por zero
        
    Returns:
        Vantagens normalizadas
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def linear_schedule(start_value: float, end_value: float, progress: float) -> float:
    """
    Agenda linear de valores.
    
    Args:
        start_value: Valor inicial
        end_value: Valor final
        progress: Progresso entre 0 e 1
        
    Returns:
        Valor interpolado
    """
    return start_value + progress * (end_value - start_value)


def cosine_schedule(start_value: float, end_value: float, progress: float) -> float:
    """
    Agenda cosseno de valores.
    
    Args:
        start_value: Valor inicial
        end_value: Valor final
        progress: Progresso entre 0 e 1
        
    Returns:
        Valor interpolado usando cosseno
    """
    import math
    return end_value + (start_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))


def exponential_schedule(start_value: float, end_value: float, progress: float, decay_rate: float = 2.0) -> float:
    """
    Agenda exponencial de valores.
    
    Args:
        start_value: Valor inicial
        end_value: Valor final
        progress: Progresso entre 0 e 1
        decay_rate: Taxa de decaimento
        
    Returns:
        Valor interpolado usando decaimento exponencial
    """
    import math
    return end_value + (start_value - end_value) * math.exp(-decay_rate * progress)


class MovingAverage:
    """
    Classe para calcular média móvel de valores.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        """
        Atualiza média móvel com novo valor.
        
        Args:
            value: Novo valor
            
        Returns:
            Média móvel atual
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
    
    def get_average(self) -> float:
        """Retorna média móvel atual."""
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    def reset(self) -> None:
        """Reset da média móvel."""
        self.values = []


class EarlyStopping:
    """
    Classe para early stopping baseado em uma métrica.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, maximize: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.best_value = float('-inf') if maximize else float('inf')
        self.wait = 0
        self.stopped = False
    
    def update(self, value: float) -> bool:
        """
        Atualiza early stopping com novo valor.
        
        Args:
            value: Novo valor da métrica
            
        Returns:
            True se deve parar o treinamento
        """
        if self.maximize:
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped = True
        
        return self.stopped


def log_memory_usage() -> None:
    """Loga uso de memória GPU se disponível."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[Utils] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Valida se configuração tem todas as chaves necessárias.
    
    Args:
        config: Dicionário de configuração
        required_keys: Lista de chaves obrigatórias
        
    Raises:
        ValueError: Se alguma chave obrigatória estiver faltando
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    print(f"[Utils] Configuration validated successfully") 