"""
Configurações padrão para treinamento PPO no IsaacSim.

Este arquivo contém hiperparâmetros e configurações recomendadas
para treinamento de agentes PPO com múltiplos robôs GO2.
"""

# Configurações do ambiente
DEFAULT_ENV_CONFIG = {
    "num_envs": 16,
    "spacing": 3.0,
    "safety_margin": 0.1,
    "use_relative_control": False,
    "relative_scale": 0.1,
}

# Configurações do algoritmo PPO
DEFAULT_PPO_CONFIG = {
    # Hiperparâmetros básicos
    "learning_rate": 0.0026,
    "total_timesteps": 10_000_000,
    "num_steps": 16,
    "batch_size": None,  # Calculado automaticamente (num_envs * num_steps)
    "minibatch_size": None,  # Calculado automaticamente
    
    # PPO específico
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 2,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_vloss": False,
    "ent_coef": 0.0,
    "vf_coef": 2.0,
    "max_grad_norm": 1.0,
    "target_kl": None,
    
    # Scheduling
    "anneal_lr": False,
    
    # Regularização
    "reward_scaler": 1.0,
}

# Configurações de treinamento
DEFAULT_TRAINING_CONFIG = {
    "seed": 1,
    "torch_deterministic": True,
    "cuda": True,
    
    # Logging e tracking
    "track": False,
    "wandb_project_name": "isaacsim",
    "wandb_entity": None,
    "capture_video": False,
    
    # Experimentação
    "exp_name": "ppo_isaacsim",
}

# Configurações para diferentes tipos de experimento

# Configuração para teste rápido (desenvolvimento)
QUICK_TEST_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    **DEFAULT_PPO_CONFIG,
    **DEFAULT_TRAINING_CONFIG,
    "num_envs": 4,
    "total_timesteps": 100_000,
    "num_steps": 8,
    "track": False,
}

# Configuração para treinamento completo
FULL_TRAINING_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    **DEFAULT_PPO_CONFIG,
    **DEFAULT_TRAINING_CONFIG,
    "num_envs": 32,
    "total_timesteps": 50_000_000,
    "num_steps": 32,
    "track": True,
    "anneal_lr": True,
}

# Configuração para experimentos com múltiplos robôs
MULTI_ROBOT_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    **DEFAULT_PPO_CONFIG,
    **DEFAULT_TRAINING_CONFIG,
    "num_envs": 64,
    "spacing": 2.0,
    "total_timesteps": 100_000_000,
    "learning_rate": 0.001,
    "num_steps": 64,
    "track": True,
}

# Configuração para controle relativo
RELATIVE_CONTROL_CONFIG = {
    **DEFAULT_ENV_CONFIG,
    **DEFAULT_PPO_CONFIG,
    **DEFAULT_TRAINING_CONFIG,
    "use_relative_control": True,
    "relative_scale": 0.05,
    "learning_rate": 0.002,
    "track": True,
}

# Mapeamento de configurações predefinidas
PRESET_CONFIGS = {
    "quick": QUICK_TEST_CONFIG,
    "full": FULL_TRAINING_CONFIG,
    "multi": MULTI_ROBOT_CONFIG,
    "relative": RELATIVE_CONTROL_CONFIG,
}


def get_config(preset: str = "default") -> dict:
    """
    Obtém configuração baseada em preset.
    
    Args:
        preset: Nome do preset ("default", "quick", "full", "multi", "relative")
        
    Returns:
        Dicionário com configuração
    """
    if preset == "default":
        return {
            **DEFAULT_ENV_CONFIG,
            **DEFAULT_PPO_CONFIG,
            **DEFAULT_TRAINING_CONFIG,
        }
    
    if preset in PRESET_CONFIGS:
        return PRESET_CONFIGS[preset].copy()
    
    raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys()) + ['default']}")


def print_config(config: dict) -> None:
    """
    Imprime configuração de forma organizada.
    
    Args:
        config: Dicionário de configuração
    """
    print("📋 Configuration:")
    print("─" * 50)
    
    # Agrupar por categorias
    env_keys = ["num_envs", "spacing", "safety_margin", "use_relative_control", "relative_scale"]
    ppo_keys = ["learning_rate", "total_timesteps", "num_steps", "gamma", "gae_lambda", 
                "num_minibatches", "update_epochs", "clip_coef", "vf_coef"]
    training_keys = ["seed", "torch_deterministic", "cuda", "track", "wandb_project_name"]
    
    print("🤖 Environment:")
    for key in env_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print("\n🧠 Algorithm:")
    for key in ppo_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print("\n⚙️ Training:")
    for key in training_keys:
        if key in config:
            print(f"  {key}: {config[key]}")
    
    print("─" * 50)


def validate_config(config: dict) -> None:
    """
    Valida configuração para garantir valores consistentes.
    
    Args:
        config: Dicionário de configuração
        
    Raises:
        ValueError: Se configuração inválida
    """
    # Verificar valores obrigatórios
    required_keys = ["num_envs", "learning_rate", "total_timesteps"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    # Verificar valores válidos
    if config["num_envs"] <= 0:
        raise ValueError("num_envs must be positive")
    
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    
    if config["total_timesteps"] <= 0:
        raise ValueError("total_timesteps must be positive")
    
    if "spacing" in config and config["spacing"] <= 0:
        raise ValueError("spacing must be positive")
    
    # Calcular valores derivados se necessário
    if "batch_size" not in config or config["batch_size"] is None:
        config["batch_size"] = config["num_envs"] * config.get("num_steps", 16)
    
    if "minibatch_size" not in config or config["minibatch_size"] is None:
        num_minibatches = config.get("num_minibatches", 2)
        config["minibatch_size"] = config["batch_size"] // num_minibatches
    
    print("✅ Configuration validated successfully")


# Exemplo de uso
if __name__ == "__main__":
    # Demonstrar diferentes configurações
    configs_to_show = ["default", "quick", "full"]
    
    for preset in configs_to_show:
        print(f"\n{'='*60}")
        print(f"Preset: {preset.upper()}")
        print('='*60)
        
        config = get_config(preset)
        print_config(config)
        
        try:
            validate_config(config)
        except ValueError as e:
            print(f"❌ Validation error: {e}") 