"""
Script para busca de hiperparâmetros usando WandB Sweeps.

Este script configura e executa sweeps de hiperparâmetros
para otimizar o treinamento de agentes PPO no IsaacSim.

Exemplo de uso:
    python scripts/sweep_hyperparams.py --sweep-config configs/sweep.yaml
"""

import sys
import os
import yaml
import argparse
from typing import Dict, Any

# Adicionar paths necessários
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cleanrl_isaacsim'))

from cleanrl_isaacsim.algorithms.ppo_isaacsim import train, Args
from cleanrl_isaacsim.utils.wandb_utils import init_wandb
import wandb


# Configuração padrão de sweep
DEFAULT_SWEEP_CONFIG = {
    'method': 'bayes',  # random, grid, bayes
    'metric': {
        'name': 'charts/episodic_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'num_steps': {
            'values': [8, 16, 32, 64]
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.95,
            'max': 0.999
        },
        'gae_lambda': {
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.98
        },
        'clip_coef': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.3
        },
        'vf_coef': {
            'values': [0.5, 1.0, 2.0, 4.0]
        },
        'num_minibatches': {
            'values': [1, 2, 4, 8]
        },
        'update_epochs': {
            'values': [2, 4, 8, 10]
        }
    }
}

# Configuração simplificada para testes rápidos
QUICK_SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {
        'name': 'charts/episodic_return',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.003, 0.01]
        },
        'num_steps': {
            'values': [8, 16]
        },
        'clip_coef': {
            'values': [0.1, 0.2, 0.3]
        }
    }
}


def load_sweep_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega configuração de sweep de arquivo YAML.
    
    Args:
        config_path: Caminho para arquivo de configuração
        
    Returns:
        Dicionário com configuração de sweep
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded sweep config from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("📝 Using default configuration")
        return DEFAULT_SWEEP_CONFIG


def create_sweep_function(base_args: Args):
    """
    Cria função de sweep que será executada pelo WandB.
    
    Args:
        base_args: Argumentos base para treinamento
        
    Returns:
        Função de sweep
    """
    def sweep_function():
        # Inicializar WandB run
        wandb.init()
        
        # Obter hiperparâmetros do sweep
        config = wandb.config
        
        # Atualizar argumentos com hiperparâmetros do sweep
        args = base_args
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"⚠️ Unknown parameter in sweep: {key}")
        
        # Configurar nomes únicos
        args.exp_name = f"sweep_{wandb.run.name}"
        args.track = True  # Sempre trackear durante sweeps
        
        print(f"🔄 Starting sweep run: {wandb.run.name}")
        print(f"📊 Parameters: {dict(config)}")
        
        try:
            # Executar treinamento
            train(args)
        except Exception as e:
            print(f"❌ Sweep run failed: {e}")
            wandb.log({"error": str(e)})
            raise
    
    return sweep_function


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for PPO IsaacSim")
    
    # Configuração do sweep
    parser.add_argument("--sweep-config", type=str, default=None,
                       help="Path to sweep configuration YAML file")
    parser.add_argument("--sweep-name", type=str, default="ppo-isaacsim-sweep",
                       help="Name for the sweep")
    parser.add_argument("--project", type=str, default="cleanrl-isaacsim-sweeps",
                       help="WandB project name")
    parser.add_argument("--entity", type=str, default=None,
                       help="WandB entity")
    
    # Configuração base do experimento
    parser.add_argument("--base-num-envs", type=int, default=4,
                       help="Base number of environments (for quick testing)")
    parser.add_argument("--base-total-timesteps", type=int, default=500_000,
                       help="Base total timesteps (reduced for sweeps)")
    parser.add_argument("--quick", action="store_true",
                       help="Use quick sweep configuration")
    
    # Controle do sweep
    parser.add_argument("--count", type=int, default=None,
                       help="Number of sweep runs to execute")
    parser.add_argument("--create-only", action="store_true",
                       help="Only create sweep, don't run it")
    
    return parser.parse_args()


def main():
    """Função principal do script de sweep."""
    args = parse_args()
    
    print("="*80)
    print("🔍 CleanRL + IsaacSim Hyperparameter Sweep")
    print("="*80)
    
    # Carregar configuração de sweep
    if args.sweep_config:
        sweep_config = load_sweep_config(args.sweep_config)
    elif args.quick:
        sweep_config = QUICK_SWEEP_CONFIG
        print("🚀 Using quick sweep configuration")
    else:
        sweep_config = DEFAULT_SWEEP_CONFIG
        print("⚙️ Using default sweep configuration")
    
    # Adicionar metadados ao sweep
    sweep_config['name'] = args.sweep_name
    sweep_config['description'] = f"PPO hyperparameter sweep for IsaacSim multi-robot training"
    
    print(f"\n📋 Sweep Configuration:")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"  Parameters: {len(sweep_config['parameters'])} parameters")
    print(f"  Project: {args.project}")
    
    # Configurar argumentos base para treinamento
    base_args = Args(
        env_id="Go2MultiEnv",
        num_envs=args.base_num_envs,
        total_timesteps=args.base_total_timesteps,
        track=True,
        wandb_project_name=args.project,
        wandb_entity=args.entity,
        # Configurações para sweep (mais conservadoras)
        spacing=2.0,  # Menor espaçamento para sweep mais rápido
        safety_margin=0.1,
        use_relative_control=False,
        relative_scale=0.1,
        # Seed variável para cada run
        seed=1,  # Seed padrão (WandB pode sobrescrever)
    )
    
    # Criar sweep
    print(f"\n🔄 Creating sweep...")
    try:
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity
        )
        print(f"✅ Sweep created: {sweep_id}")
        print(f"🌐 Dashboard: https://wandb.ai/{args.entity or 'your-username'}/{args.project}/sweeps/{sweep_id}")
        
        if args.create_only:
            print("🛑 Sweep created but not started (--create-only flag)")
            print(f"To run the sweep manually:")
            print(f"  wandb agent {args.entity}/{args.project}/{sweep_id}")
            return
        
        # Executar sweep
        print(f"\n🚀 Starting sweep agent...")
        wandb.agent(
            sweep_id,
            function=create_sweep_function(base_args),
            count=args.count,
            project=args.project,
            entity=args.entity
        )
        
        print("✅ Sweep completed!")
        
    except Exception as e:
        print(f"❌ Sweep failed: {e}")
        raise


if __name__ == "__main__":
    main() 