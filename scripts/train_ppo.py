"""
Script de entrada para treinamento PPO no IsaacSim + CleanRL.

Este script configura e executa o treinamento de um agente PPO
com múltiplos robôs GO2 no IsaacSim usando CleanRL.

Exemplo de uso:
    python scripts/train_ppo.py --num-envs 16 --track --wandb-project-name "go2-multibot"
"""

import sys
import os

# Adicionar paths necessários
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cleanrl_isaacsim'))

from cleanrl_isaacsim.algorithms.ppo_isaacsim import train, Args
import tyro


def main():
    """
    Função principal do script de treinamento.
    """
    print("="*80)
    print("CleanRL + IsaacSim Multi-Robot Training")
    print("="*80)
    
    # Parse argumentos usando tyro (mesmo sistema do CleanRL)
    args = tyro.cli(Args)
    
    # Imprimir configuração
    print("\nTraining Configuration:")
    print(f"  Environment: {args.env_id}")
    print(f"  Number of robots: {args.num_envs}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Robot spacing: {args.spacing}m")
    print(f"  WandB tracking: {'Yes' if args.track else 'No'}")
    if args.track:
        print(f"  WandB project: {args.wandb_project_name}")
        print(f"   Monitor at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project_name}")
    
    # Confirmação antes de começar
    response = input("\n Start training? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    print("\n🏁 Starting training...")
    
    try:
        # Executar treinamento
        train(args)
        print("\n Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        raise
    
    finally:
        print("\n Training session ended")


if __name__ == "__main__":
    main() 