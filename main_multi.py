"""
Script principal para teste e demonstração do ambiente Isaac Sim GO2 Multi-Ambiente.

Este script executa um loop básico de teste do ambiente de reinforcement learning
com múltiplos robôs GO2, demonstrando o uso da API do Gymnasium com ações aleatórias
em paralelo. É útil para verificar se o ambiente multi-robô está funcionando 
corretamente antes de iniciar o treinamento com algoritmos de RL.

Baseado no main.py original, mas adaptado para múltiplos robôs usando a estrutura
existente do isaac_gym_env.py como fundamentação.

Example:
    Para executar o teste:
    $ ./python.sh main_multi.py
    
Note:
    Este script usa ações aleatórias e não treina um agente real.
    Para treinamento com algoritmos de RL, usaremos bibliotecas como
    Stable-Baselines3, Ray RLlib, ou implementações customizadas.
"""

from isaac_gym_multi_env import IsaacSimGo2MultiEnv
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main_multi")


def main():
    """
    Função principal que executa um loop de teste com ações aleatórias para múltiplos robôs.
    
    Inicializa o ambiente multi-robô, executa passos com ações aleatórias,
    monitora recompensas e reseta automaticamente quando episódios terminam.
    Útil para debug e verificação básica do funcionamento do ambiente.
    
    Estrutura baseada no main.py original, mas estendida para múltiplos ambientes.
    """
    # Configuração do ambiente multi-robô
    NUM_ENVS = 16  # Número de robôs (2x2 grid) - reduzido de 16 para 4
    SPACING = 0.5  # spaçamento entre robôs em metros
    USE_RELATIVE_CONTROL = False  # Usar controle absoluto inicialmente
    MAX_STEPS = 10000  # Número máximo de passos - reduzido para teste mais rápido

    logger.info("="*60)
    logger.info("INICIANDO TESTE DO AMBIENTE ISAAC SIM GO2 MULTI-ROBÔ")
    logger.info("="*60)

    # Inicializa o ambiente multi-robô (baseado na estrutura do isaac_gym_env.py)
    env = IsaacSimGo2MultiEnv(
        num_envs=NUM_ENVS,
        spacing=SPACING,
        safety_margin=0.1,
        use_relative_control=USE_RELATIVE_CONTROL,
        relative_scale=0.1
    )

    # Primeira observação do ambiente (similar ao main.py original)
    obs, info = env.reset()
    
    # Variáveis de controle de episódio (estendidas para múltiplos robôs)
    total_rewards = np.zeros(NUM_ENVS)
    episode_counts = np.zeros(NUM_ENVS, dtype=int)
    step_counts = np.zeros(NUM_ENVS, dtype=int)

    logger.info(f"Ambiente inicializado com sucesso!")
    logger.info(f"Dimensão das observações: {obs.shape} [{NUM_ENVS} robôs, {obs.shape[1]} obs cada]")
    logger.info(f"Dimensão das ações: {env.action_space.shape} [{NUM_ENVS} robôs, {env.n_joints} juntas cada]")
    logger.info(f"Controle: {'Relativo' if USE_RELATIVE_CONTROL else 'Absoluto'}")
    logger.info("-" * 60)

    # Loop principal de teste (estrutura baseada no main.py original)
    for step in range(MAX_STEPS):
        # Gera ações aleatórias dentro do espaço de ações para todos os robôs
        # Formato: [num_envs, n_joints] com valores [-1, 1]
        actions = env.action_space.sample()

        # Executa as ações no ambiente (todos os robôs simultaneamente)
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Acumula recompensa total de cada robô
        total_rewards += rewards
        step_counts += 1
        
        # Log básico do progresso (adaptado para múltiplos robôs)
        if step % 50 == 0:  # Print a cada 50 passos para evitar spam
            logger.info(f"\n=== Passo {step} ===")
            logger.info(f"Recompensas instantâneas: {rewards}")
            logger.info(f"Recompensas totais acumuladas: {total_rewards}")
            logger.info(f"Robôs terminados: {np.sum(terminated)}/{NUM_ENVS}")
            logger.info(f"Episódios por robô: {episode_counts}")

        # Verifica se algum episódio terminou (baseado no main.py original)
        for env_id in range(NUM_ENVS):
            if terminated[env_id]:
                logger.info(f"\n--- Robô {env_id} ---")
                logger.info(f"Episódio {episode_counts[env_id]} finalizado após {step_counts[env_id]} passos")
                logger.info(f"Recompensa total: {total_rewards[env_id]:.3f}")
                logger.info(f"Razão da terminação: {info['termination_reasons'][env_id]}")
                
                # Reset automático já é feito pelo environment, apenas atualizamos contadores
                episode_counts[env_id] += 1
                total_rewards[env_id] = 0.0
                step_counts[env_id] = 0
                
                logger.info(f"Iniciando episódio {episode_counts[env_id]} para robô {env_id}")
                logger.info("-" * 30)

        # Demonstração de mudança de modo de controle no meio da simulação
        if step == 1000:
            logger.info("\n🔄 Mudando para controle relativo...")
            env.set_control_mode(use_relative=True, scale=0.05)
            
        if step == 2000:
            logger.info("\n🔄 Voltando para controle absoluto...")
            env.set_control_mode(use_relative=False)

        # Log detalhado a cada 200 passos
        if step % 200 == 0 and step > 0:
            env.log_environment_states(step, detailed=True)

        # Interrompe se todos os robôs tiveram pelo menos 3 episódios
        if np.all(episode_counts >= 3):
            logger.info(f"\n🎯 Todos os robôs completaram pelo menos 3 episódios!")
            logger.info("Finalizando teste...")
            break

    # Estatísticas finais (baseadas no main.py original, mas estendidas)
    logger.info("\n" + "="*60)
    logger.info("ESTATÍSTICAS FINAIS DO TESTE")
    logger.info("="*60)
    
    for env_id in range(NUM_ENVS):
        logger.info(f"Robô {env_id}:")
        logger.info(f"  - Episódios completados: {episode_counts[env_id]}")
        logger.info(f"  - Passos no episódio atual: {step_counts[env_id]}")
        logger.info(f"  - Recompensa acumulada: {total_rewards[env_id]:.3f}")
    
    logger.info(f"\nTotal de passos executados: {step}")
    logger.info(f"Total de episódios: {np.sum(episode_counts)}")
    logger.info(f"Média de episódios por robô: {np.mean(episode_counts):.2f}")

    # Demonstração de ações seguras 
    logger.info("\nExecutando demonstração de ações seguras...")
    try:
        safe_demo_obs = env.demo_safe_actions(num_steps=100)
        logger.info(f"Demonstração concluída com {len(safe_demo_obs)} observações coletadas")
    except Exception as e:
        logger.error(f"Erro durante demonstração: {e}")

    # Informações sobre limites das juntas 
    logger.info("\nInformações sobre limites das juntas:")
    try:
        joint_info = env.get_joint_limits_info()
        logger.info(f"Total de robôs: {joint_info['num_envs']}")
        logger.info(f"Juntas por robô: {joint_info['n_joints']}")
        
        # Mostra info detalhada apenas do primeiro robô
        if joint_info['environments']:
            env_0_info = joint_info['environments'][0]
            logger.info(f"Posições atuais robô 0: {env_0_info['current_positions'][:3]}... (primeiras 3)")
    except Exception as e:
        logger.error(f"Erro ao obter informações das juntas: {e}")

    # Fecha o ambiente e limpa recursos 
    logger.info("\nTeste finalizado. Fechando ambiente...")
    env.close()
    logger.info("Ambiente fechado com sucesso!")


if __name__ == "__main__":
    main()