import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

from core.multi_sim_helper import MultiSimHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("isaac_gym_multi_env")


class IsaacSimGo2MultiEnv(gym.Env):
    """
    Ambiente Gymnasium para treinamento de RL com múltiplos robôs GO2 no Isaac Sim.
    
    Este ambiente implementa a interface padrão do Gymnasium para permitir
    o treinamento de algoritmos de reinforcement learning com múltiplos robôs quadrúpedes
    GO2 simulados em paralelo no Isaac Sim, com controle seguro das juntas.
    
    Baseado no IsaacSimGo2Env original, mas estendido para múltiplos ambientes paralelos.
    
    Attributes:
        multi_helper (MultiSimHelper): Interface para controle da simulação multi-robô
        num_envs (int): Número de ambientes paralelos
        n_joints (int): Número de juntas controláveis por robô
        obs_dim (int): Dimensão do vetor de observações por robô
        action_space (Box): Espaço de ações (posições das juntas normalizadas) para todos os robôs
        observation_space (Box): Espaço de observações para todos os robôs
        use_relative_control (bool): Se True, usa controle relativo (mais suave)
        relative_scale (float): Escala para controle relativo
    
    Example:
        >>> env = IsaacSimGo2MultiEnv(num_envs=4)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, done, truncated, info = env.step(action)
        >>> env.close()
    """

    def __init__(self, num_envs: int = 4, spacing: float = 3.0, safety_margin: float = 0.1, 
                 use_relative_control: bool = False, relative_scale: float = 0.1):
        """
        Inicializa o ambiente Gymnasium para múltiplos robôs GO2.
        
        Args:
            num_envs (int): Número de ambientes paralelos (robôs)
            spacing (float): Espaçamento entre robôs em metros
            safety_margin (float): Margem de segurança para os limites das juntas (0.1 = 10%)
            use_relative_control (bool): Se True, usa controle relativo ao invés de absoluto
            relative_scale (float): Escala para o controle relativo (menor = mais suave)
        """
        super().__init__()

        # Configurações de controle
        self.num_envs = num_envs
        self.use_relative_control = use_relative_control
        self.relative_scale = relative_scale

        # Instancia o helper multi-robô
        self.multi_helper = MultiSimHelper(
            num_envs=num_envs,
            spacing=spacing,
            safety_margin=safety_margin
        )

        # --- Espaços de ação e observação ---
        self.n_joints = 12  # GO2 tem 12 juntas controláveis
        # Observação baseada no formato do isaac_gym_env.py original
        self.obs_dim = self.n_joints * 2 + 3 + 6  # joints pos + joints vel + base pos + base vel

        # Ações para todos os robôs: shape (num_envs, n_joints)
        # Sempre normalizadas entre -1 e 1 para segurança
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.n_joints),
            dtype=np.float32
        )

        # Observações para todos os robôs: shape (num_envs, obs_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.obs_dim),
            dtype=np.float32
        )

        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )
        
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Variáveis de estado
        self.episode_steps = np.zeros(self.num_envs, dtype=int)
        self.episode_rewards = np.zeros(self.num_envs, dtype=float)

        # Reseta no início
        self.reset()
        
        logger.info(f"[INFO] Ambiente multi-robô inicializado:")
        logger.info(f"  - Número de robôs: {num_envs}")
        logger.info(f"  - Espaçamento: {spacing}m")
        logger.info(f"  - Safety margin: {safety_margin}")
        logger.info(f"  - Relative control: {use_relative_control}")
        logger.info(f"  - Action space: {self.action_space.shape}")
        logger.info(f"  - Observation space: {self.observation_space.shape}")

    def reset(self, seed=None, options=None):
        """
        Reseta todos os ambientes para um novo episódio.
        
        Args:
            seed (int, optional): Semente para geração de números aleatórios
            options (dict, optional): Opções adicionais para o reset
        
        Returns:
            tuple: (observations, info) onde:
                - observations (np.ndarray): Estados iniciais de todos os robôs [num_envs, obs_dim]
                - info (dict): Informações adicionais sobre o reset
        """
        # Reset do helper multi-robô
        self.multi_helper.reset_all_environments()
        
        # Reset das variáveis de episódio
        self.episode_steps.fill(0)
        self.episode_rewards.fill(0.0)
        
        # Coleta observações iniciais
        observations = self._get_observations_array()
        
        info = {
            "reset_successful": True,
            "num_envs": self.num_envs,
            "control_mode": "relative" if self.use_relative_control else "absolute",
            "episode_steps": self.episode_steps.copy(),
            "episode_rewards": self.episode_rewards.copy(),
            "_episode": np.zeros(self.num_envs, dtype=bool),
            "episode": {
                'r': np.zeros(self.num_envs, dtype=float),
                'l': np.zeros(self.num_envs, dtype=int)
            }
        }
        
        return observations, info

    def step(self, actions):
        """
        Executa ações em todos os ambientes e avança um passo da simulação.
        
        Args:
            actions (np.ndarray): Ações para todos os robôs [num_envs, n_joints]
                                com valores normalizados entre -1.0 e 1.0
        
        Returns:
            tuple: (observations, rewards, terminated, truncated, info)
                - observations (np.ndarray): Novos estados [num_envs, obs_dim]
                - rewards (np.ndarray): Recompensas obtidas [num_envs]
                - terminated (np.ndarray): Flags de terminação [num_envs]
                - truncated (np.ndarray): Flags de truncamento [num_envs] (sempre False)
                - info (dict): Informações adicionais
        """
        # Valida formato das ações
        if actions.shape != (self.num_envs, self.n_joints):
            raise ValueError(f"Actions shape {actions.shape} doesn't match expected {(self.num_envs, self.n_joints)}")

        # Aplica ações
        if self.use_relative_control:
            # Para controle relativo, precisamos aplicar uma por vez
            for env_id in range(self.num_envs):
                current_positions = self.multi_helper.go2_robots.get_joint_positions()
                action_clipped = np.clip(actions[env_id], -1.0, 1.0)
                
                # Calcula mudança relativa baseada no SimHelper original
                observer = self.multi_helper.robot_observers[env_id]
                lower, upper = observer.get_joint_limits()
                joint_range_safe = upper - lower
                position_changes = action_clipped * self.relative_scale * joint_range_safe
                
                # Aplica mudança
                new_positions = current_positions.copy()
                new_positions[env_id] += position_changes
                
                # Limita às zonas seguras
                new_positions[env_id] = np.clip(
                    new_positions[env_id],
                    lower,
                    upper
                )
                
                self.multi_helper.go2_robots.set_joint_positions(new_positions)
        else:
            # Controle absoluto - aplica diretamente
            self.multi_helper.apply_actions(actions)
        
        # Avança a simulação
        self.multi_helper.step_simulation(render=True)

        # Coleta observações, recompensas e terminações
        observations = self._get_observations_array()
        rewards = self._get_rewards_array()
        terminated = self._get_terminations_array()
        truncated = np.zeros(self.num_envs, dtype=bool)  # Nunca truncamos por tempo

        # Atualiza estatísticas de episódio
        self.episode_steps += 1
        self.episode_rewards += rewards

        # Track completed episodes for this step
        episode_flags = np.zeros(self.num_envs, dtype=bool)
        episode_rewards = np.zeros(self.num_envs, dtype=float)
        episode_lengths = np.zeros(self.num_envs, dtype=int)

        # Reset automático de ambientes que terminaram
        for env_id in range(self.num_envs):
            if terminated[env_id]:
                logger.info(f"Environment {env_id} terminated at step {self.episode_steps[env_id]} with reward {self.episode_rewards[env_id]:.3f}")
                
                # Record completed episode data
                episode_flags[env_id] = True
                episode_rewards[env_id] = self.episode_rewards[env_id]
                episode_lengths[env_id] = self.episode_steps[env_id]
                
                # Reset environment
                self.multi_helper.reset_environment(env_id)
                self.episode_steps[env_id] = 0
                self.episode_rewards[env_id] = 0.0

        # Prepare info dict in standard Gymnasium SyncVectorEnv format
        info = {
            "episode_steps": self.episode_steps.copy(),
            "episode_rewards": self.episode_rewards.copy(),
            "control_mode": "relative" if self.use_relative_control else "absolute",
            "num_terminated": np.sum(terminated),
            "termination_reasons": self._get_termination_reasons()
        }
        
        # Add standard episode information (always present, even if empty)
        info["_episode"] = episode_flags
        info["episode"] = {
            'r': episode_rewards,
            'l': episode_lengths
        }
        
        # if np.any(episode_flags):
        #     completed_envs = np.where(episode_flags)[0]
        #     logger.debug(f"Completed episodes in environments {completed_envs}: rewards = {episode_rewards[completed_envs]}, lengths = {episode_lengths[completed_envs]}")

        return observations, rewards, terminated, truncated, info

    def _get_observations_array(self) -> np.ndarray:
        """
        Coleta observações de todos os ambientes no formato compatível com isaac_gym_env.py.
        
        Returns:
            np.ndarray: Array de observações [num_envs, obs_dim]
        """
        observations = []
        
        for env_id in range(self.num_envs):
            observer = self.multi_helper.robot_observers[env_id]
            
            # Coleta dados básicos (formato compatível com SimHelper original)
            joint_states = observer.get_joint_states()
            position, _ = observer.get_robot_pose()
            lin_vel, ang_vel = observer.get_robot_velocities()
            
            # Concatena no mesmo formato do isaac_gym_env.py original
            obs = np.concatenate([
                joint_states['positions'],  # n_joints
                joint_states['velocities'], # n_joints  
                position,                   # 3 (x, y, z)
                np.concatenate([lin_vel, ang_vel])  # 6 (lin_vel + ang_vel)
            ])
            
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)

    def _get_rewards_array(self) -> np.ndarray:
        """
        Calcula recompensas para todos os ambientes usando o sistema de recompensas avançado.
        
        Returns:
            np.ndarray: Array de recompensas [num_envs]
        """
        rewards = []
        reward_infos = self.multi_helper.calculate_rewards(
            target_velocity=1.0,
            target_height=0.5
        )
        
        for reward_info in reward_infos:
            rewards.append(reward_info['total_reward'])
        
        return np.array(rewards, dtype=np.float32)

    def _get_terminations_array(self) -> np.ndarray:
        """
        Verifica condições de terminação para todos os ambientes.
        
        Returns:
            np.ndarray: Array de flags de terminação [num_envs]
        """
        terminations = self.multi_helper.check_terminations()
        terminated = np.array([term[0] for term in terminations], dtype=bool)
        return terminated

    def _get_termination_reasons(self) -> list:
        """
        Obtém as razões de terminação para todos os ambientes.
        
        Returns:
            list: Lista de razões de terminação
        """
        terminations = self.multi_helper.check_terminations()
        reasons = [term[1] for term in terminations]
        return reasons

    def get_joint_limits_info(self) -> dict:
        """
        Retorna informações detalhadas sobre os limites das juntas para todos os robôs.
        
        Returns:
            dict: Informações completas sobre limites e estado das juntas
        """
        joint_limits_info = []
        for env_id in range(self.num_envs):
            observer = self.multi_helper.robot_observers[env_id]
            lower, upper = observer.get_joint_limits_normalized()
            joint_states = observer.get_joint_states()
            
            env_info = {
                "env_id": env_id,
                "current_positions": joint_states['positions'],
                "lower_limits": lower,
                "upper_limits": upper,
                "joint_names": observer.joint_names
            }
            joint_limits_info.append(env_info)
        
        return {
            "environments": joint_limits_info,
            "num_envs": self.num_envs,
            "n_joints": self.n_joints
        }

    def set_control_mode(self, use_relative: bool, scale: float = None):
        """
        Altera o modo de controle dos robôs em tempo real.
        
        Args:
            use_relative (bool): Se True, usa controle relativo
            scale (float, optional): Nova escala para controle relativo
        """
        self.use_relative_control = use_relative
        if scale is not None:
            self.relative_scale = scale
        
        logger.info(f"[INFO] Modo de controle alterado para: {'relativo' if use_relative else 'absoluto'}")
        if use_relative:
            logger.info(f"  - Escala relativa: {self.relative_scale}")

    def demo_safe_actions(self, num_steps: int = 100) -> list:
        """
        Demonstra ações seguras para todos os robôs simultaneamente.
        
        Gera padrões de movimento seguros baseados no número de juntas do robô,
        sem hardcoding valores específicos. Funciona com qualquer robô quadrúpede.
        
        Args:
            num_steps (int): Número de passos para a demonstração
            
        Returns:
            list: Lista de observações durante a demonstração
        """
        logger.info(f"[INFO] Iniciando demonstração de ações seguras para {self.num_envs} robôs...")
        logger.info(f"[INFO] Robô detectado com {self.n_joints} juntas")
        
        observations_history = []
        self.reset()
        
        # Gera padrões de movimento baseados na estrutura das juntas
        def generate_safe_movement_pattern(env_id: int, step: int) -> np.ndarray:
            """
            Gera padrão de movimento seguro genérico para qualquer robô quadrúpede.
            
            Assume estrutura típica: [perna1_joints, perna2_joints, perna3_joints, perna4_joints]
            onde cada perna tem o mesmo número de juntas (hip, thigh, calf, etc.)
            """
            t = step * 0.01
            phase = env_id * np.pi / self.num_envs  # Fase única por robô
            
            # Calcula quantas juntas por perna (assume 4 pernas)
            joints_per_leg = self.n_joints // 4
            
            # Gera ação inicializada com zeros
            action = np.zeros(self.n_joints)
            
            # Para cada perna
            for leg in range(4):
                leg_start_idx = leg * joints_per_leg
                
                # Padrão de caminhada: pernas opostas em fase
                if leg in [0, 3]:  # Pernas diagonalmente opostas
                    leg_phase = t + phase
                else:  # Outras pernas em contra-fase
                    leg_phase = t + np.pi + phase
                
                # Para cada junta da perna
                for joint_in_leg in range(joints_per_leg):
                    joint_idx = leg_start_idx + joint_in_leg
                    
                    if joint_in_leg == 0:  # Hip joint (primeira junta)
                        action[joint_idx] = 0.15 * np.sin(leg_phase)
                    elif joint_in_leg == 1:  # Thigh joint (segunda junta)
                        action[joint_idx] = 0.25 * np.cos(leg_phase)
                    elif joint_in_leg == 2:  # Calf joint (terceira junta)
                        action[joint_idx] = 0.2 * np.sin(leg_phase + np.pi/4)
                    else:  # Juntas adicionais (se houver)
                        action[joint_idx] = 0.1 * np.sin(leg_phase + joint_in_leg)
            
            # Clipa para garantir que está no range seguro [-1, 1]
            return np.clip(action, -1.0, 1.0)
        
        for step in range(num_steps):
            # Gera ações para todos os robôs
            actions = []
            
            for env_id in range(self.num_envs):
                action = generate_safe_movement_pattern(env_id, step)
                actions.append(action)
            
            actions = np.array(actions)
            obs, rewards, terminated, _, info = self.step(actions)
            observations_history.append(obs)
            
            if step % 20 == 0:
                logger.info(f"  Passo {step}: Terminados = {info['num_terminated']}, Rewards = {rewards}")
                
            # Para se muitos robôs terminaram
            if info['num_terminated'] > self.num_envs // 2:
                logger.warning(f"[WARNING] Muitos robôs terminaram no passo {step}")
                break
        
        logger.info("[INFO] Demonstração concluída!")
        return observations_history

    def demo_custom_pattern(self, pattern_type: str = "walking", num_steps: int = 100) -> list:
        """
        Demonstra padrões de movimento customizados.
        
        Args:
            pattern_type (str): Tipo de padrão ('walking', 'trotting', 'static', 'random')
            num_steps (int): Número de passos
            
        Returns:
            list: Lista de observações
        """
        logger.info(f"[INFO] Demonstrando padrão '{pattern_type}' para {self.num_envs} robôs...")
        
        observations_history = []
        self.reset()
        
        def get_pattern_function(pattern: str):
            """Retorna função de geração baseada no padrão escolhido."""
            
            if pattern == "walking":
                def walking_pattern(env_id, step):
                    t = step * 0.01
                    phase = env_id * np.pi / self.num_envs
                    action = np.zeros(self.n_joints)
                    
                    # Padrão de caminhada suave
                    for i in range(self.n_joints):
                        if i % 3 == 0:  # Hip joints
                            action[i] = 0.1 * np.sin(t + phase + i)
                        elif i % 3 == 1:  # Thigh joints  
                            action[i] = 0.2 * np.cos(t + phase + i)
                        else:  # Calf joints
                            action[i] = 0.15 * np.sin(t + phase + i + np.pi/2)
                    
                    return np.clip(action, -1.0, 1.0)
                return walking_pattern
                
            elif pattern == "trotting":
                def trotting_pattern(env_id, step):
                    t = step * 0.02  # Mais rápido
                    phase = env_id * np.pi / self.num_envs
                    action = np.zeros(self.n_joints)
                    
                    # Padrão de trote
                    for i in range(self.n_joints):
                        action[i] = 0.3 * np.sin(2*t + phase + i * np.pi/6)
                    
                    return np.clip(action, -1.0, 1.0)
                return trotting_pattern
                
            elif pattern == "static":
                def static_pattern(env_id, step):
                    # Pequenas oscilações para manter equilíbrio
                    t = step * 0.005
                    action = np.zeros(self.n_joints)
                    
                    for i in range(self.n_joints):
                        action[i] = 0.05 * np.sin(t + i)
                    
                    return np.clip(action, -1.0, 1.0)
                return static_pattern
                
            else:  # random
                def random_pattern(env_id, step):
                    # Movimento aleatório suave
                    return np.random.uniform(-0.2, 0.2, self.n_joints)
                return random_pattern
        
        pattern_func = get_pattern_function(pattern_type)
        
        for step in range(num_steps):
            actions = []
            
            for env_id in range(self.num_envs):
                action = pattern_func(env_id, step)
                actions.append(action)
            
            actions = np.array(actions)
            obs, rewards, terminated, _, info = self.step(actions)
            observations_history.append(obs)
            
            if step % 25 == 0:
                logger.info(f"  Passo {step}: Padrão '{pattern_type}', Terminados = {info['num_terminated']}")
        
        logger.info(f"[INFO] Demonstração de padrão '{pattern_type}' concluída!")
        return observations_history

    def log_environment_states(self, step: int, detailed: bool = False):
        """
        Log do estado de todos os ambientes (compatível com multi_sim_helper).
        
        Args:
            step (int): Passo atual da simulação
            detailed (bool): Se deve incluir informações detalhadas
        """
        self.multi_helper.log_environment_states(step, detailed)

    def close(self):
        """
        Fecha o ambiente e libera recursos.
        """
        logger.info("[INFO] Fechando ambiente multi-robô...")
        self.multi_helper.close()
        logger.info("[INFO] Ambiente fechado.") 