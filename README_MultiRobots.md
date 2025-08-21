# MultiRobotSimHelper - Múltiplos Robôs GO2 no Isaac Sim

Esta extensão permite instanciar e controlar múltiplos robôs GO2 no Isaac Sim usando configurações similares ao Isaac Lab, mas utilizando apenas as APIs nativas do Isaac Sim.

## 🚀 Características Principais

- **Múltiplos Robôs**: Instancie quantos robôs GO2 quiser em uma única simulação
- **Configuração Flexível**: APIs similares ao Isaac Lab para facilitar migração
- **Compatibilidade**: Wrapper disponível para código existente
- **Performance**: Controle eficiente de múltiplos robôs simultaneamente
- **Segurança**: Limites de juntas com margens de segurança configuráveis

## 📁 Arquivos

```
ambiente_rl/
├── sim_helper.py                    # Classe original (1 robô)
├── multi_robot_sim_helper.py        # Nova classe (múltiplos robôs)
├── exemplo_multi_robots.py          # Exemplos de uso
├── migrar_para_multi_robots.py      # Script de migração
└── README_MultiRobots.md           # Este arquivo
```

## 🔧 Instalação e Dependências

Certifique-se de ter o Isaac Sim instalado e as seguintes dependências:

```python
# Dependências básicas (já incluídas no Isaac Sim)
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdPhysics, PhysxSchema
```

## 📚 Uso Básico

### Exemplo Simples (4 robôs)

```python
from multi_robot_sim_helper import MultiRobotSimHelper, DEFAULT_GO2_CONFIG
import numpy as np

# Cria ambiente com 4 robôs
sim = MultiRobotSimHelper(
    num_robots=4,
    config=DEFAULT_GO2_CONFIG,
    robot_spacing=3.0,  # 3 metros entre robôs
    safety_margin=0.1
)

try:
    for step in range(100):
        # Gera ações para todos os robôs
        actions = np.random.uniform(-0.5, 0.5, (sim.num_robots, 12))
        
        # Aplica ações
        sim.apply_actions(actions)
        
        # Avança simulação
        sim.step_simulation(render=True)
        
        # Coleta dados
        observations = sim.get_observations()  # Shape: (num_robots, obs_dim)
        rewards = sim.calculate_rewards()      # Shape: (num_robots,)
        terminations = sim.check_terminations() # Shape: (num_robots,)
        
        # Reseta se necessário
        if np.any(terminations):
            sim.reset()

finally:
    sim.close()
```

### Configuração Customizada

```python
from multi_robot_sim_helper import (
    MultiRobotSimHelper, Go2Config, InitialState, 
    RigidBodyProperties, ArticulationRootProperties, DCMotorConfig
)

# Configuração personalizada
custom_config = Go2Config(
    rigid_props=RigidBodyProperties(
        linear_damping=0.1,
        angular_damping=0.1,
        max_linear_velocity=500.0,
    ),
    articulation_props=ArticulationRootProperties(
        enabled_self_collisions=True,
        solver_position_iteration_count=6,
    ),
    init_state=InitialState(
        pos=(0.0, 0.0, 0.5),  # Altura inicial
        joint_positions={
            ".*L_hip_joint": 0.05,
            ".*R_hip_joint": -0.05,
            ".*_thigh_joint": 0.6,
            ".*_calf_joint": -1.2,
        }
    ),
    actuators={
        "base_legs": DCMotorConfig(
            effort_limit=20.0,
            stiffness=30.0,
            damping=0.8,
        ),
    }
)

# Usar configuração customizada
sim = MultiRobotSimHelper(
    num_robots=6,
    config=custom_config,
    robot_spacing=2.5,
    safety_margin=0.15
)
```

## 🔄 Migração do Código Existente

### Opção 1: Wrapper de Compatibilidade

Para manter compatibilidade com código existente:

```python
from migrar_para_multi_robots import MigrationWrapper

# Substitua: sim = SimHelper()
# Por:
sim = MigrationWrapper(use_multi_robot=False)  # Comportamento original

# Ou para múltiplos robôs:
sim = MigrationWrapper(use_multi_robot=True, num_robots=4)
```

### Opção 2: Migração Direta

```python
# ANTES (SimHelper):
sim = SimHelper()
action = np.random.uniform(-1, 1, 12)
sim.apply_action(action)
obs = sim.get_observation()
reward = sim.calculate_reward()

# DEPOIS (MultiRobotSimHelper):
sim = MultiRobotSimHelper(num_robots=1)
actions = np.random.uniform(-1, 1, (1, 12))  # Shape: (num_robots, 12)
sim.apply_actions(actions)
observations = sim.get_observations()  # Shape: (num_robots, obs_dim)
rewards = sim.calculate_rewards()      # Shape: (num_robots,)
```

## 🎮 Controle Avançado

### Ações Diferentes para Cada Robô

```python
actions = np.zeros((sim.num_robots, 12))

for i in range(sim.num_robots):
    if i == 0:
        # Robô 0: caminhada lenta
        actions[i] = 0.3 * np.sin(step * 0.1 + np.arange(12) * 0.5)
    elif i == 1:
        # Robô 1: movimento estático
        actions[i] = 0.1 * np.sin(step * 0.05 + np.arange(12))
    else:
        # Outros: movimentos aleatórios
        actions[i] = 0.2 * np.random.uniform(-1, 1, 12)

sim.apply_actions(actions)
```

### Monitoramento Individual

```python
# Informações de um robô específico
robot_info = sim.get_robot_info(robot_index=0)
print(f"Robô 0 - Juntas: {robot_info['current_positions']}")

# Posições de todos os robôs
observations = sim.get_observations()
for i, obs in enumerate(observations):
    base_pos = obs[24:27]  # Posição da base
    print(f"Robô {i}: posição {base_pos}")
```

### Layout dos Robôs

Os robôs são automaticamente dispostos em grade:

```python
# 4 robôs -> grade 2x2
# 6 robôs -> grade 2x3
# 9 robôs -> grade 3x3

# Espaçamento configurável
sim = MultiRobotSimHelper(
    num_robots=9,
    robot_spacing=2.0  # 2 metros entre robôs
)
```

## 📊 Estrutura de Dados

### Observações
```python
observations = sim.get_observations()
# Shape: (num_robots, obs_dim)
# obs_dim = 12 (joint_pos) + 12 (joint_vel) + 3 (base_pos) + 6 (base_vel) = 33

for i, obs in enumerate(observations):
    joint_positions = obs[0:12]    # Posições das juntas
    joint_velocities = obs[12:24]  # Velocidades das juntas
    base_position = obs[24:27]     # Posição da base (x, y, z)
    base_velocity = obs[27:33]     # Velocidade da base (linear + angular)
```

### Recompensas
```python
rewards = sim.calculate_rewards()
# Shape: (num_robots,)
# Cada elemento é a recompensa individual de um robô
```

### Terminações
```python
terminations = sim.check_terminations()
# Shape: (num_robots,) dtype=bool
# True = robô deve ser resetado, False = continua episódio
```

## ⚙️ Configuração Detalhada

### RigidBodyProperties
```python
RigidBodyProperties(
    disable_gravity=False,          # Desabilitar gravidade
    retain_accelerations=False,     # Manter acelerações
    linear_damping=0.0,            # Damping linear
    angular_damping=0.0,           # Damping angular
    max_linear_velocity=1000.0,    # Velocidade linear máxima
    max_angular_velocity=1000.0,   # Velocidade angular máxima
    max_depenetration_velocity=1.0 # Velocidade máxima de separação
)
```

### ArticulationRootProperties
```python
ArticulationRootProperties(
    enabled_self_collisions=False,         # Auto-colisões
    solver_position_iteration_count=4,     # Iterações do solver de posição
    solver_velocity_iteration_count=0      # Iterações do solver de velocidade
)
```

### InitialState
```python
InitialState(
    pos=(0.0, 0.0, 0.4),          # Posição inicial (x, y, z)
    joint_positions={              # Posições iniciais das juntas
        ".*L_hip_joint": 0.1,      # Regex: qualquer junta com "L_hip_joint"
        ".*R_hip_joint": -0.1,     # Regex: qualquer junta com "R_hip_joint"
        "F[L,R]_thigh_joint": 0.8, # Regex: FL_thigh ou FR_thigh
        ".*_calf_joint": -1.5,     # Regex: qualquer calf joint
    },
    joint_velocities={".*": 0.0}   # Velocidades iniciais (todas zero)
)
```

### DCMotorConfig
```python
DCMotorConfig(
    joint_names_expr=[             # Expressões para nomes das juntas
        ".*_hip_joint", 
        ".*_thigh_joint", 
        ".*_calf_joint"
    ],
    effort_limit=23.5,             # Limite de força (Nm)
    saturation_effort=23.5,        # Força de saturação (Nm)
    velocity_limit=30.0,           # Velocidade máxima (rad/s)
    stiffness=25.0,                # Rigidez do PD controller
    damping=0.5,                   # Damping do PD controller
    friction=0.0                   # Fricção das juntas
)
```

## 🐛 Solução de Problemas

### Problema: Robôs não aparecem
```python
# Verificar caminho do USD
config = Go2Config()
print(f"Caminho USD: {config.usd_path}")

# Verificar se o asset existe
import os
if not os.path.exists(config.usd_path):
    print("ERRO: Arquivo USD não encontrado!")
```

### Problema: Robôs se comportam estranhamente
```python
# Verificar limites das juntas
robot_info = sim.get_robot_info(0)
print("Limites das juntas:")
print(f"Lower: {robot_info['lower_limits']}")
print(f"Upper: {robot_info['upper_limits']}")

# Verificar ações aplicadas
actions = np.random.uniform(-1, 1, (sim.num_robots, 12))
print(f"Actions shape: {actions.shape}")
print(f"Actions range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
```

### Problema: Performance baixa
```python
# Desabilitar renderização para simulação mais rápida
sim.step_simulation(render=False)

# Reduzir número de iterações do solver
config = Go2Config(
    articulation_props=ArticulationRootProperties(
        solver_position_iteration_count=2,  # Padrão: 4
        solver_velocity_iteration_count=0
    )
)
```

## 📝 Exemplos Avançados

Execute os scripts de exemplo:

```bash
# Exemplo básico com 4 robôs
python exemplo_multi_robots.py

# Script de migração
python migrar_para_multi_robots.py
```

## 🤝 Contribuição

Para adicionar novas funcionalidades:

1. Estenda a classe `MultiRobotSimHelper`
2. Mantenha compatibilidade com a interface existente
3. Adicione testes nos scripts de exemplo
4. Documente mudanças neste README

## 📄 Licença

Mesmo que o projeto Isaac Sim principal.

---

**Nota**: Esta implementação usa apenas APIs nativas do Isaac Sim, não requer Isaac Lab. Para funcionalidades ainda mais avançadas, considere migrar para Isaac Lab completo. 