# 🤖 CleanRL + IsaacSim Multi-Robot RL

Integração do CleanRL com IsaacSim para treinamento de reinforcement learning com múltiplos robôs GO2 em paralelo.

## 📋 Visão Geral

Este projeto implementa uma pipeline completa de treinamento de RL usando:
- **CleanRL**: Framework de RL limpo e bem documentado
- **IsaacSim**: Simulação física de alta fidelidade da NVIDIA
- **Multi-Robot**: Treinamento paralelo com múltiplos robôs GO2
- **WandB**: Tracking e monitoramento de experimentos

## 🏗️ Estrutura do Projeto

```
multi_ambiente_rl/
├── cleanrl_isaacsim/           # 📦 Pacote principal CleanRL+IsaacSim
│   ├── envs/                   # 🌍 Ambientes e wrappers
│   │   ├── multi_env_wrapper.py    # Wrapper principal (adapta ambiente existente)
│   │   ├── go2_env.py              # Ambiente single-robot (placeholder)
│   │   └── wrappers.py             # Wrappers adicionais
│   ├── algorithms/             # 🧠 Algoritmos de RL
│   │   ├── ppo_isaacsim.py         # PPO adaptado para IsaacSim
│   │   └── utils.py                # Utilitários para algoritmos
│   └── utils/                  # 🔧 Utilitários gerais
│       ├── wandb_utils.py          # Integração WandB
│       └── evaluation.py           # Avaliação de modelos
├── core/                       # 🎯 Código core do IsaacSim
│   ├── isaac_gym_multi_env.py      # Ambiente multi-robô original
│   ├── multi_sim_helper.py         # Helper de simulação
│   └── sim_launcher.py             # Launcher do IsaacSim
├── scripts/                    # 🚀 Scripts de execução
│   ├── train_ppo.py                # Script principal de treinamento
│   ├── eval_model.py               # Avaliação de modelos
│   └── sweep_hyperparams.py        # Busca de hiperparâmetros
├── configs/                    # ⚙️ Configurações
│   └── default_ppo.py              # Configurações padrão PPO
├── experiments/                # 📊 Logs e resultados
├── models/                     # 💾 Modelos salvos
└── requirements.txt            # 📋 Dependências
```

## 🚀 Instalação

### 1. Pré-requisitos

- Isaac Sim 4.5.0+ instalado
- CUDA 11.8+ (para GPU)
- Python 3.8+ (use o Python do Isaac Sim)

### 2. Dependências

```bash
# No ambiente do Isaac Sim
cd multi_ambiente_rl
pip install -r requirements.txt
```

### 3. Verificar Instalação

```bash
# Teste rápido do ambiente
/isaac-sim/python.sh -c "from cleanrl_isaacsim.envs.multi_env_wrapper import make_env; print('✅ Installation OK')"
```

## 🎯 Uso Rápido

### Treinamento Básico

```bash
# Treinamento rápido (sem tracking)
/isaac-sim/python.sh scripts/train_ppo.py --num-envs 4 --total-timesteps 100000

# Treinamento com WandB tracking
/isaac-sim/python.sh scripts/train_ppo.py \
    --num-envs 16 \
    --total-timesteps 10000000 \
    --track

# Treinamento com WandB e nome customizado
/isaac-sim/python.sh scripts/train_ppo.py \
    --num-envs 16 \
    --exp-name "meu_experimento" \
    --track
```

### Avaliação de Modelo

```bash
# Avaliar modelo treinado
/isaac-sim/python.sh scripts/eval_model.py \
    --model-path models/ppo_model.pt \
    --num-episodes 20 \
    --render
```

### Busca de Hiperparâmetros

```bash
# Sweep rápido
/isaac-sim/python.sh scripts/sweep_hyperparams.py --quick

# Sweep completo no projeto isaacsim
/isaac-sim/python.sh scripts/sweep_hyperparams.py \
    --project "isaacsim" \
    --count 50
```

## 📊 Configurações Predefinidas

### Teste Rápido (Desenvolvimento)

```bash
/isaac-sim/python.sh scripts/train_ppo.py \
    --num-envs 4 \
    --total-timesteps 100000 \
    --num-steps 8
```

### Treinamento Completo com WandB

```bash
/isaac-sim/python.sh scripts/train_ppo.py \
    --num-envs 32 \
    --total-timesteps 50000000 \
    --num-steps 32 \
    --track \
    --anneal-lr
```

### Multi-Robot (64 robôs) com Tracking

```bash
/isaac-sim/python.sh scripts/train_ppo.py \
    --num-envs 64 \
    --spacing 2.0 \
    --learning-rate 0.001 \
    --track
```

## 🔧 Customização

### Novo Ambiente

1. Criar wrapper em `cleanrl_isaacsim/envs/`
2. Registrar em `cleanrl_isaacsim/envs/__init__.py`
3. Usar em scripts com `--env-id`

### Novo Algoritmo

1. Criar em `cleanrl_isaacsim/algorithms/`
2. Seguir padrão do `ppo_isaacsim.py`
3. Implementar função `train(args)`

### Métricas Customizadas

```python
# Em cleanrl_isaacsim/utils/wandb_utils.py
def log_custom_metrics(step, custom_data):
    wandb.log({
        "custom/metric": custom_data,
        "custom/robot_speed": robot_speed,
    }, step=step)
```

## 📈 Monitoramento

### WandB Dashboard

Acompanhe métricas em tempo real:
- Recompensas por episódio
- Performance por robô
- Métricas de treinamento (loss, KL divergence)
- FPS de simulação
- Vídeos dos robôs (opcional)

### TensorBoard (Local)

```bash
tensorboard --logdir experiments/runs/
```

## 🐛 Debugging

### Problemas Comuns

1. **Erro de Import**: Verificar se está usando Python do Isaac Sim
2. **GPU Out of Memory**: Reduzir `--num-envs`
3. **Simulação Lenta**: Verificar drivers NVIDIA
4. **WandB Login**: `wandb login` no ambiente do Isaac Sim

### Logs Detalhados

```bash
# Habilitar logs detalhados
export ISAAC_SIM_LOG_LEVEL=DEBUG
/isaac-sim/python.sh scripts/train_ppo.py --verbose
```

## 🔬 Experimentos Avançados

### Curriculum Learning

```bash
# Começar com poucos robôs e aumentar gradualmente
/isaac-sim/python.sh scripts/train_ppo.py \
    --num-envs 4 \
    --curriculum \
    --max-envs 32
```

### Transfer Learning

```bash
# Treinar em ambiente simples, transferir para complexo
/isaac-sim/python.sh scripts/train_ppo.py \
    --load-model models/simple_env.pt \
    --env-complexity high
```

### Multi-Task Learning

```bash
# Treinar múltiplas tarefas simultaneamente
/isaac-sim/python.sh scripts/train_ppo.py \
    --tasks "walk,turn,jump" \
    --task-weights "0.5,0.3,0.2"
```

## 📚 Recursos

- [CleanRL Documentation](https://docs.cleanrl.dev/)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [WandB Guides](https://docs.wandb.ai/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 🤝 Contribuição

1. Fork o projeto
2. Crie feature branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para branch (`git push origin feature/nova-funcionalidade`)
5. Abra Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja `LICENSE` para detalhes.

## 🙏 Agradecimentos

- **CleanRL Team**: Framework de RL excepcional
- **NVIDIA Isaac Sim**: Simulação física de alta qualidade
- **WandB**: Platform de tracking de experimentos
- **Unitree**: Robô GO2 de referência

---

**Developed with ❤️ for Multi-Robot Reinforcement Learning** 