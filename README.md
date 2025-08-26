# Ambiente RL Isaac Sim - Go2 Multi-Robot

Sistema de treinamento de Reinforcement Learning para robôs Go2 usando Isaac Sim e CleanRL.

## Estrutura do Ambiente

```
ambiente_rl_isaacsim/
├── cleanrl_isaacsim/           # Algoritmos RL customizados
│   ├── algorithms/             # Implementações de algoritmos
│   │   ├── ppo_isaacsim.py    # PPO para Isaac Sim
│   │   └── utils.py           # Utilitários
│   ├── envs/                  # Ambientes customizados
│   │   ├── go2_env.py         # Ambiente Go2
│   │   ├── multi_env_wrapper.py # Wrapper multi-ambiente
│   │   └── wrappers.py        # Wrappers adicionais
│   └── utils/                 # Utilitários gerais
│       ├── evaluation.py     # Avaliação de modelos
│       └── wandb_utils.py     # Integração WandB
├── core/                      # Componentes principais
│   ├── isaac_gym_multi_env.py # Ambiente multi-robô
│   ├── multi_sim_helper.py    # Helper para simulação
│   └── sim_launcher.py        # Launcher da simulação
├── configs/                   # Configurações de treinamento
│   └── default_ppo.py        # Configurações PPO
├── scripts/                   # Scripts de execução
│   ├── train_ppo.py          # Script principal de treinamento
│   ├── eval_model.py         # Avaliação de modelos
│   └── sweep_hyperparams.py  # Sweep de hiperparâmetros
├── experiments/               # Resultados de experimentos
├── models/                    # Modelos treinados
├── docker/                    # Configuração Docker
│   └── Dockerfile            # Imagem Isaac Sim + CleanRL
├── docker-compose.yml         # Orquestração Docker
├── run_docker.sh             # Script para iniciar container
├── python-cleanrl.sh         # Script para rodar script python
└── stop.sh                   # Script para parar container
```

## Instalação e Utilização com Docker

### Pré-requisitos

- Docker com suporte a GPU (nvidia-docker)
- Docker Compose
- Sistema X11 para GUI (Linux)

### Configuração WandB (Opcional)

Para usar tracking WandB sem precisar passar as credenciais toda vez, configure o arquivo `config.env`:

1. **Criar arquivo de configuração na raiz do repositório**:
```bash
touch config.env
```

2. **Editar arquivo `config.env`** com suas credenciais WandB:
```bash
# Configurações WandB para Docker Compose
# Copie este arquivo para config.env e configure suas variáveis

# API Key do WandB (obtenha em https://wandb.ai/settings)
WANDB_API_KEY=your_wandb_api_key_here

# Entidade/time WandB (seu usuário ou organização)
WANDB_ENTITY=your_username_or_team

# Projeto padrão WandB (pode ser sobrescrito via --wandb-project-name)
WANDB_PROJECT=your-project-name

```

> **📝 Nota**: Obtenha sua API key em [wandb.ai/settings](https://wandb.ai/settings)  
> **🔒 Segurança**: O arquivo `config.env` deve estar no `.gitignore` para não expor credenciais

### Início Rápido

1. **Iniciar ambiente (primeira vez - build automático)**:
```bash
./run_docker.sh --build
```

2. **Iniciar ambiente (execuções subsequentes)**:
```bash
./run_docker.sh
```

3. **Parar ambiente**:
```bash
./stop.sh
```

O script `run_docker.sh` automaticamente:
- Builda a imagem Isaac Sim + CleanRL (se `--build` especificado)
- Inicia o container em background
- Entra automaticamente no container
- Carrega variáveis WandB do arquivo `config.env` (se existir)

### Estrutura do Container

- **Workspace**: `/isaac-sim/ambiente_rl_isaacsim`
- **Python**: Ambiente virtual em `/isaac-sim/venv-cleanrl`
- **Imagem**: `ambiente-rl-isaacsim:4.5.0`

## Treinamento com Exemplos

### Flags Principais

#### Controle de Experimento
- `--exp-name`: Nome do experimento
- `--seed`: Seed para reprodutibilidade
- `--total-timesteps`: Total de passos de treinamento

#### Ambiente
- `--num-envs`: Número de robôs em paralelo
- `--spacing`: Espaçamento entre robôs (metros)
- `--use-relative-control`: Usar controle relativo
- `--relative-scale`: Escala para controle relativo
- `--safety-margin`: Margem de segurança para limites

#### Algoritmo PPO
- `--learning-rate`: Taxa de aprendizado
- `--num-steps`: Passos por rollout
- `--gamma`: Fator de desconto
- `--gae-lambda`: Lambda para GAE
- `--clip-coef`: Coeficiente de clipping
- `--ent-coef`: Coeficiente de entropia
- `--vf-coef`: Coeficiente da função valor
- `--num-minibatches`: Número de mini-batches
- `--update-epochs`: Épocas de atualização

#### WandB Tracking
- `--track`: Habilitar tracking WandB
- `--wandb-project-name`: Nome do projeto WandB (padrão: `WANDB_PROJECT` env var)
- `--wandb-entity`: Entidade/time WandB (padrão: `WANDB_ENTITY` env var)

#### Visualização
- `--webrtc`: Habilitar WebRTC para visualização remota

### Exemplos de Uso

#### 1. Teste Rápido (desenvolvimento)
```bash
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 4 \
    --total-timesteps 100000 \
    --num-steps 8 \
    --exp-name "quick_test"
```

#### 2. Treinamento Básico com WandB
```bash
# Com variáveis configuradas no config.env (recomendado)
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 16 \
    --total-timesteps 10000000 \
    --track \
    --exp-name "baseline_training"

# Ou especificando manualmente (sobrescreve config.env)
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 16 \
    --total-timesteps 10000000 \
    --track \
    --exp-name "baseline_training"
```

#### 3. Treinamento Completo Otimizado
```bash
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 32 \
    --total-timesteps 50000000 \
    --learning-rate 0.0026 \
    --num-steps 32 \
    --spacing 2.0 \
    --track \
    --exp-name "production_v1" \
    --anneal-lr
```

#### 4. Experimento com Controle Relativo
```bash
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 16 \
    --use-relative-control \
    --relative-scale 0.05 \
    --learning-rate 0.002 \
    --track \
    --exp-name "relative_control_test"
```

#### 5. Treinamento Multi-Robô Massivo
```bash
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 64 \
    --total-timesteps 100000000 \
    --spacing 1.5 \
    --num-steps 64 \
    --learning-rate 0.001 \
    --track \
    --exp-name "massive_parallel"
```

#### 6. Sweep de Hiperparâmetros
```bash
./python-cleanrl.sh scripts/sweep_hyperparams.py \
    --project "go2-hyperparams" \
    --entity "seu-time" \
    --count 20 \
    --base-num-envs 8 \
    --base-total-timesteps 1000000
```

#### 7. Treinamento com WebRTC (Visualização Remota)
```bash
./python-cleanrl.sh scripts/train_ppo.py \
    --num-envs 16 \
    --webrtc \
    --track \
    --exp-name "remote_visualization"
```

### Configurações WandB Avançadas

#### Tracking Detalhado
```bash
./python-cleanrl.sh scripts/train_ppo.py \
    --track \
    --capture-video \
    --exp-name "detailed_analysis" \
    --num-envs 16
```

#### Múltiplos Experimentos com Seeds
```bash
# Seed 1
./python-cleanrl.sh scripts/train_ppo.py --track --seed 1 --exp-name "multi_seed_1"

# Seed 42  
./python-cleanrl.sh scripts/train_ppo.py --track --seed 42 --exp-name "multi_seed_42"

# Seed 123
./python-cleanrl.sh scripts/train_ppo.py --track --seed 123 --exp-name "multi_seed_123"
```

### Monitoramento

- **WandB Dashboard**: `https://wandb.ai/[entity]/[project]`
- **TensorBoard**: `tensorboard --logdir experiments/runs`
- **Logs do Container**: `docker compose logs -f isaac-sim`

> **⚠️ Nota sobre WandB**: 
> - **Recomendado**: Configure suas credenciais no arquivo `config.env` (veja seção "Configuração WandB")
> - **Alternativa**: Na primeira execução com `--track`, o WandB solicitará sua API key. Obtenha em [wandb.ai/settings](https://wandb.ai/settings)
> - Se não configurar `WANDB_ENTITY`, use `--wandb-entity seu_usuario` no comando

### Avaliação de Modelos

```bash
./python-cleanrl.sh scripts/eval_model.py \
    --model-path "models/ppo_model.pt" \
    --num-envs 4 \
    --num-episodes 10
``` 