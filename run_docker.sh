#!/bin/bash

echo "Iniciando Isaac Sim com docker compose..."

# Verifica parâmetros
BUILD_FLAG=""
if [[ "$1" == "--build" ]]; then
    BUILD_FLAG="--build"
    echo "Modo build habilitado - imagem será reconstruída"
else
    echo "Usando imagem existente (use --build para reconstruir)"
fi

# Habilita acesso ao X11 para GUI
xhost +local:

# Verifica se docker compose está disponível
if ! docker compose version &> /dev/null; then
    echo "docker compose não encontrado. Tentando usar 'docker-compose'..."
    if ! command -v docker-compose &> /dev/null; then
        echo "Nem 'docker compose' nem 'docker-compose' estão disponíveis."
        echo "Por favor, instale Docker Compose."
        exit 1
    fi
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo "Usando comando: $COMPOSE_CMD"

# Para qualquer container anterior se estiver rodando
echo "Parando containers anteriores..."
$COMPOSE_CMD down

# Builda e roda o container em background
if [[ -n "$BUILD_FLAG" ]]; then
    echo "Buildando imagem e iniciando container..."
    $COMPOSE_CMD up -d --build
else
    echo "Iniciando container..."
    $COMPOSE_CMD up -d
fi

# Verifica se o container está rodando
if [ $? -eq 0 ]; then
    echo "Container iniciado com sucesso!"
    echo ""
    echo "Entrando no container..."
    echo "Para sair do container, digite 'exit'"
    echo "Para parar o container completamente, use './stop.sh'"
    echo ""
    
    # Entra automaticamente no container
    $COMPOSE_CMD exec isaac-sim bash
else
    echo "Falha ao iniciar o container"
    exit 1
fi