#!/bin/bash

echo "Parando Isaac Sim containers..."

# Verifica se docker compose está disponível
if ! docker compose version &> /dev/null; then
    echo "docker compose não encontrado. Tentando usar 'docker compose'..."
    if ! command -v docker compose &> /dev/null; then
        echo "Nem 'docker compose' nem 'docker compose' estão disponíveis."
        echo "Por favor, instale Docker Compose."
        exit 1
    fi
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker compose"
fi

echo "Usando comando: $COMPOSE_CMD"

# Para todos os containers do compose
$COMPOSE_CMD down

if [ $? -eq 0 ]; then
    echo "Containers parados com sucesso!"
    
    # Opcional: remover imagens órfãs e limpar cache
    read -p "Deseja limpar imagens e containers não utilizados? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Limpando recursos não utilizados..."
        docker system prune -f
        echo "Limpeza concluída!"
    fi
else
    echo "Erro ao parar containers"
    exit 1
fi

echo "Removendo acesso X11..."
xhost -local: 