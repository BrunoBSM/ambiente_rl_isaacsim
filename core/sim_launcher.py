"""
Launcher para inicialização do Isaac Sim.

Este módulo inicializa o SimulationApp do Isaac Sim antes que outros módulos
sejam importados. É essencial que este import seja feito primeiro para garantir
que o ambiente de simulação esteja configurado corretamente.

Attributes:
    simulation_app (SimulationApp): Instância global do aplicativo de simulação
                                   configurado para execução com interface gráfica.

Note:
    - O parâmetro "headless": False permite visualização da simulação
    - Este módulo deve ser importado ANTES de qualquer outro módulo do Isaac Sim
    - A instância simulation_app é compartilhada globalmente para controle da simulação
    - WebRTC pode ser habilitado via variável de ambiente ISAAC_WEBRTC=1

Warning:
    Não modifique a ordem de importação. O SimulationApp deve ser inicializado
    antes de qualquer importação de módulos do Isaac Sim ou Isaac Lab.
"""

import logging
import os

# Configurar logging ANTES do SimulationApp
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

from isaacsim import SimulationApp

# Configurações opcionais - lidas de variáveis de ambiente
ENABLE_WEBRTC = os.getenv("ISAAC_WEBRTC", "0").lower() in ("1", "true", "yes")
ENABLE_MOUSE_DRAW = True  # Mude para False se não quiser desenhar o mouse

# Log da configuração WebRTC
if ENABLE_WEBRTC:
    logging.info("WebRTC será habilitado (ISAAC_WEBRTC=1)")
else:
    logging.info("WebRTC desabilitado (ISAAC_WEBRTC=0 ou não definido)")

# Configuração do Isaac Sim
if ENABLE_WEBRTC:
    CONFIG = {
        "width": 1280,
        "height": 720,
        "window_width": 1920,
        "window_height": 1080,
        "headless": True,
        "hide_ui": False,
        "renderer": "RaytracedLighting",
        "display_options": 3286,
    }
    # Inicia o Simulation App com configuração completa
    simulation_app = SimulationApp(launch_config=CONFIG)
    
    # Habilita extensões opcionais
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.livestream.webrtc")
    logging.info("WebRTC habilitado")
    
    if ENABLE_MOUSE_DRAW:
        simulation_app.set_setting("/app/window/drawMouse", True)
        logging.info("Mouse drawing habilitado")
else:
    # Configuração simples quando WebRTC está desabilitado
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1280,
        "height": 720,
        "window_width": 1280,
        "window_height": 720,
        "renderer": "RaytracedLighting"
    })
    logging.info("Simulação iniciada com configuração visual (janela visível)")