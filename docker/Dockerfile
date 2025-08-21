FROM nvcr.io/nvidia/isaac-sim:4.5.0

SHELL ["/bin/bash", "-lc"]
WORKDIR /isaac-sim/ambiente_rl_isaacsim

# Copia os arquivos do projeto
COPY . .

# Cria venv para CleanRL e instala dependências
RUN /isaac-sim/python.sh -m venv --system-site-packages /isaac-sim/venv-cleanrl && \
    source /isaac-sim/venv-cleanrl/bin/activate && \
    pip install --no-deps cleanrl && \
    pip install --constraint isaac-constraints.txt -r cleanrl-requirements.txt

# Variáveis de ambiente para usar sempre o venv
ENV VIRTUAL_ENV=/isaac-sim/venv-cleanrl
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copia script auxiliar
COPY python-cleanrl.sh /usr/local/bin/python-cleanrl
RUN chmod +x /usr/local/bin/python-cleanrl

RUN echo "source /isaac-sim/venv-cleanrl/bin/activate" >> /root/.bashrc