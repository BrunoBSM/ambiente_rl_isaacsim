#!/bin/bash
VENV_SITE=/isaac-sim/venv-cleanrl/lib/python3.10/site-packages
PYTHONPATH=$VENV_SITE:$PYTHONPATH /isaac-sim/python.sh "$@"
