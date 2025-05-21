#!/bin/bash

module load miniconda3/22.11.1-1
module load gcc/12.3.0
module load qmio-run/0.5.1-python-3.11.9
module load qmio-tools/0.2.0-python-3.11.9

jupyter-lab --no-browser --ip=$(hostname -i) --log-level='WARN' --notebook-dir='.'
