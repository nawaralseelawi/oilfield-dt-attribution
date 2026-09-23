#!/bin/bash
# Full reproduction (about 40 min on 1 CPU). Run in the foreground or under a process manager that
# survives your session: in our sandbox, background jobs were killed when the session was idle.
set -e
export PYTHONPATH=$(pwd)
mkdir -p results figures
python3 -W ignore run_main.py 1 --aux
python3 -W ignore run_main.py 2
python3 -W ignore run_main.py 3
python3 -W ignore analysis.py
python3 -W ignore paired_ablation.py
python3 -W ignore breakeven.py
python3 -W ignore make_figures.py
node build_manuscript.js      # optional; needs: npm install docx ; writes the .docx to the working directory
