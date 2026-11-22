cd /p/scratch/training2666/malik4/unlearning
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
$VENV/bin/python -m py_compile unlearning/salun.py && echo "SYNTAX OK"
$VENV/bin/python -c "import sys; sys.path.insert(0,'.'); import unlearning.salun as s; print('IMPORT OK'); print('objectives:', s.DEFAULTS['forget_objective'], '| projection:', s.DEFAULTS['projection'])"
echo "--- sweep script present at root: ---"
ls -la jureca_salun_sweep.sh
echo "--- smoke job ---"
squeue -u malik4 -o "%.10i %.10T %.6M %.20S"
