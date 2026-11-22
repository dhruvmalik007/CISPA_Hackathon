cd /p/scratch/training2666/malik4/unlearning
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
$VENV/bin/python -m py_compile unlearning/salun.py && echo "SYNTAX OK on cluster"
$VENV/bin/python -c "import sys; sys.path.insert(0,'.'); import unlearning.salun as s; print('IMPORT OK'); print('count_low/high:', s.COUNT_LOW, s.COUNT_HIGH)"
grep -n "COUNT_LOW\|kept_untrained_baseline" unlearning/salun.py | head -5
