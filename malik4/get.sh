B=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning
echo "########## common.py ##########"
cat $B/common.py
echo "########## task_npo.py (first 120) ##########"
sed -n '1,120p' $B/naninithin/task_npo.py
