B=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning
echo "=== runs dir ==="
ls -la $B/runs/ 2>&1 | head -30
echo "=== npo_400* dirs ==="
ls -d $B/runs/npo_400* 2>&1
echo "=== npo run contents ==="
for d in $B/runs/npo_400*; do echo "--- $d ---"; ls -la "$d" 2>&1 | head -20; done
echo "=== any npo python files ==="
ls -la $B/*.py 2>&1 | head -30
echo "=== grep for proxyKL definition ==="
grep -rln "proxyKL\|proxy_kl" $B --include=*.py 2>/dev/null | head -10
