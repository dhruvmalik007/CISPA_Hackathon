echo "=== MY JOBS ==="
squeue -u malik4 -o "%.10i %.18j %.10T %.8M %D %b" 2>&1
echo "=== MY JOB HISTORY ==="
sacct -u malik4 --format=JobID,JobName,State,ExitCode,Elapsed -P 2>&1 | tail -6
echo "=== RUNNING NODES PER USER ==="
squeue -p dc-gpu -t RUNNING -h -o "%u|%D|%M" | awk -F'|' '{n[$1]+=$2; c[$1]++} END {for (u in n) printf "%-16s jobs=%-3d nodes=%d\n", u, c[u], n[u]}' | sort -t= -k3 -rn | head -12
echo "=== LONGEST RUNNING ==="
squeue -p dc-gpu -t RUNNING -h -o "%u|%i|%M|%D" | sort -t'|' -k3 -r | head -8
echo "=== MY PENDING REASON ==="
squeue -u malik4 -h -o "%.12i|%.10T|%r|%S"
echo "=== DC-GPU AVAIL ==="
sinfo -p dc-gpu -o "%20P %6a %6D %10t" | head -6
