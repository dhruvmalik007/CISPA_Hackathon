# Unlearning challenge notes
- Rules/metric: Unlearning.pdf (only model.pt + forget_images.npz may be used; no external data). doing.pdf = chat log of past experiments (image-only PDF; render with PyMuPDF to read).
- Never request GPUs yourself. The user supplies an A100 job id; run steps with
  `srun --jobid=<ID> --overlap bash -c "cd $PWD && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=<g> python -u <script> ..."` (the `--pty` shell does not work from the agent).
  The local RTX 8000s (GPU0) are for small diagnostics only. The user submits to the leaderboard; the agent never runs submission.py.
- Disk quota is tight (a run crashed with "Disk quota exceeded"); checkpoints are 328 MB, so save few.
- One script per method, importing common.py + task_template.py.
- Best public score 0.1956: runs/soup_4way_tb10_s42.pt (local: exact acc 0.49 CE 2.26/2.84, altered "mix" acc 0.32 CE 4.61/4.80; orig on mix 0.885 / 0.61).
- Known bad: 9-class head-bias drop (0.1249), soup_5way_plus_ca_cons900 (0.1616), classaware step0600 (0.1825), hybrid step0600 (<0.1956).
- task_soup_kd.py: forward-KL(T=2) distillation of the soup's exact-image outputs onto exact + altered copies, mosaic retain anchor (runs/skd_*).
- task_soup_kd.py result: runs/skd_h00/step0300.pt scored 0.1681 (worse than the 0.1956 soup). Pulling altered copies toward exact-image outputs / lowering forget CE hurts. Leaderboard allows 1 submission per 10 min; train one model at a time.
- runs/soup_anti_skd05.pt = 1.5*soup - 0.5*skd_h00 (extrapolation away from the bad direction), unscored.
