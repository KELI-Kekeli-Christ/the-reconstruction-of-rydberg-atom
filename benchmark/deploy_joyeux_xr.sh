#!/usr/bin/env bash
# Déploie + lance export x/rij (n=4,10) SANS tuer le job n4_50 en cours
set -euo pipefail

REMOTE_USER="${REMOTE_USER:?Set REMOTE_USER to your Joyeux login}"
REMOTE_HOST="${REMOTE_HOST:-joyeux.univ-avignon.fr}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="$(ssh "$REMOTE" 'echo $HOME/stage_LMA_2026/benchmark_Lalpha')"

echo "==> Sync vers $REMOTE:$REMOTE_DIR"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/test_real' '$REMOTE_DIR/logs' '$REMOTE_DIR/results'"

rsync -avz --progress \
  "$LOCAL_ROOT/formulations.jl" \
  "$LOCAL_ROOT/utils.jl" \
  "$REMOTE:$REMOTE_DIR/"

rsync -avz --progress \
  "$LOCAL_ROOT/test_real/utils.jl" \
  "$LOCAL_ROOT/test_real/formulations.jl" \
  "$LOCAL_ROOT/test_real/formulations_Lvar.jl" \
  "$LOCAL_ROOT/test_real/config_L_alpha.toml" \
  "$LOCAL_ROOT/test_real/load_config_L_alpha.jl" \
  "$LOCAL_ROOT/test_real/exp_L_alpha_xr.jl" \
  "$REMOTE:$REMOTE_DIR/test_real/"

echo "==> Lance nohup export x/rij (PID séparé)"
ssh "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
export PATH="\$HOME/bin:\$PATH"
REMOTE_DIR='$REMOTE_DIR'
cd "\$REMOTE_DIR"
mkdir -p logs results

export OUT_DIR="\$REMOTE_DIR/results"
export N_VALUES="4,10"
export N_INSTANCES="5"
export ALPHAS="1e-6,1e-4,1e-2,1.0"
export L_INIT="10.0"
export FORMS="Standard,Simple,Simple_square,Taylor,Taylor_square"
export OUT_RUNS_NAME="exp_L_alpha_xr_runs.csv"
export OUT_PAIRS_NAME="exp_L_alpha_xr_pairs.csv"
export OUT_LOG_NAME="exp_L_alpha_xr.log"

LOG="\$REMOTE_DIR/logs/exp_L_alpha_xr_\$(date +%Y%m%d_%H%M%S).out"
PIDFILE="\$REMOTE_DIR/logs/exp_L_alpha_xr.pid"

# Stop uniquement l'ancien job XR (pas le n4_50)
if [[ -f "\$PIDFILE" ]]; then
  oldpid=\$(cat "\$PIDFILE" || true)
  if [[ -n "\${oldpid:-}" ]] && kill -0 "\$oldpid" 2>/dev/null; then
    echo "Stop ancien XR PID=\$oldpid"
    kill "\$oldpid" || true
    sleep 2
  fi
fi

echo "Lancement nohup → \$LOG"
nohup env \\
  OUT_DIR="\$OUT_DIR" \\
  OUT_RUNS_NAME="\$OUT_RUNS_NAME" \\
  OUT_PAIRS_NAME="\$OUT_PAIRS_NAME" \\
  OUT_LOG_NAME="\$OUT_LOG_NAME" \\
  N_VALUES="\$N_VALUES" \\
  N_INSTANCES="\$N_INSTANCES" \\
  ALPHAS="\$ALPHAS" \\
  L_INIT="\$L_INIT" \\
  FORMS="\$FORMS" \\
  julia test_real/exp_L_alpha_xr.jl > "\$LOG" 2>&1 &
echo \$! > "\$PIDFILE"
echo "PID=\$(cat "\$PIDFILE")"
echo "CSV runs  : \$REMOTE_DIR/results/\$OUT_RUNS_NAME"
echo "CSV pairs : \$REMOTE_DIR/results/\$OUT_PAIRS_NAME"
sleep 8
head -n 40 "\$LOG" || true
REMOTE_SCRIPT

echo "==> Suivi :"
echo "  ssh $REMOTE 'tail -f $REMOTE_DIR/logs/exp_L_alpha_xr_*.out'"
echo "  scp $REMOTE:$REMOTE_DIR/results/exp_L_alpha_xr_runs.csv ."
echo "  scp $REMOTE:$REMOTE_DIR/results/exp_L_alpha_xr_pairs.csv ."
