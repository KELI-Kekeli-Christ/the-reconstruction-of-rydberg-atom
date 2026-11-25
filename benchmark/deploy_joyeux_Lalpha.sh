#!/usr/bin/env bash
# Déploiement + lancement nohup sur Joyeux
set -euo pipefail

REMOTE_USER="${REMOTE_USER:?Set REMOTE_USER to your Joyeux login}"
REMOTE_HOST="${REMOTE_HOST:-joyeux.univ-avignon.fr}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Résoudre le chemin absolu côté distant (évite ~ et $HOME non expansés)
REMOTE_DIR="$(ssh "$REMOTE" 'echo $HOME/stage_LMA_2026/benchmark_Lalpha')"

echo "==> Sync vers $REMOTE:$REMOTE_DIR"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/test_real' '$REMOTE_DIR/pasqal' '$REMOTE_DIR/logs' '$REMOTE_DIR/results'"

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
  "$LOCAL_ROOT/test_real/exp_L_alpha.jl" \
  "$LOCAL_ROOT/test_real/exp_L_alpha_full.jl" \
  "$REMOTE:$REMOTE_DIR/test_real/"

rsync -avz --progress \
  "$LOCAL_ROOT/pasqal/" \
  "$REMOTE:$REMOTE_DIR/pasqal/"

echo "==> Lance nohup (packages déjà installés)"
ssh "$REMOTE" bash -s <<REMOTE_SCRIPT
set -euo pipefail
export PATH="\$HOME/bin:\$PATH"
REMOTE_DIR='$REMOTE_DIR'
cd "\$REMOTE_DIR"
mkdir -p logs results

julia --version

export PASQAL_ROOT="\$REMOTE_DIR/pasqal"
export OUT_DIR="\$REMOTE_DIR/results"
# n étendu pour Exp1/Exp2 (Exp3 Pasqal : n fixé par les GML)
export N_VALUES="4,10,20,30,40,50"
export N_INSTANCES="5"
export ALPHAS="1e-6,1e-4,1e-2,1.0,5.0,10.0"
export L_INIT="10.0"
export L_CHOSEN="10.0"
export RUN_EXP1="1"
export RUN_EXP2="1"
# Pasqal déjà calculé (n=25 fixe) — on ne le relance pas
export RUN_EXP3="0"
export FORMS="Standard,Simple,Simple_square,Taylor,Taylor_square"
export OUT_CSV_NAME="exp_L_alpha_full_results_n4_50.csv"
export OUT_LOG_NAME="exp_L_alpha_full_n4_50.log"

# Sauvegarde de l'ancien CSV complet (avec Pasqal)
mkdir -p "\$REMOTE_DIR/results/backup"
if [[ -f "\$REMOTE_DIR/results/exp_L_alpha_full_results.csv" ]]; then
  cp -n "\$REMOTE_DIR/results/exp_L_alpha_full_results.csv" \\
    "\$REMOTE_DIR/results/backup/exp_L_alpha_full_results_n4_10.csv" 2>/dev/null || true
fi

LOG="\$REMOTE_DIR/logs/exp_L_alpha_n4_50_\$(date +%Y%m%d_%H%M%S).out"
PIDFILE="\$REMOTE_DIR/logs/exp_L_alpha_full.pid"

if [[ -f "\$PIDFILE" ]]; then
  oldpid=\$(cat "\$PIDFILE" || true)
  if [[ -n "\${oldpid:-}" ]] && kill -0 "\$oldpid" 2>/dev/null; then
    echo "Stop ancien PID=\$oldpid"
    kill "\$oldpid" || true
    sleep 2
  fi
fi

echo "Lancement nohup → \$LOG"
echo "N_VALUES=\$N_VALUES  RUN_EXP3=\$RUN_EXP3  → \$OUT_CSV_NAME"
nohup env \\
  PASQAL_ROOT="\$PASQAL_ROOT" \\
  OUT_DIR="\$OUT_DIR" \\
  OUT_CSV_NAME="\$OUT_CSV_NAME" \\
  OUT_LOG_NAME="\$OUT_LOG_NAME" \\
  N_VALUES="\$N_VALUES" \\
  N_INSTANCES="\$N_INSTANCES" \\
  ALPHAS="\$ALPHAS" \\
  L_INIT="\$L_INIT" \\
  L_CHOSEN="\$L_CHOSEN" \\
  RUN_EXP1="\$RUN_EXP1" \\
  RUN_EXP2="\$RUN_EXP2" \\
  RUN_EXP3="\$RUN_EXP3" \\
  FORMS="\$FORMS" \\
  julia test_real/exp_L_alpha_full.jl > "\$LOG" 2>&1 &
echo \$! > "\$PIDFILE"
echo "PID=\$(cat "\$PIDFILE")"
echo "CSV attendu : \$REMOTE_DIR/results/\$OUT_CSV_NAME"
sleep 8
head -n 60 "\$LOG" || true
REMOTE_SCRIPT

echo "==> Déployé. Suivi :"
echo "  ssh $REMOTE 'tail -f $REMOTE_DIR/logs/exp_L_alpha_n4_50_*.out'"
echo "  scp $REMOTE:$REMOTE_DIR/results/exp_L_alpha_full_results_n4_50.csv ."
