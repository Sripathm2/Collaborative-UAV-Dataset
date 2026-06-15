#!/bin/bash
# run_layer3.sh — End-to-end Layer-3 replay pipeline.
#
# Usage:
#   bash run_layer3.sh --list
#   bash run_layer3.sh --flight_id testbed_1004vol1_flight28
#   bash run_layer3.sh --flight_ids fid1 fid2 fid3
#   bash run_layer3.sh --all
#   bash run_layer3.sh --all --resume
#   bash run_layer3.sh --all --sweep          # 4 variants per flight + compare
#   bash run_layer3.sh --all --sweep --resume # skip variant runs already done
#
# Variants for --sweep (pathloss_model x shadow):
#   __logdist_shadow      logdist + stochastic shadow fading
#   __logdist_noshadow    logdist mean-only
#   __3gpp_shadow         3GPP TR 36.777 + shadow
#   __3gpp_noshadow       3GPP TR 36.777 mean-only
#
# Outputs files with the variant suffix in their names so nothing is
# overwritten. --resume skips a (flight, variant) when sim_<fid><suffix>.csv
# already exists. After all variants finish, a single
# metrics_variant_compare.csv is emitted summarising each.
#
# Stages per (flight, variant):
#   1. layer3_prepare.py   (run once per flight; reused across variants)
#   2. layer3_run.py       trajectory -> sim_<fid><suffix>.csv
#   3. layer3_analysis.py  per-flight + summary CSVs with same suffix
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PY="python3"
RESULTS_DIR="../results/layer3"

# ----- defaults -----
SOURCE="aadm_testbed"
ALT_MIN="5.0"
PATHLOSS_MODEL="logdist"
MODULATION="adaptive"
TX_POWER="30.0"
NOISE_FLOOR="-58.0"
SEED="41"
NO_SHADOW=""
NO_FIG=""
LIST_ONLY=0
ALL=0
RESUME=0
SWEEP=0
FLIGHT_IDS=()

# variant table: pathloss_model;no_shadow_flag;suffix
SWEEP_VARIANTS=(
  "logdist;;__logdist_shadow"
  "logdist;--no_shadow;__logdist_noshadow"
  "3gpp;;__3gpp_shadow"
  "3gpp;--no_shadow;__3gpp_noshadow"
)

# ----- parse args -----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)              LIST_ONLY=1; shift ;;
    --all)               ALL=1; shift ;;
    --resume)            RESUME=1; shift ;;
    --sweep)             SWEEP=1; shift ;;
    --flight_id)         FLIGHT_IDS+=("$2"); shift 2 ;;
    --flight_ids)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        FLIGHT_IDS+=("$1"); shift
      done ;;
    --source)            SOURCE="$2"; shift 2 ;;
    --alt_min)           ALT_MIN="$2"; shift 2 ;;
    --pathloss_model)    PATHLOSS_MODEL="$2"; shift 2 ;;
    --modulation)        MODULATION="$2"; shift 2 ;;
    --tx_power)          TX_POWER="$2"; shift 2 ;;
    --noise_floor)       NOISE_FLOOR="$2"; shift 2 ;;
    --seed)              SEED="$2"; shift 2 ;;
    --no_shadow)         NO_SHADOW="--no_shadow"; shift ;;
    --no_fig)            NO_FIG="--no_fig"; shift ;;
    -h|--help)           sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ----- list mode -----
if [[ $LIST_ONLY -eq 1 ]]; then
  $PY "$SCRIPT_DIR/layer3_prepare.py" --list
  exit 0
fi

# ----- discover flights for --all -----
if [[ $ALL -eq 1 ]]; then
  if [[ ${#FLIGHT_IDS[@]} -gt 0 ]]; then
    echo "WARNING: --all overrides explicit --flight_id(s)" >&2
    FLIGHT_IDS=()
  fi
  echo "Discovering all flights under source=$SOURCE ..."
  mapfile -t FLIGHT_IDS < <(
    $PY "$SCRIPT_DIR/layer3_prepare.py" --list \
      | awk -v target="=== $SOURCE ===" '
          $0 == target { in_block=1; next }
          /^=== / && in_block { exit }
          in_block && NF { print $1 }
        '
  )
  if [[ ${#FLIGHT_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no flights discovered for --source $SOURCE" >&2
    exit 2
  fi
  echo "  found ${#FLIGHT_IDS[@]} flights"
fi

if [[ ${#FLIGHT_IDS[@]} -eq 0 ]]; then
  echo "ERROR: provide --flight_id <id>, --flight_ids ..., --all, or --list" >&2
  exit 2
fi

# ----- build the run plan -----
# Each entry: pathloss_model;no_shadow_flag;suffix
if [[ $SWEEP -eq 1 ]]; then
  RUN_VARIANTS=("${SWEEP_VARIANTS[@]}")
else
  # single variant from CLI flags
  RUN_VARIANTS=("$PATHLOSS_MODEL;$NO_SHADOW;")
fi

echo "=== Layer-3 pipeline ==="
echo "  source         : $SOURCE"
echo "  alt_min        : $ALT_MIN m"
echo "  resume         : $([ $RESUME -eq 1 ] && echo on || echo off)"
echo "  sweep          : $([ $SWEEP -eq 1 ] && echo on  || echo off)"
echo "  flights        : ${#FLIGHT_IDS[@]}"
echo "  variants       : ${#RUN_VARIANTS[@]}"
echo

# ----- stage 1: prepare each flight exactly once -----
echo "----- prepare (once per flight) -----"
i=0
for FID in "${FLIGHT_IDS[@]}"; do
  i=$((i + 1))
  TRAJ="$RESULTS_DIR/trajectory_${FID}.csv"
  GT="$RESULTS_DIR/ground_truth_${FID}.csv"
  if [[ $RESUME -eq 1 && -s "$TRAJ" && -s "$GT" ]]; then
    echo "[$i/${#FLIGHT_IDS[@]}] SKIP prepare $FID (trajectory + ground_truth exist)"
    continue
  fi
  echo "[$i/${#FLIGHT_IDS[@]}] prepare $FID"
  $PY "$SCRIPT_DIR/layer3_prepare.py" \
    --source "$SOURCE" \
    --flight_id "$FID" \
    --alt_min "$ALT_MIN"
done
echo

# ----- stage 2: per variant, run + analyze -----
ALL_SUFFIXES=()
for VARIANT in "${RUN_VARIANTS[@]}"; do
  IFS=';' read -r V_PL V_NS V_SUFFIX <<< "$VARIANT"
  echo "########## variant: pathloss=$V_PL  no_shadow=$([ -n "$V_NS" ] && echo yes || echo no)  suffix='$V_SUFFIX' ##########"
  ALL_SUFFIXES+=("$V_SUFFIX")

  N_DONE=0; N_SKIP=0; N_FAIL=0
  i=0
  for FID in "${FLIGHT_IDS[@]}"; do
    i=$((i + 1))
    SIM_FILE="$RESULTS_DIR/sim_${FID}${V_SUFFIX}.csv"
    if [[ $RESUME -eq 1 && -s "$SIM_FILE" ]]; then
      echo "[$i/${#FLIGHT_IDS[@]}] SKIP run $FID$V_SUFFIX (sim exists)"
      N_SKIP=$((N_SKIP + 1))
      continue
    fi
    echo "[$i/${#FLIGHT_IDS[@]}] run $FID$V_SUFFIX"
    if ! $PY "$SCRIPT_DIR/layer3_run.py" \
          --flight_id "$FID" \
          --pathloss_model "$V_PL" \
          --modulation "$MODULATION" \
          --tx_power_dbm "$TX_POWER" \
          --noise_floor_dbm "$NOISE_FLOOR" \
          --seed "$SEED" \
          --out_suffix "$V_SUFFIX" \
          $V_NS; then
      echo "  FAIL run $FID$V_SUFFIX" >&2
      N_FAIL=$((N_FAIL + 1))
      continue
    fi
    N_DONE=$((N_DONE + 1))
  done
  echo "  variant '$V_SUFFIX' :  done=$N_DONE  skip=$N_SKIP  fail=$N_FAIL"

  # collect successful flights for analysis
  ANALYZE_FIDS=()
  for FID in "${FLIGHT_IDS[@]}"; do
    if [[ -s "$RESULTS_DIR/sim_${FID}${V_SUFFIX}.csv" ]]; then
      ANALYZE_FIDS+=("$FID")
    fi
  done
  if [[ ${#ANALYZE_FIDS[@]} -eq 0 ]]; then
    echo "  no flights produced sim for variant '$V_SUFFIX'; skipping analyze" >&2
    continue
  fi
  echo "  analyze variant '$V_SUFFIX' (${#ANALYZE_FIDS[@]} flights)"
  $PY "$SCRIPT_DIR/layer3_analysis.py" \
    --flight_ids "${ANALYZE_FIDS[@]}" \
    --out_suffix "$V_SUFFIX" \
    $NO_FIG
  echo
done

# ----- stage 3: cross-variant comparison -----
if [[ $SWEEP -eq 1 || ${#ALL_SUFFIXES[@]} -gt 1 ]]; then
  echo "########## variant comparison ##########"
  $PY "$SCRIPT_DIR/layer3_compare_variants.py" \
    --suffixes "${ALL_SUFFIXES[@]}"
fi

echo
echo "=== Layer-3 pipeline complete ==="
echo "Outputs under: $RESULTS_DIR/"