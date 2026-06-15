#!/bin/bash
# run_layer4.sh — End-to-end Layer-4 three-way fidelity pipeline.
#
# Prerequisite: layer3 must have been run first so that
# results/layer3/sim_<fid>{variant}.csv files exist. Layer-4 reuses
# those as the "Sim" source for RSS.
#
# Usage:
#   bash run_layer4.sh --all                            # AUTO: discover every
#                                                       # layer3 variant present
#                                                       # and run layer4 for each
#   bash run_layer4.sh --sim_flight_ids fid1 fid2       # manual
#   bash run_layer4.sh --variant_suffix __logdist_noshadow \
#                       --sim_flight_ids fid1 fid2       # manual + single variant
#   bash run_layer4.sh --all --skip_packet_stats
#
# With --all, after each variant finishes a single
# metrics_variant_compare.csv is emitted ranking variants by
# H(Maeng,Sim)+H(AFAR,Sim).  Lower = more realistic sim.
#
# Stages per variant:
#   1. layer4_prepare.py            extract RSS samples; sim file = variant
#   2. layer4_run.py                Hellinger + 5-stat + figure (suffixed)
# Then once across all variants:
#   3. layer4_sim_packet_stats.py   variant-independent
#   4. layer4_compare_variants.py   one row per variant
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PY="python3"
L3_DIR="../results/layer3"

# defaults
ALT_MIN="5.0"
RSS_MIN="-130.0"
RSS_MAX="-20.0"
N_BINS="50"
ROW_CAP="0"
SIM_FIDS=()
AFAR_FIDS=()
SKIP_PACKET=0
ALL=0
VARIANT_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)               ALL=1; shift ;;
    --variant_suffix)    VARIANT_SUFFIX="$2"; shift 2 ;;
    --sim_flight_ids)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SIM_FIDS+=("$1"); shift
      done ;;
    --afar_flight_ids)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        AFAR_FIDS+=("$1"); shift
      done ;;
    --alt_min)           ALT_MIN="$2"; shift 2 ;;
    --rss_min)           RSS_MIN="$2"; shift 2 ;;
    --rss_max)           RSS_MAX="$2"; shift 2 ;;
    --n_bins)            N_BINS="$2"; shift 2 ;;
    --row_cap)           ROW_CAP="$2"; shift 2 ;;
    --skip_packet_stats) SKIP_PACKET=1; shift ;;
    -h|--help)           sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- discover variants present under $L3_DIR ----
# Returns list of suffix strings (e.g. "__logdist_noshadow"). Empty string
# means "legacy sim_<fid>.csv with no suffix".
discover_variants() {
  local found
  found=$(
    ls "$L3_DIR"/sim_*.csv 2>/dev/null \
      | sed -n 's|.*/sim_.*\(__[A-Za-z0-9_]\+\)\.csv$|\1|p' \
      | sort -u
  )
  if [[ -n "$found" ]]; then
    echo "$found"
  else
    # any plain sim_*.csv (no suffix)?
    if compgen -G "$L3_DIR/sim_*.csv" > /dev/null; then
      echo ""
    fi
  fi
}

# ---- discover flight_ids for a given variant suffix ----
discover_fids_for_variant() {
  local v="$1"
  local out=()
  shopt -s nullglob
  local files=( "$L3_DIR"/sim_*"$v".csv )
  shopt -u nullglob
  local f bn fid
  for f in "${files[@]}"; do
    bn=$(basename "$f")
    fid="${bn#sim_}"
    fid="${fid%${v}.csv}"
    if [[ -z "$v" && "$fid" == *__* ]]; then
      # defensive: when v is empty, skip files that DO have variant suffixes
      continue
    fi
    out+=("$fid")
  done
  printf '%s\n' "${out[@]}"
}

# ---- build VARIANTS list ----
VARIANTS=()
if [[ $ALL -eq 1 ]]; then
  if [[ ${#SIM_FIDS[@]} -gt 0 ]]; then
    echo "WARNING: --all overrides explicit --sim_flight_ids" >&2
    SIM_FIDS=()
  fi
  if [[ -n "$VARIANT_SUFFIX" ]]; then
    # user pinned a specific variant
    VARIANTS=("$VARIANT_SUFFIX")
  else
    mapfile -t VARIANTS < <(discover_variants)
  fi
  if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    echo "ERROR: no sim_*.csv files found under $L3_DIR" >&2
    echo "       run layer3 first (run_layer3.sh --all [--sweep])" >&2
    exit 2
  fi
  echo "Variants discovered:"
  for v in "${VARIANTS[@]}"; do
    echo "  '${v:-<none>}'"
  done
else
  # manual single-variant mode
  if [[ ${#SIM_FIDS[@]} -eq 0 ]]; then
    echo "ERROR: --sim_flight_ids fid1 [fid2 ...] required (or --all)" >&2
    exit 2
  fi
  VARIANTS=("$VARIANT_SUFFIX")
fi

# ---- per-variant: prepare + run ----
for V in "${VARIANTS[@]}"; do
  echo
  echo "########## variant: '${V:-<none>}' ##########"

  # Resolve FIDS for this variant
  if [[ $ALL -eq 1 ]]; then
    mapfile -t V_FIDS < <(discover_fids_for_variant "$V")
  else
    V_FIDS=("${SIM_FIDS[@]}")
  fi

  if [[ ${#V_FIDS[@]} -eq 0 ]]; then
    echo "  no sim flights for variant '$V'; skipping"
    continue
  fi

  echo "  sim flights : ${#V_FIDS[@]}"
  echo "  alt_min     : $ALT_MIN m"
  echo "  rss bins    : [$RSS_MIN, $RSS_MAX] / $N_BINS"
  if [[ ${#AFAR_FIDS[@]} -gt 0 ]]; then
    echo "  afar filter : ${AFAR_FIDS[*]}"
  fi

  # 1. prepare
  echo "----- prepare -----"
  PREP_ARGS=(--alt_min "$ALT_MIN" --sim_flight_ids "${V_FIDS[@]}")
  if [[ -n "$V" ]]; then
    PREP_ARGS+=(--variant_suffix "$V")
  fi
  if [[ ${#AFAR_FIDS[@]} -gt 0 ]]; then
    PREP_ARGS+=(--afar_flight_ids "${AFAR_FIDS[@]}")
  fi
  $PY "$SCRIPT_DIR/layer4_prepare.py" "${PREP_ARGS[@]}"

  # 2. analyze
  echo
  echo "----- analyze -----"
  $PY "$SCRIPT_DIR/layer4_run.py" \
    --rss_min "$RSS_MIN" \
    --rss_max "$RSS_MAX" \
    --n_bins  "$N_BINS" \
    --out_suffix "$V"
done

# ---- sim-only packet stats (variant-independent) ----
if [[ $SKIP_PACKET -eq 0 ]]; then
  echo
  echo "########## sim packet stats (variant-independent) ##########"
  PKT_ARGS=()
  [[ "$ROW_CAP" != "0" ]] && PKT_ARGS+=(--row_cap "$ROW_CAP")
  $PY "$SCRIPT_DIR/layer4_sim_packet_stats.py" "${PKT_ARGS[@]}"
fi

# ---- variant comparison ----
if [[ ${#VARIANTS[@]} -gt 1 ]]; then
  echo
  echo "########## variant comparison ##########"
  $PY "$SCRIPT_DIR/layer4_compare_variants.py" \
    --suffixes "${VARIANTS[@]}"
fi

echo
echo "=== Layer-4 pipeline complete ==="
echo "Outputs under: ../results/layer4/"