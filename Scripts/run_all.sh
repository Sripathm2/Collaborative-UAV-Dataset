#!/bin/bash
# run_all.sh
#
# STATE OF PLAY:
#   - Step_0/Step_1 calibration: already done (layer1_comparison.json +
#     layer2a_mobility_library.json present in results/Step_1).
#   - Step_2 tables.tex: already produced.
#   - Step_3/4/5: dataset CSVs + diversity/eval .tex files already present.
#   - Step_6 stat.csv: already present in Datasets/.
#   - Step_7/Step_8 figures: MUST BE REGENERATED for ACM formatting.
#
# ONLY uncomment the data/calibration sections after the 7-day v2 dataset
# re-collection (process_csvs.py v2 + build_*_csv.py v2). Until then,
# every figure-producing script reads files that ALREADY EXIST on disk.
# -----------------------------------------------------------------------------

# ============ preprocessing — SKIP (already done) ============
# mkdir ../results
# python 0-step1_process_aadm.py   --aadm_dir   ../Datasets/Finetuning-raw/aadm/AADM2025Dryad/USRP
# python 0-step1_process_afar.py   --afar_dir   ../Datasets/Finetuning-raw/afar/AFAR\ \ 2023_SigMF
# python 0-step1_process_gurses.py --gurses_dir ../Datasets/Finetuning-raw/gurses_channel
# python 0-step1_process_maeng.py  --maeng_dir  ../Datasets/Finetuning-raw/maeng_rsrp/extracted
#
# python 1-step2_layer1_compare.py
# python 1-step3a_layer2a_mobility_library.py

# ============ Step_2 calibration tables — SKIP (already done) ============
# mkdir ../results/Step_2
# python3 2-fill_calibration_tables.py \
#       --layer1 ../results/Step_1/layer1_comparison.json \
#       --layer2 ../results/Step_1/layer2a_mobility_library.json \
#       --out ../results/Step_2/tables.tex

# ============ Step_3 dataset CSVs — SKIP for now (re-run AFTER 7-day v2 collection) ============
# These build the four UAV-CAS_ts*.csv variants. v1 CSVs already exist on
# disk. Only re-run once process_csvs.py v2 has produced 4-tuple flow files
# from the new 7-day collection — at that point uncomment all four.
# mkdir ../results/Step_3
# python3 3-build_ts_csv.py --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_ts.csv --out-tex ../results/Step_3/table7_dataset_stats.tex
# python3 3-build_ts_csv.py --root ../UAV-cas-dataset --embed_config --out-csv ../Datasets/UAV-CAS_ts_cfg.csv
# python3 3-build_ts_csv.py --root ../UAV-cas-dataset --include_flags --embed_config --out-csv ../Datasets/UAV-CAS_ts_flags_cfg.csv
# python3 3-build_ts_csv.py --root ../UAV-cas-dataset --include_flags --out-csv ../Datasets/UAV-CAS_ts_flags.csv
# python3 3-build_ts_csv_FlowFusion.py --root ../UAV-cas-dataset --include_flags --out-csv ../Datasets/UAV-CAS_ts_flowfusion.csv

# ============ Step_4 diversity table — SKIP (already done) ============
# mkdir ../results/Step_4
# python3 4-diversity_table.py

# ============ Step_5 eval tables — SKIP (already done) ============
# mkdir ../results/Step_5
# python3 5-fill_eval_tables.py --root ../UAV-cas-dataset

# ============ Step_6 stat.csv — SKIP for now (re-run AFTER 7-day v2 collection) ============
# v1 stat.csv already exists in Datasets/. v2 (with 22 Fwd/Bwd direction
# features) needs new 4-tuple flow files. Same trigger as Step_3.
# python3 6-build_stat_csv.py --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_stat.csv
# python3 6-build_stat_csv.py --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_stat_cfg.csv --embed_config

# ============ Step_7 figures — RUN (ACM formatting overhaul) ============
mkdir -p ../results/Step_7
python3 7-fig_attack_distributions.py  --in-csv ../Datasets/UAV-CAS_stat.csv  --out-pdf ../results/Step_7/fig_attack_distributions.pdf
python3 7-fig_internal_diversity.py    --in-csv ../Datasets/UAV-CAS_stat.csv  --out-pdf ../results/Step_7/fig_internal_diversity.pdf
python3 7-fig_internal_diversity.py    --in-csv ../Datasets/UAV-CAS_stat.csv  --class-filter all  --out-pdf ../results/Step_7/fig_internal_diversity_all.pdf

python3 7-fig_topology.py \
  --simlib-path ../Network-simulator \
  --layer1 ../results/Step_1/layer1_comparison.json \
  --layer2 ../results/Step_1/layer2a_mobility_library.json \
  --config-str '10-2-image-logdist-adaptive-random-30-95' \
  --attack-prefix benign \
  --out-pdf ../results/Step_7/fig_topology.pdf

python3 7-fig_tsne.py --in-csv ../Datasets/UAV-CAS_stat.csv

# ============ Step_8 figures — RUN (ACM formatting overhaul) ============
mkdir -p ../results/Step_8
python3 8-fig_mobility_traces.py     --simlib-path ../Network-simulator  --layer2 ../results/Step_1/layer2a_mobility_library.json  --out-pdf ../results/Step_8/fig_mobility_traces.pdf
python3 8-fig_pathloss_crossval.py   --layer1 ../results/Step_1/layer1_comparison.json  --gurses ../Datasets/Finetuning-processed/gurses_channel.csv  --out-pdf ../results/Step_8/fig_pathloss_crossval.pdf
python3 8-fig_pathloss_curves.py     --layer1 ../results/Step_1/layer1_comparison.json  --maeng  ../Datasets/Finetuning-processed/maeng_rsrp.csv      --out-pdf ../results/Step_8/fig_pathloss_curves.pdf
python3 8-fig_velocity_validation.py --simlib-path ../Network-simulator  --layer2 ../results/Step_1/layer2a_mobility_library.json  --afar-csv ../Datasets/Finetuning-processed/afar_testbed.csv  --out-pdf ../results/Step_8/fig_velocity_validation.pdf

# ============ NOT in this file — but also need to be re-run for ACM ============
# Run from your Scripts dir (or wherever the updated make_figures.py lives):
#   python3 make_figures.py            # Fig 10 (confusion best+worst) + bar chart
# Layer-3 / Layer-4 figs re-generated automatically when you next run:
#   bash run_layer3.sh --all --sweep --resume   # produces Fig 6 link-quality PDFs
#   bash run_layer4.sh --all                    # produces Fig 5 three-way PDF