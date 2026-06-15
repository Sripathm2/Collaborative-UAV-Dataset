# UAV-CAS — Collaborative-Attack Security Dataset for UAV Networks

A measurement-calibrated, fully reproducible dataset for evaluating intrusion detection in unmanned aerial vehicle (UAV) networks under both single and **collaborative** attacks. Generated through a four-layer calibration pipeline anchored to public empirical UAV campaigns (AERPAW AADM, AERPAW AFAR, Gurses-Sichitiu, Maeng-Lim-Shin) and an extensible Containernet-based swarm digital twin. 

Github url - https://github.com/Sripathm2/Collaborative-UAV-Dataset. 
Dataset url - https://dx.doi.org/10.21227/zgrg-z865.

> Paper currently under journal review; arXiv preprint forthcoming. See **Citation** below.

---

## Table of contents

1. [Features](#features)
2. [Repository layout](#repository-layout)
3. [Tested platforms](#tested-platforms)
4. [Setup on a fresh machine](#setup-on-a-fresh-machine)
5. [Collection workflow](#collection-workflow)
6. [Cleanup](#cleanup)
7. [Dataset format](#dataset-format)
8. [Reproducing the dataset](#reproducing-the-dataset)
9. [Reproducing the calibration pipeline](#reproducing-the-calibration-pipeline)
10. [Reproducing IDS baselines](#reproducing-ids-baselines)
11. [Reproducing figures](#reproducing-figures)
12. [File-naming conventions](#file-naming-conventions)
13. [Citation](#citation)
14. [License](#license)
15. [Acknowledgments](#acknowledgments)

---

## Features

- **1,024 configurations** sampled from a 9-axis design space (5 attacks, 5–20 drones, 1–4 base stations, image / control / mixed payloads, two propagation models, two modulation policies, four mission types, three TX powers, three noise floors).
- **Six attack classes**: Benign + DoS, DDoS, Blackhole, Wormhole, Replay.
- **Nine collaborative compositions**: every Synchronized or Complementary pair of the five attacks, exactly as taxonomised in the accompanying paper.
- **Four-layer calibration** against published empirical UAV measurements:
  - *Layer 1* — path-loss model fit (Maeng-Lim-Shin RSRP, cross-validated on Gurses-Sichitiu).
  - *Layer 2a* — mobility-trace library mined from AERPAW AADM development flights.
  - *Layer 3* — packet-level link-quality replay against AADM testbed flights (median |RSS error|, SNR Pearson r).
  - *Layer 4* — three-way fidelity (Hellinger distance) against AERPAW AFAR and Maeng RSRP.
- **Direction-aware per-flow data**: each packet carries a direction bit so Fwd/Bwd statistics can be recovered (47-feature flow tables, not the usual 25).
- **Optional config embedding** for generative or controllable-traffic research (`--embed_config` flag attaches the full 9-axis scenario string to every label).
- **Cross-dataset baselines** out of the box: ten IDS architectures (1D-CNN, LSTM, RF, SGD, LR, MLP, LightGBM, ConvNet, TinyML, CNN-BiLSTM) on UAV-CAS, UNSW-NB15, CICIOT2023, UAV-NIDD, CICIDS2017.

---

## Repository layout

```
.
├── Docker/                  Containernet node image (Dockerfile.node + build helpers)
├── Network-simulator/       Containernet topology + simlib (propagation, mobility,
│                            modulation, shadow fading, routing) +
│                            Layer-3 / Layer-4 calibration scripts and runners
├── Scripts/                 Preprocessing, dataset build, calibration tables,
│                            figure generation, validation tooling
├── IDS/                     Ten baseline architectures + cross-dataset
│                            evaluation pipelines (T11, T12, T13)
├── Makefile                 One-command setup of Containernet + venv + node image
├── requirements.txt         Python dependencies for the venv used by Scripts/ and IDS/
└── start_collection.sh      Driver that launches dataset collection over the
                             configured 9-axis Cartesian product
```

Each top-level folder is self-contained; the only inter-folder dependency is that `Scripts/` and `IDS/` consume the CSVs produced by running `Network-simulator/` inside the Docker image.

---

## Tested platforms

The codebase was developed and tested on:

- **CloudLab** Clemson `c8220` nodes (Intel Xeon E5-2683 v3, 256 GB RAM) running Ubuntu 22.04 LTS.
- A workstation-class Linux box with ≥ 16 GB RAM and root access for `tc netem` / OVS kernel datapath.

Anything Ubuntu 22.04 / Python 3.10 / Docker ≥ 20.10 should work. macOS is not supported (Containernet requires Linux network namespaces).

---

## Setup on a fresh machine

The repository ships a `Makefile` that installs all simulator dependencies in two phases. Both phases are idempotent — re-running them is safe.

### 0. Edit two paths in the Makefile

Open `Makefile` and change the two variables at the top to match your machine:

```make
home_dir := /users/<your-user>/        # absolute path to a writable HOME on this machine
root_dir := /mydata/local/             # absolute path to where you cloned this repo
```

On CloudLab `c8220` nodes the defaults are typically `/users/<NetID>/` for `home_dir` and `/mydata/local/` for `root_dir` (CloudLab mounts a dedicated data volume at `/mydata`). If you cloned the repo into your home directory on a non-CloudLab box, set both to that path.

### 1. Phase 1 — system packages, Containernet, node image

```bash
make install-containernet-and-requirements-part1
```

This target:

1. Refreshes apt indexes.
2. Installs `ansible`, `python3.10-venv`, `tshark`, `parallel`, and `htop`.
3. Clones [Containernet](https://github.com/containernet/containernet) into `$(home_dir)`.
4. Runs Containernet's Ansible playbook (`containernet/ansible/install.yml`), which sets up Mininet, Open vSwitch, and Docker on the host.
5. Creates a Python 3.10 virtualenv at `$(home_dir)/venv`.
6. Drops you into a root shell, where the final command builds the UAV node container image:
   ```bash
   sudo docker build --no-cache --tag=uav_nodes -f Docker/Dockerfile.node Docker/
   ```

> The Ansible step takes 10–20 minutes the first time. If the apt update fails on `us.archive.ubuntu.com`, uncomment the `sed -i ...` line at the top of the target — it rewrites the mirror URL to `https://`, which fixes intermittent TLS-cert errors seen on some CloudLab images.

### 2. Phase 2 — Python deps, calibration data, and start collection

After phase 1 finishes and you're back in your normal shell:

```bash
make install-containernet-and-requirements-part2
```

This target:

1. Installs Containernet's Python bindings into the venv.
2. Installs the project's Python requirements from `requirements.txt`.
3. Unzips `UAV_data.zip` (preprocessed empirical calibration data: AERPAW AADM/AFAR, Maeng RSRP, Gurses channel) into `Network-simulator/`. See **note** below if you don't have this archive.
4. Copies CloudLab-specific helper scripts into the repo root.
5. Re-enters a root shell and immediately launches a detached `tmux` session running `start_collection.sh`, which iterates the 9-axis Cartesian product and invokes `Topo.py` for every configuration assigned to this node.

To watch the running collection:

```bash
sudo tmux attach -t mysession
```

To detach without killing it, press `Ctrl-b` then `d`.

> **Note on `UAV_data.zip`.** The empirical calibration archive is distributed separately from the GitHub repo to keep the clone small (it's ~few GB after extraction). It contains the preprocessed `Finetuning-processed/` CSVs produced by `Scripts/0-step1_process_{aadm,afar,gurses,maeng}.py`. If you don't have the zip yet, either (a) request a copy from the corresponding author, or (b) regenerate it yourself by downloading the upstream raw datasets (linked in the License section) and running the four `0-step1_process_*.py` scripts manually.

### 3. Quick sanity check

Verify the node image built and the simulator runs:

```bash
sudo docker images | grep uav_nodes
cd Network-simulator
sudo /users/<your-user>/venv/bin/python Topo.py --config_idx 0
```

A successful run prints scenario info, per-link propagation values, and tc-verification summaries to stdout, and emits `pcaps/*.pcap`, `attack_details_*.txt`, and `tc_diagnostics_*.log`.

---

## Collection workflow

For full-campaign collection across many nodes (e.g. 10 CloudLab servers running the 9-axis Cartesian product in parallel), the recommended pattern is:

1. **Per-node assignment.** Each server picks a disjoint slice of `config_idx` values via `start_collection.sh`. By default the script distributes work by `hostname` hash; edit it if you want explicit ranges.
2. **Detached tmux session.** Phase 2 of the Makefile launches `start_collection.sh` inside `tmux new-session -d -s mysession`, so the work survives SSH disconnects. Monitor with `sudo tmux attach -t mysession`.
3. **Validate as you go.** From a separate shell:
   ```bash
   cd Scripts
   python validate_run.py     # only validates new (unseen) tags
   python validate_run.py --sum
   ```
4. **Gather artifacts.** After all nodes finish, rsync the per-node `pcaps/flows-*.txt` and `pcaps/attack_details_*.txt` files back to a single host and run the dataset builders (`Scripts/3-build_ts_csv.py`, `Scripts/6-build_stat_csv.py`).

A helper script for multi-server collection on CloudLab is shipped in `Cloudlab-utilities/` (copied into root by phase 2 of the Makefile).

---

## Cleanup

After a run, residual Docker containers, OVS bridges, and Mininet state can accumulate. The Makefile ships a `clean` target:

```bash
make clean
```

This stops and removes every `mn.*` container, tears down Mininet (`mn -c`), and deletes stray `.png` files from the working directory. Run this if `Topo.py` fails to start due to "namespace exists" or "bridge already exists" errors.

---

## Dataset format

The simulator emits per-flow records into two complementary CSV files. See `Scripts/3-build_ts_csv.py` and `Scripts/6-build_stat_csv.py` for the exact emitters.

### `UAV-CAS_ts.csv` — per-flow packet sequences

| Column | Type | Description |
|---|---|---|
| `packet_time` | list-of-float | Per-packet Unix timestamps (s). |
| `packet_size` | list-of-int | Per-packet bytes, aligned to `packet_time`. |
| `packet_dir` | list-of-int | Per-packet direction bit (0 = sender is the alphabetically smaller IP of the sorted pair, 1 = larger). |
| `Label` | str | Canonical class. Single: `Benign`, `DoS`, `DDoS`, `Blackhole`, `Wormhole`, `Replay`. Collaborative: `+`-joined alphabetically (e.g. `Blackhole+DoS`). |

An optional `packet_flag` column (list-of-hex-string TCP flags, e.g. `['0x10','0x18']`) is inserted between `packet_size` and `packet_dir` if the file was built with `--include_flags`.

### `UAV-CAS_stat.csv` — per-flow aggregated features (47 features)

**Meta (11):** `config_idx, num_drones, num_bs, payload, pathloss, modulation, mission, tx_power, noise, src_ip, dst_ip`.

**Flow-level, direction-agnostic (25):** `Flow Duration`, `Total Packets`, `Total Length of Packets`, `Flow Bytes/s`, `Flow Packets/s`, `Flow IAT {Total, Mean, Std, Max, Min}`, `Min/Max Packet Length`, `Packet Length {Mean, Std, Variance}`, `{FIN, SYN, RST, PSH, ACK, URG, CWE, ECE} Flag Count`, `Header Length`, `Average Packet Size`.

**Direction-aware (22):** `Total Fwd/Bwd Packets`, `Total Length of Fwd/Bwd Packets`, `Fwd/Bwd Packets/s`, `Fwd/Bwd Packet Length {Max, Min, Mean, Std}`, `Fwd/Bwd IAT {Total, Mean, Std}`, `Fwd/Bwd Header Length`.

**Label (1):** same convention as `ts.csv`.

### Config-embedded variants

Built by passing `--embed_config` to either build script. The `Label` becomes `<canonical>|<config_string>` where `<config_string>` is the verbatim 9-axis scenario, e.g.

```
DDoS+Replay|ddos=1000+replay=50,200,10,5,inc-20-2-image-logdist-adaptive-random-30-95
```

Split on `|` to recover canonical and config independently. Neither piece contains `|`.

### Parsing example

```python
import ast, pandas as pd

df = pd.read_csv("UAV-CAS_ts.csv", low_memory=False)
df["pkt_t"]   = df["packet_time"].apply(ast.literal_eval)
df["pkt_sz"]  = df["packet_size"].apply(ast.literal_eval)
df["pkt_dir"] = df["packet_dir"].apply(ast.literal_eval)

# Reconstruct per-direction byte counts for the first flow:
row = df.iloc[0]
fwd_bytes = sum(s for s, d in zip(row["pkt_sz"], row["pkt_dir"]) if d == 0)
bwd_bytes = sum(s for s, d in zip(row["pkt_sz"], row["pkt_dir"]) if d == 1)
```

---

## Reproducing the dataset

The dataset is generated by running the Containernet topology once per configuration, parsing the resulting pcaps into flow records, then aggregating them into CSV form.

### 1. Generate per-config pcaps

Each configuration is identified by a single integer (`config_idx`) that indexes into a Cartesian product of the 9-axis design space.

**For a full collection campaign**, the recommended path is the `tmux`-detached `start_collection.sh` workflow described in **Setup on a fresh machine → Phase 2**. Phase 2 of the Makefile launches it automatically; on subsequent runs you can restart it manually with:

```bash
sudo tmux new-session -d -s mysession \
    'source <home_dir>/venv/bin/activate && ./start_collection.sh'
```

**For a single-config debug run**, invoke `Topo.py` directly:

```bash
cd Network-simulator
sudo <home_dir>/venv/bin/python Topo.py --config_idx <N>
```

Outputs per `config_idx`:
- `pcaps/mn.<host>-<YYYYMMDD>-<HHMM>-<config_idx>.pcap` — raw packet capture per node.
- `attack_details_<YYYYMMDD>-<HHMM>-<config_idx>.txt` — ground-truth attack log (attacker, victim, parameters, per-link propagation, mobility windows).
- `tc_diagnostics_<...>.log` — verification trace of applied `tc netem` impairments.

### 2. Convert pcaps to flow records

```bash
cd Scripts
python process_csvs.py
```

Walks every `mn.*.pcap.csv` (produced from pcaps by `tshark`), groups by sorted IP pair, and emits `flows-<tag>.txt` files. Each per-packet tuple in v2 is `(timestamp, size, tcp_flags, direction_bit)`.

### 3. Sanity-check each configuration

```bash
python validate_run.py
```

Cross-references the attack-details log against the flows file: confirms tc applied to every container, verifies attack dispatch matches config, checks per-attack flow rates against expected values, etc. Maintains `validation_results.txt` so re-runs only validate new configurations.

### 4. Build the dataset CSVs

```bash
# canonical labels only (used by IDS)
python 3-build_ts_csv.py   --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_ts.csv \
                          --out-tex ../results/Step_3/table7_dataset_stats.tex
python 6-build_stat_csv.py --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_stat.csv

# config-embedded variants (for generative or controllable-traffic work)
python 3-build_ts_csv.py   --root ../UAV-cas-dataset --embed_config \
                          --out-csv ../Datasets/UAV-CAS_ts_cfg.csv
python 6-build_stat_csv.py --root ../UAV-cas-dataset --embed_config \
                          --out-csv ../Datasets/UAV-CAS_stat_cfg.csv
```

Add `--include_flags` to either ts builder to add the optional `packet_flag` list column. Add `--single-only` to drop collaborative-attack rows.

---

## Reproducing the calibration pipeline

The calibration pipeline anchors the simulator's propagation, mobility, and traffic behavior to public empirical UAV measurements. All four layers are reproducible from the raw datasets cited in the paper.

### Layer 1 — path-loss model fit

```bash
cd Scripts

# Preprocess each public dataset into a common schema
python 0-step1_process_aadm.py   --aadm_dir   <path-to-AADM-USRP-dir>
python 0-step1_process_afar.py   --afar_dir   <path-to-AFAR-SigMF-dir>
python 0-step1_process_gurses.py --gurses_dir <path-to-gurses-channel-dir>
python 0-step1_process_maeng.py  --maeng_dir  <path-to-maeng-extracted-dir>

# Fit log-distance, Two-Ray, and 3GPP TR 36.777 models on Maeng RSRP,
# then cross-validate on Gurses
python 1-step2_layer1_compare.py
```

Outputs `results/Step_1/layer1_comparison.json` containing fitted parameters and per-model RMSE/bias/r.

### Layer 2a — mobility library

```bash
python 1-step3a_layer2a_mobility_library.py
```

Mines AERPAW AADM development flights, classifies each into one of {spiral, grid, hover_transit, random}, fits per-category velocity and altitude distributions, and writes `results/Step_1/layer2a_mobility_library.json`.

### Layer 3 — packet-level link-quality replay

```bash
cd Network-simulator
bash run_layer3.sh --list                       # list available AERPAW flights
bash run_layer3.sh --all --sweep --resume       # replay every flight across 4 propagation variants
```

For each flight, produces `sim_<flight_id>__<variant>.csv` (sim RSS/SNR per packet), `metrics_<flight_id>__<variant>.csv` (median |RSS error|, mean bias, SNR Pearson r), and a per-flight figure. `metrics_variant_compare.csv` ranks the four propagation variants by overall fidelity.

### Layer 4 — three-way fidelity

```bash
bash run_layer4.sh --all
```

For each Layer-3 variant, computes pairwise Hellinger distances between Maeng RSRP, AFAR DT (digital twin), and simulator RSS distributions. Emits `metrics_threeway__<variant>.csv` and a two-panel figure. `metrics_variant_compare.csv` ranks variants by `H(Maeng,Sim) + H(AFAR,Sim)`.

---

## Reproducing IDS baselines

```bash
cd IDS

python run_t11.py          # Cross-dataset binary Benign-vs-DoS (5 datasets × 2 inputs × 3 scalers × 10 models)
python run_t12.py          # 15-class native multi-class on UAV-CAS
python run_t13.py          # Train-on-single / test-on-collaborative AUROC
python run_confusion.py    # 80/20 split, per-model 15×15 confusion matrices

python make_tables.py      # Emit LaTeX tabulars from the CSVs
python make_figures.py     # Emit fig_confusion.pdf + fig_baseline_bars.pdf
```

`run_t11.py` is checkpoint-resumable via `t11_checkpoint.csv` (each completed (dataset, input, scaler, model) combination is flushed and fsync'd immediately). `run_t12.py` / `run_t13.py` / `run_confusion.py` run from scratch every invocation.

To see which IDS architecture wins on the 15-class task before regenerating the confusion figure:

```bash
python make_figures.py --rank
```

prints the macro-F1 ranking of every model and exits without rendering.

---

## Reproducing figures

The paper uses 11 figures. Each is produced by a single script and authored at its final ACM column width (3.33 in single-column, 7.0 in `figure*`) so no PDF downscaling is required.

| Fig | Script | Description |
|---|---|---|
| 1 | `Scripts/8-fig_pathloss_curves.py` | Path-loss models vs. distance at 5 altitudes |
| 2 | `Scripts/8-fig_pathloss_crossval.py` | Cross-validation scatter (Gurses) for 3 models |
| 3 | `Scripts/8-fig_mobility_traces.py` | 3D trajectories for 4 mission types |
| 4 | `Scripts/8-fig_velocity_validation.py` | Velocity CDFs sim vs. AFAR, 4 missions |
| 5 | `Network-simulator/layer4_run.py` | Three-way RSS fidelity (PDFs + Hellinger bars) |
| 6 | `Network-simulator/layer3_analysis.py` | Per-flight RSS / SNR / throughput time series |
| 7 | `Scripts/7-fig_internal_diversity.py` | Per-axis flow-feature spread (violin) |
| 8 | `Scripts/7-fig_attack_distributions.py` | Per-attack IAT + packet-rate distributions |
| 9 | `Scripts/7-fig_tsne.py` | t-SNE projection of stat features by class |
| 10 | `IDS/make_figures.py` | Confusion matrices: best vs. worst model |
| 11 | `Scripts/7-fig_topology.py` | Topology evolution across mobility windows |

Each script's `--out-pdf` argument controls the output path. See `Scripts/run_all.sh` for a one-shot regeneration of all figures from existing data.

---

## File-naming conventions

| Pattern | Meaning |
|---|---|
| `mn.<host>-<YYYYMMDD>-<HHMM>-<idx>.pcap` | Raw per-host packet capture |
| `flows-<YYYYMMDD>-<HHMM>-<idx>.txt` | Aggregated per-flow packet sequences |
| `attack_details_<YYYYMMDD>-<HHMM>-<idx>.txt` | Ground-truth attack log + tc verification |
| `tc_diagnostics_<YYYYMMDD>-<HHMM>-<idx>.log` | Detailed `tc netem` apply/verify trace |
| `sim_<flight_id>__<variant>.csv` | Layer-3 replay output per (flight, variant) |
| `rss_sim__<variant>.csv` | Layer-4 sim RSS samples for a propagation variant |
| `cm_<model>.npy` | Per-model 15×15 raw confusion matrix |

Propagation variants: `__logdist_shadow`, `__logdist_noshadow`, `__3gpp_shadow`, `__3gpp_noshadow`.

---

## Citation

If you use UAV-CAS in your research, please cite:

```bibtex
@misc{uavcas2026,
  author       = {Mishra, Sripath and {Bhargava}, Bharat},
  title        = {{UAV-CAS}: A Calibrated, Reproducible Dataset for Collaborative-Attack Detection in {UAV} Networks},
  year         = {2026},
  note         = {Under journal review. arXiv preprint forthcoming.},
}
```

Once the arXiv preprint is posted, the BibTeX entry will be updated with the official identifier.

---

## License

Code and documentation in this repository are released under the **MIT License** (see `LICENSE`).

The dataset CSVs themselves are released under **CC BY 4.0** with the request that downstream papers cite the work as above.

The public empirical datasets used during calibration retain their original licenses:

- **AERPAW AADM** — Drone Mobility Dataset (NCSU). See [AERPAW](https://aerpaw.org/).
- **AERPAW AFAR** — A-CCI Find a Rover. See [AERPAW](https://aerpaw.org/).
- **Maeng-Lim-Shin RSRP** — see the cited Dryad release.
- **Gurses-Sichitiu channel sounder** — see the cited Zenodo release.

If you redistribute calibration artifacts derived from these datasets, retain the upstream attribution.

---

## Acknowledgments

The authors acknowledge NCSU's AERPAW platform for providing the AADM and AFAR datasets that anchor the propagation and mobility layers of this work, and the authors of the Maeng and Gurses-Sichitiu releases for making their measurements public.

---

## Questions / issues

Please open a GitHub issue. For correspondence regarding the paper, contact mishra60 at purdue.edu.