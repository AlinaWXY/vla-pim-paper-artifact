# VLA-PIM: Simulator and Paper Artifact

Anonymous artifact for **VLA-PIM: LPDDR-PIM Architecture–Dataflow Co-Design for Heterogeneous Acceleration of Edge VLA Inference**. Snapshot: 18 November 2026.

This repository contains the latest cycle-accurate simulator, frozen implementations used for the paper, Thor measurements, generated PIM programs, native simulation outputs, and numerical figure inputs and plotting scripts. Model checkpoints and commercial EDA tools/PDKs are obtained separately from their providers.

## Quick start

Requires Linux, GCC 14 with C++20, CMake >= 3.18, and Python >= 3.10. Simulation requires no GPU. CMake downloads pinned fmt and yaml-cpp dependencies on the first build.

```bash
python3 tools/artifact.py verify
python3 tools/artifact.py paper-checks
JOBS=2 bash build.sh
```

Large JSON/CSV files use deterministic, lossless gzip compression. The replay helper unpacks required inputs automatically. To unpack everything, allow several GB of free space and run `python3 tools/artifact.py unpack`.

## Contents and versions

| Location | Contents |
| --- | --- |
| Repository root | Latest simulator, including WCK rank-switch synchronization |
| `data/paper-simulator/` | Frozen π0.5 paper simulator |
| `data/gr00t-simulator/` | Frozen GR00T paper simulator |
| `data/pi05/` | Paper inputs, configs, ablations, native outputs and DQ traces |
| `data/gr00t/` | H=15, 640×480 inputs, calibration, Thor captures and repeated native runs |
| `data/wck/` | Latest b–g runs with `wck_rank_switch=resync` |
| `data/timeline/` | Layer-0 timeline native inputs and outputs: 15 vs 19 completed VLM kernels |
| `data/power/`, `data/area/` | Power counters, estimation inputs and synthesis reports |
| `paper/figures/` | Literal paper chart data and plotting scripts |

The π0.5 paper figure uses the archived always-on WCK model. Its b/c/d/e labels map to native b/c/e/g, with cycle times 122.88/86.77/83.57/77.68 ms. GR00T maps to n16-b/n16-c/n16-d/n16-f, with 104.58/78.36/77.82/74.41 ms. WCK resynchronization results are a separate, newer experiment. See [the artifact guide](docs/paper-artifact.md) for provenance and limits.

## Native reproduction

Build the frozen π0.5 implementation:

```bash
cmake -S data/paper-simulator -B data/paper-simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build data/paper-simulator/build --target pim_vla_sim -j2
```

The short timeline examples replay one AE layer and a 32-kernel VLM prefix:

```bash
python3 tools/artifact.py replay --config data/timeline/config-e.json \
  --output replay-output/timeline-e --expected data/timeline/results-e/raw.json
python3 tools/artifact.py replay --config data/timeline/config-g.json \
  --output replay-output/timeline-g --expected data/timeline/results-g/raw.json
```

Complete inference runs can take hours. Examples:

```bash
python3 tools/artifact.py replay --config data/pi05/configs/combined-g.json \
  --output replay-output/pi05-g
python3 tools/artifact.py replay --config data/wck/configs/g.json \
  --output replay-output/wck-g
```

The helper selects the frozen simulator for historical π0.5/timeline configs and the latest root simulator for WCK. `--latest` applies the latest simulator to historical inputs. For GR00T, build `data/gr00t-simulator` using the equivalent CMake commands, then:

```bash
python3 tools/artifact.py replay --config data/gr00t/configs/n16-f.json \
  --output replay-output/gr00t-f --expected data/gr00t/native/n16-f/repeat-0/raw.json
```

## Paper charts

```bash
python3 -m pip install -r paper/requirements.txt
python3 paper/figures/plot-async-vla-performance.py
python3 paper/figures/plot-decoupled-rates.py
python3 paper/figures/plot-layout-dq.py
python3 paper/figures/plot-timeline.py
```

The scripts require Arial: install it or set `VLA_PIM_FONT_DIR` to a directory containing its regular, italic, bold and bold-italic fonts. Font files are obtained separately. The timeline applies a disclosed monotone display transform for readability; drawn widths are not an absolute time axis. Native CK timestamps are retained.

## License and anonymity

Our additions, scripts and numerical data use the [MIT license](LICENSE). Bundled third-party components retain their original notices and licenses; see [third-party notices](THIRD-PARTY-NOTICES.md). Author and machine-specific text paths are anonymized. `ARTIFACT-MANIFEST.json` records released and original source checksums. Original Git histories, checkpoints, credentials and commercial tool/PDK files are excluded.
