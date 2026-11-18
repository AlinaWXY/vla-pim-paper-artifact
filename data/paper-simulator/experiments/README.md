# Native experiment workflow

The current user-authorized scope is **Version2, step 0 / layer 0 only**.
Calibration, full AE and end-to-end runs remain paused. Use the separate entry:

```bash
python programs/gen_pim_program.py --output programs/generated --compact
python experiments/run_single_layer.py --binary build/pim_vla_sim \
  --output results/version2-single-layer --repeats 2
```

This runs the five AE structures with no VLM path, no memory-traffic fit and no
call to the full campaign. It includes both layer residuals and normalization,
plus the measured streaming PV-to-O transpose. Global prefix/final work is
outside this layer boundary. See [the Version2 report](../docs/version2-single-layer.md)
for exact scope, source bindings and results. Every run requires a fresh output
directory. The workflow below describes the paused full-campaign capability.

`run.py` invokes `build/pim_vla_sim`; it does not schedule memory commands or
reconstruct timing from traces. The source ledgers under `input-data/measured`
are Thor measurements. The normalized inputs retain every measured kernel
duration unchanged. The sum of per-unit VLM durations is 105.2205782 ms;
the independent measured phase target is 104.950496035 ms. Their difference
is reported, never removed by normalization.

Generate the programs and prepare native configurations:

```bash
python programs/gen_pim_program.py --output programs/generated --compact
python experiments/run.py --output results/rewrite --prepare-only
```

Run the acceptance matrix after native spec, checker, and microbenchmark
checks pass:

```bash
python experiments/run.py --output results/rewrite
```

This runs all five standalone AE designs, then standalone VLM calibration,
then b–g and e-frfcfs/e-lookahead/e-gap, twice each. The concurrent matrix is
not entered if VLM calibration exceeds 5% error. Existing result directories
are protected; `--resume` requires the same configuration and existing native
output. Each run writes input configuration, native log, raw timestamps and
provenance. The experiment output adds the main CSV, per-stage/kernel/bus
tables, conservation checks, and byte-identical repetition checks.

The calibration search varies only alpha, per-channel outstanding window,
and write watermarks. It first diagnoses window 64 with alpha 1, then scans
window 1024 and two window-256 write-drain configurations. Alpha candidates
are 1, 0.5, 0.2, 0.1, and 0.05, in that order. The positive 0.05 search floor is
explicit and is not presented as a measured DRAM writeback lower bound.
The first passing full-workload candidate is frozen for all cases. Exact
memcpy sectors are never multiplied by alpha; only kernel L2 write sectors
are a writeback proxy. Per-kernel median and p90 relative errors compare
native standalone kernel duration to that kernel's measured duration.

For Slurm arrays, run calibration candidates independently and collect them
after completion. The collector waits for earlier candidate indices before
freezing the first passing candidate, so completion order cannot change the
chosen model parameters:

```bash
python calib/calibrate_vlm.py --config results/rewrite/configs/b.json --output results/calibration --candidate 0
python calib/calibrate_vlm.py --config results/rewrite/configs/b.json --output results/calibration --collect
```

Run standalone designs once, then share their verified outputs among
independent case jobs. `--solo-only --cases e` selects one AE design, so the
five standalone designs may themselves be run independently:

```bash
python experiments/run.py --output results/solos --solo-only
python experiments/run.py --output results/case-e --cases e --reuse-solos results/solos --calibration results/calibration/calibration.json
python experiments/run.py --output results/matrix --aggregate-from results/case-*
```

The aggregate requires all requested cases, identical frozen calibration,
verified executable and loaded `libramulator.so` hashes, and successful
byte-identical native repetitions. Resume and standalone reuse also check the
loaded library, so replacing a shared library invalidates prior calibration.

For the full server campaign, commit the tested source and use the prepared
Slurm workflow. The first command freezes the source, binary and shared
library, then regenerates all five PIM programs using the archived source.
It verifies all input provenance and program source bindings, then creates
the nine configurations before any parallel job can write them. `prepare`
performs no simulation and rejects stale source/data hashes:

```bash
RUN=/artifact-source/pim_vla_sim2_20261109/results/full-k4
bash tools/run-campaign.sh prepare "$RUN"
bash tools/run-campaign.sh submit "$RUN"
```

The dependency chain is five AE standalone array tasks (`b,d,e,f,g`), then
calibration candidates 0–5, then deterministic collection. After collection
creates a passing `$RUN/calibration/calibration.json`, run:

```bash
bash tools/run-campaign.sh submit-cases "$RUN"
```

If candidates 0–5 all fail, `submit-rest` runs the remaining 6–15 and collects
again. Concurrent cases remain gated on successful calibration. Case jobs
read shared `$RUN/solos/ae-solo/<variant>` results, execute both repetitions
under `$RUN/cases/<case>/native/<case>/repeat-{0,1}`, then an `afterok`
aggregate writes `$RUN/matrix/evaluation-b-g.csv` and the report. Nine case
tasks cover b–g and e-frfcfs/e-lookahead/e-gap. Each array task requests one
CPU and 8 GiB; `PIM_PARTITION=dev` selects the 12-hour partition instead of
the default 24-hour `cpu-scavenger`. `PIM_MAC_FREQ_MULT` can select 1, 2, or 4
at prepare time; a different multiplier needs a separate campaign and fit.

Every job verifies source, prepared configuration, executable and library SHA
manifests before execution. The binary comes from `$RUN/frozen/bin` and
`LD_LIBRARY_PATH` starts with `$RUN/frozen/lib`; the source commit is bound to
the archive rather than a later mutable checkout. Output provenance includes
the embedded spec's canonical JSON hash as well as input and raw-result hashes.
If a checkout's Git newline conversion has invalidated recorded byte hashes,
`tools/verify_input_provenance.py --rebind-line-endings` can update provenance
only after proving that the previous hash matches the same bytes with LF/CRLF
conversion. It records both hashes and leaves all data files untouched. Commit
that metadata repair before preparing the campaign; ordinary content changes
are rejected.

The complete AE event schedule has 18 layers × 10 denoising steps. It retains
per-step natural-cache GPU kernel durations and replaces only the offloaded
operators with PIM stage references. GPU RoPE kernels appear between the
PIM `rope_export` and `rope_import` stages. Local RoPE uses offline
coefficients and omits GPU coefficient-generation and apply kernels.
Cross-rank score/O reduction and Gate/Up/Down cast helpers use separately
measured CUDA device-kernel times; these are arithmetic work, not control
handoff latency. Their original measurement ledger is preserved. The fixed
control handoff cost remains zero.

Model boundaries: VLM addresses are sequential logical tensor regions mapped
into the fixed 16 GiB rank0 capacity by row modulo the spec's row count.
Regions can be reused because kernels never overlap one another; each single
kernel's region is smaller than the physical capacity. This is an explicit
synthetic ring-address model, not a larger fabricated memory configuration.
Requests are uniformly released over measured kernel durations, and actual
writeback traffic is fitted. GPU transfer SM occupancy and extra control
handoff costs are omitted. AE GPU blocks and VLM kernels share the GPU
without preemption. Results are conditional simulation estimates, not a
Thor hardware concurrency measurement.
