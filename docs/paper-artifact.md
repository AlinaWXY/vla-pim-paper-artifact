# Paper data and reproduction guide

The root implementation includes optional WCK resynchronization. Historical paper timing uses `always_on`. New results should state the source directory and WCK mode explicitly.

## Provenance

`ARTIFACT-MANIFEST.json` records SHA-256 digests of released bytes, decompressed content and original source bytes. Metadata redaction is recorded separately. Checksums embedded in historical reports describe original inputs; use the release manifest to verify anonymous files.

Figure inputs in `paper/figures/data/` record the mapping from paper a–e labels to native labels. π0.5 GPU measurements/calibration are under `reference/2026-11-15/`; complete archived input evidence is under `data/pi05/evidence/`. GR00T data include repeat-0 and repeat-1, standalone runs, calibration, generated programs and Thor measurements.

GR00T's serial GPU baseline sums separately measured phase means rather than an independently timed serial pair. The supplemental dual-stream value 98.27 ms is AE inference latency p50; an end-to-end pair-cycle p50 was not supplied. Its metadata preserves that distinction. The packed-SDPA VLM capture reports a 10.71% compiled-versus-eager relative RMS difference; it supports performance analysis of that implementation.

Decoupled PIM rates are first-overlap-window estimates, without multi-frame native validation. The archived fresh-cache energy estimate uses a 135.75 ms steady-state baseline period. The paper instead multiplies recorded board power by the 137.07 ms coupled dual-stream period; `data/power/paper-energy.json` makes the calculation explicit. GPU rail power is a proxy. PIM logic assumes 256 units remain at reference active power throughout the interval. These are model estimates, rather than measured heterogeneous-board energy.

Area data are cell-area synthesis estimates. DRAM scaling uses 63.4/1.38, normalized to 16 banks and a 60.92 mm² reference die. This does not establish implementation area or timing closure in a DRAM process. Cell-area reports and assumptions are included; commercial software and PDK libraries are obtained separately.

## Timeline

The two configs under `data/timeline/` replay step 0, layer 0, with the first 32 VLM kernels. Omitting `vlm_gpu_quantum_us` completes an active VLM kernel before pending AE GPU operators run. The frozen π0.5 source produces 15 completed VLM kernels for streaming and 19 for preloading.

The AE spans are independently normalized and narrow blocks adjusted for legibility. All lanes share the same monotone transform per panel. Original CK timestamps remain unchanged. `paper/figure-validation.json` records the display adjustments.

## Portable replay

Configuration input paths are relative to the repository root. Use `tools/artifact.py replay` to resolve paths, unpack required files and write a concrete config in the output directory. `verify` uses only Python's standard library. Full experiments can take hours; the one-layer timeline examples are bounded checks. Hardware recapture requires the providers' software/weights and the documented Thor environment.
