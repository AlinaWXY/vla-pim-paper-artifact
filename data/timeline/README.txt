Figure 6 native VLM-priority replay, 2026-11-18

Scope: original step-0 setup plus AE layer 0, and original VLM kernels
0--31. Both AE cases finish before this VLM prefix runs out. The prefix
preserves initial GPU ownership and controller state. This is one layer
of the same AE workload, not a complete 18-layer, 10-step AE inference.

The frozen 2026-11-16 native simulator already implements kernel-boundary
handoff when vlm_gpu_quantum_us is omitted. The active VLM kernel finishes,
then queued AE GPU work executes before another VLM kernel may start.
The original 2-us yielding option is removed in these figure-only inputs.
No simulator code or original experiment outputs were edited.

Two bounded local native replays took 16.69 s (e) and 18.59 s (g), each
limited to 60 s; no cluster job, training, or full sweep was launched.

Layer-0 results:
  (a) streaming input: 372.811852 us, 15 completed VLM kernels.
  (b) input preloading: 479.892056 us, 19 completed VLM kernels.
All retained kernels complete within the AE window; none is interrupted
or overlaps an AE GPU event. Every delayed AE GPU event begins exactly
when its blocking VLM kernel ends. Kernel identifiers in the drawing are
one-based (native ordinal + 1).

Each panel's AE span is independently normalized to the same width.
Very narrow VLM blocks are widened for numeric labels. Kernels 2 and 16
are then shortened by the same 35% of their display widths in both panels;
the released width is distributed proportionally among the other intervals.
Kernels use dark-green fill, pale-green borders and white numeric labels.
QK^T is widened by the same 50% in both panels, redistributing display
width within its containing VLM kernel so all VLM block widths are retained.
In the GPU Kernels lane, kernel 2 is shortened by a further 35% in
both panels and that width is transferred to the following AE GPU interval.
This later adjustment preserves QK^T and the other VLM kernel widths.
Orange blocks show native AE GPU intervals, merging adjacent operators;
the first interval after kernel 2 is displayed at 20% of its enlarged width,
with the released width distributed over VLM blocks other than 2 and 16.
The orange text AE Non-Linear sits directly below its orange block in each panel.
The same monotone
display transform is applied to PIM stages and DQ burst starts,
so the lanes remain aligned. Native CK timestamps are retained unchanged
in the snapshot. Drawing widths must not be interpreted as a common
absolute time scale or as kernel execution times.
The gray I/O-epoch background bands and their legend are omitted from the
final figure; the native epoch timestamps remain available in the snapshot.

Reproduce the bounded native runs and regenerate the self-contained input:
  python3 experiments/2026-11-18-vlm-priority-figure/reproduce.py --run
Extract the input snapshot from existing native outputs without running:
  python3 experiments/2026-11-18-vlm-priority-figure/reproduce.py
Plot using an environment with current matplotlib, NumPy, Pillow, and Arial:
  VLA_PIM_FONT_DIR=/cache/arial-corefonts \
    /cache/dq-traffic-plot-venv/bin/python \
    VLA-PIM-paper/figures/plot-timeline.py

The figure snapshot contains input, executable, shared-library and native
output checksums. Original paper performance data are not updated by this
local scheduling illustration. The old timeline.json is retained intact.
