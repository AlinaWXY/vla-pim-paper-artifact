# Native execution model

The executable links the clean Ramulator 2.1 core pinned at
`7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`. Sixteen channel controllers each own
one native `DRAMDevice` containing both ranks. GPU and PIM candidates consult
the same device's prerequisites, timing state and command issue path. No legacy
PIM trace or external arbitration timeline is read.

The spec exports command completion and bus metadata along with its native
timing table. A passive checker records actual CA intervals and delayed DQ
intervals, splitting occupancy at global stage boundaries. It is never queried
to make an issue decision. The fault test removes a native transfer constraint
and verifies that this observer detects the resulting DQ overlap.

The PIM JSON contains a layer-relative physical program and explicit completion
dependencies. Each layer has a separate 1 MiB region per bank; denoising steps
reuse the corresponding layer. Descriptor children are the statically expanded
instruction program. One descriptor is resident at a time; its children issue
through the private command path. The stored program is not a
hardware FIFO. This branch admits at most 32 descriptor children into an
in-order FIFO. Only its head may issue, including the head's native ACT/PRE
prerequisites; younger children cannot bypass a stalled head or preopen a row.
Dispatch advances the FIFO immediately and records the completion timestamp.
It does not wait for every dispatched child to complete: independent operations
can overlap subject to native timings, resource readiness and explicit
dependencies. A completion barrier and the descriptor's final completion wait
for the outstanding tails. This supersedes the Version2 out-of-order window
introduced in `292cfbc`; archived measurements retain their original semantics.
The streaming lookahead policy can inspect at most the next 32 instructions.
SRF and output pools are checked by a separate bounded data-lifetime scoreboard.

VLM requests are uniformly released within each measured kernel duration and
bounded by a per-channel outstanding window. Each channel receives its own
balanced read/write sequence: channel striping must not accidentally segregate
reads and writes into different channels. Read and write tensor regions are
distinct and the logical address sequence wraps within the physical rank0
capacity. Requests are queued by bank and direction; the controller compares
ready heads by row hit and arrival. This is equivalent to scanning the generated
monotonic bank/direction streams, without scanning the entire outstanding window.

Both refresh modes have native command semantics. The fixed policy is **rank0
(GPU) REFpb and rank1 (PIM) REFab**. Rank0 rotates through eight paired-bank
targets (n and n+8) every nREFIpb=520 CK; rank1 refreshes all sixteen banks every
nREFI=4164 CK. Due targets drain through native precharge prerequisites before
refresh. Native nRFCpb=128 CK and nRFC=224 CK control recovery; no controller
delay is added. Rank1 all-bank operations must re-activate the closed banks.
The former global `refresh_allbank` option is rejected even when false/null.
Raw outputs record the policy and actual command counts separately for each rank.
Split ACT
commands have owned ACT2 requests: ACT2 issues as early as legal, with nAAD as
an upper bound rather than an artificial mandatory wait.

Case b implements the document's sum of independently initialized solo epochs.
After AE completes, it resets the DRAM state and refresh phase for the VLM epoch
inside the same simulator, retaining the global elapsed timestamp and all
statistics. Thus its measured total equals AE_solo + VLM_solo without a latency
correction. It is an independent-epoch serial baseline, not a measurement of
retained row-buffer or refresh-phase state across an AE-to-VLM handoff. Concurrent
cases never reset memory state during execution.

The model does not execute tensor values. Actual GPU memory addresses and request
burst traces are unavailable; its uniform arrivals and synthetic sequential
regions are the workload assumptions required by the supplied plan. Measured
arithmetic helpers are counted separately from the zero-cost control handoff.
SM contention from WRGB transport, additional control handoff latency, and
concurrent execution of AE nonlinear kernels with VLM kernels are not modeled.
