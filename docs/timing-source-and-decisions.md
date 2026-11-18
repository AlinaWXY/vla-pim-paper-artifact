# LPDDR5X-8533 timing source and modelling decisions

The clean base is CMU-SAFARI/ramulator2 commit
`7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da` (2026-11-03).
The only new timing authority is
`ramulator2/python/ramulator/dram/lpddr5x_pim.py`. Native generated C++ uses
Ramulator's DRAMDevice prerequisite, timing and state mechanisms.

The source is **JEDEC JESD209-5C, June 2023**, checked against the original
document's tables. A [readable copy of the standard](https://xjbcode.fun/Notes/006-ai/assets/JESD209-5C.pdf)
was inspected; printed pages differ from PDF page numbers by 38.
The document itself is not redistributed with this code.

| Parameter | Value (CK unless stated) | Source |
|---|---:|---|
| CK | 938 ps | 8533 bin, clock rounded upward |
| CL, CWL | 23, 12 | Tables 225 / 229; RL set 0 / WL set A |
| Write recovery | 37 | Table 230, x16, ECC/DVFSC disabled |
| Read-to-precharge | 8 | Table 225 nRBTP=6 plus BL/n_min=2 |
| Read / write RCD | 20 / 9 | Table 396, ceil(18 ns / CK), ceil(8 ns / CK) |
| Single / all bank RP | 20 / 23 | Table 396, 18 / 21 ns |
| RAS / RC | 45 / 65 | Table 396, 42 ns and RAS + single-bank RP |
| RRD / FAW | 4 / 16 | Table 396, 3.75 / 15 ns |
| WTR_S / WTR_L | 7 / 13 | Table 396, 6.25 / 12 ns |
| BL minimum / column cycle | 2 / 4 | Tables 339-342, x16 BL16 BG |
| RFCab / RFCpb (8 Gb die) | 224 / 128 | Table 240, ceil(210 / 120 ns / CK) |
| REFI / REFIpb | 4164 / 520 | Table 240; maximum intervals round downward |
| Per-bank refresh spacing / to other ACT | 96 / 8 | Table 240, 90 / 7.5 ns |

WCK remains on. DQ/NT ODT and ECC are disabled. Read preamble is 4 WCK
and read postamble 0.5 WCK; Table 377 then gives a 4 CK inter-rank
RD-to-RD issue gap (2 CK burst plus 2 CK rank guard). The 2 CK R-to-W
DQ guard represents upward rounding of the Table 377 WCK-to-DQ output
offset. These choices are fixed operating conditions, not calibration knobs.

## Resolving inconsistent examples in the task document

1. Alias only **transfer-to-transfer interface** constraints. Blindly aliasing
   every Rank rule would incorrectly add array tWR/tRTP before PREab to
   WRGB/RDRES, contradicting their SRAM destinations. Bank rules are never
   inherited by these aliases. Refresh blocking is separately explicit.
2. A buffer command addresses every BG. Consequently same-rank RDRES-to-WRGB
   is **17 CK**, from the same-BG rule `CL + BL_max + 2 - CWL`.
   Inter-rank RDRES-to-WRGB and GPU RD-to-PIM WRGB are **15 CK**.
   The document's unqualified "same-channel 15 CK" is the cross-rank case;
   15 CK cannot replace the stricter same-BG rule.
3. All-BG inheritance also retains WTR_L for same-rank WRGB-to-read (29 CK
   command gap), in addition to the Rank WTR_S rule. This follows the stated
   all-BG interface alias algorithm; removing it would contradict that rule.
4. REFpb is the LPDDR5 paired-bank command: bank n and n+8 are refreshed
   together. The upstream pair-target mechanism is retained.
5. The user-specified controller policy is fixed: GPU rank0 uses REFpb;
   PIM rank1 uses REFab. Both reuse the native intervals and recovery above.
   Refresh consumes one external CA CK and no DQ; recovery remains rank/bank
   scoped, so rank1 recovery does not freeze unrelated rank0 work.

## PIM resource assumptions

`MACAB_M{M}` / `MACQK_M{M}` cover M=1,2,4,8,16,32. Their work-batch completion
is `M*4/k` CK, and the next column issue separation is
`max(tCCD_L, M*4/k)`. The completion timestamp includes the batch's compute
pipeline; no extra unspecified tail is added. This is a model assumption
from section 5.1, not a measured PE RTL latency. Sensitivity uses k=1,2,4.

Local 32-byte array transfers use 4 CK service. SRF multicast/gather use
2 CK, cast 1 CK, rank reduction 4 CK, and RoPE pair latency 4 CK with pair
initiation interval 1 CK. Version2 specifies one RoPE PE per bank. Each
32-byte instruction contains eight pairs, so the Bank-level instruction
initiation interval is 8 CK and completion is 7 + 4 = 11 CK. A second bank
can execute independently; K can load while Q rotates. These values are
architectural assumptions, distinct from JEDEC memory protocol timings.
Descriptors occupy external CA for 21 CK and decode for 1 CK.

Native timing rules retain one logical receive tick for local commands,
so suppressing external CA does not introduce erroneous +/-1 CK timing
offsets or shift shared tFAW timestamps. Native `command_cycles` is therefore
one, while `metadata.external_ca` is zero; explicit spec bus constraints
exclude local commands from the external CA. Metadata also carries private
CA occupancy separately. Private ACT/PRE and
MAC aliases update the **same** native bank state and timing histories.
Version2 distinguishes streaming fixed-function local operations (`LOCALRD`,
`LOCALWR`, broadcast, gather, cast, reduction, and RoPE) from fused descriptor
expansion. Base local commands occupy neither CA bus nor DQ. Their `_LOCAL`
aliases occupy the fused rank controller's private CA for 1 CK, without
external CA or DQ. Alias pairs update the same native execution constraints.
Instruction issue is separate from execution: LOCALRD/LOCALWR timing remains
bank-scoped, cast and RoPE are bank-scoped, and broadcast/gather/reduction retain
their own rank-unit initiation intervals. No shared rank-wide data-port lock
serializes these independent units until completion. For example, two local
streaming reads to different banks have no rank-CA spacing, while the same bank retains
its 4 CK read-to-read constraint.
Refresh blocks SRAM IO conservatively as required by the task, and waits
for issued SRAM IO data bursts before entering refresh.
The native multi-cycle descriptor convention measures reception at the last
CA beat: after REFab, MAC_RUN/ROPE_RUN may begin at 204 CK, with their last beat
at 224 CK and decode completion at 226 CK. Their execution children still obey
the full native refresh recovery. This is not an early array operation.

## Version2 ACTAB (ACT4 x 4)

`ACTAB` occupies external CA for 2 CK. Its `ACTAB_LOCAL` alias is one fused
private instruction and occupies private CA for 1 CK. Both have a logical
native reception length of one tick, so timings are measured from the issue
timestamp specified by Version2 without a multi-cycle receive adjustment.
These command encodings are PIM architectural assumptions, not JEDEC commands.

The exported spec supplies `command_expansions`: sixteen `ACT2_PULSE`
reservations, four per batch at offsets 0, 16, 32, and 48 CK. Batch i targets
bank i in each BG. DRAMDevice applies their ordinary ACT2 timing and row-state
updates using the same native nodes. LOCALRD readiness is therefore 20, 36,
52, and 68 CK by bank; the all-bank MAC waits 68 CK. Each pulse contributes
one activation to the existing shared four-entry FAW history, also used by
ordinary ACT2, private ACT2, and REFpb. ACTAB admission waits until the latest
such activation is at least one FAW old so its first batch has four slots.

Reservations update future readiness immediately. The last four pulses at
t+48 block ordinary ACT2/REFpb through t+64; thus no subsequent legal issue
can insert an earlier timestamp into this activation history. Other commands
remain governed by their own Bank/Rank constraints. The snapshot lookahead
already copies all affected row state, readiness, and shared histories; no
controller timer or separate expansion event queue is needed.

All-bank MAC prerequisites select ACTAB only for a fully closed rank. Mixed
rows or any wrong open row first request PREab. Native PREab-to-ACTAB is 23 CK.
Single-bank LOCALRD/LOCALWR continue to use ordinary ACT1/ACT2 prerequisites.

`build_spec(k)` exports native config and command completion/DQ metadata.
The bus checker consumes this metadata, not a replayed trace or a second
timing table. Its physical DQ guards are independent of the alias-derived
command constraint list, allowing fault injection to detect a deleted rule.

## Native lookahead scope

`NativeLookahead` allocates one reusable private DRAMDevice per channel using
the same full spec export. Before forecasting it copies the live node states,
open rows, readiness clocks, per-command histories, and shared rolling-window
histories. It expands at most 32 FIFO operations within the requested time
horizon using native prerequisite resolution, native timing checks, and native
issue into this snapshot. Explicit dependencies and barrier completion tails
use actual or predicted completion timestamps from spec metadata. The live
device is never changed by prediction.

Forecasts omit future GPU commands, future refresh, and PIM resource contention.
They therefore provide conservative earliest issue estimates for arbitration,
not promises of actual service times. A local head can reveal a later WRGB or
RDRES within the window; checking only the current head cannot do this. The
controller still checks every real issue against the live native device. The
bounded predictor is intended for streaming FIFO programs; it stops at a
descriptor instead of inventing a nested program schedule.

`tests/native_lookahead_tests.cpp` covers future DQ behind a local operation,
explicit dependencies, barriers, a 32-operation limit, horizon truncation,
all-bank activation prerequisites, deterministic repeated forecasts,
and byte-for-byte-equivalent live state before and after prediction.

## Validation

`python tests/test_spec.py` tests numeric provenance, all 48 transfer pairs,
all-BG broadcast, absent SRAM array prerequisites/recovery, completion
metadata, MAC scaling, refresh coverage, and fixed topology validation.
`python tests/test_spec.py --export reference/spec` writes the full native
configuration and regression command matrix. `tests/native_spec_tests.cpp`
repeats every matrix boundary against real DRAMDevice, checks prerequisite
address resolution, paired-bank refresh, all sixteen staggered ACTAB bank
boundaries, shared FAW reservations, PREab recovery, CA occupancy, and bank
RoPE pipelines. It must be built and run on the
server before claiming native validation passed.
