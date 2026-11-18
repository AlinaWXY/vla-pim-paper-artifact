# LPDDR5X-8533 WCK synchronization at rank switches

Set `"wck_rank_switch": "resync"` in a native simulator JSON config. The
default `"always_on"` preserves existing replay timing. In resync mode,
each external DQ transfer that changes rank issues CAS(WS_RD) or CAS(WS_WR)
on CA and commits the selected data command for the following CK. This
experiment isolates rank switching. The first transfer, same-rank idle
intervals and local PIM commands do not trigger a synchronization CAS.

The model uses BG mode, BL16, 938 ps CK, WCK:CK 4:1, RL set 0 (23 CK),
WL set A (12 CK), 2 CK DQ bursts, a 4 CK BL/n_max, and a 2.5 WCK
postamble. DQ ODT, non-target ODT, WS_FS broadcast and Link ECC are off.
The following values are CK:

| Transition | Always-on DQ gap | CAS issue gap from previous column | Resync DQ gap |
|---|---:|---:|---:|
| Read → read | 2 | 20 | 19 |
| Read → write | 2 | 23 | 11 |
| Write → read | 11 | 9 | 19 |
| Write → write | 1 | 12 | 11 |

The previous column to CAS gap from Table 379 is
`L_prev + BL/n_max + ceil((tWCKPST - 0.5*tWCK)/tCK) - tWCKENL_next`.
Tables 204/205 give `tWCKENL_RD=8` and `tWCKENL_WR=5`. Tables
361/362 require the column one CK after CAS. Adding the next data latency
and subtracting the old data latency and actual burst duration gives the
DQ gaps above. The controller waits longer when native column constraints
or CA traffic require it. The bus checker verifies a direction and rank
matching CAS on the preceding CK and independently checks the DQ gap.
See [JESD209-5C, §7.2.1.6 and Table 379](https://xjbcode.fun/Notes/006-ai/assets/JESD209-5C.pdf#page=555).

For π0.5, `tools/report_wck_switch.py` reports rank0 DQ utilization and
the share unoccupied by rank1 over six PIM linear stages: QKV, QKᵀ, PV,
O, Gate/Up, and Down, weighted by total channel cycles. Its fixed-trace
read-capacity estimate packs complete 2 CK GPU reads into rank1 gaps on
the recorded channel and applies both rank-switch edges. This estimate
uses frozen PIM timings and omits CA arbitration, refresh, VLM request
availability, and fused episode restrictions. It provides a DQ-only
comparison for archived c (all off) and g (all on). Case e uses token
layout and local RoPE with streaming input, and case g uses fused input
preload. See the [b–g replay results](../reference/2026-11-18/wck-rank-switch/DQ-comparison.zh-CN.md).
