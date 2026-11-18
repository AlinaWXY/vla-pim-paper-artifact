# Programs for fused kernel windows

`generate(variant, channel_count=16, execution_model="fused_kernel_windows")`
builds the new f/g programs. The corresponding CLI option is
`--execution-model fused_kernel_windows`; without `--variant`, this mode generates
only `token_gpu_after` and `token_local_after`. The default
`native_dram_requests` mode retains the earlier command sequences. Both modes
identify themselves in the top-level `execution_model` field.

Every descriptor stage in the new mode consists of input WR requests and their
completion barrier, one descriptor with private children, then optional ordinary
RD exports. The descriptor's completion includes all local result stores.
`stage.kernel_window` exposes `descriptor_index_by_channel` and
`export_begin_by_channel` under `mode: descriptor_compute`. An export index equal
to the channel program length means that stage has no external export. The
frontend admits a new VLM kernel only after descriptor delivery on every
channel, and joins the current VLM kernel before external payload export.
An already-running VLM kernel continues across a safe descriptor-only boundary.
No timing constants are encoded by this generator.

The frontend classifies a continuous boundary structurally: two adjacent PIM
events must belong to the same layer/step and each contain one descriptor on
all sixteen channels. Only completion BARRIERs may follow the outgoing
descriptor or precede the incoming one. Both descriptor bodies retain the
private-only checks. The current programs therefore allow QKV→local RoPE and
PV→O without joining VLM. Their preceding local stores and barriers still
complete before the next stage starts. No generator command, source dependency,
FIFO entry or output DRAM transfer is removed by this scheduling change.

The next descriptor is issued through the native shared CA/decode path while
the current calibrated VLM kernel continues. Its duration does not include a
new estimate of the coarse command's small GPU CA interference; PIM command
timing remains explicit. No new VLM kernel starts during descriptor delivery.
This replaces the earlier every-descriptor join policy, whose measured results
remain separately identified as historical in the execution document.

RoPE import/export and both Q-exchange stages use
`mode: exclusive_handoff`: the entire stage remains an exclusive GPU handoff,
including any LOCALRD requests mixed with external requests. In GPU RoPE mode,
Q has already been exported, so `q_exchange_export` contains only LOCALRD.
PV retains its existing local DRAM output and passes it directly to O without
an external result transfer.

O's two tiles now each gather/cast and LOCALWR their BF16 output before reusing
the existing 2 KiB per-bank output pool. Tile 1 waits for tile 0's local stores,
then both tiles are exported together after descriptor completion. Gate/Up and
Down similarly store their FP32 output inside their descriptors. The subsequent
RD commands read those exact addresses, not the output pool through RDRES.
The output location is a fixed operator contract of this mode; the descriptor
wire words and descriptor CA cost are unchanged. Native execution still consumes
the explicitly expanded children, not a binary output-address decoder.

Each layer reserves `[0x90000, 0x90800)` in every bank as ordinary result DRAM.
Packet p maps to bank `p % 16`, byte address `0x90000 + (p // 16) * 32`.
O has 1024 packets across two tiles, Gate/Up 1024, and Down 128. This occupies
2 KiB, 2 KiB, and 256 B per bank respectively; operators reuse the region only
after the preceding complete export. The region is separate from input staging,
scores and weights, and fits the existing 1 MiB per-layer address stride.
The generator checks all reserved ranges for bounds, alignment and overlap.
SRF and output-pool capacities are unchanged; no new SRAM is introduced.

External bytes are unchanged for both variants. Per channel and layer, 2176
former RDRES packets become 2176 LOCALWR plus 2176 ordinary RD. Local-write
completion and array prerequisites for the RD exports are therefore new real
costs; old native stage latencies cannot be reused for this mode.
