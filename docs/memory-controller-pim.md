# Memory-controller command control at MAC 2×

The top-level memory controller issues the row and column commands for both
ranks. Fused descriptor children remain a software representation of the
existing operation order. Their decode takes zero cycles and issues no CA
command. Fixed-function GATHER, CAST, and adder-tree work remains on die.

## Rank and column state

- A split ACT to the last row of bank 0 followed by PREpb enters PIM mode.
  The penultimate row of bank 1 followed by PREpb restores DRAM mode.
- In PIM mode, a single-bank ACT to a configured input row selects broadcast.
  The subsequent RD sends the column to all SRFs. PRE closes the input row
  and returns to MB idle. Fused Down packs two fragments' 32 input columns
  in half a staging row and stripes batches over banks 0 and 4. One ACT
  opens the source for all 32 broadcast reads before PRE and the weight
  ACTAB. Weight addresses and the per-bank SRF slots follow the original
  Down mapping.
- ACTAB opens all 16 banks at once. It occupies two external CA cycles and
  uses the same array timings as an ordinary split ACT, with one activation
  event in the rolling activation history.
- ACTAB to the third-last row selects SRF writes until the rank changes
  mode. The selector occupies two CA cycles, leaves the DRAM row buffers
  intact, and can coexist with a weight row open for MAC. Each Host WR sends
  32 B over DQ to one SRF slot in all banks. The MC interleaves the next
  half's WRs with MAC RDs on the current half. Its first consumer waits for
  all sixteen writes to complete. Other PIM-mode RD and WR have no
  DQ transfer. For a Host result, the controller issues an internal store
  into the existing score or fused-result address space and then reads the
  column in DRAM mode.

After an ACTAB to a data row, the column command targets the active rank row,
and the four combined BG/BA bits select the function. Single-bank array
operations use ordinary bank addressing. This includes the existing sparse
spills, read-modify-writes, and the source RD in Down broadcast.

| BG/BA | RD | WR |
| --- | --- | --- |
| `0000` | MAC into the current accumulator | WB_ACC |
| `0001` | MAC_INIT overwrites the accumulator | WB_MAC, reserved for the bypass path |
| `0010` | MAC_RED, reserved for a multi-MAC reduction | Reserved |
| `0011` | MAC_INIT_RED starts and reduces a one-MAC dot product | WB_OBUF |
| `0100` | MAC_INIT_RED with K BF16 lanes 0–7 | Reserved |
| `0101` | MAC_INIT_RED with K BF16 lanes 8–15 | Reserved |
| `0110` | RD_ROPE loads the addressed coefficient column into each bank's RoPE register | Reserved |
| `0111` | RD_ROPE_Q loads a Q vector and schedules rotation with the loaded coefficients | Reserved |
| `1000` | RD_ROPE_K loads a K vector and schedules rotation with the loaded coefficients | Reserved |

This QK program issues one MAC per key and reduces its 16 bank contributions.
Token-wise uses the entire 32 B column. Feature-wise uses its original packed
K columns and selects one half with the opcode. Both half-column forms use the
same MAC timing and reduction as the whole-column form.
MAC_INIT_RED schedules the original 4 CK reduction at MAC completion. It
fills the original per-bank output slots without another CA command. Fused
QK uses one WB_OBUF per score column and an implicit bank-enable mask for
banks 0 and 4, matching the two existing output packets. Fused O, Gate/Up,
and Down use one WB_OBUF per 16-bank result stripe. The controller drains
these stripes to DRAM while still in PIM mode, then reads the stage's
results in DRAM mode.

MC-controlled streaming cases b–e batch result exports by default. Each
batch writes all results into their assigned DRAM columns in PIM mode, then
switches once to DRAM mode for the Host reads. QK uses up to 496 score
columns per batch, within the existing per-bank output pool. The two score
packets per key share one WB_OBUF. PV, O and FFN batch the reads at the
existing tile or stage boundaries.

After local RoPE, Q exchange exports all local Q columns in DRAM mode.
The streaming path then uses the SRF selector and bank-addressed Host WRs
to load both query parities directly into the SRFs. Column bit 5 selects
bank-addressed writes, BG/BA selects the destination bank, and column bits
0–4 select the slot. Each bank owns different Q features. The
fused path writes the remote half into DRAM, then switches to PIM mode
once and reads the local and remote columns into their banks' SRFs.
The fused export and import stages reserve the DQ window for the exchange
and its setup. Physical burst counts are reported separately.

Local RoPE runs after the BF16 Q/K projection stores. In each bank, the
working row holds eight coefficient columns at 0–7, Q at 8–15, and K
at 16–23. The MC can open this row with one ACTAB, subject to refresh and
any existing row conflict. For each local token, RD_ROPE reads its explicit
coefficient column in all banks, RD_ROPE_Q/K read their addressed vectors,
and ordinary WB_OBUF stores the rotations back to those columns. Every bank
reads its own coefficients. These commands occupy CA and transfer zero DQ
bytes. Bank-local RoPE still has an 8 CK vector initiation interval and
11 CK rotation latency. The model checks the lifetimes of the coefficient,
input and output slots. GPU-RoPE cases retain their GPU steps.

Q and K can have different token counts and can span multiple 32 B blocks
per token. A coefficient block describes one position and one block of eight
feature pairs. One activation per bank suffices for the entire operation
when the Q, K and distinct coefficient blocks all fit in the same row
(64 columns per bank), at aligned column positions across the rank. Feature
pairs must be within the block processed by one bank. Larger layouts use
one activation per row tile when the corresponding coefficient and data
columns are co-located. The current experiment generates the fixed 8-token
layout and models the command protocol and timing.

## Reproduction

Run the original and new configurations with the same MAC frequency.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pim_vla_sim --parallel
c++ -std=c++20 -O3 tools/gap_replay.cpp -o build/gap_replay

python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/baseline-ca-traces --mac-freq-mult 2
python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/mc-default-batch-traces --mac-freq-mult 2 --mc-control \
  --cases b c d e f g
python3 tools/estimate_dq_gaps.py --traces results/mc-default-batch-traces \
  --spec results/mc-default-batch-traces/spec-k2.json --engine build/gap_replay \
  --output results/mc-default-batch/replay
python3 tools/plot_pim_buses.py --traces results/mc-default-batch-traces \
  --output results/mc-default-batch/buses --title 'Top-level MC | MAC 2x' --cases b c d e f g
```

The per-packet control uses the same input stream and `--no-batch-output`.
Compare the generated programs with this control to verify the streaming
input order, Host read addresses and result spill addresses.

```sh
python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/mc-rd-rope-traces --mac-freq-mult 2 \
  --mc-control --no-batch-output
python3 tools/audit_mc_protocol.py --traces results/mc-default-batch-traces \
  --baseline results/baseline-ca-traces --per-packet results/mc-rd-rope-traces \
  --spec results/mc-default-batch-traces/spec-k2.json \
  --output results/mc-default-batch/protocol-audit.json
```

The matched-slice e2e estimator repeats each sampled slice 180 times. Case b
adds the VLM profile duration after the serial AE run. Cases c–e use the
prior α=1.0 and same-first DQ replay. Cases f/g use the prior
shared-I/O accounting and ideal overlap outside I/O epochs. The estimator
omits shared CA contention with VLM. The CA and DQ graphs use native AE-solo
timestamps and show 0.5 µs bins, including AE GPU intervals on the time
axis. The full projected AE plots tile those samples 180 times and
average them into 1,600 bins per case. The PIM-stage utilization denominator
is the sum of PIM-stage CK
over the 16 channels. CA counts both ranks' refresh and separately reports
rank 1 CA. DQ counts the physical Host transfers during those stages.
