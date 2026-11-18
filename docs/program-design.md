# PIM program interface and retained tensor contracts

`programs/gen_pim_program.py` generates JSON directly from tensor definitions.
It does not read an old native trace, a CA/DQ event file, or a replay schedule.
All five valid layout/RoPE/dataflow variants are generated with source hashes.
This document describes the Version2 dataflow with the 2026-11-09 FIFO32
in-order correction. The current requested measurement is **one layer and one
denoising step**, with coefficients and weights already resident, comparing
simulated latency and native wall time against the `6bcd57b` baseline. This
comparison does not start calibration or full end-to-end runs.
The workload format retains the full 18-layer, 10-step schedule, but that
capability is not a claim that Version2 full-workload experiments have run.
The JSON is a layer-relative address template: the controller adds
`layer * 512` to its array row index, equivalent to 1MiB per bank per layer.
Descriptor base fields are also relative to the current layer; their
expanded children receive the same relocation. Thus all eighteen layers
have distinct weights, KV and temporary regions; denoising steps reuse
the same layer's allocation. The 18MiB per-bank reservation fits within
the 64MiB bank capacity of the 8Gb/rank organization. The largest relative
reservation ends at 1MiB, so adjacent layers cannot overlap.

Each stage contains sixteen channel programs. Stage completion is a
global barrier. `deps` refers to **completion**, with indices local to a
stage/channel program; a descriptor's children use a separate index space.
Streaming outer requests and fused descriptor children issue in program order.
The descriptor's 32-entry FIFO bounds decoded buffering; it is not an
out-of-order selection window. Only the head and its native prerequisites may
issue, subject to explicit dependencies, resource identities, and native DRAM
checks. Dispatch advances the head without waiting for completion, so legally
independent operations can remain in flight together. `issue_deps` waits
for a producer's **issue**, allowing overlap with its execution; it does not
replace a completion dependency on consumed data. `BARRIER` has no command or
latency. It fences the window and waits for preceding issued completion tails
as well as its explicit dependencies.

Addresses are `[channel,rank,BG,bank,row,column]`; column is a 32-byte column
within a 2KiB row. Whole-rank MAC, WRGB and RDRES use BG/bank `-1`.
`slot`, `dest_slot`, and `result_slot` use **32-byte units**. Input SRF slots
are 0–31; output-pool slots are 0–63. `source_regions` identifies exact
FP32 source bytes by bank and byte offset, including a packet that collects
data from two banks. Gather/cast operates on this existing output pool; the
program introduces no rank result cache or unlimited packet scratch.
Transfers consume these explicit words, so a later gather of disjoint words
can overlap an earlier RDRES/LOCALWR completion tail; gather metadata remains
ordered through CAST, and real RMW scratch retains its own lifetime checks.

`pim_resources.h` enforces those identities independently of DRAM timing.
It tracks validity, ready timestamps and in-flight readers for 32 SRF slots
and 512 FP32 output words per bank, plus the bounded RoPE input/output state.
QK reduction results occupy words in this existing bank output pool, not a
separate rank score cache. First-fragment MAC initializes a word;
subsequent fragments must accumulate the immediately preceding partial sum.
RDRES locks exactly its source words through the spec-supplied completion,
and a subsequent MAC cannot overwrite them before that timestamp. Invalid
dataflow raises an error rather than converting missing producers into time.

MAC output geometry is `rows * output_columns * 4` bytes per bank. QKV uses
1536B; token PV 1024B; feature PV 512B; Gate/Up and each O tile 2048B; Down
256B. Fused GEMMs fill batches of up to 1024B and wait for the last MAC
before reusing the SRF. Streaming GEMMs split the same 1KiB SRF into two
512B halves. WRGB fills the idle half between MAC issues. The next batch's
first MAC waits for its complete input, and each refill waits for the final
MAC from the previous use of that half. Token QKV holds two 256B views per
half. O's second tile cannot fill its input or overwrite output until all
512 first-tile exports complete.

`MAC_RUN`/`ROPE_RUN` carries eight version-2 wire words and CRC16,
plus a timing-free expansion. The controller must feed these children through
its finite 32-entry internal FIFO and complete the descriptor after all
children finish. Younger instructions and their ACT/PRE prerequisites cannot
issue before the head. Native prerequisites prepare only the current head;
there is no younger-instruction prefetch. Named I/O epochs retain external
bus ownership, and barriers drain preceding completion tails. The generator
retains explicit RAW, WAR and WAW
completion dependencies for exact array columns, SRF slots, output words,
gather/RMW state and RoPE slots, rather than relying on textual issue order.
Only the RoPE overlap edges use `issue_deps` deliberately. Array and local
commands in children use their private-CA aliases. Streaming local operations
use fixed-function units with no external or private CA; their ACT/PRE
prerequisites still use external CA. External
RDRES operations between O tiles still use external CA/DQ and their named
I/O epoch. A local-RoPE fused layer has six MAC descriptors plus one RoPE
descriptor (147 CA CK under the spec); GPU-RoPE has six (126 CA CK). Decode
latency and CA occupancy are defined exclusively by the DRAM spec.

## Down co-resident input and weight rows

The requested Down layout uses twelve fragments per 2KiB bank row. Fragment f
occupies weight columns `5*(f%12)+0..3` and input column `5*(f%12)+4`, in row
`DOWN_BASE/2048 + f//12`. Query q's 32B input is in bank q; LOCALRD loads that
bank's SRF slot `(f%2)*16+q`, then BCAST supplies the same slot in all sixteen
banks. This preserves the two-fragment/1KiB SRF batch and all tensor ownership.

All 256 fragments reserve 22 rows per bank: 32KiB weights, 8KiB inputs and
4KiB total unused space (including the partial last row), or 44KiB in
`[0x50000,0x5B000)`. Full rows use columns 0..59; the final four-fragment row
uses 0..19. Both streaming and fused read this identical weight mapping.
Streaming still imports directly with WRGB and leaves the array input columns
unused. Fused still executes 4096 ordinary WR, 4096 LOCALRD, 4096 BCAST, 1024
MACAB_M16 and 128 RDRES per channel; input/output DQ remains 128KiB/4KiB.
The fused input WR epoch is fragment-major and visits query banks in the order
`0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15`, so adjacent WR requests alternate bank
groups. The old two-fragment/two-bank staging interleave would write the same
bank twice under the new mapping. This reorders identical addressed payloads;
the completion fence still covers every input write before descriptor execution.

Each of the 22 fused row groups begins with an explicit ACTAB request after
all MACs from the previous row complete. Native prerequisites perform PREab
when required. First-batch LOCALRD instructions depend on ACTAB **issue**, not
its 68CK completion, so the native 20/36/52/68CK bank readiness remains visible.
Later batches preserve SRF data dependencies. Refresh can require additional
native reactivation; 22 is the explicit program count, not a promised count of
all dynamically issued activation commands. No timing constant is subtracted
or encoded into the program to realize the layout benefit.

The Down descriptor uses previously reserved word-0 bit 29 for row-colocated
addressing. Its A base is `0x50080`, B base `0x50000`, both fragment strides are
160B, the row period (`split`) is 12 and padding (`jump`) is 128B. In this mode
the offset is `base + f*stride + floor(f/split)*jump`; A selects bank=query,
and B adds output-column*32B. Descriptor length and native CA/decode cost stay
unchanged. Other operators retain their wire words. The simulator executes
the explicitly expanded children, rather than decoding binary words at runtime;
an independent wire-field test verifies that the Down words describe those
physical requests.

The Q exchange has two global stages, `q_exchange_export` and
`q_exchange_import`. The old bank-addressed Q_REMOTE behavior cannot be
represented as WRGB's same-data-to-all-banks broadcast. The new program uses
ordinary addressed WR to a dedicated per-bank `[0x15000,0x15800)` Q-import
region. For each token-layout channel/rank, all **128 ordinary WR** operations
form one `q_import` I/O epoch; a completion barrier follows them, then **128
LOCALRD** operations fill the proper query SRF slots. The program does not
alternate a WR with its matching LOCALRD. Q export and import
retain their original external byte counts. All peer exports finish before
imports start. This adds real row/write/local-read costs relative to the old
targeted-buffer-write interpretation and is an intentional physicalization.

Feature layout owns all 16 queries and eight feature lanes per bank; token
layout owns eight local queries and sixteen pair-major feature lanes per
bank. Token parity, Q/K addresses, V-cache tiling, FFN weight placement,
two-bank staging, O weight residency, and BF16/FP32 boundaries retain the
1108 definitions. The generator's computed per-layer external totals across
16 channels are 6096/5352/5160/4968/4776 KiB for the five variants. Historical
totals serve only as an independent byte-conservation assertion.

The feature rank's feature sets are `[0:64,128:192]` and
`[64:128,192:256]`; each bank owns eight consecutive members of its set.
Token layout uses the pair-major mapping `bank=(feature%128)//8`,
`lane=2*(feature%8)+feature//128`, with all 256 features present in both
parity ranks. Feature Q and K packets contain two token vectors per 32B:
Q load metadata selects the appropriate half into the query slot, while
each MACQK identifies its K `weight_lane_offset` of zero or eight. It still
issues one dot per K token, without assuming double-dot packed throughput.

| Operator | Token layout: matrices per rank | Feature layout: matrices per rank | Scalar MACs, all channels |
|---|---|---|---:|
| QKV | `[8,1024] × [1024,768]` | `[16,1024] × [1024,384]` | 100,663,296 |
| QKᵀ | `[16,256] × [496,256]ᵀ` | `[16,128] × [992,128]ᵀ` | 32,505,856 |
| PV | `[16,496] × [496,256]` | `[16,992] × [992,128]` | 32,505,856 |
| O | `[16,256] × [256,1024]` | `[16,128] × [128,1024]` | 67,108,864 / 33,554,432 |
| Gate/Up | `[16,1024] × [1024,512]` | same | 134,217,728 |
| Down | `[16,4096] × [4096,64]` | same | 67,108,864 |

Token O performs twice the arithmetic of feature O: each KV parity rank
projects its partial PV output before the GPU adds the sixteen rank
outputs. This existing distributed projection choice must not be described
as equal arithmetic across the layout ablation. Both layouts retain a full
`[2048,1024]` BF16 W_O copy per rank, but access only 512KiB/token rank or
256KiB/feature rank. QKV active weights are 1.5MiB/token rank and
0.75MiB/feature rank. Initialization is outside the steady-state program.

The V tail RMW preserves different lanes in the two layouts. Token layout's
last local block contains four prefix tokens, eight new tokens, and four
padding tokens. Feature layout has two final blocks: eight prefix/eight new,
then eight new/eight padding. Gather source regions explicitly select the
right QKV accumulator rows and lanes. Gate/Up exports FP32 complete feature
shards; Down exports FP32 complete output shards, with no cross-rank sum.

Streaming QK reductions export each 64B query-score column immediately as
two RDRES bursts. Fused token QK uses the existing 2KiB output pool in every
bank: FP32 words **0–15** are MAC scratch, while words **16–511** hold one
query's reduced scores for **496 K columns**. Query q's reduced score resides
in bank q. Thus all 496 token-layout columns fit before any score spill;
there is no separate score tile cache. After all reductions complete, a
barrier permits **992 LOCALWR** packets to write the scores to DRAM, two
32-byte packets per column. Each packet gathers eight 4-byte score words
from eight banks. The descriptor completes only after this spill, then
**992 ordinary RD** packets export the score array in `score_export`.

Whole-rank MAC prerequisites use Version2 `ACTAB` (ACT4 x 4), not sixteen
controller-issued split activations. Its spec-defined internal batches
activate one bank per BG at offsets 0, 16, 32 and 48 CK. Bank reads become
ready at 20, 36, 52 and 68 CK; all-bank MAC waits for the last batch. Native
activation histories enforce the rolling FAW window shared with ordinary
ACT, and a nonclosed rank first requires PREab. Single-bank local transfers
retain ordinary ACT1/ACT2 prerequisites. Timing and completion differences
from the legacy simulator include this activation model, MAC completion
tails, correct WRGB/RDRES CL/CWL and turnaround, and addressed Q import.
Reports must explain these differences without applying a constant correction.

RoPE coefficient values are precomputed BF16 sin/cos for the actual model
position IDs. The generated operations cover every pair, input and output,
but this is a timing/layout model and does not execute tensor numerics.
Version2 has **one independent RoPE PE per bank**, each with two input slots
and a two-entry output FIFO. Its validity records describe vector views in
the existing SRF/output pools, not additional unbounded SRAM. RoPE follows
completed QKV cast/spills and precedes Q residency, so these views do not
overlap live GEMV or QK inputs. Each rank/layer executes **384 LOCALRD, 256
LOCALWR and 256 ROPE_EXEC** operations. Every EXEC rotates one 32-byte vector
(eight pairs): the pair II is 1 CK, vector II is **8 CK**, and the final pair
completes after **11 CK**. The spec applies this pipeline constraint per bank;
the resource model enforces input lifetime and FIFO capacity/order without
locking the PE until completion.

The double schedule loads coefficients and Q into slot 0, issues Q EXEC,
then loads K into slot 1 while Q executes. K load and K EXEC have an issue
dependency on Q EXEC, while K EXEC also waits for its own input/coefficient
completion. Q and K stores consume their completed FIFO entries in order.
No execution-completion edge serializes different banks. Fused ROPE_RUN
expands this same sequence through the private-CA in-order FIFO32. In-order
dispatch does not add completion edges between otherwise independent banks.

Feature W_O full residency is arranged as sixteen rotated feature-shard
slots, each 16KiB per bank, placing the active shard in slot zero. Token W_O
uses eight rotated head slots, each 32KiB per bank. Both occupy the same
256KiB per-bank reservation and preserve their active MAC addresses. This
is a layout/capacity contract for preinstalled weights; the simulator does
not execute initialization or derive a numerical accuracy claim from it.
