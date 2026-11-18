# Fused execution with calibrated VLM kernels

This execution model follows the user's latest 2026-11-09 scheduling clarification.
`execution_model=fused_kernel_windows` applies only to concurrent fused f/g.
Continuous PIM work keeps the current VLM kernel running across PV→O and
QKV→local RoPE. These boundaries deliver the next coarse command as soon as
native timing permits; they do not wait for the current VLM kernel to finish.
This supersedes the earlier rule that joined after every descriptor.
The older `native_dram_requests` path remains an explicit, separate model.

Each descriptor stage follows these boundaries across all sixteen channels:

1. Upload inputs and deliver the coarse command using native PIM-side DRAM
   requests, prerequisites and timing. Wait for every channel's input DQ and
   descriptor delivery/decode to complete.
2. Run the descriptor's private FIFO32 program while the GPU executes VLM
   kernels in dependency order. The GPU starts another kernel whenever the
   current one finishes and the private computation is still active; there is
   no kernel-size fitting or preemption.
3. Finish the entire private program, including output writes to DRAM. At a
   continuous boundary, retire the preceding stage's completion barriers and
   deliver the next descriptor while the already-running VLM kernel continues.
   At a payload transfer or AE GPU boundary, close admission of new VLM kernels
   and wait for the current kernel's exact calibrated completion event.
4. Export all outputs using native ordinary DRAM reads and execute the next AE
   GPU helper/nonlinear block with VLM admission closed. Input payload uploads
   and pure transfer/setup stages also remain exclusive. Only a descriptor-only
   continuation may overlap an existing VLM kernel.

A continuation requires adjacent PIM events in the same layer and denoising
step, with one descriptor in each stage on all sixteen channels. Only BARRIER
operations may follow the outgoing descriptor or precede the incoming one.
The descriptor bodies must still pass the private-command checks. In the current
f/g programs this admits PV→O, and in g also QKV→local RoPE. RoPE export/import
and Q exchange retain their exclusive handoff behavior. A program name alone
does not grant a continuation, and no payload request is skipped.

The outgoing stage must fully finish before its controller cursor is replaced.
The next descriptor still consumes native shared CA time and decode latency
(21 CK + 1 CK under the current spec), including any prerequisite/refresh delay.
All channels must finish delivery before the next private compute phase begins.
An active VLM kernel retains its original start and completion timestamps across
this descriptor-only launch. If it finishes during launch, a new VLM kernel
starts only when the next compute phase opens.

Native zero-time barriers and descriptor completion are settled before admitting
a kernel at the same cycle. This prevents a kernel from starting just as PIM
results become ready. The native resource scoreboard is preserved across stages,
including Q residency. Native refresh, FIFO dependencies, CA/DQ constraints and
completion tails continue to apply to PIM; only the host's phase ownership gates
issue. This path does not use FIFO-head I/O arbitration against GPU requests.

O, Gate/Up and Down now actually spill results to a bounded DRAM region before
export; O finishes both tiles before returning data. See
[program layout](fused-kernel-programs.md). These additional writes and ordinary
read prerequisites are included in the simulation. Earlier RDRES-based stage
durations must not be reused for this model.

## Calibrated GPU abstraction

`tools/export_calibrated_vlm.py` reads the existing accepted VLM-solo calibration
and checks its hashes, identities, clocks, byte totals and complete coverage. It
exports the observed integer-cycle duration of every work unit, including four
memory units: 2357 units totaling 113958594 cycles / 106.893161172 ms at 938 ps.
The old calibration's binary and source identity are retained as provenance;
this is reuse of its durations, not a newly calibrated binary.

The frontend advances VLM by these completion events. It does not generate the
619790815 GPU read/write transactions and does not claim to model their current
rank0 row state or command-level interference. Their bytes are retained as
`vlm_accounted_*` metadata. `gpu_*_bytes` and native GPU command counts are zero.
This is a deliberate model change under the requested exclusive handoff rules,
not a claim that the detailed concurrent DRAM trajectory is unchanged.

During a descriptor-only continuation, calibrated VLM time advances continuously.
The model does not add the small shared-CA contention delay of that coarse
command to the VLM kernel duration. The PIM-side command still pays its full
native CA/decode cost. This is an explicit limitation of using calibrated kernel
durations instead of GPU request-level simulation.

`phases` distinguishes payload upload, descriptor-only launch, private compute,
zero-time continuation, join, download and transfer, with per-channel ready
times; `ae_gpu` records exclusive GPU helper intervals. `AE_VLM_join_us` measures
waiting **before exclusive handoff**, including payload transfers, AE GPU work
and setup boundaries. `AE_GPU_wait_us` retains its older meaning and is zero
here. Stage duration includes its join wait; subtract the explicit join when
reporting native PIM phase time, and do not label that value an AE-solo result.
Current raw output uses schema 3 and
`window_policy: continuous_pim_descriptors`. `PIM_descriptor_continuations`
counts safe boundaries: the full current f schedule has 180 and g has 360.
These identifiers distinguish new runs from the historical every-descriptor
join results below.

## Execution and audit

Generate f/g with `programs/gen_pim_program.py --execution-model
fused_kernel_windows --compact --output programs/generated/fused-kernel`.
`experiments/run_fused_kernels.py --prepare-only` freezes the binary, runtime,
programs, schedules, VLM input and calibrated-duration ledger. Run each case with
the same tool's `--case f` / `--case g`, then audit with
`tools/audit_fused_kernels.py --run <campaign>`.

The audit verifies native PIM byte conservation and refresh, all 18 layers × 10
steps, exact calibrated kernel durations, zero native GPU transactions, no
kernel overlap with payload handoff/nonlinear intervals, and full phase/timestamp
identities. A descriptor-only launch can overlap an already-running VLM kernel;
it cannot admit a new one. Native completion barriers, all-channel readiness and
exact kernel timestamps still constrain continuous PIM boundaries.

## Completed continuous execution run, 2026-11-09

Full f/g runs each cover 18 layers × 10 steps and all 2357 calibrated VLM units.
Source `924141c285811ce26bde6e58bcf011ff3b435e88`, native binary SHA-256
`9cb7f3b806a1ec69b7277edb65cb2a1f0a3f7984294cf4c571a2e20e5a71c248`.
The full array `1983025` and audit `1983030` passed. Remote validation passed
24 native frontend fixtures and 91 Python tests; ten additional auditor unit
tests subsequently passed locally, bringing that unit-test suite to 22.

| Metric / ms unless stated | f / GPU RoPE | g / local RoPE |
|---|---:|---:|
| Total | 137.737998608 | 131.832380624 |
| AE end | 124.515315582 | 118.609697598 |
| Explicit VLM join | 59.276970970 | 59.036419932 |
| Native PIM phases, excluding join | 41.432692532 | 41.409574584 |
| AE GPU helpers | 23.805652080 | 18.163703082 |
| VLM tail | 13.222683026 | 13.222683026 |
| Continuations overlapping an active VLM kernel | 180 | 360 |
| Wall seconds | 478.752 | 477.272 |

All private computation and continuation launches overlap VLM execution in
these runs. The measured g total decomposes into 106.893161172 ms calibrated
VLM + 18.163703082 ms AE GPU work + 6.775516370 ms exclusive PIM transfers.
Join time is already covered by VLM time and must not be added again.
Compared with the historical policy below, g finishes AE 15.371028512 ms
earlier, but its VLM tail grows by 13.222683026 ms; total latency therefore
falls by 2.148345486 ms (1.60347%). For f, AE finishes 11.756668756 ms earlier
and total latency falls by 0.026616688 ms (0.01932%).

The [saved audit](../reference/fused-kernel-continuous-e2e/summary.json) and
[policy comparison](../reference/fused-kernel-continuous-e2e/policy-comparison.json)
bind both runs to identical workload/calibration input hashes, non-path
configuration values and runtime libraries. Both new cases ran on einstein;
the historical g ran on infinite-frontier, so wall-time differences are not a
controlled same-host speedup measurement. Each case ran once; repeat consistency
is not claimed. The result archive SHA-256 is
`c242bb798f1b4ecf000a350cfc8801ebdbba9d730b32fb333a8ee76d1b25214f`.

## Historical run with a join after every descriptor, 2026-11-09

The following measurements used the superseded every-descriptor join policy,
including joins at QKV→local RoPE and PV→O. They are retained unchanged as a
historical baseline and are not measurements of the continuation policy above.

Source `0046e8bfd29fb272a98fd945d135e5ac967d1a1c`, native binary SHA-256
`76d1b07e6c1d89839abc438a0a5079aa4aef6cee24103d50f00cb92e65d9176b`.
Validation job `1982834`, full f/g array `1982852`, audit `1982856` all completed
successfully. Validation includes 91 Python tests, 13 new frontend fixtures,
224 native spec checks, 12 sequencer regressions, five refresh regressions,
resource checks and the original frontend smoke.

| Metric / ms unless stated | f / GPU RoPE | g / local RoPE |
|---|---:|---:|
| Total | 137.764615296 | 133.980726110 |
| AE end | 136.271984338 | 133.980726110 |
| Explicit VLM join | 71.018671122 | 74.431107618 |
| Native PIM phases, excluding join | 41.447661136 | 41.385915410 |
| AE GPU helpers | 23.805652080 | 18.163703082 |
| VLM tail | 1.492630958 | 0 |
| Calibrated VLM duration sum | 106.893161172 | 106.893161172 |
| Native commands | 131946000 | 133220480 |
| Wall seconds | 484.587571230 | 487.233322663 |

This model change removes the 619790815 GPU transactions per case, but retains
roughly 132 million PIM/refresh commands. The wall times were observed on
einstein (f) and infinite-frontier (g); they are not a controlled same-host speedup
benchmark. In that historical run every descriptor handoff waited for whole
kernels, and O/FFN outputs already spilled to DRAM before export.

The [historical independent audit](../reference/fused-kernel-e2e/summary.json) verifies full
coverage, exact calibrated durations, zero native GPU transactions, exclusive
handoffs, FIFO order, byte conservation and source/runtime/input hashes. Raw
results and binaries are retained in the server campaign
`/artifact-source/pim_vla_fused_kernel_20261109/results/fused-kernel-e2e-20261109`.
The local result archive SHA-256 is
`a78a41486ea1f6e55dd697705144f95c1bc06babe1112eab57018ed1ea812cb3`.
