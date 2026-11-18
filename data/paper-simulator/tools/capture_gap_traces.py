"""Capture native AE solo slices, with a byte-for-byte trace control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, sha256, write_json
from experiments.config import case_config, variant_name

COMPARISON_SLICES = ((0, 0), (9, 17))
SLICES = dict.fromkeys("cdefg", COMPARISON_SLICES)
REPEATS = 180


def native(binary, config, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "input-config.json"
    write_json(path, config)
    with (output / "native.log").open("wb") as log:
        run = subprocess.run([str(binary), "--config", str(path), "--output", str(output)],
                             stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f"native exit {run.returncode}: {(output / 'native.log').read_text()}")


def generate(variant, directory, model, *, mc_control=False, batch_output=None):
    protocol = ["--mc-control"] if mc_control else []
    if batch_output is not None:
        protocol.append("--batch-output" if batch_output else "--no-batch-output")
    subprocess.run([sys.executable, str(ROOT / "programs/gen_pim_program.py"),
                    "--variant", variant, "--compact", "--output", str(directory),
                    "--execution-model", model, *protocol], check=True)
    return directory / f"{variant}.json"


def capture(case, binary, output, ledger, slices, spec, mac_freq_mult, mc_control, batch_output):
    config = case_config(case, mac_freq_mult=mac_freq_mult, run_kind="ae_solo")
    variant = variant_name(config)
    schedule_path = ROOT / f"input-data/ae-{variant}.json"
    program = generate(variant, output / "programs", "native_dram_requests", mc_control=mc_control,
                       batch_output=batch_output)
    schedule = read_json(schedule_path)
    vlm = ROOT / "input-data/vlm.json"
    config.update(spec_path=str(spec), program_path=str(program), vlm_path=str(vlm))
    for step, layer in slices:
        directory = output / f"{case}-s{step}-l{layer}"
        schedule_slice = directory / "ae-slice.json"
        events = [e for e in schedule["events"] if (e["step"], e["layer"]) == (step, layer)]
        if not events:
            raise ValueError(f"empty AE slice: {case}, {step}, {layer}")
        write_json(schedule_slice, {**schedule, "events": events})
        sliced = dict(config, ae_schedule_path=str(schedule_slice))
        native(binary, sliced, directory / "off")
        native(binary, dict(sliced, trace_ae_dq=True), directory / "on")
        before, after = ((directory / mode / "raw.json").read_bytes() for mode in ("off", "on"))
        if before != after:
            raise ValueError(f"raw.json changed with trace: {directory}")
        write_json(directory / "verification.json",
                   dict(byte_identical=True, raw_sha256=sha256(directory / "on/raw.json"),
                        schedule_sha256=sha256(schedule_slice), program_sha256=sha256(program),
                        spec_sha256=sha256(spec), binary_sha256=sha256(binary),
                        trace_sha256={name: sha256(directory / "on" / name) for name in
                                      ("ca-commands.csv", "dq-bursts.csv", "dq-epochs.csv",
                                       "dq-timeline.json")}))
        print(f"{case} step {step} layer {layer}: raw.json identical", flush=True)

    if case in "fg" and ledger:
        program = generate(variant, output / "fused-programs", "fused_kernel_windows")
        calibration = read_json(ledger)
        fused = dict(config, program_path=str(program), ae_schedule_path=str(schedule_path),
                     run_kind="combined", execution_model="fused_kernel_windows",
                     calibration=calibration["calibration_parameters"], calibrated_vlm_path=str(ledger))
        # The existing fused frontend reads the embedded spec.
        fused["spec"] = read_json(spec)
        native(binary, fused, output / f"{case}-fused")
        print(f"{case}: full fused kernel windows completed", flush=True)


def repeat_fused_g(binary, output, step, layer):
    """Use the existing g windows frontend on a repeated measured schedule slice."""
    config = read_json(output / "g-fused/input-config.json")
    label = f"g-s{step}-l{layer}"
    schedule = read_json(output / label / "ae-slice.json")
    # Keep the original layer/address offset. The end of each slice is an AE
    # GPU block, so no PIM descriptor continuation crosses repetitions.
    events = schedule["events"]
    if events[-1]["kind"] != "gpu":
        raise ValueError(f"g slice has no final GPU boundary: {label}")
    directory = output / f"{label}-fused-repeated"
    directory.mkdir(parents=True)
    repeated = directory / "ae-repeated.json"
    write_json(repeated, {**schedule, "events": events * REPEATS})
    native(binary, dict(config, ae_schedule_path=str(repeated)), directory)
    print(f"{label}: {REPEATS} repeats using fused kernel windows completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mac-freq-mult", type=int, choices=(1, 2, 4), default=4,
                        help="MAC frequency for newly captured traces (default: 4)")
    parser.add_argument("--mc-control", action="store_true", help="use the rank-mode CA protocol")
    parser.add_argument("--four-batch-actab", action="store_true",
                        help="activate four groups of four banks per data-row ACTAB under MC control")
    parser.add_argument("--batch-output", action=argparse.BooleanOptionalAction, default=None,
                        help="batch result exports (default for MC-controlled streaming)")
    parser.add_argument("--cases", choices=list("bcdefg"), nargs="+", default=list("cdefg"))
    parser.add_argument("--slice", type=int, nargs=2, metavar=("STEP", "LAYER"),
                        help="capture only one step/layer for the selected cases")
    parser.add_argument("--fused-ledger", type=Path, help="also rerun the existing full f/g model")
    parser.add_argument("--repeat-fused-g", action="store_true",
                        help="reuse prepared g-fused config and repeat each g slice using fused windows")
    args = parser.parse_args()
    if args.four_batch_actab and not args.mc_control:
        parser.error("--four-batch-actab requires --mc-control")
    if args.batch_output and (not args.mc_control or any(case in "fg" for case in args.cases)):
        parser.error("--batch-output requires --mc-control and streaming cases b/c/d/e")
    if args.mc_control and (args.fused_ledger or args.repeat_fused_g):
        parser.error("--mc-control uses the native matched-slice paths")
    output = args.output.resolve()
    if args.repeat_fused_g:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(repeat_fused_g, args.binary.resolve(), output, step, layer)
                       for step, layer in SLICES["g"]]
            for future in futures:
                future.result()
        return
    if len(set(args.cases)) != len(args.cases):
        parser.error("--cases must be unique")
    slices = (tuple(args.slice),) if args.slice is not None else COMPARISON_SLICES
    for case in args.cases:
        for step, layer in slices:
            destination = output / f"{case}-s{step}-l{layer}"
            if destination.exists():
                raise FileExistsError(destination)
    ledger = args.fused_ledger.resolve() if args.fused_ledger else None
    spec = ROOT / "reference/spec/spec.json"
    if args.mac_freq_mult != 4 or args.mc_control:
        sys.path.insert(0, str(ROOT / "ramulator2/python"))
        from ramulator.dram.lpddr5x_pim import build_spec

        spec = output / f"spec-k{args.mac_freq_mult}.json"
        write_json(spec, build_spec(args.mac_freq_mult, mc_control=args.mc_control,
                                    four_batch_actab=args.four_batch_actab))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(capture, case, args.binary.resolve(), output, ledger, slices,
                               spec, args.mac_freq_mult, args.mc_control, args.batch_output)
                   for case in args.cases if case != "b"]
        for future in futures:
            future.result()
    if "b" in args.cases:
        # b and c share the feature-wise program. Generate b after c's readers
        # have finished to avoid writing the same program file concurrently.
        capture("b", args.binary.resolve(), output, ledger, slices, spec,
                args.mac_freq_mult, args.mc_control, args.batch_output)


if __name__ == "__main__":
    main()
