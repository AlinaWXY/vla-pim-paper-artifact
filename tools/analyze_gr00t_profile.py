"""Export per-launch CUDA timings and isolated AE operator timings from a Thor profile."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no measurements for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    return dict(samples=len(values), mean_us=statistics.mean(values), p50_us=statistics.median(values),
                p90_us=statistics.quantiles(values, n=10, method="inclusive")[8] if len(values) > 1 else values[0])


def analyze(root, trace_directory="640x480"):
    launches = []
    by_name = defaultdict(list)
    by_name_call = defaultdict(lambda: defaultdict(float))
    by_position = defaultdict(list)
    operator_metadata = {}
    validation = {"compiled_trace_directory": trace_directory}
    phase_report = json.loads((root / trace_directory / "profile-summary.json").read_text())
    for phase in ("vlm", "ae"):
        traces = sorted((root / trace_directory).glob(f"{phase}-compile*.trace.json"))
        if len(traces) != 5:
            raise ValueError(f"expected five compiled {phase} traces, found {len(traces)}")
        sequences = []
        kernel_sums, gpu_spans = [], []
        for repeat, path in enumerate(traces):
            events = json.loads(path.read_text())["traceEvents"]
            kernels = sorted((event for event in events if event.get("cat") == "kernel"),
                             key=lambda event: event["ts"])
            if not kernels:
                raise ValueError(f"no CUDA kernels in {path}")
            sequences.append([event["name"] for event in kernels])
            kernel_sums.append(sum(event["dur"] for event in kernels) / 1000)
            gpu_spans.append((max(event["ts"] + event["dur"] for event in kernels)
                              - min(event["ts"] for event in kernels)) / 1000)
            for ordinal, event in enumerate(kernels):
                duration = event["dur"]
                if duration <= 0:
                    raise ValueError("CUDA kernel duration must be positive")
                args = event.get("args", {})
                grid = json.dumps(args.get("grid", []))
                block = json.dumps(args.get("block", []))
                launches.append(dict(phase=phase, repeat=repeat, ordinal=ordinal, name=event["name"],
                                     duration_us=duration, start_us=event["ts"], stream=args.get("stream", ""),
                                     grid=grid, block=block))
                by_name[phase, event["name"], grid, block].append(duration)
                by_name_call[phase, event["name"], grid, block][repeat] += duration
                by_position[phase, ordinal, event["name"], grid, block].append(duration)
        validation[phase] = dict(kernel_counts=[len(sequence) for sequence in sequences],
                                 identical_kernel_sequence=all(sequence == sequences[0] for sequence in sequences),
                                 unprofiled_cuda_p50_ms=phase_report["phases"][phase]["compile"]["cuda_ms"]["p50"],
                                 profiled_kernel_sum_p50_ms=statistics.median(kernel_sums),
                                 profiled_gpu_span_p50_ms=statistics.median(gpu_spans))
        detailed = json.loads((root / "640x480" / f"{phase}-compile-0.trace.json").read_text())["traceEvents"]
        detailed_kernels = sorted((event for event in detailed if event.get("cat") == "kernel"),
                                  key=lambda event: event["ts"])
        same_sequence = [event["name"] for event in detailed_kernels] == sequences[0]
        validation[phase]["detailed_metadata_sequence_match"] = same_sequence
        if same_sequence:
            cpu_by_id = {event.get("args", {}).get("External id"): event for event in detailed
                         if event.get("cat") == "cpu_op" and "External id" in event.get("args", {})}
            for ordinal, kernel in enumerate(detailed_kernels):
                cpu = cpu_by_id.get(kernel.get("args", {}).get("External id"), {})
                args = cpu.get("args", {})
                operator_metadata[phase, ordinal] = dict(
                    cpu_operator=cpu.get("name", ""),
                    input_shapes=json.dumps(args.get("Input Dims", [])),
                    input_dtypes=json.dumps(args.get("Input type", [])),
                )
    write_csv(root / "compiled-kernel-launches.csv", launches)
    write_csv(root / "compiled-kernels-by-name.csv", [
        dict(phase=phase, name=name, grid=grid, block=block, **summary(values),
             launches_per_call=len(values) / 5,
             call_total_p50_us=statistics.median([
                 by_name_call[phase, name, grid, block].get(repeat, 0) for repeat in range(5)
             ]))
        for (phase, name, grid, block), values in by_name.items()
    ])
    write_csv(root / "compiled-kernels-by-position.csv", [
        dict(phase=phase, ordinal=ordinal, name=name, grid=grid, block=block, **summary(values),
             **operator_metadata.get((phase, ordinal), dict(cpu_operator="", input_shapes="[]", input_dtypes="[]")))
        for (phase, ordinal, name, grid, block), values in by_position.items()
    ])

    operators = json.loads((root / "ae-nonlinear.json").read_text())
    operator_rows = []
    isolated_launches = []
    for name, operator in operators["consumers"].items():
        for cache, measurement in (("cold_L2", operator), ("warm_L2", operator["warm"])):
            for backend in ("eager", "compile"):
                value = measurement[backend]
                operator_rows.append(dict(
                    operator=name, cache=cache, backend=backend, inputs=json.dumps(operator["inputs"]),
                    kernel_count_each_call=json.dumps(value["kernel_count_each_call"]),
                    device_p50_us=value["kernel"]["p50_us"], device_p95_us=value["kernel"]["p95_us"],
                    call_event_p50_us=statistics.median(measurement["event_samples"][backend]["event"]),
                    samples=len(value["kernel"]["samples_us"]),
                ))
                for repeat, kernels in enumerate(value["kernel_launches"]):
                    for ordinal, kernel in enumerate(kernels):
                        isolated_launches.append(dict(operator=name, cache=cache, backend=backend,
                                                       repeat=repeat, ordinal=ordinal, **kernel))
    write_csv(root / "ae-nonlinear-latencies.csv", operator_rows)
    write_csv(root / "ae-nonlinear-summary.csv", [
        dict(operator=name, inputs=json.dumps(operator["inputs"]),
             cold_eager_us=operator["eager"]["kernel"]["p50_us"],
             cold_compile_us=operator["compile"]["kernel"]["p50_us"],
             warm_eager_us=operator["warm"]["eager"]["kernel"]["p50_us"],
             warm_compile_us=operator["warm"]["compile"]["kernel"]["p50_us"])
        for name, operator in operators["consumers"].items()
    ])
    write_csv(root / "ae-nonlinear-kernel-launches.csv", isolated_launches)
    validation["isolated_operator_count"] = len(operators["consumers"])
    validation["isolated_launch_rows"] = len(isolated_launches)
    (root / "kernel-validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--trace-directory", default="640x480")
    args = parser.parse_args()
    analyze(args.root, args.trace_directory)
