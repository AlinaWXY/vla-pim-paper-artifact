"""Export continuous dual-stream GR00T latency, cache, and overlap evidence."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def csv_file(path, rows):
    if not rows:
        raise ValueError(f"empty data: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def overlap(a, b):
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def trace_overlap(path):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [v for v in events if v.get("cat") == "kernel"]
    streams = {}
    for k in kernels:
        streams.setdefault(k["args"]["stream"], []).append(k)
    ae_streams = {k["args"]["stream"] for k in kernels if "normal" in k["name"].lower()}
    if len(ae_streams) != 1:
        raise ValueError(f"cannot identify the AE stream by its fresh-noise kernels: {ae_streams}")
    ae = ae_streams.pop()
    vlm = max((s for s in streams if s != ae), key=lambda s: sum(k["dur"] for k in streams[s]))
    start = min(k["ts"] for k in streams[ae])
    end = max(k["ts"] + k["dur"] for k in streams[ae])
    intervals = {}
    for stream in (ae, vlm):
        intervals[stream] = merge([(max(start, k["ts"]), min(end, k["ts"] + k["dur"]))
                                   for k in streams[stream] if k["ts"] < end and k["ts"] + k["dur"] > start])
    ae_busy = sum(b - a for a, b in intervals[ae])
    vlm_busy = sum(b - a for a, b in intervals[vlm])
    intersection = overlap(intervals[ae], intervals[vlm])
    return dict(ae_cupti_stream=ae, vlm_cupti_stream=vlm,
                ae_kernel_count=len(streams[ae]), vlm_kernel_count=len(streams[vlm]),
                ae_window_us=end - start, ae_busy_us=ae_busy, vlm_busy_us=vlm_busy,
                kernel_overlap_us=intersection, overlap_fraction_of_ae_busy=intersection / ae_busy,
                actual_kernel_overlap_observed=intersection > 0,
                scope="Separate profiler trace, including trace warmup; overlap evidence only"), [
        dict(role="ae" if k["args"]["stream"] == ae else "vlm", stream=k["args"]["stream"],
             name=k["name"], start_us=k["ts"] - start, duration_us=k["dur"])
        for k in sorted(kernels, key=lambda k: k["ts"]) if k["args"]["stream"] in (ae, vlm)
    ]


def analyze(root):
    report = json.loads((root / "results/dual-stream-summary.json").read_text())
    if not report.get("completed_at_utc"):
        raise ValueError("unfinished dual-stream run")
    summaries, ae_launches, refreshes, trace_kernels = [], [], [], []
    validation = dict(vit_input_shapes=report["vit_input_shapes"], cache_shapes=report["cache_shapes"],
                      cache_bytes=report["kv_bytes"], cache_correctness=report["cache_correctness"], modes={})
    for backend, mode in report["modes"].items():
        for condition, row in mode["results"].items():
            p50 = lambda name: row[name]["p50"] if row[name] else None
            measured_launches = [v for run in row["blocks"] for v in run["ae_launches"]
                                 if run["window_start_ms"] < v["end_ms"] <= run["window_end_ms"]]
            summaries.append(dict(
                backend=backend, condition=condition, ae_chunk_hz=row["ae_chunk_hz"],
                vlm_refresh_hz=row["vlm_refresh_hz"], ae_latency_p50_ms=p50("ae_latency_ms"),
                ae_latency_p90_ms=row["ae_latency_ms"]["p90"], ae_period_mean_ms=row["period_ms"]["mean"],
                ae_period_p50_ms=p50("period_ms"), ae_idle_p50_ms=p50("ae_idle_ms"),
                producer_vlm_p50_ms=p50("vlm_latency_ms"), kv_projection_p50_ms=p50("projection_latency_ms"),
                full_refresh_p50_ms=p50("refresh_latency_ms"),
                cache_age_p50_ms=p50("cache_age_ms") if condition == "with_vlm" else None,
                cache_age_p90_ms=row["cache_age_ms"]["p90"] if condition == "with_vlm" else None,
                cache_age_at_finish_p50_ms=statistics.median(v["cache_age_ms"] + v["latency_ms"]
                    for v in measured_launches) if condition == "with_vlm" else None,
                cache_reuse_fraction=row["cache_reuse_fraction"], gpu_start_races=row["gpu_start_races"]))
            for index, run in enumerate(row["blocks"]):
                producer_starts = {v["version"]: v["start_ms"] for v in run["producer_refreshes"]}
                for launch in run["ae_launches"]:
                    measured = run["window_start_ms"] < launch["end_ms"] <= run["window_end_ms"]
                    source = producer_starts.get(launch["cache_version"])
                    ae_launches.append(dict(backend=backend, condition=condition, block=index,
                                            measured=measured, **launch,
                                            cache_age_at_finish_ms=launch["cache_age_ms"] + launch["latency_ms"],
                                            vlm_start_age_at_ae_start_ms=launch["start_ms"] - source if source is not None else None,
                                            vlm_start_age_at_ae_finish_ms=launch["end_ms"] - source if source is not None else None))
                refreshes.extend(dict(backend=backend, condition=condition, block=index,
                                      in_measurement_window=run["window_start_ms"] < v["end_ms"] <= run["window_end_ms"],
                                      **v) for v in run["producer_refreshes"])
        evidence, launches = trace_overlap(root / "results" / f"dual-{backend}.trace.json")
        validation["modes"][backend] = dict(no_recompilation=mode["no_recompilation_in_timing"],
                                            overlap=evidence,
                                            latest_observed_checks=all(b["latest_observed_checks_passed"]
                                                for c in mode["results"].values() for b in c["blocks"]))
        trace_kernels.extend(dict(backend=backend, **row) for row in launches)
    csv_file(root / "dual-stream-latencies.csv", summaries)
    csv_file(root / "ae-cache-timeline.csv", ae_launches)
    csv_file(root / "vlm-refresh-timeline.csv", refreshes)
    csv_file(root / "dual-trace-kernels.csv", trace_kernels)
    (root / "dual-stream-validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in validation.items() if key != "cache_shapes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
