"""Compare two native binaries on identical frozen step-0/layer-0 AE inputs."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline-binary", "fifo-binary", "program-dir", "spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-commit", required=True)
    args = parser.parse_args()
    require(args.repeats >= 2, "at least two repetitions are required for determinism checks")
    require(sys.platform.startswith("linux"), "this benchmark requires Linux GNU /usr/bin/time")
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), "use a fresh output directory")
    require(Path("/usr/bin/time").is_file(), "GNU /usr/bin/time is required")
    # Both executables and all repetitions use the same allocated logical CPU.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    sys.path.insert(0, str(source))
    from experiments.common import read_json, sha256, write_csv, write_json, object_sha256
    from experiments.native import runtime_libraries
    from experiments.run_single_layer import prepare, summarize

    binaries = {"baseline": args.baseline_binary.resolve(), "fifo": args.fifo_binary.resolve()}
    binary_hashes = {name: sha256(path) for name, path in binaries.items()}
    require(binary_hashes["baseline"] != binary_hashes["fifo"], "the two binaries are identical")
    libraries = {name: runtime_libraries(binary) for name, binary in binaries.items()}
    require(libraries["baseline"] == libraries["fifo"], "both binaries must resolve the same library path and hash")
    shared_libraries = libraries["baseline"]
    run_env = os.environ.copy()  # One unchanged LD_LIBRARY_PATH for every timed process.
    spec_path = args.spec.resolve()
    spec = read_json(spec_path)
    configs = prepare(output, input_dir=source / "input-data", program_dir=args.program_dir.resolve(), spec=spec)
    require(len(configs) == 5, "expected exactly five AE variants")
    for cfg in configs.values():
        require(cfg["run_kind"] == "ae_solo" and cfg["mode"] == "serial" and
                "vlm_path" not in cfg and "calibration" not in cfg, "single-layer scope violated")
    inputs = {spec_path}
    for cfg in configs.values():
        inputs.update(Path(cfg[key]) for key in ("program_path", "ae_schedule_path"))
    inputs.update((source / "input-data").glob("ae-*.json"))
    inputs.update((source / "input-data/measured").glob("*.json"))
    input_hashes = {str(path): sha256(path) for path in sorted(inputs)}
    source_files = [*sorted((source / "experiments").glob("*.py")), Path(__file__).resolve()]
    driver_hashes = {str(path): sha256(path) for path in source_files}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True)
    require(not dirty, "commit tracked source changes before benchmarking")
    manifest = dict(scope="step 0 / layer 0 AE only", no_vlm=True, no_full_campaign=True,
                    source=str(source), source_commit=revision, baseline_commit=args.baseline_commit,
                    source_status_porcelain=dirty, hostname=platform.node(),
                    binary_paths={k: str(v) for k, v in binaries.items()}, binary_sha256=binary_hashes,
                    runtime_libraries=shared_libraries, inputs=input_hashes, driver_sha256=driver_hashes,
                    spec_sha256=object_sha256(spec), repetitions=args.repeats,
                    LD_LIBRARY_PATH=run_env.get("LD_LIBRARY_PATH", ""), platform=platform.platform(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    timing_scope="native subprocess including startup and input parsing; excludes compilation, generation and Python validation; monotonic additionally includes GNU time launcher",
                    execution_order="variant then repetition, baseline/fifo; reverse pair order on odd repetitions")
    write_json(output / "manifest.json", manifest)
    records, all_stages, all_gpu, paired, execution_order = [], [], [], {}, []
    time_format = '{"wall_seconds":%e,"user_seconds":%U,"sys_seconds":%S,"maxrss_kib":%M,"exit_status":%x}'

    def stable_runtime():
        for name, binary in binaries.items():
            require(sha256(binary) == binary_hashes[name], f"{name} binary changed")
            require(runtime_libraries(binary) == shared_libraries, f"{name} runtime library changed")

    for variant, cfg in configs.items():
        program, schedule = read_json(cfg["program_path"]), read_json(cfg["ae_schedule_path"])
        paired[variant] = {"baseline": [], "fifo": []}
        for repetition in range(args.repeats):
            order = ("baseline", "fifo") if repetition % 2 == 0 else ("fifo", "baseline")
            for version in order:
                stable_runtime()
                for key in ("program_path", "ae_schedule_path"):
                    require(sha256(cfg[key]) == input_hashes[cfg[key]], f"{key} changed")
                destination = output / "native" / variant / version / f"repeat-{repetition}"
                destination.mkdir(parents=True, exist_ok=False)
                config_path = destination / "input-config.json"
                write_json(config_path, cfg)
                command = ["/usr/bin/time", "-f", time_format, "-o", str(destination / "time.json"),
                           str(binaries[version]), "--config", str(config_path), "--output", str(destination)]
                print(f"RUN {variant} {version} repeat={repetition}", flush=True)
                with (destination / "native.log").open("w", encoding="utf-8") as log:
                    started = time.monotonic()
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=run_env, cwd=source)
                    monotonic_seconds = time.monotonic() - started
                timing = read_json(destination / "time.json")
                timing.update(monotonic_seconds=monotonic_seconds,
                              cpu_seconds=timing["user_seconds"] + timing["sys_seconds"])
                write_json(destination / "timing.json", timing)
                provenance = dict(version=version, variant=variant, repetition=repetition, command=command,
                                  binary_sha256=binary_hashes[version], runtime_libraries=shared_libraries,
                                  config_sha256=sha256(config_path), spec_sha256=object_sha256(spec),
                                  inputs={key: input_hashes[cfg[key]] for key in ("program_path", "ae_schedule_path")},
                                  manifest_sha256=sha256(output / "manifest.json"), returncode=process.returncode,
                                  native_log_sha256=sha256(destination / "native.log"),
                                  time_sha256=sha256(destination / "time.json"), timing_sha256=sha256(destination / "timing.json"))
                if (destination / "raw.json").is_file():
                    provenance["raw_sha256"] = sha256(destination / "raw.json")
                write_json(destination / "provenance.json", provenance)
                require(process.returncode == 0 and timing["exit_status"] == 0, f"native run failed: {destination}")
                stable_runtime()
                raw = read_json(destination / "raw.json")
                row, stages, gpu = summarize(variant, raw, schedule, program, spec)
                if version == "fifo":
                    require(row["PIM_out_of_order_issues"] == row["PIM_prefetched_prerequisites"] == 0,
                            f"FIFO emitted an out-of-order instruction or prefetched prerequisite: {variant}")
                row.update(version=version, repetition=repetition, **timing,
                           raw_sha256=provenance["raw_sha256"], config_sha256=provenance["config_sha256"],
                           provenance_sha256=sha256(destination / "provenance.json"))
                print(f"DONE {variant} {version}: latency={row['total_us']:.6f} us wall={timing['monotonic_seconds']:.3f} s", flush=True)
                records.append(row)
                paired[variant][version].append(row)
                all_stages.extend(dict(s, version=version, repetition=repetition) for s in stages)
                all_gpu.extend(dict(s, version=version, repetition=repetition) for s in gpu)
                execution_order.append(dict(variant=variant, version=version, repetition=repetition))
                write_csv(output / "runs.csv", records)
                write_json(output / "runs.json", records)

    comparisons, checks, stage_comparisons = [], {}, []
    timing_keys = ("wall_seconds", "monotonic_seconds", "user_seconds", "sys_seconds", "cpu_seconds", "maxrss_kib")
    for variant, versions in paired.items():
        for version, rows in versions.items():
            require(len({r["raw_sha256"] for r in rows}) == 1, f"{variant}/{version}: nondeterministic raw output")
        before, after = versions["baseline"][0], versions["fifo"][0]
        require(before["config_sha256"] == after["config_sha256"], f"{variant}: configurations differ")
        if configs[variant]["dataflow"] == "streaming":
            require(before["raw_sha256"] == after["raw_sha256"], f"{variant}: streaming output changed")
        for key in ("pim_read_bytes", "pim_write_bytes", "GPU_us", "GPU_measured_us", "GPU_blocks", "PIM_stages"):
            require(math.isclose(before[key], after[key], rel_tol=0, abs_tol=1e-6), f"{variant}: {key} changed")
        result = dict(variant=variant, repeats=args.repeats, baseline_total_us=before["total_us"],
                      fifo_total_us=after["total_us"], latency_delta_us=after["total_us"] - before["total_us"],
                      latency_delta_pct=100 * (after["total_us"] / before["total_us"] - 1),
                      baseline_PIM_us=before["PIM_us"], fifo_PIM_us=after["PIM_us"], GPU_us=before["GPU_us"])
        for key in timing_keys:
            for version, rows in versions.items():
                result[f"{version}_median_{key}"] = statistics.median(r[key] for r in rows)
        for key in ("wall_seconds", "monotonic_seconds", "cpu_seconds"):
            b, f = result[f"baseline_median_{key}"], result[f"fifo_median_{key}"]
            result[f"{key}_speedup"] = b / f if f else None
            result[f"{key}_delta_pct"] = 100 * (f / b - 1) if b else None
        comparisons.append(result)
        checks[variant] = dict(repetitions_byte_identical=True, inputs_identical=True, bytes_and_GPU_time_conserved=True,
                               fifo_out_of_order_zero=True, fifo_prerequisite_prefetch_zero=True,
                               streaming_byte_identical=before["raw_sha256"] == after["raw_sha256"]
                               if configs[variant]["dataflow"] == "streaming" else None)
        baseline_stages = [s for s in all_stages if s["variant"] == variant and s["version"] == "baseline" and s["repetition"] == 0]
        fifo_stages = [s for s in all_stages if s["variant"] == variant and s["version"] == "fifo" and s["repetition"] == 0]
        require(len(baseline_stages) == len(fifo_stages), "stage count changed")
        for b, f in zip(baseline_stages, fifo_stages):
            require((b["name"], b["read_bytes"], b["write_bytes"]) ==
                    (f["name"], f["read_bytes"], f["write_bytes"]), "stage identity or traffic changed")
            stage_comparisons.append(dict(variant=variant, name=b["name"], baseline_us=b["duration_us"],
                                          fifo_us=f["duration_us"], delta_us=f["duration_us"] - b["duration_us"],
                                          delta_pct=100 * (f["duration_us"] / b["duration_us"] - 1)
                                          if b["duration_us"] else None, read_bytes=b["read_bytes"], write_bytes=b["write_bytes"]))
    for path, digest in {**input_hashes, **driver_hashes}.items():
        require(sha256(path) == digest, f"input or benchmark driver changed: {path}")
    stable_runtime()
    for name, rows in (("comparison", comparisons), ("stage-comparison", stage_comparisons),
                       ("stage-runs", all_stages), ("gpu-blocks", all_gpu)):
        write_csv(output / f"{name}.csv", rows)
        write_json(output / f"{name}.json", rows)
    write_json(output / "summary.json", dict(checks_passed=True, no_vlm=True, no_full_campaign=True,
               scope=manifest["scope"], checks=checks, comparison=comparisons, execution_order=execution_order,
               manifest_sha256=sha256(output / "manifest.json"), wall_timing_excludes_python_validation=True))
    print(f"PASS: {len(records)} native runs; comparison saved to {output}", flush=True)


if __name__ == "__main__":
    main()
