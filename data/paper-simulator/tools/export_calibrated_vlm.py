"""Export integer-cycle VLM durations from an accepted native solo calibration.

This tool only reads saved results. It neither calibrates nor runs a simulator.
The source binary/library identities remain those of the original calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def finite_number(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and value >= 0, f"{label} must be finite and nonnegative")
    return value


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _export_ledger(*, calibration, raw, provenance, config, vlm):
    paths = dict(calibration=calibration, raw=raw, provenance=provenance, config=config, vlm=vlm)
    hashes = {key: sha256(path) for key, path in paths.items()}
    selected, native, binding, cfg, workload = (read_json(paths[key]) for key in paths)
    validation = selected["validation"]
    require(validation["passed"] is True, "calibration has not passed")
    require(validation["work_units"] == 2357 and validation["kernel_count"] == 2353,
            "calibration must cover all 2357 work units / 2353 kernels")
    require(cfg["run_kind"] == "vlm_solo", "source execution must be VLM solo")
    require(selected["raw_sha256"] == binding["raw_sha256"] == hashes["raw"], "raw hash mismatch")
    require(binding["config_sha256"] == hashes["config"], "configuration hash mismatch")
    require(selected["vlm_sha256"] == binding["inputs"]["vlm_path"]["sha256"] == hashes["vlm"],
            "VLM input hash mismatch")
    require(isinstance(cfg["spec"], dict), "an embedded source spec is required")
    spec_hash = object_sha256(cfg["spec"])
    require(selected["spec_sha256"] == binding["spec_sha256"] == spec_hash and
            object_sha256(selected["spec"]) == object_sha256(binding["spec"]) == spec_hash,
            "source spec hash mismatch")
    params = selected["parameters"]
    require(params == cfg["calibration"] == binding["calibration"], "calibration parameters differ")
    require(set(params) == {"alpha", "window", "write_high", "write_low"}, "invalid calibration parameter fields")
    require(0.05 <= finite_number(params["alpha"], "alpha") <= 1, "invalid writeback alpha")
    require(all(isinstance(params[k], int) and not isinstance(params[k], bool)
                for k in ("window", "write_high", "write_low")) and
            0 <= params["write_low"] < params["write_high"] <= params["window"] <= 4096,
            "invalid calibration window/watermarks")
    binary_hash = selected["binary_sha256"]
    require(valid_digest(binary_hash) and binary_hash == binding["binary_sha256"], "source binary hash mismatch")
    revision = selected["source_commit"]
    require(isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
            and revision == binding["source_commit"], "source commit mismatch")
    libraries = selected["runtime_libraries"]
    require(libraries == binding["runtime_libraries"] and "libramulator.so" in libraries and
            all(valid_digest(record["sha256"]) for record in libraries.values()), "source library provenance mismatch")
    clock_ps = cfg["spec"]["clock_ps"]
    require(isinstance(clock_ps, int) and not isinstance(clock_ps, bool) and clock_ps > 0 and
            native["clock_ps"] == clock_ps, "source clock mismatch")

    def cycles(value, label):
        finite_number(value, label)
        count = round(value * 1000000 / clock_ps)
        # Native JSON prints decimal microseconds. Accept only its formatting
        # error, never a fractional-cycle duration that would need rescaling.
        require(abs(value - count * clock_ps / 1000000) <= max(1e-9, clock_ps / 1e11),
                f"{label} is not an integer number of source clock cycles")
        return count

    metrics = native["metrics"]
    require(not native["stages"] and metrics["pim_read_bytes"] == metrics["pim_write_bytes"] == 0 and
            metrics["AE_elapsed_us"] == metrics["AE_GPU_wait_us"] == 0, "source contains AE/PIM work")
    observed, inputs = native["kernels"], workload["units"]
    require(len(observed) == len(inputs) == 2357, "missing work units")
    ordinals = [unit["ordinal"] for unit in inputs]
    require(all(isinstance(x, int) and not isinstance(x, bool) for x in ordinals) and len(set(ordinals)) == 2357,
            "input ordinals must be unique integers")
    require([unit["ordinal"] for unit in observed] == ordinals, "native work-unit ordinal order differs")
    require(Counter(unit["kind"] for unit in inputs) == {"kernel": 2353, "memory": 4}, "work-unit kinds differ")
    units, previous_end, read_bytes, write_bytes = [], 0, 0, 0
    for source, result in zip(inputs, observed):
        ordinal = source["ordinal"]
        require(isinstance(source["name"], str) and bool(source["name"]), f"unit {ordinal}: missing name")
        measured_us = finite_number(source["us"], f"unit {ordinal} measured duration")
        duration = cycles(result["duration_us"], f"unit {ordinal} duration")
        start = cycles(result["start_us"], f"unit {ordinal} start")
        end = cycles(result["end_us"], f"unit {ordinal} end")
        require(start == previous_end and end - start == duration, f"unit {ordinal}: noncontiguous or inconsistent timing")
        require(duration >= math.ceil(measured_us * 1000000 / clock_ps - 1e-10),
                f"unit {ordinal}: calibrated duration is shorter than measured compute")
        rd, wr = source["read_sectors"], source["write_sectors"]
        require(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (rd, wr)),
                f"unit {ordinal}: invalid sector counts")
        if source.get("write_is_proxy", True):
            wr = math.ceil(params["alpha"] * wr)
        require((result["read_bytes"], result["write_bytes"]) == (rd * 32, wr * 32),
                f"unit {ordinal}: source GPU byte conservation failed")
        read_bytes += rd * 32
        write_bytes += wr * 32
        units.append(dict(ordinal=ordinal, name=source["name"], kind=source["kind"], duration_cycles=duration,
                          us=duration * clock_ps / 1000000, measured_us=measured_us))
        previous_end = end
    total_cycles = sum(unit["duration_cycles"] for unit in units)
    require(total_cycles == previous_end == cycles(metrics["total_us"], "native total") ==
            cycles(metrics["VLM_remaining_us"], "VLM solo total") == cycles(validation["total_us"], "validated total"),
            "total duration does not equal the complete calibrated unit sequence")
    require((metrics["gpu_read_bytes"], metrics["gpu_write_bytes"]) == (read_bytes, write_bytes),
            "source total GPU byte conservation failed")
    total_us = total_cycles * clock_ps / 1000000
    target_us = finite_number(validation["target_us"], "calibration target")
    require(target_us > 0 and abs(total_us / target_us - 1) <= 0.05, "source calibration exceeds the 5% acceptance gate")
    require(math.isclose(validation["relative_error"], total_us / target_us - 1, rel_tol=0, abs_tol=1e-12),
            "reported calibration error is inconsistent")
    require(workload["target_us"] == target_us, "VLM source target differs from calibration")
    return dict(schema_version=1, kind="calibrated_vlm_durations", clock_ps=clock_ps,
                total_cycles=total_cycles, total_us=total_us, work_units=len(units), kernel_count=2353, memory_count=4,
                calibration_parameters=params,
                source=dict(**{key + "_sha256": value for key, value in hashes.items()}, spec_sha256=spec_hash,
                            source_commit=revision, binary_sha256=binary_hash, runtime_libraries=libraries,
                            selected_candidate=selected["selected_candidate"], raw_path=selected["raw_path"],
                            calibration_validation=validation),
                semantics="Native calibrated VLM-solo elapsed duration per unit, including its memory stalls; no added traffic or rescaling",
                units=units)


def export_ledger(**paths):
    try:
        return _export_ledger(**paths)
    except (KeyError, TypeError) as error:
        raise ValueError(f"incomplete or invalid source provenance: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration", "raw", "provenance", "config", "vlm", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = export_ledger(**{key: getattr(args, key) for key in ("calibration", "raw", "provenance", "config", "vlm")})
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        require(args.output.read_text(encoding="utf-8") == data, "refusing to overwrite a different ledger")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output), work_units=result["work_units"],
                          total_cycles=result["total_cycles"], total_us=result["total_us"])))


if __name__ == "__main__":
    main()
