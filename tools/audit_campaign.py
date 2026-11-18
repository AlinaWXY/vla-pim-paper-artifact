"""Independent read-only audit of a frozen campaign; imports no runner code.

Default scope is full nine-case acceptance. --scope frozen checks inputs only.
No jobs run and no inputs/results change. --output saves only an audit report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile

CASES = ("b", "c", "d", "e", "f", "g", "e-frfcfs", "e-lookahead", "e-gap")
VARIANTS = dict(zip(CASES, ("feature_gpu_before", "feature_gpu_before", "token_gpu_before",
                          "token_local_before", "token_gpu_after", "token_local_after",
                          "token_local_before", "token_local_before", "token_local_before")))
LEGACY_FIELDS = "case,variant,total_ms,AE_solo_ms,PIM_interference_ms,AE_GPU_wait_ms,net_overlap_ms,AE_native_schedule_ms,VLM_memory_stall_ms,VLM_completed_model_ms,VLM_remaining_model_ms,raw_model_total_ms,completed_units,GPU_act1,GPU_act2,GPU_prepb,GPU_preab,GPU_reads,GPU_writes,GPU_row_hits,GPU_row_misses".split(",")
GPU_COUNTERS = LEGACY_FIELDS[13:]
INPUT_FIELDS = ("program_path", "vlm_path", "ae_schedule_path")
TOLERANCE_US = 0.002


class AuditError(Exception):
    pass


def need(condition, description):
    if not condition:
        raise AuditError(description)


def objhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def close(actual, expected, description, tolerance=TOLERANCE_US):
    need(math.isfinite(float(actual)) and math.isfinite(float(expected)), description + ": nonfinite")
    need(abs(float(actual) - float(expected)) <= tolerance,
         f"{description}: observed={actual}, expected={expected}")


def span(record):
    value = record["end_us"] - record["start_us"]
    need(value >= 0, "negative native interval")
    if "duration_us" in record:
        close(record["duration_us"], value, "native duration differs from endpoints", 0.000002)
    return value


def percentile(values, quantile):
    values = sorted(values)
    at = (len(values) - 1) * quantile
    low, high = math.floor(at), math.ceil(at)
    return values[low] if low == high else values[low] * (high - at) + values[high] * (at - low)


class Auditor:
    def __init__(self, run, expected_source=None):
        self.run = Path(run).resolve()
        self.source = self.run / "frozen/source"
        self.expected_source = expected_source
        self.hashes, self.programs, self.schedules = {}, {}, {}
        self.report = {"schema_version": 1, "campaign": str(self.run), "passed": False,
                       "checks": [], "native_runs": [], "files": {},
                       "auditor_sha256": self.digest(Path(__file__))}

    def digest(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            need(path.is_file(), f"missing file: {path}")
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            self.hashes[path] = value.hexdigest()
        return self.hashes[path]

    def record_file(self, path):
        path = Path(path).resolve()
        self.report["files"][str(path.relative_to(self.run))] = self.digest(path)

    def read(self, path):
        self.record_file(path)
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)

    def csv(self, path):
        self.record_file(path)
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def relocated(self, path):
        """Support copied archives without reading former external locations."""
        value = str(path).replace("\\", "/")
        for prefix in (str(self.run).replace("\\", "/"), self.original_root):
            if value.startswith(prefix + "/"):
                result = (self.run / value[len(prefix) + 1:]).resolve()
                need(result.is_relative_to(self.run), "reference escapes campaign")
                return result
        raise AuditError(f"reference is outside frozen campaign: {path}")

    def sums(self, manifest, base, *, inventory=False):
        self.record_file(manifest)
        listed = set()
        for line in Path(manifest).read_text().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
            need(match is not None, f"malformed SHA256SUMS line: {line!r}")
            expected, name = match.groups()
            path = (base / name).resolve()
            need(path.is_relative_to(base.resolve()), f"manifest escapes base: {name}")
            need(path not in listed, f"duplicate manifest entry: {name}")
            listed.add(path)
            need(self.digest(path) == expected, f"frozen hash mismatch: {name}")
        need(listed, f"empty manifest: {manifest}")
        if inventory:
            present = {p.resolve() for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
            need(present == listed, "frozen source inventory differs from manifest")
        self.report["checks"].append({"manifest": str(manifest.relative_to(self.run)), "files": len(listed)})

    def frozen(self):
        self.commit = (self.run / "frozen/SOURCE_COMMIT").read_text().strip()
        need(re.fullmatch("[0-9a-f]{40}", self.commit), "invalid SOURCE_COMMIT")
        if self.expected_source:
            need(self.commit.startswith(self.expected_source), "campaign source differs from expected commit")
        self.report["source_commit"] = self.commit
        for name in ("frozen/SOURCE_COMMIT", "python-path", "partition", "jobs/worker.sbatch"):
            self.record_file(self.run / name)
        self.sums(self.run / "frozen/runtime-SHA256SUMS", self.run)
        self.sums(self.run / "frozen/source-SHA256SUMS", self.source, inventory=True)
        self.sums(self.run / "frozen/config-SHA256SUMS", self.run)
        self.binary = self.digest(self.run / "frozen/bin/pim_vla_sim")
        self.library = self.digest(self.run / "frozen/lib/libramulator.so")
        self.configs = {case: self.read(self.run / "configs" / f"{case}.json") for case in CASES}
        path = self.configs["b"]["program_path"].replace("\\", "/")
        need("/frozen/source/" in path, "program is not inside frozen source")
        self.original_root = path.split("/frozen/source/", 1)[0]
        self.spec, frequency = self.configs["b"]["spec"], self.configs["b"]["mac_freq_mult"]
        self.spec_hash = objhash(self.spec)
        need(frequency in (1, 2, 4), "invalid MAC hardware multiplier")
        need(self.read(self.run / "configs" / f"spec-k{frequency}.json") == self.spec, "exported spec differs")
        for case, cfg in self.configs.items():
            need(cfg["spec"] == self.spec and cfg["mac_freq_mult"] == frequency, "case hardware differs")
            need(cfg["case"] == case, "prepared case identity differs")
            need(cfg["mode"] == ("serial" if case == "b" else "concurrent"), "case mode differs")
            need(f"{cfg['layout']}_{cfg['rope']}_{'after' if cfg['dataflow']=='fused' else 'before'}" == VARIANTS[case], "case design differs")
            policy = {"c": "pim_head", "d": "pim_head", "e": "pim_head", "e-frfcfs": "frfcfs",
                      "e-lookahead": "pim_lookahead", "e-gap": "gap_batch"}.get(case)
            need(cfg.get("arbitration") == policy and (("arbitration" in cfg) == bool(policy)), "case arbitration differs")
            need(not any(key.startswith("pim_") or key in ("shift", "constant_offset", "dq_latency", "row_hit") for key in cfg), "retired config field")
            for key in INPUT_FIELDS:
                self.digest(self.relocated(cfg[key]))
        self.vlm = self.read(self.relocated(self.configs["b"]["vlm_path"]))
        units = self.vlm["units"]
        need(len(units) == 2357 and len({u["ordinal"] for u in units}) == 2357, "VLM lacks 2357 unique units")
        # The normalized ledger names memcpy records "memory".
        need(Counter(u["kind"] for u in units) == {"kernel": 2353, "memory": 4}, "VLM kernel/memcpy counts differ")
        close(sum(u["us"] for u in units), 105220.5782, "measured VLM durations were rescaled", 0.00001)
        close(self.vlm["target_us"], 104950.49603462219, "VLM phase target differs", 0.000001)
        self.units = {u["ordinal"]: u for u in units}
        input_provenance = self.read(self.source / "input-data/provenance.json")
        for name, record in input_provenance["sources"].items():
            need(self.digest(self.source / "input-data/measured" / name) == record["sha256"],
                 f"measured input provenance mismatch: {name}")
        for name, expected in input_provenance["generated"].items():
            need(self.digest(self.source / "input-data" / name) == expected,
                 f"normalized input provenance mismatch: {name}")
        for case in ("b", "d", "e", "f", "g"):
            cfg, variant = self.configs[case], VARIANTS[case]
            program = self.read(self.relocated(cfg["program_path"]))
            total = Counter(read=0, write=0)
            def count_operations(ops):
                for op in ops:
                    if op["op"] in ("RD", "RDA", "RDRES"):
                        total["read"] += 32 * op.get("count", 1)
                    if op["op"] in ("WR", "WRA", "WRGB"):
                        total["write"] += 32 * op.get("count", 1)
                    if "children" in op:
                        count_operations(op["children"])
            for stage in program["stages"]:
                need(len(stage["channels"]) == 16, "program channel count differs")
                for ops in stage["channels"]:
                    count_operations(ops)
            expected = program.get("expected_dq_bytes_per_layer", program.get("expected_dq_bytes"))
            if isinstance(expected, dict):
                expected = expected["total"]
            need(sum(total.values()) == expected, f"{variant}: DQ metadata is not request-derived")
            self.programs[variant] = dict(total)
            schedule = self.read(self.relocated(cfg["ae_schedule_path"]))
            need(schedule["layers"] == 18 and schedule["denoising_steps"] == 10, "AE schedule dimensions differ")
            self.schedules[variant] = [e for e in schedule["events"] if e["kind"] == "pim"]
            counts = Counter((e["step"], e["layer"]) for e in self.schedules[variant])
            need(set(counts) == {(s, l) for s in range(10) for l in range(18)}, "AE layer/step coverage differs")
            need(set(counts.values()) == {len(program["stages"])}, "AE per-layer stage count differs")
        self.report["checks"].append({"frozen_inputs": "verified", "mac_freq_mult": frequency,
                                      "VLM_units": 2357, "spec_sha256": self.spec_hash})

    def git_source(self, repository):
        """Bind tracked frozen bytes to the actual Git object, without checkout."""
        archive = subprocess.check_output(["git", "-C", str(repository), "archive",
                                           "--format=tar", self.commit], stderr=subprocess.STDOUT)
        count = 0
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tree:
            for member in tree:
                if member.isdir():
                    continue
                need(member.isfile(), f"unsupported source archive member: {member.name}")
                path = (self.source / member.name).resolve()
                need(path.is_relative_to(self.source), "Git archive path escapes source")
                content = tree.extractfile(member)
                need(content is not None, "Git archive member lacks content")
                expected = hashlib.sha256(content.read()).hexdigest()
                need(self.digest(path) == expected, f"frozen file differs from source commit: {member.name}")
                count += 1
        self.report["checks"].append({"git_source_commit": self.commit, "tracked_files_verified": count})

    def provenance(self, value):
        need(value["source_commit"] == self.commit, "result source differs from frozen source")
        need(value["binary_sha256"] == self.binary, "result executable differs")
        libs = value["runtime_libraries"]
        need(set(libs) == {"libramulator.so"}, "runtime library provenance missing/unexpected")
        need(libs["libramulator.so"]["sha256"] == self.library, "result runtime library differs")
        need(self.relocated(libs["libramulator.so"]["path"]) == self.run / "frozen/lib/libramulator.so", "runtime did not use frozen library")
        need(value["spec_sha256"] == self.spec_hash, "result spec hash differs")

    def native(self, directory, expected_config):
        config = self.read(directory / "input-config.json")
        need(config == expected_config, f"unexpected native config: {directory}")
        provenance = self.read(directory / "provenance.json")
        self.provenance(provenance)
        need(provenance["config_sha256"] == self.digest(directory / "input-config.json"), "native config hash mismatch")
        need(provenance["spec"] == config["spec"] and provenance["calibration"] == config["calibration"], "native provenance differs")
        for key in INPUT_FIELDS:
            need(provenance["inputs"][key]["sha256"] == self.digest(self.relocated(config[key])), f"native {key} hash mismatch")
        raw = self.read(directory / "raw.json")
        raw_hash = self.digest(directory / "raw.json")
        need(provenance["raw_sha256"] == raw_hash, f"native raw hash mismatch: {directory}")
        self.record_file(directory / "native.log")
        kind = config.get("run_kind", "combined")
        need(f"completed {kind} " in (directory / "native.log").read_text(), "native completion log missing")
        m = raw["metrics"]
        need(m["total_us"] > 0 and m["checked_commands"] > 0, "empty native run")
        variant = VARIANTS[config["case"]]
        pim = self.programs[variant] if kind != "vlm_solo" else {"read": 0, "write": 0}
        for direction in ("read", "write"):
            need(m[f"pim_{direction}_bytes"] == pim[direction] * 180, "full AE PIM DQ mismatch")
        gpu = {"read": 0, "write": 0}
        if kind != "ae_solo":
            alpha = config["calibration"]["alpha"]
            gpu["read"] = sum(u["read_sectors"] for u in self.units.values()) * 32
            gpu["write"] = sum(math.ceil(u["write_sectors"] * alpha) if u["write_is_proxy"] else u["write_sectors"] for u in self.units.values()) * 32
        for direction in ("read", "write"):
            need(m[f"gpu_{direction}_bytes"] == gpu[direction], "full VLM DQ mismatch")
        need(m["GPU_reads"] * 32 == gpu["read"] and m["GPU_writes"] * 32 == gpu["write"], "GPU command/byte count mismatch")
        need(m["GPU_row_hits"] + m["GPU_row_misses"] == m["GPU_reads"] + m["GPU_writes"], "GPU hit/miss conservation failure")
        expected_stages = self.schedules[variant] if kind != "vlm_solo" else []
        need(len(raw["stages"]) == len(expected_stages), "native stage coverage differs")
        for index, (stage, expected) in enumerate(zip(raw["stages"], expected_stages)):
            need((stage["stage_id"], stage["name"], stage["step"], stage["layer"]) ==
                 (index, expected["stage"], expected["step"], expected["layer"]), "native stage identity differs")
            span(stage)
        kernels = raw["kernels"]
        need(len(kernels) == (0 if kind == "ae_solo" else 2357), "native VLM coverage differs")
        if kernels:
            need([k["ordinal"] for k in kernels] == list(self.units), "native kernel order/identity differs")
        previous_end = 0
        for kernel in kernels:
            source = self.units[kernel["ordinal"]]
            need(kernel["start_us"] + 0.000001 >= previous_end, "VLM kernels overlap or reorder")
            need(span(kernel) + 0.000001 >= source["us"], "native kernel faster than measured compute floor")
            writes = math.ceil(source["write_sectors"] * config["calibration"]["alpha"]) if source["write_is_proxy"] else source["write_sectors"]
            need(kernel["read_bytes"] == source["read_sectors"] * 32 and kernel["write_bytes"] == writes * 32,
                 f"kernel {kernel['ordinal']} DQ differs from measured read / fitted write contract")
            previous_end = kernel["end_us"]
        need({b["channel"] for b in raw["bus"]} == set(range(16)), "bus channel coverage incomplete")
        for entry in raw["bus"]:
            need(0 <= entry["ca_cycles"] <= entry["duration_cycles"] and
                 0 <= entry["dq_cycles"] <= entry["duration_cycles"], "bus utilization outside physical bounds")
        for field in ("gpu_read_bytes", "gpu_write_bytes", "pim_read_bytes", "pim_write_bytes"):
            need(sum(b[field] for b in raw["bus"]) == m[field], f"bus byte accounting mismatch: {field}")
        self.report["native_runs"].append({"directory": str(directory.relative_to(self.run)),
                                           "raw_sha256": raw_hash, "kind": kind, "units": len(kernels)})
        return raw

    def calibration(self):
        directory = self.run / "calibration"
        frozen = self.read(directory / "calibration.json")
        self.provenance(frozen)
        need(frozen["spec"] == self.spec and frozen["no_latency_offsets"] is True, "calibration spec/offset violation")
        need(frozen["vlm_sha256"] == self.digest(self.relocated(self.configs["b"]["vlm_path"])), "calibration input hash mismatch")
        parameters = [dict(alpha=1.0, window=64, write_high=48, write_low=16)]
        for window, high, low in ((1024, 768, 256), (256, 192, 64), (256, 224, 128)):
            parameters += [dict(alpha=alpha, window=window, write_high=high, write_low=low) for alpha in (1., .5, .2, .1, .05)]
        selected = frozen["selected_candidate"]
        need(isinstance(selected, int) and 0 <= selected < len(parameters), "invalid selected candidate")
        need(frozen["parameters"] == parameters[selected], "calibration differs from declared search")
        assessments, chosen_raw = {}, None
        for index, params in enumerate(parameters):
            where = directory / f"candidate-{index:02d}"
            if not (where / "assessment.json").exists():
                need(index > selected, "earlier calibration candidate is absent")
                continue
            raw = self.native(where, dict(self.configs["b"], run_kind="vlm_solo", calibration=params))
            errors = [(span(k) - self.units[k["ordinal"]]["us"]) / self.units[k["ordinal"]]["us"]
                      for k in raw["kernels"] if self.units[k["ordinal"]]["kind"] == "kernel"]
            target, total = self.vlm["target_us"], raw["metrics"]["total_us"]
            relative = (total - target) / target
            validation = dict(total_us=total, target_us=target, relative_error=relative,
                              kernel_relative_error_median=percentile(errors, .5),
                              kernel_relative_error_p90=percentile(errors, .9), work_units=2357,
                              kernel_count=2353, passed=abs(relative) <= .05)
            assessment = self.read(where / "assessment.json")
            need(assessment["candidate"] == index and all(assessment[k] == v for k, v in params.items()), "candidate identity/parameters differ")
            for key, expected in validation.items():
                close(assessment[key], expected, f"calibration assessment {key}", 0.00000001)
            assessments[index] = validation
            if index == selected:
                chosen_raw = raw
                need(frozen["raw_sha256"] == self.digest(where / "raw.json") and self.relocated(frozen["raw_path"]) == where / "raw.json", "selected calibration raw differs")
                for key, expected in validation.items():
                    close(frozen["validation"][key], expected, f"frozen calibration {key}", 0.00000001)
        need(assessments[selected]["passed"] and not any(assessments[i]["passed"] for i in range(selected)), "selected candidate is not the first passing full candidate")
        sweep = self.read(directory / "calibration-sweep.json")
        _, sweep_csv = self.csv(directory / "calibration-sweep.csv")
        need(len(sweep) == len(sweep_csv) == len(assessments), "calibration sweep coverage differs")
        for record, csv_record, index in zip(sweep, sweep_csv, sorted(assessments)):
            expected = dict(candidate=index, **parameters[index], **assessments[index])
            self.compare_row(record, expected, "calibration sweep JSON")
            self.compare_row(csv_record, expected, "calibration sweep CSV")
        self.report["checks"].append({"calibration": "independently recomputed", "selected_candidate": selected,
                                      "validation": assessments[selected]})
        return frozen, chosen_raw

    def evaluation(self, case, combined, solo, vlm_solo):
        m = combined["metrics"]
        ae, vlm, total = solo["metrics"]["total_us"], vlm_solo["metrics"]["total_us"], m["total_us"]
        wait, elapsed = m["AE_GPU_wait_us"], m["AE_elapsed_us"]
        interference = sum(span(a) - span(b) for a, b in zip(combined["stages"], solo["stages"]))
        close(elapsed, ae + interference + wait, f"{case}: AE accounting")
        close(total, elapsed + m["VLM_remaining_us"], f"{case}: total accounting / no offsets")
        if case == "b":
            close(total, ae + vlm, "serial independent-epoch identity")
            need(m["serial_epoch_reset"] is True, "serial epoch definition missing")
        else:
            need(m["serial_epoch_reset"] is False, "concurrent run reset memory state")
        base_kernels = {k["ordinal"]: k for k in vlm_solo["kernels"]}
        complete = [k["ordinal"] for k in combined["kernels"] if k["end_us"] <= elapsed]
        stall = sum(span(k) - span(base_kernels[k["ordinal"]]) for k in combined["kernels"])
        return dict(case=case, variant=VARIANTS[case], total_ms=total / 1000, AE_solo_ms=ae / 1000,
                    PIM_interference_ms=interference / 1000, AE_GPU_wait_ms=wait / 1000,
                    net_overlap_ms=(ae + vlm - total) / 1000, AE_native_schedule_ms=ae / 1000,
                    VLM_memory_stall_ms=stall / 1000, VLM_completed_model_ms=sum(span(base_kernels[k]) for k in complete) / 1000,
                    VLM_remaining_model_ms=m["VLM_remaining_us"] / 1000, raw_model_total_ms=total / 1000,
                    completed_units=len(complete), **{key: m[key] for key in GPU_COUNTERS},
                    AE_parallel_ms=elapsed / 1000, VLM_solo_ms=vlm / 1000,
                    VLM_remaining_ms=m["VLM_remaining_us"] / 1000,
                    PIM_stages=len(combined["stages"]), VLM_work_units=2357)

    def compare_row(self, actual, expected, description):
        need(set(actual) == set(expected), description + ": fields differ")
        for key, value in expected.items():
            if isinstance(value, bool):
                need(actual[key] in (value, str(value)), description + f": {key} differs")
            elif isinstance(value, str):
                need(actual[key] == value, description + f": {key} differs")
            else:
                close(actual[key], value, description + f": {key}", TOLERANCE_US / 1000)

    def details(self, directory, case, raw, solo, vlm_solo):
        _, rows = self.csv(directory / f"{case}-PIM-and-AE-waits.csv")
        need(len(rows) == len(raw["stages"]), "stage CSV coverage differs")
        for row, live, base in zip(rows, raw["stages"], solo["stages"]):
            expected = dict(stage_id=live["stage_id"], name=live["name"], step=live["step"], layer=live["layer"],
                            solo_us=span(base), concurrent_us=span(live), interference_us=span(live) - span(base),
                            AE_GPU_wait_us=live.get("gpu_wait_us", 0), start_us=live["start_us"], end_us=live["end_us"])
            self.compare_row(row, expected, f"{case} stage {live['stage_id']}")
        _, rows = self.csv(directory / f"{case}-VLM-placement.csv")
        need(len(rows) == 2357, "kernel CSV coverage differs")
        for row, live, base in zip(rows, raw["kernels"], vlm_solo["kernels"]):
            expected = dict(ordinal=live["ordinal"], start_us=live["start_us"], end_us=live["end_us"],
                            solo_us=span(base), duration_us=span(live), stall_us=span(live) - span(base))
            self.compare_row(row, expected, f"{case} kernel {live['ordinal']}")
        _, rows = self.csv(directory / f"{case}-bus.csv")
        need(len(rows) == len(raw["bus"]), "bus CSV coverage differs")
        for row, live in zip(rows, raw["bus"]):
            wall = live["duration_cycles"]
            expected = dict(live, dq_utilization=live["dq_cycles"] / wall if wall else 0,
                            ca_utilization=live["ca_cycles"] / wall if wall else 0)
            self.compare_row(row, expected, f"{case} bus record")

    def full(self):
        solos = {}
        for case in ("b", "d", "e", "f", "g"):
            variant = VARIANTS[case]
            solos[variant] = self.native(self.run / "solos/ae-solo" / variant,
                                         dict(self.configs[case], run_kind="ae_solo"))
        calibration, vlm_solo = self.calibration()
        expected_rows, repeated = [], []
        for case in CASES:
            directory = self.run / "cases" / case
            cfg = dict(self.configs[case], calibration=calibration["parameters"])
            first = self.native(directory / "native" / case / "repeat-0", cfg)
            second = self.native(directory / "native" / case / "repeat-1", cfg)
            hashes = [self.digest(directory / "native" / case / f"repeat-{i}/raw.json") for i in (0, 1)]
            need(hashes[0] == hashes[1], f"{case}: native repetitions differ byte-for-byte")
            need(first == second, f"{case}: repeated JSON contents differ")
            manifest = self.read(directory / "provenance.json")
            self.provenance(manifest)
            need(manifest["no_external_trace_replay"] is True and manifest["no_constant_correction"] is True, "case provenance permits retiming")
            need(manifest["calibration"] == calibration and manifest["cases"] == [case], "case calibration/identity differs")
            program_hashes = {VARIANTS[c]: self.digest(self.relocated(self.configs[c]["program_path"])) for c in CASES}
            schedule_hashes = {VARIANTS[c]: self.digest(self.relocated(self.configs[c]["ae_schedule_path"])) for c in CASES}
            need(manifest["programs"] == program_hashes, "case manifest program hashes differ")
            need(manifest["measured_inputs"] == dict(vlm=self.digest(self.relocated(cfg["vlm_path"])), **schedule_hashes),
                 "case manifest measurement hashes differ")
            expected = self.evaluation(case, first, solos[VARIANTS[case]], vlm_solo)
            expected_rows.append(expected)
            summary = self.read(directory / "summary.json")
            need(summary["latency_correction"] is None and summary["calibration"] == calibration, "case summary calibration/offset differs")
            repeat = {"case": case, "raw_sha256": hashes, "byte_identical": True}
            repeated.append(repeat)
            need(summary["repeat_checks"] == [repeat], "case repeat hash manifest differs")
            need(manifest["reproducibility"] == summary["repeat_checks"], "case provenance repeat manifest differs")
            need(len(summary["evaluation"]) == 1, "unexpected case summary rows")
            self.compare_row(summary["evaluation"][0], expected, case + " summary")
            header, rows = self.csv(directory / "evaluation-b-g.csv")
            need(header[:21] == LEGACY_FIELDS and len(rows) == 1, "legacy CSV columns/order differ")
            self.compare_row(rows[0], expected, case + " CSV")
            checks = self.read(directory / f"{case}-checks.json")
            need(checks["PIM_DQ_actual_bytes"] == checks["PIM_DQ_expected_bytes"] == sum(self.programs[VARIANTS[case]].values()) * 180, "reported PIM DQ check differs")
            for direction in ("read", "write"):
                need(checks[f"GPU_DQ_actual_{direction}_bytes"] == checks[f"GPU_DQ_expected_{direction}_bytes"] == first["metrics"][f"gpu_{direction}_bytes"], "reported GPU DQ check differs")
            close(checks["AE_timestamp_accounting_error_us"], 0, "reported AE accounting failed")
            need(checks["stage_coverage"] == len(first["stages"]) and checks["vlm_coverage"] == 2357, "reported coverage differs")
            self.details(directory, case, first, solos[VARIANTS[case]], vlm_solo)
            self.record_file(directory / "REPORT.zh-CN.md")
        matrix = self.run / "matrix"
        manifest = self.read(matrix / "provenance.json")
        self.provenance(manifest)
        need(manifest["no_external_trace_replay"] is True and manifest["no_constant_correction"] is True, "matrix permits retiming")
        need(manifest["cases"] == list(CASES) and manifest["calibration"] == calibration, "matrix case list/calibration differs")
        need(len(manifest["parts"]) == 9, "matrix provenance parts incomplete")
        for part, case in zip(manifest["parts"], CASES):
            need(self.relocated(part["directory"]) == self.run / "cases" / case, "aggregate source directory differs")
            need(part["provenance_sha256"] == self.digest(self.run / "cases" / case / "provenance.json"), "aggregate source manifest hash differs")
        summary = self.read(matrix / "summary.json")
        need(summary["calibration"] == calibration and summary["latency_correction"] is None, "matrix calibration/offset differs")
        need(summary["repeat_checks"] == repeated, "aggregate repetition evidence differs")
        header, rows = self.csv(matrix / "evaluation-b-g.csv")
        need(header[:21] == LEGACY_FIELDS and len(rows) == 9 and len(summary["evaluation"]) == 9, "matrix header/row coverage differs")
        for actual, saved, expected in zip(rows, summary["evaluation"], expected_rows):
            self.compare_row(actual, expected, expected["case"] + " aggregate CSV")
            self.compare_row(saved, expected, expected["case"] + " aggregate JSON")
        for case in CASES:
            for suffix in ("PIM-and-AE-waits.csv", "VLM-placement.csv", "bus.csv", "checks.json"):
                path = matrix / f"{case}-{suffix}"
                self.record_file(path)
                need(self.digest(path) == self.digest(self.run / "cases" / case / path.name), "aggregated detail differs from source")
        self.record_file(matrix / "REPORT.zh-CN.md")
        for directory in (self.run / "jobs", self.run / "logs"):
            need(directory.is_dir() and any(directory.iterdir()), "scheduler provenance/log directory absent")
            for path in directory.iterdir():
                if path.is_file():
                    self.record_file(path)
        self.report["checks"].append({"complete_matrix": "independently recomputed", "cases": list(CASES),
                                      "native_repeats_per_case": 2, "legacy_columns": LEGACY_FIELDS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--scope", choices=("full", "frozen"), default="full")
    parser.add_argument("--expected-source", help="expected full commit or unambiguous prefix")
    parser.add_argument("--git-repo", type=Path, help="also compare every tracked frozen file to git archive of SOURCE_COMMIT")
    parser.add_argument("--output", type=Path, help="save audit report outside frozen/; default stdout only")
    args = parser.parse_args()
    auditor = Auditor(args.campaign, args.expected_source)
    auditor.report.update(scope=args.scope, full_experiment_accepted=False)
    try:
        auditor.frozen()
        if args.git_repo:
            auditor.git_source(args.git_repo)
        if args.scope == "full":
            auditor.full()
            auditor.report["full_experiment_accepted"] = True
        auditor.report["passed"] = True
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        auditor.report["error"] = f"{type(error).__name__}: {error}"
    encoded = json.dumps(auditor.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        need(not args.output.resolve().is_relative_to((args.campaign / "frozen").resolve()), "audit must not modify frozen/")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if auditor.report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
