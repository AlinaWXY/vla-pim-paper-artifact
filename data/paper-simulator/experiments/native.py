"""Invoke the single native Ramulator2 execution path and bind results to inputs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from experiments.common import object_sha256, read_json, sha256, source_revision, write_json
from experiments.config import validate_config


def runtime_libraries(binary):
    """Hash the library the ELF loader will actually use in this environment."""
    linked = subprocess.check_output(["ldd", str(Path(binary).resolve())], text=True, stderr=subprocess.STDOUT)
    for line in linked.splitlines():
        if line.strip().startswith("libramulator.so =>"):
            path = line.split("=>", 1)[1].rsplit("(", 1)[0].strip()
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise RuntimeError(f"Ramulator runtime library cannot be resolved: {line.strip()}")
            return {"libramulator.so": {"path": str(resolved), "sha256": sha256(resolved)}}
    raise RuntimeError("native binary did not resolve libramulator.so; check LD_LIBRARY_PATH")


def library_hashes(manifest):
    libraries = manifest.get("runtime_libraries")
    if not libraries or "libramulator.so" not in libraries:
        raise ValueError("runtime library provenance is missing; rerun using the frozen library")
    return {name: record["sha256"] for name, record in libraries.items()}


def execute(binary, config, output, *, resume=False):
    output = Path(output)
    config = validate_config(dict(config))
    if config.get("execution_model") == "fused_kernel_windows":
        ledger = read_json(config["calibrated_vlm_path"])
        if ledger.get("kind") != "calibrated_vlm_durations":
            raise ValueError("invalid calibrated VLM duration ledger")
        if ledger["source"]["vlm_sha256"] != sha256(config["vlm_path"]):
            raise ValueError("calibrated ledger belongs to another VLM input")
        if ledger["source"]["spec_sha256"] != object_sha256(config["spec"]):
            raise ValueError("calibrated ledger belongs to another DRAM spec")
        if ledger["calibration_parameters"] != config["calibration"]:
            raise ValueError("calibrated ledger parameter mismatch")
    libraries = runtime_libraries(binary)
    binary_hash = sha256(binary)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing to overwrite native run {output}")
        old = read_json(output / "input-config.json")
        if old != config:
            raise ValueError(f"resume configuration differs in {output}")
        provenance = read_json(output / "provenance.json")
        if provenance["binary_sha256"] != binary_hash or provenance["raw_sha256"] != sha256(output / "raw.json"):
            raise ValueError(f"resume binary or result hash differs in {output}")
        if library_hashes(provenance) != library_hashes({"runtime_libraries": libraries}):
            raise ValueError(f"resume runtime library differs in {output}")
        for key in ("program_path", "vlm_path", "ae_schedule_path", "calibrated_vlm_path", "mlp_path"):
            if config.get(key) and provenance["inputs"][key]["sha256"] != sha256(config[key]):
                raise ValueError(f"resume input {key} differs in {output}")
        return read_json(output / "raw.json")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input-config.json", config)
    with (output / "native.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(Path(binary).resolve()), "--config", str((output / "input-config.json").resolve()),
                        "--output", str(output.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = read_json(output / "raw.json")
    if sha256(binary) != binary_hash or runtime_libraries(binary) != libraries:
        raise ValueError("native executable or runtime library changed during execution")
    inputs = {}
    for key in ("program_path", "vlm_path", "ae_schedule_path", "calibrated_vlm_path", "mlp_path"):
        if config.get(key):
            inputs[key] = {"name": Path(config[key]).name, "sha256": sha256(config[key])}
    spec = config.get("spec")
    if isinstance(spec, str) and Path(spec).is_file():
        inputs["spec"] = {"name": Path(spec).name, "sha256": sha256(spec)}
    write_json(output / "provenance.json", {
        "schema_version": 1, "source_commit": source_revision(), "binary_sha256": binary_hash,
        "runtime_libraries": libraries,
        "config_sha256": sha256(output / "input-config.json"), "inputs": inputs,
        "calibration": config.get("calibration"), "spec": config.get("spec"),
        "execution_model": config.get("execution_model", "native_dram_requests"),
        "spec_sha256": object_sha256(config.get("spec")),
        "raw_sha256": sha256(output / "raw.json"),
    })
    return raw


def metrics(raw):
    return raw.get("metrics", raw)


def total_us(raw):
    value = metrics(raw).get("total_us")
    if value is None:
        raise ValueError("native raw.json has no total_us")
    return float(value)
