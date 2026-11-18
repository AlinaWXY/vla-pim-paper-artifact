"""Compare every native Operation field with independent Python JSON parsing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


FIELDS = [("op", ""), ("epoch", ""), ("id", None), ("slot", 0), ("result_slot", 0),
          ("rows", 1), ("pairs", 1), ("dest_slot", -1), ("source_bank", 0), ("tensor_fragment", -1),
          ("output_columns", 1), ("output_column", 0), ("reduction_word", 16), ("fragment", 0), ("output_tile", 0),
          ("kv_token", 0), ("lanes", 16), ("elements", 16), ("bytes", 32), ("token", 0),
          ("release_result", 0), ("dest", ""), ("source", "")]


def semantic_digest(program):
    hash64, count = 14695981039346656037, 0

    def add(value):
        nonlocal hash64
        data = str(value).encode("utf-8")
        for b in str(len(data)).encode("ascii") + b":" + data:
            hash64 = ((hash64 ^ b) * 1099511628211) & ((1 << 64) - 1)

    def operations(ops):
        nonlocal count
        add(len(ops))
        for i, op in enumerate(ops):
            count += 1
            for name, default in FIELDS:
                value = op.get(name, i if name == "id" else default)
                add(int(value) if isinstance(value, bool) else value)
            for name in ("addr", "deps", "issue_deps"):
                values = op.get(name, [])
                add(len(values))
                for v in values:
                    add(v)
            regions = op.get("source_regions", [])
            add(len(regions))
            for region in regions:
                for name in ("bank", "byte_offset", "bytes"):
                    add(region[name])
            operations(op.get("children", []))

    stages = sorted(program["stages"], key=lambda s: s["name"])
    add(len(stages))
    for stage in stages:
        add(stage["name"])
        for channel in stage["channels"]:
            operations(channel)
    return dict(stages=len(stages), operations=count, semantic_digest64=f"{hash64:016x}",
                layer_stride_bytes=program.get("tensor_contract", {}).get("layer_stride_bytes_per_bank", 1048576))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("programs", nargs="+", type=Path)
    args = parser.parse_args()
    output = []
    for path in args.programs:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        native = json.loads(subprocess.check_output([str(args.binary.resolve()), str(path)], text=True))
        expected = semantic_digest(json.loads(path.read_text(encoding="utf-8")))
        for key, value in expected.items():
            if native[key] != value:
                raise AssertionError((path, key, native[key], value))
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise AssertionError(f"program changed while verifying: {path}")
        output.append(dict(program=path.name, sha256=before, fields_match=True, **native))
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        invalid = [
            {"schema_version": 2, "stages": []},
            {"schema_version": 1, "stages": [{"name": "bad"}]},
            {"schema_version": 1, "stages": [{"name": "bad", "channels": [[]]}]},
            {"schema_version": 1, "stages": [{"name": "bad", "channels": [[{"op": "WRGB", "addr": [0, 0, -1, -1, 0, 0]}]] + [[]] * 15}]},
            {"schema_version": 1, "stages": [{"name": "bad", "channels": [[{"op": "WRGB", "addr": [0, 1, -1, -1, 0, 0], "deps": [0]}]] + [[]] * 15}]},
            {"schema_version": 1, "stages": [{"name": "bad", "channels": [[{"op": "WRGB", "addr": [0, 1, -1, -1, 0, 0], "issue_deps": [0]}]] + [[]] * 15}]},
            {"schema_version": 1, "stages": [{"name": "bad", "channels": [[{"op": "REDUCE", "addr": [0, 1, -1, -1, 0, 0], "reduction_word": 512}]] + [[]] * 15}]},
        ]
        for i, program in enumerate(invalid):
            path = directory / f"invalid-{i}.json"
            path.write_text(json.dumps(program), encoding="utf-8")
            result = subprocess.run([str(args.binary.resolve()), str(path)], capture_output=True)
            if result.returncode == 0:
                raise AssertionError(f"invalid fixture {i} was accepted")
    print(json.dumps({"programs": output, "malformed_inputs_rejected": len(invalid)}, indent=2))


if __name__ == "__main__":
    main()
