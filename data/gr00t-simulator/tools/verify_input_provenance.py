"""Verify persisted inputs; optionally rebind only proven LF/CRLF changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_inputs(root, *, rebind_line_endings=False, check_programs=True):
    root = Path(root).resolve()
    provenance_path = root / "input-data/provenance.json"
    manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
    changes, checked = [], 0
    for group in ("sources", "generated"):
        for name, record in manifest[group].items():
            relative = "input-data/" + ("measured/" if group == "sources" else "") + name
            data = (root / relative).read_bytes()
            expected = record["sha256"] if group == "sources" else record
            actual = digest(data)
            checked += 1
            if actual == expected:
                continue
            lf = data.replace(b"\r\n", b"\n")
            alternatives = {digest(lf), digest(lf.replace(b"\n", b"\r\n"))}
            if not rebind_line_endings or expected not in alternatives:
                raise ValueError(f"input hash differs: {relative}; expected {expected}, actual {actual}")
            changes.append(dict(path=relative, previous_sha256=expected, sha256=actual,
                                reason="Only LF/CRLF byte conversion; no JSON data or timing change"))
            if group == "sources":
                record["sha256"] = actual
            else:
                manifest[group][name] = actual
    programs = []
    if check_programs:
        for path in sorted((root / "programs/generated").glob("*.json")):
            program = json.loads(path.read_text(encoding="utf-8"))
            bindings = program["source_sha256"]
            if set(bindings) != {"programs/gen_pim_program.py", "programs/descriptors.py", "workloads/layout.py"}:
                raise ValueError(f"incomplete program source bindings: {path}")
            for name, expected in bindings.items():
                if digest((root / name).read_bytes()) != expected:
                    raise ValueError(f"program {path.name} source binding differs: {name}")
                checked += 1
            programs.append(path.name)
        if len(programs) != 5:
            raise ValueError(f"expected five generated PIM programs, found {len(programs)}")
    if changes:
        manifest.setdefault("line_ending_rebindings", []).extend(changes)
        provenance_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8", newline="\n")
    return dict(verified_hashes=checked, programs=programs, line_ending_rebindings=changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rebind-line-endings", action="store_true",
                        help="update metadata only when the old hash matches a pure LF/CRLF conversion")
    args = parser.parse_args()
    print(json.dumps(verify_inputs(args.root, rebind_line_endings=args.rebind_line_endings), indent=2))


if __name__ == "__main__":
    main()
