"""Delete one native alias constraint and require the passive checker to fail."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--binary",type=Path,default=Path("build/native_microbench"))
    parser.add_argument("--spec",type=Path,default=Path("reference/spec/spec.json"))
    args=parser.parse_args()
    pristine=json.loads(args.spec.read_text())
    spec=copy.deepcopy(pristine)
    commands=spec["metadata"]["commands"]
    rd,wrgb=commands["RD"]["id"],commands["WRGB"]["id"]
    changed=0
    for rule in spec["dram"]["timing_constraints"]:
        if rule[0]==1 and len(rule)>5 and rule[5] and rd in rule[1] and wrgb in rule[2] and rule[3]==15:
            rule[2].remove(wrgb)
            changed+=1
    assert changed==1, f"expected exactly one derived sibling rule, got {changed}"
    with tempfile.TemporaryDirectory(prefix="pim-spec-fault-") as directory:
        mutated=Path(directory)/"mutated-spec.json"
        mutated.write_text(json.dumps(spec),encoding="utf-8")
        run=subprocess.run([str(args.binary.resolve()),str(mutated),"--fault-case"],capture_output=True,text=True)
    assert run.returncode==1, (run.returncode,run.stdout,run.stderr)
    assert "DQ overlap or return reordering" in run.stderr, run.stderr
    # The fault must be caught by the observer after native check_timing
    # accepted the command, not by scheduler-side DQ reservations.
    assert "scheduler bypassed" not in run.stderr,run.stderr
    print(json.dumps({"fault_injection":"one_removed_RD_to_WRGB_sibling_rule",
                      "status":"detected_by_passive_DQ_checker","diagnostic":run.stderr.strip()},sort_keys=True))


if __name__=="__main__":
    main()
