"""Export semantic requests for the dependency-free C++ resource replay test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from programs.gen_pim_program import generate, VARIANTS, walk

FIELDS = dict(slot=0, dest_slot=-1, source_bank=0, tensor_fragment=-1, rows=1,
              output_columns=1, output_column=0, fragment=0, output_tile=0,
              kv_token=0, elements=16, result_slot=0, token=0, release_result=False, reduction_word=16)

with Path(sys.argv[1]).open("w", encoding="ascii", newline="\n") as stream:
    for variant in VARIANTS:
        for stage in generate(variant, 1)["stages"]:
            stream.write("stage " + stage["name"] + "\n")
            for op in walk(stage["channels"][0]):
                if op["op"] in ("MAC_RUN", "ROPE_RUN"):
                    continue
                row = [op["op"], op.get("dest", "-"), op.get("source", "-"),
                       op["addr"][2], op["addr"][3]]
                row.extend(int(op.get(k, v)) for k, v in FIELDS.items())
                regions = op.get("source_regions", [])
                row.append(len(regions))
                for region in regions:
                    row.extend(region[k] for k in ("bank", "byte_offset", "bytes"))
                stream.write(" ".join(map(str, row)) + "\n")
