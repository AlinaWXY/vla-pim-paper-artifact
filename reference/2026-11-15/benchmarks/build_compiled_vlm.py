"""Match the compiled prefix's CUDA timeline to single-pass Thor L2 counters."""

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from prepare_uniform_demand import canonical_name


READS = "lts__t_sectors_op_read_lookup_miss.sum"
WRITES = "lts__t_sectors_op_write.sum"
AUTOTUNED = ("triton_red_fused__to_copy_add_convolution_embedding_native_layer_norm_transpose_view_5",
             "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_20")


def counter_name(name):
    value = canonical_name(name).replace("cudnn::cnn::kern_precompute_indices", "cudnn::kern_precompute_indices")
    return re.sub(r"(?<=[<,])(false|true)(?=[,>])", lambda x: "0" if x[0] == "false" else "1", value)


def timings(path):
    trace = json.loads(path.read_bytes())["traceEvents"]
    events = [e for e in trace if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    return sorted(events, key=lambda e: e["ts"])


def identity(event):
    data = event["args"]
    return (event["cat"], canonical_name(event["name"]), data.get("grid"), data.get("block"), data.get("bytes"))


def counters(path, metric):
    with path.open(newline="") as handle:
        header = next(line for line in handle if line.startswith('"ID","Process ID"'))
        rows = list(csv.DictReader(handle, fieldnames=next(csv.reader([header]))))
    assert rows.pop(0)[metric] == "sector", (path, metric)
    for i, row in enumerate(rows):
        assert int(row["ID"]) == i and float(row["profiler__replayer_passes"]) == 1
        sectors = float(row[metric].replace(",", ""))
        assert sectors >= 0 and sectors.is_integer(), (path, metric, i)
        row["sectors"] = int(sectors)
    return rows


def build(directory, *, counter_identity=counter_name, launch_variant_validator=None):
    paths = [directory / f"compiled-prefix-{i}.json" for i in range(5)]
    series = [timings(path) for path in paths]
    first = series[0]
    assert first and all(list(map(identity, sample)) == list(map(identity, first)) for sample in series)
    reads_path = directory / "reads.csv"
    writes_path = directory / "writes.csv"
    reads, writes = counters(reads_path, READS), counters(writes_path, WRITES)
    assert len(reads) == len(writes) == sum(e["cat"] == "kernel" for e in first)
    pairs = iter(zip(reads, writes))
    units = []
    alternate_launches = []
    cursor = 0
    for i, event in enumerate(first):
        name, category, data = event["name"], event["cat"], event["args"]
        if category == "kernel":
            r, w = next(pairs)
            for direction, row in (("read", r), ("write", w)):
                assert counter_identity(row["Kernel Name"]) == counter_identity(name), (i, name, row["Kernel Name"])
                grid, block = (list(ast.literal_eval(row[field])) for field in ("Grid Size", "Block Size"))
                if (grid, block) != (data["grid"], data["block"]):
                    assert name in AUTOTUNED or (launch_variant_validator is not None and
                        launch_variant_validator(event, row)), (i, name, grid, block, data)
                    alternate_launches.append(dict(ordinal=i, name=name, direction=direction, sectors=row["sectors"],
                                                   counter_grid=grid, counter_block=block,
                                                   timing_grid=data["grid"], timing_block=data["block"]))
            nr, nw = r["sectors"], w["sectors"]
        else:
            count = math.ceil(data["bytes"] / 32)
            assert category == "gpu_memset" or any(direction in name for direction in ("DtoH", "HtoD", "DtoD")), name
            nr = count if "DtoH" in name or "DtoD" in name else 0
            nw = count if "DtoH" not in name else 0
        cursor = (cursor + 1023) // 1024 * 1024
        units.append(dict(ordinal=i, name=name, kind="kernel" if category == "kernel" else "memory",
                          read_sectors=nr, write_sectors=nw, write_is_proxy=category == "kernel",
                          us=statistics.median(sample[i]["dur"] for sample in series), base_sector=cursor))
        cursor += max(1, nr + nw)
    meta = directory / "capture-summary.json"
    source = json.loads(meta.read_bytes())
    target = source["gpu_ms"]["compile"]["mean"] * 1000
    return dict(schema_version=1, units=units, target_us=target, phase=source["phase"],
                kernel_count=len(reads), memory_units=len(first)-len(reads),
                measured_duration_sum_us=math.fsum(u["us"] for u in units),
                read_bytes=32 * sum(u["read_sectors"] for u in units),
                write_proxy_bytes=32 * sum(u["write_sectors"] for u in units if u["write_is_proxy"]),
                alternate_launch_counter_proxies=alternate_launches,
                timeline_spans_us=[max(e["ts"] + e["dur"] for e in xs) - xs[0]["ts"] for xs in series],
                source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in (*paths, reads_path, writes_path, meta)},
                assumptions=dict(addresses="Sequential synthetic sectors with the same channel striping as the old model",
                                 arrivals="Uniform releases over the measured kernel interval",
                                 timings="Median duration of five ordinal-aligned CUPTI captures in microseconds",
                                 writeback="Fit the fraction of measured L2 write sectors that reach DRAM"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = build(args.directory)
    (args.directory / "vlm.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "units"}, indent=2))
