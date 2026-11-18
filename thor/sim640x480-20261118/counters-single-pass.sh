#!/bin/bash
set -euo pipefail
project="GR00T N1.6"
run=/var/tmp/anonymous-gr00t16/sim640x480-20261118
runtime=/var/tmp/anonymous-gr00t16
scratch=/dev/shm/anonymous-gr00t-sim640x480-20261118
out="$project/thor/sim640x480-20261118"
python=/var/tmp/vla-venv/bin/python
exec 9>"$runtime/capture-local.lock"
flock 9
export PYTHONPATH="$run/deps:$runtime/profile-640x480-20261118/deps:$runtime/deps"
export HF_HOME=/var/tmp/anonymous-gr00t16-hf HF_MODULES_CACHE="$scratch/hfmodules"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR=/var/tmp/anonymous-gr00t16-inductor TRITON_CACHE_DIR="$scratch/triton"
export TMPDIR="$scratch/tmp" TORCHINDUCTOR_COMPILE_THREADS=4 OPENBLAS_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1
ncu=/opt/nvidia/nsight-compute/2026.1.1/ncu
args=(-u "$run/source/tools/capture_gr00t_thor.py" --source "$runtime/source" --checkpoint "$scratch/checkpoint"
      --output "$run/capture" --image-width 640 --image-height 480 --default-image-preprocess
      --expected-vit-size 336 252 --cameras 3 --samples 20 --mode counter)
for item in 'reads:lts__t_sectors_op_read_lookup_miss.sum' 'writes:lts__t_sectors_op_write.sum'; do
 short=${item%%:*}; metric=${item#*:}
 "$ncu" --config-file off --target-processes all --kernel-name-base demangled --profile-from-start off   --replay-mode kernel --cache-control none --metrics "$metric,profiler__replayer_passes"   --csv --page raw --log-file "$run/capture/$short.csv" --force-overwrite   "$python" "${args[@]}" 2>&1 | tee "$run/capture/$short.log"
 "$python" - "$run/capture/$short.csv" "$run/capture/compiled-prefix-0.json" <<'PY'
import csv,json,sys
with open(sys.argv[1]) as f:
 header=next(line for line in f if line.startswith('"ID","Process ID"'))
 rows=list(csv.DictReader(f,fieldnames=next(csv.reader([header]))))[1:]
expected=sum(e.get("cat")=="kernel" and e.get("ph")=="X" for e in json.load(open(sys.argv[2]))["traceEvents"])
assert len(rows)==expected,(len(rows),expected)
assert all(int(r["ID"])==i and float(r["profiler__replayer_passes"])==1 for i,r in enumerate(rows))
print("Verified single-pass counters",len(rows),flush=True)
PY
 cp "$run/capture/$short.csv" "$run/capture/$short.log" "$out/"
done
cp "$run/counters-single-pass.sh" "$out/"
date -Is | tee "$out/completed.txt"

