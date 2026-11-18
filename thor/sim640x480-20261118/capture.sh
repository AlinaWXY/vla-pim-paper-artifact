#!/usr/bin/env bash
set -euo pipefail
project="GR00T N1.6"
run=/var/tmp/anonymous-gr00t16/sim640x480-20261118
runtime=/var/tmp/anonymous-gr00t16
scratch=/dev/shm/anonymous-gr00t-sim640x480-20261118
python=/var/tmp/vla-venv/bin/python
out="$project/thor/sim640x480-20261118"
mkdir -p "$scratch/checkpoint" "$scratch/tmp" "$scratch/hfmodules" "$run/capture" "$out"
date -Is
for name in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  if [ "$name" = model-00001-of-00002.safetensors ]; then size=4991091456; else size=1582283096; fi
  if [ "$(stat -c %s "$scratch/checkpoint/$name" 2>/dev/null || true)" != "$size" ]; then
    cp "$project/thor/checkpoint/$name" "$scratch/checkpoint/$name"
  fi
done
cp "$project/thor/checkpoint/"*.json "$scratch/checkpoint/"
echo "Simulator checkpoint staged independently"
profile="$project/thor/profile-640x480-20261118/640x480/profile-summary.json"
while ! "$python" - "$profile" <<'PY'
import json,sys
try:
    p=json.load(open(sys.argv[1]))
    sys.exit(0 if p.get("completed_at_utc") else 1)
except (FileNotFoundError,json.JSONDecodeError): sys.exit(1)
PY
do sleep 10; done
"$python" - "$profile" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["completed_at_utc"] and p["image_size"] == [480, 640]
assert p["vit_input_shapes"] == [[1,3,252,336]]*3, p["vit_input_shapes"]
assert p["processed_image_sizes"] == [[252,336]]*3
assert p["dit_query_shapes"] == [[1,16,1536]]*4
print("640x480 profile finished with verified 336x252 ViT tensors",flush=True)
PY
printf '%s  %s\n' \
  9710d31d331b79cfa229fd23605ba9ad47e207cb0f4c7722fe0bacdc666c8326 "$scratch/checkpoint/model-00001-of-00002.safetensors" \
  eb79bbd7893068897cda18c86ed91064e92e20e356dea97bc398bf9c8bf2fa35 "$scratch/checkpoint/model-00002-of-00002.safetensors" | sha256sum -c -
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
      --expected-vit-size 336 252 --cameras 3 --samples 20)
put() { cp "$@" "$out/"; }
{
 date -Is
 hostname
 nvidia-smi -L
 nvpmodel -q
 "$python" -V
 sha256sum "$run/source/tools/"*.py "$run/source/workloads/"*.py
} > "$run/capture/environment.txt"
put "$run/capture/environment.txt"
cp -a "$run/source" "$out/"
"$python" "${args[@]}" --mode fixture 2>&1 | tee "$run/capture/fixture.log"
put "$run/capture/fixture.json" "$run/capture/fixture.log"
"$python" "${args[@]}" --mode trace 2>&1 | tee "$run/capture/vlm.log"
put "$run/capture/capture-summary.json" "$run/capture"/compiled-prefix-?.json "$run/capture/vlm.log"
"$python" "${args[@]}" --mode ae 2>&1 | tee "$run/capture/ae.log"
put "$run/capture/ae-latency.json" "$run/capture/ae-ledger.json" "$run/capture/ae.log"
"$python" -u "$run/source/tools/capture_gr00t_cold_consumers.py" --fixture "$run/capture/fixture.json" \
 --output "$run/capture/cold-consumers.json" --flush-mib 192 2>&1 | tee "$run/capture/cold-consumers.log"
put "$run/capture/cold-consumers.json" "$run/capture/cold-consumers.log"
for item in 'reads:lts__t_sectors_op_read_lookup_miss.sum' 'writes:lts__t_sectors_op_write.sum'; do
 short=${item%%:*}; metric=${item#*:}
 "$ncu" --config-file off --kernel-name-base demangled --target-processes all --profile-from-start off --replay-mode kernel --cache-control none \
  --metrics "$metric,profiler__replayer_passes" --csv --page raw --log-file "$run/capture/$short.csv" --force-overwrite \
  "$python" "${args[@]}" --mode counter 2>&1 | tee "$run/capture/$short.log"
 put "$run/capture/$short.csv" "$run/capture/$short.log"
done
date -Is | tee "$out/completed.txt"

