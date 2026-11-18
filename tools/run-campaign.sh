#!/usr/bin/env bash
# Freeze once, then submit dependency-separated Slurm phases. No simulation
# starts during prepare, and no case is submitted without a passing calibration.
set -euo pipefail
action=${1:?Usage: run-campaign.sh prepare|submit|submit-rest|submit-cases RUN_DIRECTORY}
run_dir=$(realpath -m "${2:?RUN_DIRECTORY is required}")
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
partition=${PIM_PARTITION:-cpu-scavenger}
python_exec=${PYTHON:-/artifact-source/pim_vla_reproduce_anonymous_1024/.venv/bin/python}
case "$action" in prepare|submit|submit-rest|submit-cases) ;; *) echo "Unknown action: $action" >&2; exit 2;; esac

if [[ "$action" == prepare ]]; then
  test ! -e "$run_dir" || { echo "Refusing to overwrite campaign $run_dir" >&2; exit 1; }
  test -z "$(git -C "$source_root" status --porcelain --untracked-files=normal)" || {
    echo "Commit the tested source tree before freezing a campaign." >&2; exit 1;
  }
  source_commit=$(git -C "$source_root" rev-parse HEAD)
  mkdir -p "$run_dir"/{frozen/bin,frozen/lib,frozen/source,jobs,logs}
  git -C "$source_root" archive "$source_commit" | tar -x -C "$run_dir/frozen/source"
  cp -p -- "$source_root/build/pim_vla_sim" "$run_dir/frozen/bin/pim_vla_sim"
  cp -p -- "$source_root/ramulator2/libramulator.so" "$run_dir/frozen/lib/libramulator.so"
  printf '%s\n' "$source_commit" > "$run_dir/frozen/SOURCE_COMMIT"
  printf '%s\n' "$python_exec" > "$run_dir/python-path"
  printf '%s\n' "$partition" > "$run_dir/partition"
  export PYTHONDONTWRITEBYTECODE=1 PIM_VLA_SOURCE_COMMIT="$source_commit"
  "$python_exec" "$run_dir/frozen/source/programs/gen_pim_program.py" --output "$run_dir/frozen/source/programs/generated" --compact
  "$python_exec" "$run_dir/frozen/source/tools/verify_input_provenance.py" --root "$run_dir/frozen/source"
  "$python_exec" "$run_dir/frozen/source/experiments/run.py" --output "$run_dir" --prepare-only --vlm-release uniform --mac-freq-mult "${PIM_MAC_FREQ_MULT:-4}"
  (cd "$run_dir" && sha256sum frozen/bin/pim_vla_sim frozen/lib/libramulator.so > frozen/runtime-SHA256SUMS)
  (cd "$run_dir/frozen/source" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > ../source-SHA256SUMS)
  (cd "$run_dir" && find configs -type f -print0 | sort -z | xargs -0 sha256sum > frozen/config-SHA256SUMS)
  cat > "$run_dir/jobs/worker.sbatch" <<'BATCH'
#!/usr/bin/env bash
set -euo pipefail
: "${PIM_RUN:?}" "${PIM_PHASE:?}"
source /opt/rh/gcc-toolset-14/enable
export LD_LIBRARY_PATH="$PIM_RUN/frozen/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIM_VLA_SOURCE_COMMIT="$(cat "$PIM_RUN/frozen/SOURCE_COMMIT")"
python_exec=$(cat "$PIM_RUN/python-path")
(cd "$PIM_RUN" && sha256sum -c frozen/runtime-SHA256SUMS --quiet && sha256sum -c frozen/config-SHA256SUMS --quiet)
(cd "$PIM_RUN/frozen/source" && sha256sum -c ../source-SHA256SUMS --quiet)
cd "$PIM_RUN/frozen/source"
export PYTHONPATH="$PWD/ramulator2/python"
exec "$python_exec" tools/campaign_worker.py --run "$PIM_RUN" --phase "$PIM_PHASE" --index "${SLURM_ARRAY_TASK_ID:-0}"
BATCH
  printf 'Prepared immutable campaign: %s\nSource commit: %s\n' "$run_dir" "$source_commit"
  exit 0
fi

test -f "$run_dir/jobs/worker.sbatch" || { echo "Run prepare first." >&2; exit 1; }
partition=$(cat "$run_dir/partition")
python_exec=$(cat "$run_dir/python-path")
submit_job() {
  local phase=$1; shift
  local wall_time=23:55:00
  [[ "$partition" == dev ]] && wall_time=11:55:00
  [[ "$phase" == collect || "$phase" == aggregate ]] && wall_time=00:15:00
  sbatch --parsable --partition="$partition" --ntasks=1 --cpus-per-task=1 --mem=8G --time="$wall_time" \
    --job-name="pim1109-$phase" --output="$run_dir/logs/$phase-%A_%a.log" \
    --export="ALL,PIM_RUN=$run_dir,PIM_PHASE=$phase" "$@" "$run_dir/jobs/worker.sbatch"
}

if [[ "$action" == submit ]]; then
  test ! -f "$run_dir/jobs/solo.id" || { echo "Initial campaign jobs already submitted." >&2; exit 1; }
  solo_id=$(submit_job solo --array=0-4%5)
  printf '%s\n' "$solo_id" > "$run_dir/jobs/solo.id"
  calibration_id=$(submit_job calibrate --array=0-5%6 --dependency="afterok:$solo_id")
  printf '%s\n' "$calibration_id" > "$run_dir/jobs/calibration.id"
  collect_id=$(submit_job collect --dependency="afterok:$calibration_id")
  printf '%s\n' "$collect_id" > "$run_dir/jobs/collect.id"
  printf 'AE solos %s -> calibration candidates 0..5 %s -> collect %s\n' "$solo_id" "$calibration_id" "$collect_id"
  echo "After collect passes, run submit-cases. If all initial candidates fail, run submit-rest."
elif [[ "$action" == submit-rest ]]; then
  test ! -f "$run_dir/jobs/calibration-rest.id" || { echo "Remaining candidates already submitted." >&2; exit 1; }
  solo_id=$(cat "$run_dir/jobs/solo.id")
  initial_id=$(cat "$run_dir/jobs/calibration.id")
  calibration_id=$(submit_job calibrate --array=6-15%6 --dependency="afterok:$solo_id:$initial_id")
  printf '%s\n' "$calibration_id" > "$run_dir/jobs/calibration-rest.id"
  collect_id=$(submit_job collect --dependency="afterok:$calibration_id")
  printf '%s\n' "$collect_id" > "$run_dir/jobs/collect.id"
  printf 'Remaining calibration candidates %s -> collect %s\n' "$calibration_id" "$collect_id"
else
  test ! -f "$run_dir/jobs/cases.id" || { echo "Case jobs already submitted." >&2; exit 1; }
  "$python_exec" -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["validation"]["passed"], "VLM calibration has not passed"' "$run_dir/calibration/calibration.json"
  collect_id=$(cat "$run_dir/jobs/collect.id")
  case_id=$(submit_job case --array=0-8%9 --dependency="afterok:$collect_id")
  printf '%s\n' "$case_id" > "$run_dir/jobs/cases.id"
  aggregate_id=$(submit_job aggregate --dependency="afterok:$case_id")
  printf '%s\n' "$aggregate_id" > "$run_dir/jobs/aggregate.id"
  printf 'Cases b..g plus e arbitration %s -> aggregate %s\n' "$case_id" "$aggregate_id"
fi
