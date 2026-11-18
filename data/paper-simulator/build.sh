#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"${JOBS:-4}"
PYTHONPATH="$PWD/ramulator2/python${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -m unittest discover -s tests -p 'test_*.py' -v
build/native_spec_tests reference/spec/spec.json reference/spec/command-pair-matrix.json
build/native_microbench reference/spec/spec.json
build/native_lookahead_tests reference/spec/spec.json
build/native_sequencer_tests reference/spec/spec.json
build/native_refresh_tests reference/spec/spec.json
build/test_pim_resources
"${PYTHON:-python3}" tests/test_fault_injection.py
"${PYTHON:-python3}" tests/smoke_native.py
