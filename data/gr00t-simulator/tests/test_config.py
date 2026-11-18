"""Acceptance contracts for design switches and measured frontend inputs."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import CASES, case_config, validate_calibration, validate_config, variant_name
from experiments.prepare_inputs import ae_schedule, normalize_vlm, layer_boundaries, single_layer_schedule, gpu_measurements
from experiments.run_single_layer import prepare as prepare_single_layer, VARIANTS as SINGLE_LAYER_VARIANTS, refresh_counts
from calib.calibrate_vlm import assess
from experiments.native import runtime_libraries, library_hashes
from analysis.report import summarize
from tools.verify_input_provenance import verify_inputs


class ConfigurationTests(unittest.TestCase):
    def test_calibrated_windows_are_only_concurrent_fused_combined(self):
        good = dict(case_config("f"), execution_model="fused_kernel_windows", calibrated_vlm_path="durations.json")
        validate_config(good)
        for change in (dict(mode="serial"), dict(dataflow="streaming"), dict(run_kind="ae_solo"),
                       dict(calibrated_vlm_path=""), dict(execution_model="unknown"), dict(execution_model="native_dram_requests")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(dict(good, **change))

    def test_matrix(self):
        self.assertEqual(len(CASES), 9)
        self.assertEqual(len({variant_name(c) for c in CASES.values()}), 5)
        for case in CASES:
            validate_config(case_config(case))

    def test_vlm_gpu_quantum_requires_native_concurrent_combined(self):
        valid = case_config("g", vlm_gpu_quantum_us=2.0)
        for change in (dict(vlm_gpu_quantum_us=0), dict(vlm_gpu_quantum_us=True),
                       dict(vlm_gpu_quantum_us=float("inf")), dict(mode="serial"),
                       dict(run_kind="ae_solo"), dict(execution_model="fused_kernel_windows")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "vlm_gpu_quantum_us"):
                validate_config(dict(valid, **change))

    def test_only_five_structural_combinations(self):
        accepted = []
        for layout, rope, dataflow in itertools.product(("feature", "token"), ("gpu", "local"), ("streaming", "fused")):
            c = dict(layout=layout, rope=rope, dataflow=dataflow, mode="serial")
            try:
                validate_config(c)
                accepted.append((layout, rope, dataflow))
            except ValueError:
                pass
        self.assertEqual(len(accepted), 5)

    def test_arbitration_presence_is_strict(self):
        for case in ("b", "f", "g"):
            for value in (None, "frfcfs", "pim_head"):
                with self.assertRaises(ValueError):
                    validate_config(dict(case_config(case), arbitration=value))
        for case in ("c", "d", "e"):
            c = case_config(case)
            del c["arbitration"]
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_retired_switches_rejected(self):
        for retired in ("pim_data_path", "pim_whole_srf", "shift", "constant_offset", "dq_latency"):
            with self.assertRaises(ValueError):
                validate_config(dict(case_config("b"), **{retired: 0}))

    def test_global_refresh_switch_is_rejected_even_when_false(self):
        for value in (True, False, None, 0, 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fixed hardware rule"):
                validate_config(dict(case_config("b"), refresh_allbank=value))

    def test_hardware_multiplier(self):
        for k in (1, 2, 4):
            validate_config(case_config("g", mac_freq_mult=k))
        for k in (0, 3, 8, True):
            with self.assertRaises(ValueError):
                validate_config(case_config("g", mac_freq_mult=k))

    def test_watermark_validation(self):
        good = dict(alpha=0.2, window=64, write_low=16, write_high=48)
        validate_calibration(good)
        for key, value in (("alpha", -0.1), ("alpha", 0), ("alpha", 0.049), ("alpha", 1.1), ("alpha", float("nan")),
                           ("write_high", 65), ("write_low", 48), ("window", True)):
            with self.assertRaises(ValueError):
                validate_calibration(dict(good, **{key: value}))


class RuntimeProvenanceTests(unittest.TestCase):
    def test_loaded_library_hash_tracks_content_not_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "libramulator.so"
            path.write_bytes(b"first library")
            line = f"libramulator.so => {path} (0x0000000123450000)\n"
            with mock.patch("experiments.native.subprocess.check_output", return_value=line):
                first = runtime_libraries(Path(temporary) / "simulator")
                path.write_bytes(b"changed library")
                second = runtime_libraries(Path(temporary) / "simulator")
            self.assertNotEqual(library_hashes({"runtime_libraries": first}), library_hashes({"runtime_libraries": second}))

    def test_missing_runtime_is_rejected(self):
        with mock.patch("experiments.native.subprocess.check_output", return_value="libramulator.so => not found\n"):
            with self.assertRaises(RuntimeError):
                runtime_libraries("simulator")
        with self.assertRaises(ValueError):
            library_hashes({"binary_sha256": "executable alone is insufficient"})

    def test_line_ending_rebind_proves_byte_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "input-data"
            directory.mkdir()
            payload = b'{"duration_us": 3}\n'
            original = hashlib.sha256(payload.replace(b"\n", b"\r\n")).hexdigest()
            data_path = directory / "vlm.json"
            data_path.write_bytes(payload)
            provenance = directory / "provenance.json"
            provenance.write_text(json.dumps(dict(sources={}, generated={"vlm.json": original})), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_inputs(root, check_programs=False)
            result = verify_inputs(root, rebind_line_endings=True, check_programs=False)
            self.assertEqual(len(result["line_ending_rebindings"]), 1)
            self.assertEqual(data_path.read_bytes(), payload)
            verify_inputs(root, check_programs=False)
            data_path.write_bytes(b'{"duration_us": 4}\n')
            with self.assertRaises(ValueError):
                verify_inputs(root, rebind_line_endings=True, check_programs=False)


class ReportingTests(unittest.TestCase):
    def test_single_layer_refresh_requires_both_fixed_policies_and_activity(self):
        raw = dict(refresh_policy={"rank0": "REFpb", "rank1": "REFab"},
                   refresh_commands={"rank0": {"REFpb": 128, "REFab": 0},
                                     "rank1": {"REFpb": 0, "REFab": 16}})
        self.assertEqual(refresh_counts(raw), dict(rank0_REFpb=128, rank0_REFab=0,
                                                  rank1_REFpb=0, rank1_REFab=16))
        for policy in (None, {"rank0": "REFpb", "rank1": "REFpb"},
                       {"rank0": "REFab", "rank1": "REFab"}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                refresh_counts(dict(raw, refresh_policy=policy))
        for rank, command, value in (("rank0", "REFab", 1), ("rank1", "REFpb", 1),
                                     ("rank0", "REFpb", 0), ("rank1", "REFab", 0),
                                     ("rank0", "REFpb", True), ("rank1", "REFab", -1),
                                     ("rank0", "REFpb", 1.5)):
            invalid = json.loads(json.dumps(raw))
            invalid["refresh_commands"][rank][command] = value
            with self.subTest(rank=rank, command=command, value=value), self.assertRaises(ValueError):
                refresh_counts(invalid)
        for counters in (None, {"rank0": raw["refresh_commands"]["rank0"]},
                         {"rank0": {"REFpb": 128}, "rank1": raw["refresh_commands"]["rank1"]}):
            with self.subTest(counters=counters), self.assertRaises(ValueError):
                refresh_counts(dict(raw, refresh_commands=counters))

    def test_compatibility_uses_native_completion_and_keeps_negative_interference(self):
        stage = dict(stage_id=0, name="qkv", step=0, layer=0, start_us=0, end_us=10, duration_us=10)
        solo = dict(metrics={"total_us": 20}, stages=[stage])
        vlm = dict(metrics={"total_us": 15}, kernels=[
            dict(ordinal=0, start_us=0, end_us=5), dict(ordinal=1, start_us=5, end_us=15)])
        counters = {name: index for index, name in enumerate(("GPU_act1", "GPU_act2", "GPU_prepb", "GPU_preab",
                    "GPU_reads", "GPU_writes", "GPU_row_hits", "GPU_row_misses"))}
        combined = dict(metrics=dict(total_us=25, AE_elapsed_us=18, AE_GPU_wait_us=0,
                                    VLM_remaining_us=7, **counters),
                        stages=[dict(stage, end_us=8, duration_us=8)], bus=[],
                        kernels=[dict(ordinal=0, start_us=0, end_us=5), dict(ordinal=1, start_us=5, end_us=25)])
        with tempfile.TemporaryDirectory() as temporary:
            row = summarize("c", combined, solo, vlm, temporary)
        self.assertEqual(row["PIM_interference_ms"], -0.002)
        self.assertEqual(row["completed_units"], 1)
        self.assertEqual(row["VLM_completed_model_ms"], 0.005)
        self.assertEqual(row["VLM_remaining_model_ms"], 0.007)
        self.assertEqual(row["AE_native_schedule_ms"], row["AE_solo_ms"])
        self.assertEqual(row["net_overlap_ms"], 0.01)
        self.assertEqual(row["GPU_row_misses"], counters["GPU_row_misses"])


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / "input-data"
        cls.vlm = json.loads((cls.data / "vlm.json").read_text(encoding="utf-8"))

    def test_complete_vlm_preserves_measurements(self):
        units = self.vlm["units"]
        self.assertEqual(len(units), 2357)
        self.assertEqual(sum(u["kind"] == "kernel" for u in units), 2353)
        self.assertAlmostEqual(sum(u["us"] for u in units), 105220.5782, places=5)
        self.assertAlmostEqual(self.vlm["target_us"], 104950.49603462219, places=5)
        self.assertEqual(sum(u["read_sectors"] for u in units) * 32, 11804375072)
        self.assertTrue(all(u["write_is_proxy"] == (u["kind"] == "kernel") for u in units))

    def test_logical_tensor_regions_and_physical_ring_capacity(self):
        units = self.vlm["units"]
        for left, right in zip(units, units[1:]):
            self.assertLessEqual(left["base_sector"] + left["read_sectors"] + left["write_sectors"], right["base_sector"])
            self.assertEqual(right["base_sector"] % 1024, 0)
        capacity = self.vlm["physical_rank0_capacity_bytes"]
        self.assertEqual(capacity, 16 * 1024**3)
        self.assertTrue(all((u["read_sectors"] + u["write_sectors"] + 16) * 32 <= capacity for u in units))

    def test_ae_stage_coverage_and_rope_boundary(self):
        for path in self.data.glob("ae-*.json"):
            schedule = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(all(n == 180 for n in schedule["stage_occurrences"].values()))
            events = schedule["events"]
            for step in range(10):
                for layer in range(18):
                    subset = [e for e in events if e["step"] == step and e["layer"] == layer]
                    stages = [e["stage"] for e in subset if e["kind"] == "pim"]
                    self.assertEqual(stages[0], "qkv")
                    self.assertEqual(stages[-1], "down")
                    if "_gpu_" in schedule["variant"]:
                        a = next(i for i, e in enumerate(subset) if e.get("stage") == "rope_export")
                        b = next(i for i, e in enumerate(subset) if e.get("stage") == "rope_import")
                        self.assertEqual(sum(e.get("name") == "rope_apply" for e in subset[a:b]), 10)
                    else:
                        self.assertFalse(any(e.get("name") == "rope_apply" for e in subset))

    def test_streaming_transpose_is_measured_once_between_pv_and_o(self):
        _, transpose = gpu_measurements(self.data)
        self.assertAlmostEqual(transpose["feature"], 4.352, places=9)
        self.assertAlmostEqual(transpose["token"], 4.992, places=9)
        for variant in SINGLE_LAYER_VARIANTS:
            schedule = json.loads((self.data / f"ae-{variant}.json").read_text())
            positions = [i for i, e in enumerate(schedule["events"]) if e.get("name") == "pv_to_o_transpose"]
            self.assertEqual(len(positions), 180 if variant.endswith("before") else 0)
            for i in positions:
                self.assertEqual(schedule["events"][i - 1]["stage"], "pv")
                self.assertEqual(schedule["events"][i + 1]["stage"], "o")
                self.assertEqual(schedule["events"][i]["us"], transpose[variant.split("_")[0]])

    def test_single_layer_has_both_residuals_and_excludes_global_work(self):
        ledger = json.loads((self.data / "measured/natural-ae-kernel-ledger.json").read_text())
        self.assertEqual(layer_boundaries(ledger)[0], (30, 85))
        for variant in SINGLE_LAYER_VARIANTS:
            full = json.loads((self.data / f"ae-{variant}.json").read_text())
            schedule = single_layer_schedule(full, ledger)
            self.assertTrue(all(e["step"] == 0 and e["layer"] == 0 for e in schedule["events"]))
            natural = [e["source_ordinal"] for e in schedule["events"] if e["kind"] == "gpu" and e["source_ordinal"] >= 0]
            self.assertEqual((min(natural), max(natural)), (30, 84))
            self.assertTrue({64, 65, 83, 84}.issubset(natural))
            self.assertNotIn(62, natural)
            full["events"] = [e for e in full["events"] if not (e["step"] == 0 and e["layer"] == 0 and e.get("source_ordinal") == 84)]
            with self.assertRaises(ValueError):
                single_layer_schedule(full, ledger)

    def test_single_layer_runner_prepares_only_ae_without_vlm_or_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            programs = root / "programs"
            programs.mkdir()
            for variant in SINGLE_LAYER_VARIANTS:
                (programs / f"{variant}.json").write_text("{}")
            configs = prepare_single_layer(root / "output", input_dir=self.data, program_dir=programs,
                                           spec={"mac_freq_mult": 4})
            self.assertEqual(len(configs), 5)
            for cfg in configs.values():
                self.assertEqual(cfg["run_kind"], "ae_solo")
                self.assertEqual(cfg["mode"], "serial")
                self.assertTrue({"vlm_path", "calibration", "arbitration"}.isdisjoint(cfg))
                schedule = json.loads(Path(cfg["ae_schedule_path"]).read_text())
                self.assertTrue(all(count == 1 for count in schedule["stage_occurrences"].values()))

    def test_calibration_cannot_accept_partial_or_shortened_kernel(self):
        with self.assertRaises(ValueError):
            assess({"kernels": [], "metrics": {"total_us": self.vlm["target_us"]}}, self.vlm)
        rows = [dict(ordinal=u["ordinal"], start_us=0, end_us=u["us"], duration_us=u["us"]) for u in self.vlm["units"]]
        raw = {"kernels": rows, "metrics": {"total_us": sum(u["us"] for u in self.vlm["units"])}}
        self.assertTrue(assess(raw, self.vlm)["passed"])
        rows[0]["duration_us"] = 0
        with self.assertRaises(ValueError):
            assess(raw, self.vlm)


if __name__ == "__main__":
    unittest.main()
