"""Saved-calibration export integrity and integer-cycle timing regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.export_calibrated_vlm import export_ledger, object_sha256, sha256


class CalibratedVLMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = {key: Path(self.tmp.name) / f"{key}.json"
                      for key in ("calibration", "raw", "provenance", "config", "vlm")}
        clock_ps = 938
        spec = dict(clock_ps=clock_ps, mac_freq_mult=4, metadata={"test_fixture": True})
        params = dict(alpha=0.5, window=1024, write_high=768, write_low=256)
        units, observed, cycle, rd_sum, wr_sum = [], [], 0, 0, 0
        for ordinal in range(2357):
            duration = 100 + ordinal % 7
            kind = "kernel" if ordinal < 2353 else "memory"
            reads, writes = ordinal % 11, ordinal % 5
            rd, wr = reads * 32, ((writes + 1) // 2 if kind == "kernel" else writes) * 32
            units.append(dict(ordinal=ordinal, name=f"{kind}-{ordinal}", kind=kind,
                              us=(duration - 2) * clock_ps / 1000000,
                              read_sectors=reads, write_sectors=writes, write_is_proxy=kind == "kernel"))
            observed.append(dict(ordinal=ordinal, start_us=cycle * clock_ps / 1000000,
                                 end_us=(cycle + duration) * clock_ps / 1000000,
                                 duration_us=duration * clock_ps / 1000000, read_bytes=rd, write_bytes=wr))
            cycle += duration
            rd_sum += rd
            wr_sum += wr
        total = cycle * clock_ps / 1000000
        target = total / 1.02
        self.docs = dict(vlm=dict(units=units, target_us=target),
                         raw=dict(clock_ps=clock_ps, stages=[], kernels=observed,
                                  metrics=dict(total_us=total, VLM_remaining_us=total, AE_elapsed_us=0,
                                               AE_GPU_wait_us=0, pim_read_bytes=0, pim_write_bytes=0,
                                               gpu_read_bytes=rd_sum, gpu_write_bytes=wr_sum)),
                         config=dict(run_kind="vlm_solo", spec=spec, calibration=params, vlm_path="/frozen/vlm.json"))
        common = dict(binary_sha256="b" * 64, source_commit="c" * 40, spec=spec,
                      spec_sha256=object_sha256(spec),
                      runtime_libraries={"libramulator.so": {"path": "/original/library.so", "sha256": "d" * 64}})
        self.docs["provenance"] = dict(copy.deepcopy(common), calibration=copy.deepcopy(params), inputs={})
        self.docs["calibration"] = dict(copy.deepcopy(common), parameters=copy.deepcopy(params), selected_candidate=1,
                                        raw_path="/original/candidate-01/raw.json",
                                        validation=dict(passed=True, work_units=2357, kernel_count=2353,
                                                        total_us=total, target_us=target, relative_error=total / target - 1))
        self.persist()

    def save(self, key):
        self.paths[key].write_text(json.dumps(self.docs[key], ensure_ascii=False), encoding="utf-8")

    def persist(self):
        for key in ("vlm", "raw", "config"):
            self.save(key)
        provenance, calibration = self.docs["provenance"], self.docs["calibration"]
        provenance.update(raw_sha256=sha256(self.paths["raw"]), config_sha256=sha256(self.paths["config"]),
                          inputs={"vlm_path": {"name": "vlm.json", "sha256": sha256(self.paths["vlm"])}})
        calibration.update(raw_sha256=sha256(self.paths["raw"]), vlm_sha256=sha256(self.paths["vlm"]))
        self.save("provenance")
        self.save("calibration")

    def test_preserves_all_units_memory_and_exact_cycles(self):
        ledger = export_ledger(**self.paths)
        self.assertEqual(ledger, export_ledger(**self.paths))
        self.assertEqual(ledger["kind"], "calibrated_vlm_durations")
        self.assertEqual((ledger["work_units"], ledger["kernel_count"], ledger["memory_count"]), (2357, 2353, 4))
        self.assertEqual([u["kind"] for u in ledger["units"][-4:]], ["memory"] * 4)
        self.assertEqual([u["duration_cycles"] for u in ledger["units"]], [100 + i % 7 for i in range(2357)])
        self.assertEqual(sum(u["duration_cycles"] for u in ledger["units"]), ledger["total_cycles"])
        self.assertEqual(ledger["total_us"], self.docs["raw"]["metrics"]["total_us"])
        self.assertEqual(ledger["source"]["provenance_sha256"], sha256(self.paths["provenance"]))
        self.assertEqual(ledger["source"]["binary_sha256"], "b" * 64)
        self.assertEqual(ledger["units"][0]["measured_us"], self.docs["vlm"]["units"][0]["us"])

    def test_rejects_unaccepted_calibration(self):
        self.docs["calibration"]["validation"]["passed"] = False
        self.persist()
        with self.assertRaisesRegex(ValueError, "has not passed"):
            export_ledger(**self.paths)

    def test_rejects_raw_bytes_changed_without_provenance(self):
        self.paths["raw"].write_text(self.paths["raw"].read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "raw hash mismatch"):
            export_ledger(**self.paths)

    def test_rejects_unbound_vlm_input(self):
        self.docs["vlm"]["units"][0]["name"] = "changed"
        self.save("vlm")
        with self.assertRaisesRegex(ValueError, "VLM input hash mismatch"):
            export_ledger(**self.paths)

    def test_rejects_unbound_config(self):
        self.docs["config"]["extra"] = 1
        self.save("config")
        with self.assertRaisesRegex(ValueError, "configuration hash mismatch"):
            export_ledger(**self.paths)

    def test_rejects_spec_mismatch(self):
        self.docs["provenance"]["spec_sha256"] = "e" * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, "spec hash mismatch"):
            export_ledger(**self.paths)

    def test_rejects_binary_identity_mismatch(self):
        self.docs["provenance"]["binary_sha256"] = "e" * 64
        self.persist()
        with self.assertRaisesRegex(ValueError, "binary hash mismatch"):
            export_ledger(**self.paths)

    def test_rejects_missing_library_provenance(self):
        del self.docs["provenance"]["runtime_libraries"]
        self.persist()
        with self.assertRaisesRegex(ValueError, "source provenance"):
            export_ledger(**self.paths)

    def test_rejects_missing_or_reordered_native_units(self):
        saved = copy.deepcopy(self.docs["raw"]["kernels"])
        for broken in (saved[:-1], [saved[1], saved[0], *saved[2:]]):
            self.docs["raw"]["kernels"] = broken
            self.persist()
            with self.assertRaisesRegex(ValueError, "missing work units|ordinal order"):
                export_ledger(**self.paths)

    def test_rejects_duplicate_source_ordinals(self):
        self.docs["vlm"]["units"][1]["ordinal"] = 0
        self.persist()
        with self.assertRaisesRegex(ValueError, "unique integers"):
            export_ledger(**self.paths)

    def test_rejects_fractional_native_cycle(self):
        self.docs["raw"]["kernels"][0]["duration_us"] += 0.000469
        self.persist()
        with self.assertRaisesRegex(ValueError, "integer number"):
            export_ledger(**self.paths)

    def test_rejects_gap_not_silently_absorbed_into_durations(self):
        self.docs["raw"]["kernels"][1]["start_us"] += 0.000938
        self.persist()
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            export_ledger(**self.paths)

    def test_rejects_total_not_matching_units(self):
        self.docs["calibration"]["validation"]["total_us"] += 0.000938
        self.persist()
        with self.assertRaisesRegex(ValueError, "total duration"):
            export_ledger(**self.paths)

    def test_rejects_clock_mismatch(self):
        self.docs["raw"]["clock_ps"] = 1000
        self.persist()
        with self.assertRaisesRegex(ValueError, "clock mismatch"):
            export_ledger(**self.paths)

    def test_rejects_shorter_than_measured_compute(self):
        self.docs["vlm"]["units"][0]["us"] = 1.0
        self.persist()
        with self.assertRaisesRegex(ValueError, "shorter than measured"):
            export_ledger(**self.paths)

    def test_rejects_gpu_byte_loss(self):
        self.docs["raw"]["kernels"][1]["read_bytes"] -= 32
        self.persist()
        with self.assertRaisesRegex(ValueError, "byte conservation"):
            export_ledger(**self.paths)

    def test_rejects_combined_source(self):
        self.docs["config"]["run_kind"] = "combined"
        self.persist()
        with self.assertRaisesRegex(ValueError, "VLM solo"):
            export_ledger(**self.paths)


if __name__ == "__main__":
    unittest.main()
