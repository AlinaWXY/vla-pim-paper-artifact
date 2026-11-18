import json
from pathlib import Path
import tempfile
import unittest

from tools.calibrate_kernel_mlp import calibrate_unit, native_probe


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/pim_vla_sim"


class KernelMlpTests(unittest.TestCase):
    def test_bisection_and_classification(self):
        unit = dict(ordinal=0, kind="kernel", name="synthetic", us=100)
        row = calibrate_unit(lambda w: max(90, 300 - w * 10), unit, 32, 1)
        self.assertEqual((row["classification"], row["w_k"]), ("memory_limited", 20))
        row = calibrate_unit(lambda w: 80, unit, 32, 1)
        self.assertEqual((row["classification"], row["w_k"]), ("compute_limited", 32))
        row = calibrate_unit(lambda w: 101, unit, 32, 1)
        self.assertEqual((row["classification"], row["w_k"]), ("simulator_slower_than_thor", 32))

    @unittest.skipUnless(BINARY.is_file(), "requires built native simulator")
    def test_synthetic_native_window_monotonicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            workload, config = (Path(tmp) / name for name in ("vlm.json", "probe.json"))
            workload.write_text(json.dumps({"units": [
                dict(ordinal=0, kind="kernel", name="read-only", us=100,
                     read_sectors=16 * 192, write_sectors=0, base_sector=0),
            ]}), encoding="utf-8")
            config.write_text(json.dumps(dict(spec_path=str(ROOT / "reference/spec/spec.json"),
                                              vlm_path=str(workload), w_max=128)), encoding="utf-8")
            with native_probe(BINARY, config) as probe:
                windows = (1, 2, 4, 8, 16, 32, 64, 128)
                durations = [probe(0, w) for w in windows]
            self.assertEqual(durations, sorted(durations, reverse=True))
            self.assertGreater(durations[0], durations[-1])
