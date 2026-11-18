"""Fused kernel windows preserve tensors while exporting only completed DRAM data."""
import copy
from collections import Counter
import unittest

from programs.gen_pim_program import generate, kernel_window_metadata, walk
from tests.test_programs import completion_ancestors, audit_transient_array_producers
from workloads import layout as L


class FusedKernelProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = ("token_gpu_after", "token_local_after")
        cls.legacy = {v: generate(v, 1) for v in cls.variants}
        cls.programs = {v: generate(v, 1, execution_model="fused_kernel_windows") for v in cls.variants}

    def stage(self, program, name):
        return next(s for s in program["stages"] if s["name"] == name)

    def test_mode_is_explicit_and_legacy_remains_default(self):
        for variant in self.variants:
            self.assertEqual(self.legacy[variant]["execution_model"], "native_dram_requests")
            self.assertEqual(self.programs[variant]["execution_model"], "fused_kernel_windows")
            self.assertEqual(self.legacy[variant], generate(variant, 1, execution_model="native_dram_requests"))
        with self.assertRaisesRegex(ValueError, "requires a fused"):
            generate("token_gpu_before", 1, execution_model="fused_kernel_windows")
        with self.assertRaisesRegex(ValueError, "execution model"):
            generate("token_gpu_after", 1, execution_model="unknown")

    def test_external_bytes_and_arithmetic_are_preserved(self):
        for variant, program in self.programs.items():
            legacy = self.legacy[variant]
            self.assertEqual(program["expected_dq_bytes"], legacy["expected_dq_bytes"])
            for name in ("WR", "MACAB", "MACQK", "ROPE_EXEC", "BCAST", "GATHER", "CAST"):
                self.assertEqual(program["counts"].get(name, 0), legacy["counts"].get(name, 0))
            self.assertEqual(program["counts"].get("RDRES", 0), 0)
            self.assertEqual(program["counts"]["LOCALWR"] - legacy["counts"]["LOCALWR"], 2176)
            self.assertEqual(program["counts"]["RD"] - legacy["counts"]["RD"], 2176)
            for stage in program["stages"]:
                old = self.stage(legacy, stage["name"])
                self.assertEqual(stage["expected_dq_bytes"], old["expected_dq_bytes"])
                old_macs = [o for o in walk(old["channels"][0]) if o["op"] in {"MACAB", "MACQK"}]
                new_macs = [o for o in walk(stage["channels"][0]) if o["op"] in {"MACAB", "MACQK"}]
                fields = ("op", "addr", "rows", "slot", "fragment", "kv_token", "output_tile", "output_column")
                self.assertEqual([[o.get(k) for k in fields] for o in new_macs],
                                 [[o.get(k) for k in fields] for o in old_macs])

    def test_explicit_boundaries_and_private_descriptor_bodies(self):
        for program in self.programs.values():
            for stage in program["stages"]:
                window = stage["kernel_window"]
                ops = stage["channels"][0]
                if window["mode"] == "exclusive_handoff":
                    self.assertIn(stage["name"], {"rope_export", "rope_import", "q_exchange_export", "q_exchange_import"})
                    continue
                index = window["descriptor_index_by_channel"][0]
                self.assertEqual(window["export_begin_by_channel"], [index + 1])
                self.assertEqual(sum(o["op"] in {"MAC_RUN", "ROPE_RUN"} for o in ops), 1)
                self.assertTrue(all(o["op"] in {"WR", "BARRIER"} for o in ops[:index]))
                self.assertTrue(all(o["op"] == "RD" for o in ops[index + 1:]))
                self.assertFalse(any(o["op"] in {"RD", "WR", "RDRES", "WRGB", "MAC_RUN", "ROPE_RUN"}
                                     or o.get("epoch") for o in ops[index]["children"]))
                for read in ops[index + 1:]:
                    self.assertIn(index, read["deps"])

    def test_spill_packets_preserve_output_sources_and_round_trip_addresses(self):
        for variant, program in self.programs.items():
            for name, count in (("o", 1024), ("gate_up", 1024), ("down", 128)):
                outer = self.stage(program, name)["channels"][0]
                descriptor = next(o for o in outer if "children" in o)
                stores = [o for o in descriptor["children"] if o.get("source") == "fused_result"]
                reads = [o for o in outer if o["op"] == "RD"]
                old_exports = [o for o in walk(self.stage(self.legacy[variant], name)["channels"][0])
                               if o["op"] == "RDRES"]
                self.assertEqual(len(stores), count)
                self.assertEqual(len(reads), count)
                self.assertEqual(len({tuple(o["addr"]) for o in stores}), count)
                for packet, (store, read, old) in enumerate(zip(stores, reads, old_exports)):
                    self.assertEqual(store["source_regions"], old["source_regions"])
                    self.assertEqual(store["output_tile"], old["output_tile"])
                    self.assertEqual(store["result_slot"], old["result_slot"])
                    self.assertTrue(store["release_result"])
                    self.assertEqual(store["result_packet"], packet)
                    self.assertEqual(read["result_packet"], packet)
                    self.assertEqual(read["addr"], store["addr"])
                    address = store["addr"]
                    # Independently decode the physical address, not the layout helper.
                    self.assertEqual(address[2] * 4 + address[3], packet % 16)
                    self.assertEqual(address[4] * 2048 + address[5] * 32, 0x90000 + packet // 16 * 32)
                counts = Counter(o["addr"][2] * 4 + o["addr"][3] for o in stores)
                self.assertEqual(set(counts.values()), {count // 16})

    def test_o_second_tile_waits_for_local_spill_not_external_export(self):
        for program in self.programs.values():
            outer = self.stage(program, "o")["channels"][0]
            children = outer[0]["children"]
            tile0 = [o for o in children if o["op"] == "LOCALWR" and o["output_tile"] == 0]
            tile1 = next(o for o in children if o["op"] == "MACAB" and o["output_tile"] == 1)
            self.assertEqual(len(tile0), 512)
            self.assertTrue({o["id"] for o in tile0} <= completion_ancestors(children, tile1["id"]))
            self.assertFalse(any(o["op"] in {"RD", "RDRES"} for o in children))
            addresses0 = {tuple(o["addr"]) for o in tile0}
            addresses1 = {tuple(o["addr"]) for o in children if o["op"] == "LOCALWR" and o["output_tile"] == 1}
            self.assertFalse(addresses0 & addresses1)
            self.assertEqual(len(addresses1), 512)

    def test_existing_local_paths_and_q_handoffs_are_unchanged(self):
        for variant, program in self.programs.items():
            for stage in program["stages"]:
                if stage["name"] not in {"o", "gate_up", "down"}:
                    self.assertEqual(stage["channels"], self.stage(self.legacy[variant], stage["name"])["channels"])
            pv = self.stage(program, "pv")["channels"][0]
            self.assertFalse(any(o["op"] in {"RD", "RDRES"} for o in walk(pv)))
            self.assertEqual(sum(o["op"] == "LOCALWR" and o.get("source") == "pv_bf16" for o in walk(pv)), 256)
            audit_transient_array_producers(program)

    def test_dram_reservation_fits_without_adding_sram(self):
        for variant, program in self.programs.items():
            contract = program["tensor_contract"]
            self.assertEqual(contract["buffers"], self.legacy[variant]["tensor_contract"]["buffers"])
            self.assertEqual(contract["regions_per_bank"]["fused_result_spill"], [0x90000, 0x90800])
            L.validate_regions(contract["regions_per_bank"])
            for op in walk([o for stage in program["stages"] for o in stage["channels"][0]]):
                self.assertLessEqual(op.get("output_bytes_per_bank", 0), 2048)
                for region in op.get("source_regions", []):
                    self.assertLessEqual(region["byte_offset"] + region["bytes"], 2048)
        with self.assertRaisesRegex(ValueError, "capacity"):
            L.fused_result_location(1024)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            L.validate_regions({"a": [0x90000, 0x90800], "b": [0x90020, 0x90820]})
        with self.assertRaisesRegex(ValueError, "capacity"):
            L.validate_regions({"a": [0xFFFE0, 0x100020]})

    def test_boundary_audit_rejects_external_children_and_unproduced_reads(self):
        stage = self.stage(self.programs["token_local_after"], "o")
        broken = copy.deepcopy(stage["channels"])
        broken[0][0]["children"][0]["op"] = "RDRES"
        with self.assertRaisesRegex(ValueError, "private"):
            kernel_window_metadata("o", broken)
        broken = copy.deepcopy(stage["channels"])
        broken[0][1]["addr"][4] += 1
        with self.assertRaisesRegex(ValueError, "producer"):
            kernel_window_metadata("o", broken)
        broken = copy.deepcopy(stage["channels"])
        broken[0][1]["deps"] = []
        with self.assertRaisesRegex(ValueError, "producer"):
            kernel_window_metadata("o", broken)


if __name__ == "__main__":
    unittest.main()
