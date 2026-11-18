"""Independently decode the Down wire addressing and compare physical requests."""
import unittest
from programs import descriptors as D
from programs.gen_pim_program import ffn, generate
from workloads import layout as L


class DownDescriptorTests(unittest.TestCase):
    def test_mc_down_descriptor_and_staging_use_two_fragments_per_row_half(self):
        program = generate("token_local_after", 1, mc_control=True)
        outer = next(stage for stage in program["stages"] if stage["name"] == "down")["channels"][0]
        descriptor = next(op for op in outer if op["op"] == "MAC_RUN")
        words = descriptor["descriptor"]
        self.assertEqual((words[0] >> 29) & 3, 2)
        self.assertEqual(D.crc16(words), descriptor["crc16"])
        a_base, b_base = (words[2] & 0xFFFFFF) * 32, (words[4] & 0xFFFFFF) * 32
        a_stride, b_stride = (words[3] >> 16) * 32, (words[5] >> 16) * 32
        split = (words[4] >> 24) | ((words[6] >> 30) << 8)
        jump = (words[6] & 0xFFFFFF) * 32
        self.assertEqual((a_base, a_stride, b_base, b_stride, split, jump), (0x80000, 512, 0x50000, 160, 12, 128))
        writes = {}
        for op in outer:
            if op["op"] == "WR":
                f, q = op["tensor_fragment"], op["tensor_row"]
                expected = L.address(0, a_base + ((f // 4) * 2 + f % 2) * a_stride + 32 * q, 4 * (f // 2 % 2))
                self.assertEqual(op["addr"], expected)
                writes[f, q] = expected
        self.assertEqual(len(set(map(tuple, writes.values()))), 4096)
        children = descriptor["children"]
        for i, op in enumerate(children):
            if op["op"] == "LOCALRD":
                f, q = op["tensor_fragment"], op["tensor_row"]
                self.assertEqual(op["addr"], writes[f, q])
                self.assertEqual(op["dest_slot"], 16 * (f % 2) + q)
                self.assertEqual(children[i + 1]["source_bank"], 4 * (f // 2 % 2))
            elif op["op"] == "MACAB":
                f, col = op["fragment"], op["output_column"]
                self.assertEqual(op["addr"], L.address(0, b_base + f * b_stride + (f // split) * jump + 32 * col, -1))

    def test_wire_addressing_matches_scatter_and_expanded_weight_requests(self):
        outer = ffn(0, "down", True)
        descriptor = next(o for o in outer if o["op"] == "MAC_RUN")
        words = descriptor["descriptor"]
        self.assertEqual(len(words), 8)
        self.assertEqual(D.crc16(words), descriptor["crc16"])
        self.assertEqual((words[0] >> 29) & 1, 1)
        self.assertEqual(words[0] & 15, 1)
        self.assertEqual((words[0] >> 4) & 15, 2)
        m, n = (words[1] & 15) + 1, ((words[1] >> 4) & 1023) + 1
        fragments = ((words[1] >> 14) & 4095) + 1
        pack = 1 << ((words[1] >> 26) & 3)
        self.assertEqual((m, n, fragments, pack), (16, 4, 256, 2))
        a_base, b_base = (words[2] & 0xFFFFFF) * 32, (words[4] & 0xFFFFFF) * 32
        a_stride, b_stride = (words[3] >> 16) * 32, (words[5] >> 16) * 32
        split = (words[4] >> 24) | ((words[6] >> 30) << 8)
        jump = (words[6] & 0xFFFFFF) * 32
        self.assertEqual((split, jump), (12, 128))
        for op in outer + descriptor["children"]:
            if op["op"] in ("WR", "LOCALRD"):
                f, query = op["tensor_fragment"], op["tensor_row"]
                expected = a_base + f * a_stride + (f // split) * jump
                self.assertEqual(op["addr"], L.address(0, expected, query))
            elif op["op"] == "MACAB":
                f, column = op["fragment"], op["output_column"]
                expected = b_base + f * b_stride + (f // split) * jump + column * 32
                self.assertEqual(op["addr"], L.address(0, expected, -1))
        for name in ("qkv", "qkt", "pv", "o", "gate_up", "rope"):
            self.assertEqual((D.for_operator(name)[0] >> 29) & 1, 0, name)


if __name__ == "__main__":
    unittest.main()
