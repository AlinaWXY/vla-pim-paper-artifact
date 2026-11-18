"""Tensor coverage, real traffic and bounded storage tests (no timing oracle)."""
import copy
from collections import Counter, defaultdict
import hashlib
import json
import unittest
from programs.gen_pim_program import (VARIANTS, EXTERNAL, Builder, attach_hazard_dependencies,
                                      generate, walk, validate_ops)
from workloads import layout as L


def completion_ancestors(ops, index):
    pending, seen = list(ops[index]["deps"]), set()
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(ops[current]["deps"])
    return seen


def completion_masks(ops):
    """Compact transitive completion graph; issue-only edges cannot prove RAW."""
    masks = []
    for index, op in enumerate(ops):
        mask = 0
        for dependency in op["deps"]:
            if not isinstance(dependency, int) or not 0 <= dependency < index:
                raise ValueError("invalid completion dependency")
            mask |= masks[dependency] | (1 << dependency)
        masks.append(mask)
    return masks


def audit_down_fused(outer):
    """Recover tensor coordinates from physical columns, then audit producers.

    This does not call the generator's address or dependency helpers. A valid
    command count alone cannot detect duplicated scatter or a wrong SRF view.
    """
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    def input_coordinate(op):
        address = op["addr"]
        bank = address[2] * 4 + address[3]
        row, column = address[4] - 160, address[5]
        fragment = row * 12 + column // 5
        check(0 <= bank < 16 and 0 <= fragment < 256 and column % 5 == 4 and column < 60,
              "input write/read is not an input-only Down column")
        check(op["tensor_fragment"] == fragment and op["tensor_row"] == bank,
              "scatter/read tensor label differs from physical input")
        return fragment, bank

    descriptors = [op for op in outer if op["op"] == "MAC_RUN"]
    check(len(descriptors) == 1, "one Down descriptor required")
    descriptor = descriptors[0]
    outer_masks = completion_masks(outer)
    writes = [op for op in outer if op["op"] == "WR"]
    write_by_input = {}
    for op in writes:
        coordinate = input_coordinate(op)
        check(coordinate not in write_by_input, "duplicate scatter input")
        write_by_input[coordinate] = op
        check(outer_masks[descriptor["id"]] & (1 << op["id"]), "descriptor missing scatter completion dependency")
    expected = {(fragment, query) for fragment in range(256) for query in range(16)}
    check(set(write_by_input) == expected, "incomplete scatter coverage")
    ops = descriptor["children"]
    masks = completion_masks(ops)
    reads = [op for op in ops if op["op"] == "LOCALRD"]
    read_by_input = {}
    for op in reads:
        coordinate = input_coordinate(op)
        check(coordinate not in read_by_input, "duplicate input read")
        check(op["addr"] == write_by_input[coordinate]["addr"], "read has no matching scatter producer")
        fragment, query = coordinate
        check(op["dest_slot"] == (fragment % 2) * 16 + query, "read uses wrong SRF query/fragment slot")
        read_by_input[coordinate] = op
    check(set(read_by_input) == expected, "incomplete input reads")
    broadcasts = {}
    for op in ops:
        if op["op"] != "BCAST":
            continue
        coordinate = op["tensor_fragment"], op["source_bank"]
        check(coordinate in read_by_input and coordinate not in broadcasts, "invalid broadcast coverage")
        fragment, query = coordinate
        check(op["slot"] == (fragment % 2) * 16 + query, "broadcast uses wrong SRF slot")
        check(masks[op["id"]] & (1 << read_by_input[coordinate]["id"]), "broadcast missing LOCALRD completion")
        broadcasts[coordinate] = op
    check(set(broadcasts) == expected, "incomplete input broadcasts")
    macs = [op for op in ops if op["op"] == "MACAB"]
    check(Counter((op["fragment"], op["output_column"]) for op in macs)
          == Counter({(f, c): 1 for f in range(256) for c in range(4)}), "Down MAC coverage")
    for op in macs:
        address = op["addr"]
        fragment = (address[4] - 160) * 12 + address[5] // 5
        check(address[2:4] == [-1, -1] and address[5] % 5 < 4 and address[5] < 60,
              "MAC reads an input/padding column")
        check((fragment, address[5] % 5) == (op["fragment"], op["output_column"]), "MAC weight coordinates")
        check(op["rows"] == 16 and op["slot"] == (fragment % 2) * 16, "MAC uses wrong SRF view")
        for query in range(16):
            check(masks[op["id"]] & (1 << broadcasts[fragment, query]["id"]), "MAC missing input producer dependency")
    acts = [op for op in ops if op["op"] == "ACTAB"]
    check([op["addr"][4] for op in acts] == list(range(160, 182)), "Down must activate exactly 22 shared rows")
    for act in acts:
        check(act["addr"][2:4] == [-1, -1], "Down activation must cover all banks")
        row = act["addr"][4]
        first = (row - 160) * 12
        for fragment in (first, first + 1):
            for query in range(16):
                read = read_by_input[fragment, query]
                check(act["id"] in read["issue_deps"], "new-row load missing ACTAB issue dependency")
                check(not masks[read["id"]] & (1 << act["id"]), "ACTAB incorrectly adds a full-row completion barrier")
        if first:
            previous = [op for op in macs if first - 12 <= op["fragment"] < first]
            check(len(previous) == 48, "previous complete weight row MAC coverage")
            for op in previous:
                check(masks[act["id"]] & (1 << op["id"]), "new-row ACTAB can evict unfinished older weights")
    # Ignore row ACTABs when counting the 32 read/broadcast pairs + eight MACs
    # in each two-fragment batch. The final row still contains two full batches.
    work = [op for op in ops if op["op"] != "ACTAB"]
    check(len(work) == 128 * 72, "unexpected two-fragment batch work")
    previous_macs = []
    for batch in range(128):
        chunk = work[batch * 72:(batch + 1) * 72]
        check(Counter(op["op"] for op in chunk) == {"LOCALRD": 32, "BCAST": 32, "MACAB": 8}, "batch shape")
        loads = [op for op in chunk if op["op"] == "LOCALRD"]
        bcasts = [op for op in chunk if op["op"] == "BCAST"]
        computations = [op for op in chunk if op["op"] == "MACAB"]
        check({op["fragment"] for op in computations} == {batch * 2, batch * 2 + 1}, "batch fragments")
        check(len({op["addr"][4] for op in loads + computations}) == 1, "two-fragment batch crosses a row")
        check(max(op["id"] for op in bcasts) < min(op["id"] for op in computations), "batch not fully loaded before MACs")
        for op in bcasts:
            check(masks[computations[0]["id"]] & (1 << op["id"]), "first MAC missing complete batch input")
        for op in loads:
            for previous in previous_macs:
                if previous["fragment"] % 2 == op["tensor_fragment"] % 2:
                    check(masks[op["id"]] & (1 << previous["id"]), "SRF view reused before prior MAC completion")
        previous_macs = computations


def audit_transient_array_producers(program):
    """Independent address coverage audit; initialized weights/prefix are excluded."""
    producers = {}
    for stage in program["stages"]:
        name = stage["name"]
        for op in walk(stage["channels"][0]):
            addr = op["addr"]
            offset = addr[4] * 2048 + addr[5] * 32
            if op["op"] in ("WR", "LOCALWR"):
                producers[tuple(addr)] = (name, op["op"], op.get("source", ""))
            elif op["op"] in ("RD", "LOCALRD", "MACAB", "MACQK"):
                expected = None
                if L.STAGING_BASE <= offset < L.STAGING_BASE + 65536:
                    expected = "pv" if name == "o" else name
                elif L.Q_IMPORT_BASE <= offset < L.Q_IMPORT_BASE + 2048:
                    expected = "q_exchange_import"
                elif L.SCORE_BASE <= offset < L.SCORE_BASE + 16384:
                    expected = "qkt"
                elif L.Q_BASE <= offset < L.K_SUFFIX_BASE + 256:
                    if name in ("rope", "rope_export"):
                        expected = "qkv"
                    else:
                        expected = "rope_import" if program["rope"] == "gpu" else "rope"
                elif op["op"] == "MACAB" and L.V_BASE + 15360 <= offset < L.V_BASE + 15872:
                    expected = "qkv"
                if expected is not None:
                    banks = range(16) if addr[2] == -1 else (addr[2] * 4 + addr[3],)
                    for b in banks:
                        concrete = tuple(addr[:2] + [b // 4, b % 4] + addr[4:])
                        actual = producers.get(concrete)
                        if actual is None or actual[0] != expected:
                            raise ValueError(f"{name}: {op['op']} reads {concrete} without {expected} producer")
                        if name == "o" and L.STAGING_BASE <= offset < L.STAGING_BASE + 65536:
                            if actual != ("pv", "LOCALWR", "pv_bf16"):
                                raise ValueError("O input is stale P data instead of locally written PV output")


class ProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.programs = {v: generate(v, 1) for v in VARIANTS}

    def stage(self, variant, name):
        return next(s for s in self.programs[variant]["stages"] if s["name"] == name)["channels"][0]

    def test_tensor_partition_is_bijective(self):
        self.assertEqual({L.feature_from_lane(b, l) for b in range(16) for l in range(16)}, set(range(256)))
        for f in range(256):
            self.assertEqual(L.feature_from_lane(*L.feature_owner(f)), f)
        owners = {L.token_owner(h, t) for h in range(8) for t in range(992)}
        self.assertEqual(owners, {(c, t) for c in range(16) for t in range(496)})
        for head in range(8):
            shard_a = {L.feature_shard_from_lane(2 * head, b, x) for b in range(16) for x in range(8)}
            shard_b = {L.feature_shard_from_lane(2 * head + 1, b, x) for b in range(16) for x in range(8)}
            self.assertFalse(shard_a & shard_b)
            self.assertEqual(shard_a | shard_b, set(range(256)))
            self.assertEqual(shard_a, set(range(64)) | set(range(128, 192)))

    def test_operator_scalar_mac_coverage_matches_tensor_shapes(self):
        for variant, p in self.programs.items():
            shapes = p["tensor_contract"]["operator_shapes"]
            for name, shape in shapes.items():
                ops = self.stage(variant, name)
                scalar_macs = sum(o["rows"] * 16 * o.get("lanes", 16)
                                  for o in walk(ops) if o["op"] in ("MACAB", "MACQK")) * 16
                self.assertEqual(scalar_macs, shape["scalar_macs_all_channels"], (variant, name))
            # Token KV parity repeats O projection before reducing its two partials.
            self.assertEqual(shapes["o"]["scalar_macs_all_channels"],
                             16 * 2048 * 1024 * (2 if p["layout"] == "token" else 1))

    def test_feature_q_and_k_cover_both_packed_halves(self):
        q = [o for o in self.stage("feature_gpu_before", "q_exchange_export") if o["op"] == "LOCALRD"]
        self.assertEqual({(o["dest_slot"], o["source_lane_offset"]) for o in q},
                         {(i, 8 * (i % 2)) for i in range(16)})
        k = [o for o in self.stage("feature_gpu_before", "qkt") if o["op"] == "MACQK"]
        self.assertEqual({(o["kv_token"], o["weight_lane_offset"]) for o in k},
                         {(i, 8 * (i % 2)) for i in range(992)})

    def test_eighteen_physical_layers_are_disjoint_and_in_bounds(self):
        contract = L.contract("token")
        self.assertEqual(contract["layer_stride_bytes_per_bank"], 1024 * 1024)
        regions = sorted(contract["regions_per_bank"].values())
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(regions, regions[1:])))
        all_regions = sorted((start + layer * L.LAYER_STRIDE_BYTES, end + layer * L.LAYER_STRIDE_BYTES)
                             for layer in range(18) for start, end in regions)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(all_regions, all_regions[1:])))
        self.assertLessEqual(all_regions[-1][1], 64 * 1024 * 1024)  # 8Gb rank / sixteen banks
        for layer in range(18):
            a = L.layer_address([0, 1, 0, 0, 511, 63], layer)
            self.assertEqual(a[4], layer * 512 + 511)
            self.assertLess(a[4], 32768)

    def test_down_physical_tensor_is_bijective_with_input_holes_and_row_padding(self):
        self.assertEqual(L.DOWN_BASE, 0x50000)
        occupied = defaultdict(set)
        weight_bursts, input_bursts = set(), set()
        for fragment in range(256):
            weights = [L.down_weight_offset(fragment, slot) for slot in range(4)]
            self.assertEqual(weights, list(range(weights[0], weights[0] + 128, 32)))
            self.assertEqual(len({offset // 2048 for offset in weights}), 1)
            for offset in weights:
                self.assertNotIn(offset, weight_bursts)
                weight_bursts.add(offset)
                occupied[offset // 2048].add(offset % 2048 // 32)
            for query in range(16):
                bank, offset = L.down_input_location(fragment, query)
                self.assertEqual(bank, query)
                self.assertEqual(offset, weights[-1] + 32)
                self.assertNotIn((bank, offset), input_bursts)
                input_bursts.add((bank, offset))
                occupied[offset // 2048].add(offset % 2048 // 32)
            # Actual scalar addresses cover every BF16 lane exactly once in
            # each four-column packet group, without spilling into input.
            for slot in range(4):
                lanes = [L.down_weight_location(fragment * 16 + lane, slot) for lane in range(16)]
                self.assertEqual(lanes, [(0, 0, weights[slot] + 2 * lane) for lane in range(16)])
        self.assertEqual(len(weight_bursts), 1024)
        self.assertEqual(len(input_bursts), 4096)
        self.assertFalse(weight_bursts & {offset for bank, offset in input_bursts})
        self.assertEqual(set(occupied), set(range(160, 182)))
        for row in range(160, 181):
            self.assertEqual(occupied[row], set(range(60)))
            self.assertEqual(set(range(64)) - occupied[row], {60, 61, 62, 63})
        self.assertEqual(occupied[181], set(range(20)))
        self.assertEqual(L.down_weight_offset(255, 3), 181 * 2048 + 18 * 32)
        self.assertEqual(L.down_input_location(255, 15), (15, 181 * 2048 + 19 * 32))
        # Global output ownership is injective across channels/banks, even
        # though every bank reuses the same local byte offsets.
        for channel in range(16):
            physical = set()
            for output in range(channel * 64, (channel + 1) * 64):
                for fragment in range(256):
                    c, bank, offset = L.down_weight_location(fragment * 16, output)
                    self.assertEqual(c, channel)
                    self.assertEqual(c * 64 + bank * 4 + offset % 2048 // 32 % 5, output)
                    self.assertNotIn((bank, offset), physical)
                    physical.add((bank, offset))
            self.assertEqual(len(physical), 16 * 1024)
        region = L.contract("token")["regions_per_bank"]["down_weights_and_inputs"]
        self.assertEqual(region, [0x50000, 0x5B000])
        self.assertLessEqual(max(weight_bursts) + 32, region[1])
        self.assertLessEqual(max(offset for bank, offset in input_bursts) + 32, region[1])
        self.assertLess(region[1], L.QKV_BASE)
        for layer in range(18):
            last = L.layer_address(L.address(15, region[1] - 32, 15), layer)
            end = last[4] * 2048 + last[5] * 32 + 32
            self.assertLess(end, (layer + 1) * 1024 * 1024)

    def test_down_fused_scatter_srf_lifetimes_and_row_activation(self):
        for variant in ("token_gpu_after", "token_local_after"):
            audit_down_fused(self.stage(variant, "down"))
        original = self.stage("token_local_after", "down")
        # Same byte and command counts, but duplicated scatter and wrong SRF
        # slots must fail independently of the generator's own validator.
        broken = copy.deepcopy(original)
        writes = [op for op in broken if op["op"] == "WR"]
        writes[1]["addr"] = list(writes[0]["addr"])
        with self.assertRaisesRegex(ValueError, "physical input"):
            audit_down_fused(broken)
        broken = copy.deepcopy(original)
        children = next(op for op in broken if op["op"] == "MAC_RUN")["children"]
        next(op for op in children if op["op"] == "LOCALRD")["dest_slot"] = 31
        with self.assertRaisesRegex(ValueError, "SRF"):
            audit_down_fused(broken)
        broken = copy.deepcopy(original)
        descriptor = next(op for op in broken if op["op"] == "MAC_RUN")
        descriptor["deps"] = []
        with self.assertRaisesRegex(ValueError, "scatter completion"):
            audit_down_fused(broken)
        broken = copy.deepcopy(original)
        children = next(op for op in broken if op["op"] == "MAC_RUN")["children"]
        next(op for op in children if op["op"] == "BCAST")["deps"] = []
        with self.assertRaisesRegex(ValueError, "LOCALRD completion"):
            audit_down_fused(broken)
        broken = copy.deepcopy(original)
        children = next(op for op in broken if op["op"] == "MAC_RUN")["children"]
        next(op for op in children if op["op"] == "LOCALRD")["issue_deps"] = []
        with self.assertRaisesRegex(ValueError, "ACTAB issue"):
            audit_down_fused(broken)
        broken = copy.deepcopy(original)
        children = next(op for op in broken if op["op"] == "MAC_RUN")["children"]
        [op for op in children if op["op"] == "ACTAB"][1]["deps"] = []
        with self.assertRaisesRegex(ValueError, "unfinished older weights"):
            audit_down_fused(broken)

    def test_down_scatter_is_fragment_major_and_alternates_bank_groups(self):
        query_order = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
        for variant in ("token_gpu_after", "token_local_after"):
            writes = [op for op in self.stage(variant, "down") if op["op"] == "WR"]
            self.assertEqual(len(writes), 4096)
            self.assertEqual(len({tuple(op["addr"]) for op in writes}), 4096)
            self.assertEqual({op["epoch"] for op in writes}, {"down_input"})
            # Includes the boundary between successive fragments: the final
            # query is in BG3 and the next fragment starts in BG0.
            self.assertTrue(all(left["addr"][2] != right["addr"][2]
                                for left, right in zip(writes, writes[1:])))
            for fragment in range(256):
                group = writes[fragment * 16:(fragment + 1) * 16]
                self.assertEqual([op["tensor_fragment"] for op in group], [fragment] * 16)
                self.assertEqual([op["tensor_row"] for op in group], query_order)
                self.assertEqual([op["addr"][2] * 4 + op["addr"][3] for op in group], query_order)
                self.assertEqual(len({tuple(op["addr"][4:]) for op in group}), 1)
            # Independently recover all coordinates and producer dependencies
            # after reordering; command counts alone cannot prove preservation.
            audit_down_fused(self.stage(variant, "down"))

    def test_down_streaming_uses_same_weights_without_array_staging(self):
        fused = self.stage("token_local_after", "down")
        reference = [(o["fragment"], o["output_column"], o["addr"], o["rows"], o["slot"])
                     for o in walk(fused) if o["op"] == "MACAB"]
        for variant in list(VARIANTS)[:3]:
            ops = self.stage(variant, "down")
            self.assertFalse(any(o["op"] in {"MAC_RUN", "WR", "LOCALRD", "LOCALWR", "BCAST", "ACTAB"} for o in ops))
            self.assertEqual([(o["fragment"], o["output_column"], o["addr"], o["rows"], o["slot"])
                              for o in ops if o["op"] == "MACAB"], reference)
            inputs = [o for o in ops if o["op"] == "WRGB"]
            self.assertEqual(Counter((o["tensor_fragment"], o["tensor_row"]) for o in inputs),
                             Counter({(f, q): 1 for f in range(256) for q in range(16)}))
            for o in inputs:
                self.assertEqual(o["slot"], (o["tensor_fragment"] % 2) * 16 + o["tensor_row"])
        for variant in VARIANTS:
            ops = list(walk(self.stage(variant, "down")))
            self.assertEqual(sum(o["op"] in {"WR", "WRGB"} for o in ops), 4096)
            self.assertEqual(sum(o["op"] == "RDRES" for o in ops), 128)
            self.assertEqual(sum(o["rows"] * 16 * 16 for o in ops if o["op"] == "MACAB") * 16, 67108864)

    def test_non_down_stages_match_frozen_programs(self):
        # Streaming pins include the alternating-half schedule. Fused pins
        # retain the reference taken before the Down-only layout change.
        expected = {
            "feature_gpu_before": "beb2976ce8e7e2ff06eb7903d082ad702b544fb825927232a7f93f916be0127d",
            "token_gpu_before": "e3d813bff5d1db19ed35f3e6ddd76050a7e75e9a27c363393c76cf90eeccb2d6",
            "token_local_before": "fa429f142d3c444c655304d6341263f4b367ce7bbe7f0d1047fdd49a20121918",
            "token_gpu_after": "facb3fcf49aa111e8f6ddb47d65b7bfa999407a04702ad6d7baf8cd5e453930c",
            "token_local_after": "913971028380b026b7ef3d27f23aecaea22b44f00771b6b61f4545c0735a062f",
        }
        for variant, program in self.programs.items():
            unchanged = [stage for stage in program["stages"] if stage["name"] != "down"]
            digest = hashlib.sha256(json.dumps(unchanged, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(digest, expected[variant], variant)

    def test_transient_arrays_have_complete_current_producers(self):
        for p in self.programs.values():
            audit_transient_array_producers(p)
        broken = copy.deepcopy(self.programs["token_local_after"])
        # Same number of writes and bytes, but a duplicated row leaves a hole.
        broken["stages"][0]["channels"][0][0]["addr"][4] += 1
        with self.assertRaisesRegex(ValueError, "without qkv producer"):
            audit_transient_array_producers(broken)
        broken = copy.deepcopy(self.programs["token_local_after"])
        pv = next(s for s in broken["stages"] if s["name"] == "pv")
        missing = next(o for o in walk(pv["channels"][0]) if o.get("source") == "pv_bf16")
        missing["op"] = "BARRIER"
        with self.assertRaisesRegex(ValueError, "stale P data"):
            audit_transient_array_producers(broken)

    def test_dq_conservation_and_historical_tensor_contract(self):
        # Independent established byte totals; timing is deliberately not reused.
        kib = (6096, 5352, 5160, 4968, 4776)
        for (variant, p), expected in zip(self.programs.items(), kib):
            calculated = sum(op["op"] in EXTERNAL for s in p["stages"] for op in walk(s["channels"][0])) * 32
            self.assertEqual(calculated, expected * 1024 // 16, variant)
            self.assertEqual(p["expected_dq_bytes"], calculated)
            self.assertEqual(sum(s["expected_dq_bytes"] for s in p["stages"]), calculated)

    def test_rope_and_layout_are_single_variable_traffic_differences(self):
        b = {v: p["expected_dq_bytes"] for v, p in self.programs.items()}
        self.assertEqual((b["feature_gpu_before"] - b["token_gpu_before"]) * 16, 744 * 1024)
        self.assertEqual((b["token_gpu_before"] - b["token_local_before"]) * 16, 192 * 1024)
        self.assertEqual((b["token_gpu_after"] - b["token_local_after"]) * 16, 192 * 1024)

    def test_streaming_has_no_descriptor_or_staging(self):
        for v in list(VARIANTS)[:3]:
            for s in self.programs[v]["stages"]:
                ops = list(walk(s["channels"][0]))
                self.assertFalse(any(o["op"] in ("MAC_RUN", "ROPE_RUN", "BCAST") for o in ops))
                self.assertFalse(any(o["op"] == "WR" and o.get("epoch", "").endswith("_input") for o in ops))
        self.assertEqual(sum(o["op"] == "WRGB" for o in self.stage("token_local_before", "o")), 512)

    def test_streaming_fills_alternate_srf_half_during_mac_batch(self):
        for name, views in (("qkv", 2), ("down", 1)):
            ops = self.stage("token_local_before", name)
            writes = [o for o in ops if o["op"] == "WRGB"]
            macs = [o for o in ops if o["op"] == "MACAB"]
            for batch in range(3):
                current = [o for o in macs if o["fragment"] // views == batch]
                loads = [o for o in writes if o["tensor_fragment"] // views == batch]
                self.assertEqual({o["slot"] // 16 for o in current + loads}, {batch % 2})
                self.assertTrue({o["id"] for o in loads}.issubset(current[0]["deps"]))
                if batch:
                    previous = [o for o in macs if o["fragment"] // views == batch - 1]
                    self.assertLess(previous[0]["id"], loads[0]["id"])
                    self.assertLess(loads[0]["id"], previous[-1]["id"])
                    self.assertNotIn(previous[-1]["id"], loads[0]["deps"])
                if batch == 2:
                    prior_use = max(o["id"] for o in macs if o["fragment"] < views)
                    self.assertTrue(all(prior_use in o["deps"] for o in loads))

    def test_mc_streaming_interleaves_refills_with_macs_using_the_other_half(self):
        for variant in ("feature_gpu_before", "token_local_before"):
            program = generate(variant, 1, mc_control=True)
            for stage in program["stages"]:
                ops = stage["channels"][0]
                writes = [op for op in ops if op["op"] == "WRGB" and op.get("source") != "host_query"]
                with self.subTest(variant=variant, stage=stage["name"]):
                    self.assertEqual(len(writes) % 16, 0)
                    for start in range(0, len(writes), 16):
                        refill = writes[start:start + 16]
                        slot = refill[0]["slot"]
                        self.assertIn(slot, (0, 16))
                        self.assertEqual([op["slot"] for op in refill], list(range(slot, slot + 16)))
                        next_mac = next(op for op in ops[refill[-1]["id"] + 1:]
                                        if op["op"] == "MACAB" and op["slot"] // 16 == slot // 16)
                        self.assertTrue({op["id"] for op in refill} <= set(next_mac["deps"]))
                        if refill[0]["tensor_fragment"] != 0:
                            self.assertTrue(any(op["op"] == "MACAB" and op["slot"] // 16 != slot // 16
                                                for op in ops[refill[0]["id"]:refill[-1]["id"]]))

    def test_fused_has_six_mac_descriptors_and_optional_rope(self):
        for v in list(VARIANTS)[3:]:
            ops = [o for s in self.programs[v]["stages"] for o in s["channels"][0]]
            self.assertEqual(sum(o["op"] == "MAC_RUN" for o in ops), 6)
            self.assertEqual(sum(o["op"] == "ROPE_RUN" for o in ops), int("local" in v))
            self.assertFalse(any(o["op"] == "WRGB" for o in ops))

    def test_bank_rope_double_schedule_fifo_and_exact_work(self):
        for variant in ("token_local_before", "token_local_after"):
            ops = list(walk(self.stage(variant, "rope")))
            self.assertEqual(sum(o["op"] == "LOCALRD" for o in ops), 384)
            self.assertEqual(sum(o["op"] == "LOCALWR" for o in ops), 256)
            execs = [o for o in ops if o["op"] == "ROPE_EXEC"]
            self.assertEqual(sum(o["pairs"] for o in execs), 2048)
            self.assertEqual(len(execs), 256)
            self.assertEqual({o["shared_unit"] for o in execs}, {"bank_rope"})
            if variant.endswith("after"):
                ops = ops[0]["children"]
            for bank in range(16):
                local = [o for o in ops if o["addr"][2:4] == [bank // 4, bank % 4]]
                previous_k_store = None
                for token in range(8):
                    work = [o for o in local if o.get("token") == token]
                    q, k = [o for o in work if o["op"] == "ROPE_EXEC"]
                    k_load = next(o for o in work if o["op"] == "LOCALRD" and
                                  o.get("dest") == "rope_input" and o["slot"] == 1)
                    q_store, k_store = [o for o in work if o["op"] == "LOCALWR"]
                    self.assertLess(q["id"], k_load["id"])
                    self.assertLess(k_load["id"], k["id"])
                    self.assertIn(q["id"], k_load["issue_deps"])
                    self.assertIn(q["id"], k["issue_deps"])
                    self.assertNotIn(q["id"], completion_ancestors(ops, k_load["id"]))
                    self.assertNotIn(q["id"], completion_ancestors(ops, k["id"]))
                    self.assertIn(q_store["id"], k_store["deps"])
                    if previous_k_store is not None:
                        self.assertIn(previous_k_store, q_store["deps"])
                    previous_k_store = k_store["id"]
                    for execute in (q, k):
                        self.assertTrue(all(ops[d]["addr"][2:4] == execute["addr"][2:4]
                                            for d in execute["deps"]))

    def test_score_uses_existing_pool_and_fused_flushes_full_batch(self):
        for variant in VARIANTS:
            ops = self.stage(variant, "qkt")
            if variant.endswith("after"):
                ops = ops[0]["children"]
            macs = [o for o in ops if o["op"] == "MACQK"]
            self.assertEqual(len(macs), 992 if variant.startswith("feature") else 496)
            reductions = [o for o in ops if o["op"] == "REDUCE"]
            self.assertTrue(all(not o["cache_result"] for o in reductions))
            if variant.endswith("after"):
                writes = [o for o in ops if o["op"] == "LOCALWR"]
                self.assertEqual(len(writes), 992)
                self.assertGreater(min(o["id"] for o in writes), max(o["id"] for o in reductions))
                self.assertEqual({o["reduction_word"] for o in reductions}, set(range(16, 512)))
                # Each bank owns one query's scores and 16 contribution words.
                self.assertEqual((16 + len(reductions)) * 4, 2048)
                for write in writes:
                    token, packet = write["kv_token"], write["result_slot"]
                    self.assertEqual(write["source_regions"],
                                     [dict(bank=q, byte_offset=(16 + token) * 4, bytes=4)
                                      for q in range(packet * 8, packet * 8 + 8)])
                    self.assertIn(reductions[token]["id"], completion_ancestors(ops, write["id"]))
                continue
            for mac in macs:
                red, p0, p1 = ops[mac["id"] + 1:mac["id"] + 4]
                self.assertEqual(red["op"], "REDUCE")
                self.assertEqual(red["reduction_word"], 16)
                self.assertEqual([p0["op"], p1["op"]], ["RDRES"] * 2)
                self.assertIn(red["id"], p0["deps"])
                self.assertEqual(sum(r["bytes"] for r in p0["source_regions"]), 32)

    def test_output_pool_allocation_and_two_o_tile_lifetimes(self):
        for variant, p in self.programs.items():
            for s in p["stages"]:
                allocated = {"qkv": 1536, "pv": 512 if variant.startswith("feature") else 1024,
                             "o": 2048, "gate_up": 2048, "down": 256}.get(s["name"], 2048)
                for o in walk(s["channels"][0]):
                    if o["op"] == "MACAB":
                        self.assertEqual(o["output_bytes_per_bank"], o["rows"] * o["output_columns"] * 4)
                        self.assertLessEqual(o["output_bytes_per_bank"], 2048)
                    for r in o.get("source_regions", []):
                        self.assertLessEqual(r["byte_offset"] + r["bytes"], allocated)
            ops = self.stage(variant, "o")
            if variant.endswith("after"):
                ops = ops[0]["children"]
            first_tile1 = next(o for o in ops if o.get("output_tile") == 1 and o["op"] == "MACAB")
            exports = {o["id"] for o in ops if o["op"] == "RDRES" and o["output_tile"] == 0}
            self.assertEqual(len(exports), 512)
            self.assertTrue(exports <= completion_ancestors(ops, first_tile1["id"]))

    def test_q_exchange_is_addressed_memory_not_masked_broadcast(self):
        for variant in ("token_gpu_before", "token_local_before", "token_gpu_after", "token_local_after"):
            ops = self.stage(variant, "q_exchange_import")
            writes = [o for o in ops if o["op"] == "WR"]
            self.assertEqual(len(writes), 128)
            self.assertEqual({o["source_channel"] for o in writes}, {1})
            self.assertFalse(any(o["op"] == "WRGB" for o in ops))
            reads = [o for o in ops if o["op"] == "LOCALRD"]
            self.assertEqual(len(reads), 128)
            self.assertLess(max(o["id"] for o in writes), min(o["id"] for o in reads))
            self.assertEqual({o["epoch"] for o in writes}, {"q_import"})
            all_writes = {o["id"] for o in writes}
            for wr, rd in zip(writes, reads):
                self.assertEqual(rd["addr"], wr["addr"])
                self.assertTrue(all_writes <= completion_ancestors(ops, rd["id"]))

    def test_mc_local_rope_q_exchange_uses_complete_mode_phases(self):
        for variant in ("token_local_before", "token_local_after"):
            program = generate(variant, 1, mc_control=True)
            stages = {s["name"]: s["channels"][0] for s in program["stages"]}
            exported, imported = stages["q_exchange_export"], stages["q_exchange_import"]
            self.assertEqual(len(exported), 128)
            self.assertTrue(all(op["op"] == "RD" for op in exported))
            if variant.endswith("_before"):
                self.assertEqual(len(imported), 256)
                for q, bank in ((q, b) for q in range(16) for b in range(16)):
                    op = imported[q * 16 + bank]
                    self.assertEqual((op["op"], op["source"], op["slot"], op["source_channel"]),
                                     ("WRGB", "host_query", q, q % 2))
                    self.assertEqual(op["addr"], L.address(0, L.Q_BASE + q // 2 * 32, bank))
            else:
                self.assertEqual({op["epoch"] for op in exported}, {"q_exchange_exclusive"})
                self.assertTrue(all(op["op"] == "WR" and op["epoch"] == "q_exchange_exclusive"
                                    for op in imported[:128]))
                self.assertEqual(imported[128]["op"], "BARRIER")
                self.assertEqual(len(imported[129:]), 256)
                for op in imported[129:]:
                    self.assertEqual(op["op"], "LOCALRD")
                    self.assertIn(128, op["deps"])

    def test_generator_rejects_capacity_or_dependency_corruption(self):
        ops = copy.deepcopy(self.stage("token_gpu_before", "qkv"))
        ops[0]["slot"] = 32
        with self.assertRaisesRegex(ValueError, "SRF"):
            validate_ops(ops, 0)
        ops[0]["slot"] = 0
        ops[0]["deps"] = [1]
        with self.assertRaisesRegex(ValueError, "earlier"):
            validate_ops(ops, 0)

    def test_disjoint_output_transfers_do_not_serialize_gather(self):
        first = [dict(bank=0, byte_offset=0, bytes=32)]
        second = [dict(bank=0, byte_offset=32, bytes=32)]
        for transfer in ("RDRES", "LOCALWR"):
            for release in (False, True):
                with self.subTest(transfer=transfer, release=release):
                    b = Builder(0, "packet_hazards")
                    ga = b.emit("GATHER", source_regions=first)
                    ca = b.emit("CAST", deps=[ga], source_regions=first)
                    read = b.emit(transfer, 4096, -1 if transfer == "RDRES" else 0,
                                  deps=[ca], source_regions=first, result_bank=0,
                                  release_result=release)
                    gb = b.emit("GATHER", source_regions=second)
                    cb = b.emit("CAST", deps=[gb], source_regions=second)
                    # A replacement MAC does overwrite the first packet's words.
                    replace = b.emit("MACAB", 8192, -1, rows=1, slot=0,
                                     output_columns=1, output_column=0, fragment=0)
                    attach_hazard_dependencies(b.ops)
                    self.assertNotIn(read, completion_ancestors(b.ops, gb))
                    self.assertNotIn(read, completion_ancestors(b.ops, cb))
                    self.assertIn(ca, b.ops[gb]["deps"])
                    self.assertIn(gb, b.ops[cb]["deps"])
                    self.assertIn(read, completion_ancestors(b.ops, replace))

        # RMW storage is real state, independent of the gather metadata selector.
        b = Builder(0, "rmw_hazards")
        load = b.emit("LOCALRD", 4096, dest="rmw_scratch")
        gather = b.emit("GATHER", deps=[load], source="qkv_v_output", source_regions=first)
        cast = b.emit("CAST", deps=[gather], source_regions=first)
        store = b.emit("LOCALWR", 4096, deps=[cast], source="rmw_scratch",
                       source_bank=0, source_regions=first)
        reload = b.emit("LOCALRD", 4128, dest="rmw_scratch")
        attach_hazard_dependencies(b.ops)
        self.assertIn(load, b.ops[gather]["deps"])
        self.assertIn(store, b.ops[reload]["deps"])

    def test_program_is_deterministic_and_has_no_timing_fields(self):
        p = self.programs["token_local_after"]
        self.assertEqual(json.dumps(p), json.dumps(generate("token_local_after", 1)))
        forbidden = {"dq_latency", "latency_ck", "nCL", "nCWL", "ready_at", "service_ck"}
        for s in p["stages"]:
            for o in walk(s["channels"][0]):
                self.assertFalse(forbidden.intersection(o))


if __name__ == "__main__":
    unittest.main()
