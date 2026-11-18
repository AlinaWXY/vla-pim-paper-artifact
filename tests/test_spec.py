"""Spec contract tests; expected values come from JESD209-5C, not old code."""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ramulator2" / "python"))
from ramulator.dram.lpddr5x_pim import LPDDR5XPIM, build_spec, MAC_ROWS, NO_EXTERNAL_CA, PRIVATE, LOCAL_EVENTS, LOCAL_ALIASES


def issue_gap(spec, first, second, relation):
    """Inspect the one native spec table; this is test analysis, not scheduling."""
    a, b = spec["commands"][first]["id"], spec["commands"][second]["id"]
    broadcast = first in ("WRGB", "RDRES") or second in ("WRGB", "RDRES")
    gap = 0
    for entry in spec["dram"]["timing_constraints"]:
        level, preceding, following, latency = entry[:4]
        window = entry[4] if len(entry) > 4 else 1
        sibling = entry[5] if len(entry) > 5 else False
        if a not in preceding or b not in following or window != 1:
            continue
        applicable = level == 0 or (level == 1 and sibling == (relation == "different_rank"))
        if level == 2:
            applicable = not sibling and relation != "different_rank" and (relation == "same_bg" or broadcast)
        # Different bank within same BG isolates interface from array state.
        if applicable:
            gap = max(gap, latency)
    return gap


def command_matrix(spec):
    result = []
    for relation in ("same_bg", "different_bg", "different_rank"):
        for first in ("RD", "WR", "WRGB", "RDRES"):
            for second in ("RD", "WR", "WRGB", "RDRES"):
                a, b = spec["commands"][first], spec["commands"][second]
                gap = issue_gap(spec, first, second, relation)
                result.append(dict(relation=relation, first=first, second=second, issue_gap_ck=gap,
                                   dq_gap_ck=gap+b["dq_latency"]-a["dq_latency"]-a["dq_cycles"]))
    return result


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = build_spec()

    def test_jedec_numerical_values(self):
        expected = dict(tCK_ps=938,nCL=23,nCWL=12,nWR=37,nRTP=8,nRCD=20,nRCDWR=9,
                        nRP=20,nRPab=23,nRAS=45,nRC=65,nRRDS=4,nRRDL=4,nFAW=16,
                        nBL_min=2,nBL_max=4,nCCDS=2,nCCDL=4,nWTRS=7,nWTRL=13,
                        nRFC=224,nRFCpb=128,nREFI=4164,nREFIpb=520,nPBR2PBR=96,nPBR2ACT=8)
        self.assertEqual({k:self.spec["timings"][k] for k in expected}, expected)

    def test_all_48_interface_pairs(self):
        for cell in command_matrix(self.spec):
            a, b, rel = cell["first"], cell["second"], cell["relation"]
            direction = self.spec["commands"][a]["direction"] + self.spec["commands"][b]["direction"]
            if rel == "different_rank":
                expected = dict(readread=4,readwrite=15,writeread=2,writewrite=3)[direction]
            elif rel == "same_bg" or a in ("WRGB","RDRES") or b in ("WRGB","RDRES"):
                expected = dict(readread=4,readwrite=17,writeread=29,writewrite=4)[direction]
            else:
                expected = dict(readread=2,readwrite=15,writeread=21,writewrite=2)[direction]
            with self.subTest(**cell):
                self.assertEqual(cell["issue_gap_ck"], expected)
                self.assertGreaterEqual(cell["dq_gap_ck"], 0)

    def test_aliases_never_get_array_recovery(self):
        cmd = self.spec["commands"]
        aliases = {cmd[c]["id"] for c in ("WRGB","RDRES")}
        pres = {cmd[c]["id"] for c in ("PREpb","PREab","PREpb_LOCAL","PREab_LOCAL")}
        acts = {cmd[c]["id"] for c in ("ACT1","ACT2","ACT1_LOCAL","ACT2_LOCAL")}
        for level, pre, fol, latency, *_ in self.spec["dram"]["timing_constraints"]:
            if level == 0: # CA serialization is intentionally shared.
                continue
            self.assertFalse(set(pre)&aliases and set(fol)&pres)
            self.assertFalse(set(pre)&acts and set(fol)&aliases)

    def test_all_bg_is_broadcast(self):
        self.assertEqual(issue_gap(self.spec,"WRGB","WRGB","different_bg"),4)
        self.assertEqual(issue_gap(self.spec,"RD","WRGB","different_bg"),17)
        self.assertEqual(issue_gap(self.spec,"RD","WRGB","different_rank"),15)

    def test_completion_and_ca_metadata(self):
        c = self.spec["commands"]
        self.assertEqual(c["WRGB"]["completion"],14)
        self.assertEqual(c["RDRES"]["completion"],25)
        self.assertEqual(c["MAC_RUN"]["external_ca"],21)
        self.assertEqual(c["MAC_RUN"]["completion"],22)
        for name in NO_EXTERNAL_CA:
            self.assertEqual(c[name]["external_ca"],0)
            # Native command_cycles is logical reception length, not CA use.
            self.assertEqual(self.spec["dram"]["command_cycles"][c[name]["id"]],1)

    def test_private_issue_path_is_separate_from_execution(self):
        c = self.spec["commands"]
        native = self.spec["metadata"]["commands"]
        for name in PRIVATE:
            self.assertEqual(c[name]["private_ca"],1)
            self.assertEqual(native[name]["private_ca"],1)
            self.assertEqual(native[name]["external_ca"],0)
            for following in PRIVATE:
                self.assertTrue(any(e[0]==1 and c[name]["id"] in e[1] and
                                    c[following]["id"] in e[2] and e[3]==1
                                    for e in self.spec["dram"]["timing_constraints"]))
        for name in LOCAL_EVENTS:
            self.assertEqual(c[name]["dq_cycles"],0)
            self.assertEqual(c[name]["private_ca"],0)
            self.assertEqual(c[LOCAL_ALIASES[name]]["private_ca"],1)
        self.assertEqual(c["LOCALRD"]["completion"],4)
        self.assertEqual(issue_gap(self.spec,"LOCALRD","LOCALRD","same_bg"),0)
        self.assertEqual(issue_gap(self.spec,"LOCALRD","BCAST","same_bg"),0)
        self.assertEqual(issue_gap(self.spec,"LOCALRD","WRGB","same_bg"),0)
        self.assertEqual(issue_gap(self.spec,"BCAST","BCAST","same_bg"),4)
        self.assertEqual(c["BCAST"]["completion"],4)

    def test_bcast_shares_rank_gio_with_io(self):
        # No global buffer: BCAST, WRGB and RDRES share the rank GIO/LIO path.
        t=self.spec["timings"]
        for b in ("BCAST","BCAST_LOCAL"):
            for w in ("WR","WRGB"):
                self.assertEqual(issue_gap(self.spec,w,b,"same_bg"),t["nCWL"]+t["nBL_max"])
                self.assertEqual(issue_gap(self.spec,b,w,"same_bg"),t["nBL_min"])
            for r in ("RD","RDRES"):
                self.assertEqual(issue_gap(self.spec,b,r,"same_bg"),t["nCCDL"]+t["nWTRL"])
                self.assertEqual(issue_gap(self.spec,r,b,"same_bg"),t["nCL"]+t["nBL_min"])
            # GIO is per rank: GPU rank0 traffic never waits on PIM rank1 BCAST.
            for x in ("RD","WR","WRGB","RDRES"):
                self.assertEqual(issue_gap(self.spec,x,b,"different_rank"),0)
                self.assertEqual(issue_gap(self.spec,b,x,"different_rank"),0)
        self.assertEqual(issue_gap(self.spec,"REDUCE","REDUCE","same_bg"),4)

    def test_actab_native_expansion_and_ca(self):
        c=self.spec["commands"]
        self.assertEqual(c["ACTAB"]["external_ca"],2)
        self.assertEqual(c["ACTAB_LOCAL"]["private_ca"],1)
        self.assertEqual(c["ACTAB_LOCAL"]["external_ca"],0)
        self.assertEqual(c["ACTAB"]["completion"],68)
        expansions=dict(self.spec["dram"]["command_expansions"])
        expected=[[c["ACT2_PULSE"]["id"],16*i,[[2,bg],[3,i]]] for i in range(4) for bg in range(4)]
        self.assertEqual(expansions[c["ACTAB"]["id"]],expected)
        self.assertEqual(expansions[c["ACTAB_LOCAL"]["id"]],expected)
        shared=[e for e in self.spec["dram"]["timing_constraints"] if len(e)>6]
        faw=[e for e in shared if c["ACT2_PULSE"]["id"] in e[1] and e[0]==1 and e[4]==4]
        self.assertEqual(len(faw),1)
        for name in ("ACT2","ACT2_LOCAL","REFpb"):
            self.assertIn(c[name]["id"],faw[0][1])
            self.assertIn(c[name]["id"],faw[0][2])

    def test_rope_bank_vector_pipeline(self):
        c=self.spec["commands"]
        self.assertEqual(self.spec["timings"]["nROPEBatch"],8)
        self.assertEqual(c["ROPE_EXEC"]["completion"]+7*self.spec["timings"]["nROPEII"],11)
        for a in ("ROPE_EXEC","ROPE_EXEC_LOCAL"):
            for b in ("ROPE_EXEC","ROPE_EXEC_LOCAL"):
                self.assertTrue(any(e[0]==3 and c[a]["id"] in e[1] and c[b]["id"] in e[2] and e[3]==8
                                    for e in self.spec["dram"]["timing_constraints"]))
        self.assertEqual(issue_gap(self.spec,"ROPE_EXEC","ROPE_EXEC","same_bg"),0)

    def test_mac_frequency_and_row_count(self):
        for k in (1,2,4):
            spec = build_spec(k)
            for name, rows in MAC_ROWS.items():
                self.assertEqual(spec["commands"][name]["completion"],4*rows//k)
        for bad in (0,3,8,True):
            with self.assertRaises(ValueError):
                build_spec(bad)

    def test_mc_srf_selector_uses_ca_without_array_timing(self):
        spec = build_spec(2, mc_control=True)
        selector = spec["commands"]["SRF_SELECT"]
        self.assertEqual((selector["wire_command"], selector["external_ca"], selector["dq_cycles"]),
                         ("ACTAB", 2, 0))
        for level, pre, fol, *_ in spec["dram"]["timing_constraints"]:
            if level:
                self.assertNotIn(selector["id"], pre)
        self.assertEqual(issue_gap(spec, "SRF_SELECT", "WRGB", "same_bg"), 2)
        self.assertEqual(issue_gap(spec, "WRGB", "MACAB_M16", "same_bg"), 1)

    def test_refresh_covers_pim_and_buffers(self):
        ids = {c:meta["id"] for c,meta in self.spec["commands"].items()}
        for refresh, level, delay in (("REFab",1,224),("REFpb",3,128)):
            for target in ("WRGB","RDRES","MACAB_M16","MACAB_M16_LOCAL","LOCALRD","LOCALWR"):
                self.assertTrue(any(e[0]==level and ids[refresh] in e[1] and ids[target] in e[2] and e[3]==delay for e in self.spec["dram"]["timing_constraints"]))

    def test_fixed_organization_and_clock(self):
        for args in (dict(rank=1),dict(dq=8),dict(rate=6400),dict(tCK_ps=1250)):
            with self.assertRaises(ValueError):
                LPDDR5XPIM(**args).to_config()


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1] == "--export":
        directory = pathlib.Path(sys.argv[2])
        directory.mkdir(parents=True,exist_ok=True)
        spec = build_spec()
        (directory/"spec.json").write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        (directory/"command-pair-matrix.json").write_text(json.dumps(command_matrix(spec),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    else:
        unittest.main()
