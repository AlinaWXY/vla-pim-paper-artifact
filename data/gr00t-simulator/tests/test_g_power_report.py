import importlib.util
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("g_power_report",ROOT/"tools/g_power_report.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BankPowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=json.loads((ROOT/"reference/spec/micron-8533-aat-proxy.json").read_text())
        cls.spec=json.loads((ROOT/"reference/spec/four-batch-k2.json").read_text())
        cls.c=mod.coefficients(cls.model,cls.spec,.5)

    def entry(self,k,rank=0):
        hist=[0]*17;hist[k]=100
        return {"rank":rank,"active_or_refresh_banks_ck":hist,"reads":0,"writes":0,
                "io_reads":0,"io_writes":0,"activations":0,"state_ck":[1600-100*k,0,100*k,0,0]}

    def test_more_open_banks_raise_background(self):
        one=mod.entry_energy(self.entry(1),self.c)["background_J"]
        all_banks=mod.entry_energy(self.entry(16),self.c)["background_J"]
        self.assertGreater(all_banks,one)
        self.assertAlmostEqual(all_banks-one,15*self.c["per_active_bank_W"]*100*938e-12,places=20)

    def test_bank_group_column_uses_two_bank_reference(self):
        self.assertGreater(self.c["two_active_banks_W"],self.c["shared_active_W"]+self.c["per_active_bank_W"])
        active_two=mod.entry_energy(self.entry(2),self.c)["background_J"]/(100*938e-12)
        read_power=self.c["column_read_reference_J"]/(2*938e-12)
        # The three non-Q rails of the published IDD4R table sum to this.
        self.assertAlmostEqual(active_two+read_power-self.c["VDDQ_background_W"],.53628,places=10)

    def test_refresh_duration_not_command_count_controls_tail(self):
        a=self.entry(2);a["state_ck"]=[1400,0,0,0,200]
        b=self.entry(2);b["state_ck"]=[1500,0,0,0,100]
        e1=mod.entry_energy(a,self.c)["refresh_increment_J"]
        e2=mod.entry_energy(b,self.c)["refresh_increment_J"]
        self.assertAlmostEqual(e1,2*e2,places=20)

    def test_combined_row_increment_keeps_ACT_PRE_pair_identity(self):
        self.assertAlmostEqual(self.c["ACT_PRE_combined_J"],.7744128e-9,places=18)


if __name__=="__main__":unittest.main()
