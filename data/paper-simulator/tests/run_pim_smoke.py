"""Run one actual layer in every configuration before the 180-layer campaign."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run import prepare_configs

def main():
    destination=ROOT/'results/pim-smoke'
    configs=prepare_configs(destination/'configs')
    variants={}
    for name in ('b','d','e','f','g'):
        cfg=dict(configs[name],run_kind='ae_solo')
        events=json.loads(Path(cfg['ae_schedule_path']).read_text())
        events['events']=[e for e in events['events'] if e['step']==0 and e['layer']==0]
        schedule=destination/f'{name}-one-layer.json';schedule.write_text(json.dumps(events))
        cfg['ae_schedule_path']=str(schedule);cfgpath=destination/f'{name}.json'
        cfgpath.write_text(json.dumps(cfg));out=destination/name
        print('PIM_SMOKE',name,flush=True)
        subprocess.run([str(ROOT/'build/pim_vla_sim'),'--config',str(cfgpath),'--output',str(out)],check=True)
        raw=json.loads((out/'raw.json').read_text())
        program=json.loads(Path(cfg['program_path']).read_text())
        assert raw['metrics']['pim_read_bytes']+raw['metrics']['pim_write_bytes']==program['expected_dq_bytes_per_layer']
        variants[name]=raw['metrics']
    (destination/'validation.json').write_text(json.dumps(variants,indent=2))
    print('all five actual PIM programs passed one layer',flush=True)
if __name__=='__main__':main()
