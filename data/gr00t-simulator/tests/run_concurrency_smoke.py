"""Exercise every arbitration mode with a real PIM layer and finite GPU traffic."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run import prepare_configs

def main():
    destination=ROOT/'results/concurrency-smoke'
    configs=prepare_configs(destination/'configs')
    vlm=destination/'vlm.json'
    vlm.write_text(json.dumps(dict(units=[dict(ordinal=i,kind='kernel',name='bounded-memory',us=200,
        read_sectors=160000,write_sectors=80000,write_is_proxy=True,base_sector=i*262144) for i in range(2)])))
    for name in ('c','d','e','f','g','e-frfcfs','e-lookahead','e-gap'):
        cfg=dict(configs[name],vlm_path=str(vlm))
        events=json.loads(Path(cfg['ae_schedule_path']).read_text())
        events['events']=[e for e in events['events'] if e['step']==0 and e['layer']==0]
        schedule=destination/f'{name}-one-layer.json';schedule.write_text(json.dumps(events))
        cfg['ae_schedule_path']=str(schedule);cfgpath=destination/f'{name}.json'
        cfgpath.write_text(json.dumps(cfg));out=destination/name
        print('CONCURRENCY_SMOKE',name,flush=True)
        subprocess.run([str(ROOT/'build/pim_vla_sim'),'--config',str(cfgpath),'--output',str(out)],check=True)
        raw=json.loads((out/'raw.json').read_text())
        program=json.loads(Path(cfg['program_path']).read_text())
        m=raw['metrics']
        assert m['pim_read_bytes']+m['pim_write_bytes']==program['expected_dq_bytes_per_layer']
        assert (m['gpu_read_bytes'],m['gpu_write_bytes'])==(2*160000*32,2*80000*32)
        assert len(raw['kernels'])==2 and all(k['duration_us']>=200 for k in raw['kernels'])
        assert all(x['ca_cycles']<=x['duration_cycles'] and x['dq_cycles']<=x['duration_cycles'] for x in raw['bus'])
    print('all concurrent policies passed real-layer traffic and completion checks',flush=True)
if __name__=='__main__':main()
