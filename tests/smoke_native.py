"""Small causal end-to-end checks before expensive measured workloads."""
import json
from pathlib import Path
import subprocess
import tempfile
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ramulator2/python'))
from ramulator.dram.lpddr5x_pim import build_spec

def main():
    binary=ROOT/'build/pim_vla_sim'
    with tempfile.TemporaryDirectory(prefix='pim-frontend-') as directory:
        d=Path(directory)
        def save(name,data):
            p=d/name;p.write_text(json.dumps(data));return str(p)
        programs=save('program.json',dict(schema_version=1,stages=[dict(name='broadcast',channels=[[
            dict(id=i,op='WRGB',addr=[c,1,-1,-1,0,0],slot=i,deps=[]) for i in range(16)] for c in range(16)])]))
        vlm=save('vlm.json',dict(units=[dict(ordinal=i,kind='kernel',name='test',us=2.0,
            read_sectors=2048,write_sectors=2048,write_is_proxy=True,base_sector=i*8192) for i in range(3)]))
        ae=save('ae.json',dict(events=[dict(kind='gpu',name='nonlinear',us=0.2),dict(kind='pim',stage='broadcast'),dict(kind='gpu',name='tail',us=0.2)]))
        base=dict(spec=build_spec(),program_path=programs,vlm_path=vlm,ae_schedule_path=ae,
                  mode='concurrent',dataflow='streaming',arbitration='pim_head')
        raws={}
        for kind in ('ae_solo','vlm_solo','combined'):
            cfg=save(kind+'.json',dict(base,run_kind=kind));out=d/kind
            subprocess.run([str(binary),'--config',cfg,'--output',str(out)],check=True)
            raws[kind]=json.loads((out/'raw.json').read_text())
        for kind in ('vlm_solo','combined'):
            raw=raws[kind]
            assert raw['metrics']['gpu_read_bytes']==3*2048*32
            assert raw['metrics']['gpu_write_bytes']==3*2048*32
            assert len(raw['kernels'])==3
            assert raw['metrics']['GPU_reads']==3*2048
            assert raw['metrics']['GPU_writes']==3*2048
            assert raw['metrics']['GPU_row_hits']+raw['metrics']['GPU_row_misses']==3*4096
            assert all(k['duration_us']>=2 for k in raw['kernels'])
            assert raw['vlm_release']=='closed_loop' and raw['w_max']==128
            for k in raw['kernels']:
                assert round(k['duration_us']*1e6/raw['clock_ps'])==max(k['floor_ck'],k['memory_ck'])
            per_channel=[sum(x['gpu_write_bytes'] for x in raw['bus'] if x['channel']==c) for c in range(16)]
            assert len(set(per_channel))==1
        for kind in ('ae_solo','combined'):
            assert raws[kind]['metrics']['pim_write_bytes']==16*16*32
        for raw in raws.values():
            assert all(x['ca_cycles']<=x['duration_cycles'] and x['dq_cycles']<=x['duration_cycles'] for x in raw['bus'])
        cfg=save('priority.json',dict(base,vlm_gpu_quantum_us=0.25));out=d/'priority'
        subprocess.run([str(binary),'--config',cfg,'--output',str(out)],check=True)
        priority=json.loads((out/'raw.json').read_text())
        assert priority['metrics']['VLM_GPU_yields']>0
        assert priority['metrics']['AE_GPU_wait_us']<raws['combined']['metrics']['AE_GPU_wait_us']
        for field in ('gpu_read_bytes','gpu_write_bytes','pim_write_bytes'):
            assert priority['metrics'][field]==raws['combined']['metrics'][field]
        for kernel in priority['kernels']:
            ck_per_us=1e6/priority['clock_ps']
            duration_ck=round(kernel['duration_us']*ck_per_us)
            paused_ck=round(kernel['paused_us']*ck_per_us)
            assert duration_ck==max(kernel['floor_ck']+paused_ck,kernel['memory_ck'])
        serial=dict(base,run_kind='combined',mode='serial');serial.pop('arbitration')
        cfg=save('serial.json',serial);out=d/'serial'
        subprocess.run([str(binary),'--config',cfg,'--output',str(out)],check=True)
        raw=json.loads((out/'raw.json').read_text())
        expected=raws['ae_solo']['metrics']['total_us']+raws['vlm_solo']['metrics']['total_us']
        assert abs(raw['metrics']['total_us']-expected)<0.000939
        assert raw['metrics']['serial_epoch_reset'] is True
        for policy in ('pim_lookahead','gap_batch'):
            cfg=save(policy+'.json',dict(base,run_kind='combined',arbitration=policy));out=d/policy
            subprocess.run([str(binary),'--config',cfg,'--output',str(out)],check=True)
            raw=json.loads((out/'raw.json').read_text())
            assert raw['metrics']['gpu_read_bytes']==3*2048*32
        if len(sys.argv)>1:
            # Run exactly the same fixed input against the two binaries. The
            # legacy frontend ignores the new release selector.
            cfg=save('uniform.json',dict(base,vlm_release='uniform',run_kind='combined',
                                        calibration=dict(alpha=1,window=64,write_high=48,write_low=16)))
            for name,exe in (('baseline',sys.argv[1]),('uniform',binary)):
                subprocess.run([str(exe),'--config',cfg,'--output',str(d/name)],check=True)
            assert (d/'baseline/raw.json').read_bytes()==(d/'uniform/raw.json').read_bytes()
            print('uniform raw.json matches baseline byte-for-byte')
        print('native frontend smoke: byte conservation, per-channel mix, causal kernels, bus bounds passed')
if __name__=='__main__':main()
