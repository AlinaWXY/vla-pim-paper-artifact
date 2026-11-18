"""Bind real L2 sector counters to retained ordered VLM durations.

The output is explicitly a traffic proxy: read misses generate reads; every L2
write sector is charged as a DRAM write (write-through assumption). Thor does
not expose dram__ counters. Neither a memory-pressure factor nor a preset R/W
ratio is used. Uniform timing and channel striping are modeling assumptions.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def canonical_name(s):
    # CUPTI and Nsight use different demanglers. Retain kernel/template identity;
    # discard expanded function parameter lists and normalize spelling only.
    s=s.replace('(anonymous namespace)','<unnamed>').replace('(bool)0','false').replace('(bool)1','true')
    s=re.sub(r'\(instance (\d+)\)',r'#\1',s)
    while '(' in s:
        new=re.sub(r'\([^()]*\)','',s)
        if new==s:
            raise ValueError('Unbalanced demangled name')
        s=new
    s=re.sub(r'\bvoid\b','',s)
    s=re.sub(r'\b(long|float|double|int) const\b',r'const \1',s)
    s=re.sub(r'\s+','',s)
    s=re.sub(r'[\[{]lambda#?(\d*)[\]}]',r'lambda\1',s)
    return re.sub(r'(\d+)(?:ull|ul|u)\b',r'\1',s)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--reads',type=Path,required=True)
    p.add_argument('--writes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    retained=json.loads((a.gpu/'prefix-kernels.json').read_text())['ordered_mean_units']
    trace=json.loads(a.trace.read_text())['traceEvents']
    ordered=sorted([e for e in trace if e.get('cat') in ('kernel','gpu_memcpy')],key=lambda e:e['ts'])
    assert len(ordered)==len(retained)==2357
    for old,new in zip(retained,ordered):
        if old['kind']=='kernel':
            assert new['cat']=='kernel'
            assert canonical_name(old['kernel'])==canonical_name(new['name']), (old['ordinal'],old['kernel'],new['name'])
        else:
            assert new['cat']=='gpu_memcpy'
    def load(path,metric):
        lines=path.read_text().splitlines()
        header=next(i for i,s in enumerate(lines) if s.startswith('"ID","Process ID"'))
        rows=list(csv.DictReader(lines[header:]));units=rows.pop(0)
        assert units[metric]=='sector'
        assert len(rows)==2353,len(rows)
        for i,row in enumerate(rows):
            assert int(row['ID'])==i
            assert float(row['profiler__replayer_passes'])==1, 'Replay cache warming would invalidate read misses'
            x=float(row[metric].replace(',',''))
            assert x>=0 and x.is_integer()
            row['sectors']=int(x)
        return rows
    reads=iter(load(a.reads,'lts__t_sectors_op_read_lookup_miss.sum'))
    writes=iter(load(a.writes,'lts__t_sectors_op_write.sum'))
    units=[]
    for old,new in zip(retained,ordered):
        u=dict(old)
        if old['kind']=='kernel':
            r,w=next(reads),next(writes)
            for row in (r,w):
                assert canonical_name(row['Kernel Name'])==canonical_name(old['kernel']), (old['ordinal'],row['Kernel Name'])
                assert list(ast.literal_eval(row['Grid Size']))==new['args']['grid']
                assert list(ast.literal_eval(row['Block Size']))==new['args']['block']
            u.update(read_sectors=r['sectors'],write_sectors=w['sectors'],
                     counter_kernel=r['Kernel Name'],source='Separate one-pass NCU L2 counters; full writeback proxy')
        else:
            sectors=math.ceil(new['args']['bytes']/32)
            read='DtoH' in new['name']
            assert read or 'HtoD' in new['name']
            u.update(read_sectors=sectors if read else 0,write_sectors=0 if read else sectors,
                     exact_copy_bytes=new['args']['bytes'],source=new['name'])
        units.append(u)
    report=dict(model='Measured L2 read misses + write-through L2 write proxy',
        source_sha256={str(p):sha(p) for p in (a.trace,a.reads,a.writes,a.gpu/'prefix-kernels.json')},
        validation=dict(kernel_order_template_names_grid_and_block_matched=2353,memory_copies_matched=4,
                        read_counter_passes=1,write_counter_passes=1),
        assumptions=dict(arrivals='Uniform within each kernel, with protocol drain tail calibrated to retained duration',
            channels='32-byte sectors evenly striped across all 16 native channels and four bank groups',
            reads='Every measured L2 read miss produces one 32-byte memory read',
            writes='Every measured L2 write sector produces one 32-byte memory write; writeback/coalescing not known',
            rows='GPU requests are row hits; ACT/PRE and real address locality are not reconstructed',
            traffic_pressure=None,outstanding_limit=None,read_write_ratio=None),
        read_bytes=sum(u['read_sectors'] for u in units)*32,
        write_proxy_bytes=sum(u['write_sectors'] for u in units)*32,units=units)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='units'},indent=2))


if __name__=='__main__':
    main()
