import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time


def gpu_guard():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    others=[int(s) for s in raw.splitlines() if s.strip().isdigit() and int(s)!=os.getpid()]
    if others:
        raise RuntimeError('Other GPU compute process(es): '+str(others))


def stats(xs):
    values=sorted(xs)
    def percentile(q):
        pos=(len(values)-1)*q
        lo=int(pos); hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(pos-lo)
    return dict(mean=statistics.mean(xs),p50=statistics.median(xs),p95=percentile(.95),
                minimum=min(xs),maximum=max(xs),stdev=statistics.stdev(xs) if len(xs)>1 else 0.)


def read_exact(f,size):
    value=bytearray(size); view=memoryview(value); offset=0
    while offset<size:
        count=f.readinto(view[offset:offset+min(8*1024**2,size-offset)])
        if not count:raise EOFError('Truncated checkpoint')
        offset+=count
    return value


def load_weights(model,path,phase):
    import torch
    with open(path,'rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if not 0<length<32*1024**2:raise ValueError('Invalid safetensors header')
        header=json.loads(f.read(length))
    targets=model.state_dict(keep_vars=True)
    mapping={}
    for source,info in header.items():
        if source=='__metadata__':continue
        key=source.removeprefix('model.')
        candidates=[key]
        if key not in targets:
            candidates.append(key.replace('paligemma.model.language_model.','paligemma.language_model.'))
            candidates.append(key.replace('paligemma.model.vision_tower.','paligemma.vision_tower.'))
            candidates.append(key.replace('paligemma.model.multi_modal_projector.','paligemma.multi_modal_projector.'))
        found=[k for k in dict.fromkeys(candidates) if k in targets]
        if len(found)!=1:raise ValueError('Ambiguous/missing weight key: '+source+' -> '+str(found))
        key=found[0]
        if list(targets[key].shape)!=info['shape']:raise ValueError('Shape mismatch: '+source)
        mapping[source]=key
    covered=set(mapping.values())
    aliases={}
    for name,target in targets.items():
        if name not in covered:
            candidates=[k for k in covered if targets[k].data_ptr()==target.data_ptr() and targets[k].shape==target.shape]
            if candidates:aliases[name]=candidates[0]
    missing=set(targets)-covered-set(aliases)
    if missing:raise ValueError('Uncovered weights: '+str(sorted(missing)))
    audit={'source_tensors':len(mapping),'target_names':len(targets),'aliases':aliases,'missing':[],
           'renamed':{s:k for s,k in mapping.items() if s!=k},'loader':'sequential_readinto'}
    types={'F32':torch.float32,'BF16':torch.bfloat16,'F16':torch.float16,'I64':torch.int64,'I32':torch.int32}
    start=last=time.monotonic();done=0; expected=0
    with open(path,'rb',buffering=0) as f,torch.no_grad():
        f.seek(8+length)
        for source,info in sorted(((k,v) for k,v in header.items() if k!='__metadata__'),key=lambda p:p[1]['data_offsets'][0]):
            begin,end=info['data_offsets']
            if begin!=expected:raise ValueError('Noncontiguous checkpoint')
            expected=end
            raw=read_exact(f,end-begin)
            tensor=torch.frombuffer(raw,dtype=types[info['dtype']]).reshape(info['shape'])
            targets[mapping[source]].copy_(tensor)
            done+=end-begin
            del tensor,raw
            if time.monotonic()-last>20:
                phase(f'readinto {done/1024**3:.2f} GiB in {time.monotonic()-start:.1f}s')
                last=time.monotonic()
    audit.update(seconds=time.monotonic()-start,bytes_read=done)
    return audit


def compare(a,b):
    import torch
    delta=(a.float()-b.float())
    scale=a.float().square().mean().sqrt().clamp_min(1e-6)
    result=dict(max_abs=delta.abs().max().item(),relative_rms=(delta.square().mean().sqrt()/scale).item(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    result['accepted']=result['finite'] and result['max_abs']<=.03 and result['relative_rms']<=.01
    return result
