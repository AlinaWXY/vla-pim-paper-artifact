"""Synthetic MHA8: expand only K/V; preserve original 8-head Q and O."""


def convert(model):
    import torch
    if model.training:
        raise ValueError('Only eval inference is supported')
    unchanged=[(p,p._version,p.data_ptr(),tuple(p.shape)) for n,p in model.named_parameters()
               if '.self_attn.k_proj.' not in n and '.self_attn.v_proj.' not in n]
    records=[]
    for family,tr in [('vlm',model.paligemma_with_expert.paligemma.language_model),
                      ('expert',model.paligemma_with_expert.gemma_expert.model)]:
        cfg=tr.config
        if (cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)!=(8,1,256):
            raise ValueError('Expected original 8Q/1KV/head_dim256 geometry')
        for index,layer in enumerate(tr.layers):
            attn=layer.self_attn
            assert attn.q_proj.out_features==2048 and attn.o_proj.in_features==2048
            assert attn.head_dim==256 and attn.scaling==256**-.5
            for name in ('k_proj','v_proj'):
                old=getattr(attn,name)
                new=torch.nn.Linear(old.in_features,2048,bias=old.bias is not None,
                                    device=old.weight.device,dtype=old.weight.dtype)
                with torch.no_grad():
                    new.weight.copy_(old.weight.reshape(1,256,old.in_features)
                                     .repeat_interleave(8,dim=0).flatten(0,1))
                    if old.bias is not None:
                        new.bias.copy_(old.bias.repeat(8))
                new.eval()
                setattr(attn,name,new)
                records.append(dict(family=family,layer=index,projection=name,
                                    old_shape=list(old.weight.shape),new_shape=list(new.weight.shape),
                                    dtype=str(new.weight.dtype),contiguous=new.weight.is_contiguous(),
                                    physical_storage_bytes=new.weight.untyped_storage().nbytes()))
            attn.num_key_value_groups=1
            if hasattr(attn,'num_key_value_heads'):
                attn.num_key_value_heads=8
        cfg.num_key_value_heads=8
    model.paligemma_with_expert.paligemma.config.text_config.num_key_value_heads=8
    for p,version,address,shape in unchanged:
        if (p._version,p.data_ptr(),tuple(p.shape))!=(version,address,shape):
            raise RuntimeError('A non-K/V parameter was modified')
    return dict(kind='synthetic_physical_MHA8',query_heads=8,kv_heads=8,head_dim=256,
                attention_width=2048,q_o_unchanged=True,non_kv_parameters_unchanged=True,
                changes=records,
                note='Only K/V weights are tiled into eight physically stored heads. Original Q/O, residual widths, FFNs and ViT are unchanged. Not a trained MHA8 policy.')
