# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
_tensor_constant0 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85babac0
_tensor_constant1 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85f4ad50
_tensor_constant2 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85eefe80
_tensor_constant3 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85ea55e0
_tensor_constant4 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85d28370
_tensor_constant5 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85d2b7f0
_tensor_constant6 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85c91270
_tensor_constant7 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85c50870
_tensor_constant8 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85736030
_tensor_constant9 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85717ca0
_tensor_constant10 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85652f30
_tensor_constant11 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc858bbb60
_tensor_constant12 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc8585fe30
_tensor_constant13 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc858f9450
_tensor_constant14 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85a5cd70
_tensor_constant15 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85a7fc00
_tensor_constant16 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85ae5c20
_tensor_constant17 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85b0b6b0
_tensor_constant18 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85d72b70
_tensor_constant19 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85dc32a0
_tensor_constant20 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85e03ac0
_tensor_constant21 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85e46ee0
_tensor_constant22 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc859a0910
_tensor_constant23 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc859c7b10
_tensor_constant24 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc85a358b0
_tensor_constant25 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc8555bc50
_tensor_constant26 = None  # device(type='cuda', index=0) torch.int64 (1296,) (1,) fffc855be760


# kernel path: /var/tmp/anonymous-gr00t16-inductor/gd/cgdsbuycg4th6rcopu6ub722hdlmrb3gw2ybc4quz762akqlrtrg.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states => add, add_1, convert_element_type_4, convert_element_type_5, mul_3, mul_4, rsqrt, sub, var_mean
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_1 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=buf1]
#   %arg1_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type_4 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %getitem_1), kwargs = {})
#   %add : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_4 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg1_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg2_1), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   return %getitem_1,%buf1,%convert_element_type_5
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 8962560}}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp3
        tmp12 = tl.full([1, 1], 1152.0, tl.float32)
        tmp13 = (tmp7 / tmp12)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24.to(tl.bfloat16)
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/rr/crrbwkzkwefdcbec6t27m7elpn6osu5nvyclidtxcdmqrfkurdte.py
# Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attention_mask => unsqueeze_5, unsqueeze_6
#   attn_output => _scaled_dot_product_efficient_attention, expand, full_default_2, full_default_3, where
#   eq => eq
#   getitem_13 => unsqueeze_3
#   getitem_14 => unsqueeze_4
#   key => clone_13
#   keys => view_18
#   keys_1 => view_22
#   keys_2 => permute_5
#   owners => lift_fresh_copy
#   queries => view_16
#   queries_1 => view_21
#   queries_2 => permute_4
#   query => clone_12
#   value => clone_14
#   values => view_20
#   values_1 => permute_3
#   view_2 => view_23
# Graph fragment:
#   %addmm : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_16 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 1296, 1152]), kwargs = {})
#   %view_21 : Tensor "bf16[1, 1296, 16, 72][1492992, 1152, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [1, 1296, 16, 72]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 16, 1296, 72][1492992, 72, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_12 : Tensor "bf16[1, 16, 1296, 72][1492992, 93312, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_18 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 1296, 1152]), kwargs = {})
#   %view_22 : Tensor "bf16[1, 1296, 16, 72][1492992, 1152, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_18, [1, 1296, 16, 72]), kwargs = {})
#   %permute_5 : Tensor "bf16[1, 16, 1296, 72][1492992, 72, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %clone_13 : Tensor "bf16[1, 16, 1296, 72][1492992, 93312, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_20 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 1296, 1152]), kwargs = {})
#   %view_23 : Tensor "bf16[1, 1296, 16, 72][1492992, 1152, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [1, 1296, 16, 72]), kwargs = {})
#   %permute_3 : Tensor "bf16[1, 16, 1296, 72][1492992, 72, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_14 : Tensor "bf16[1, 16, 1296, 72][1492992, 93312, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %lift_fresh_copy : Tensor "i64[1296][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1296, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%lift_fresh_copy, 1), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, 1296][1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%lift_fresh_copy, 0), kwargs = {})
#   %eq : Tensor "b8[1296, 1296][1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_3, %unsqueeze_4), kwargs = {})
#   %unsqueeze_5 : Tensor "b8[1, 1296, 1296][1679616, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, 0), kwargs = {})
#   %unsqueeze_6 : Tensor "b8[1, 1, 1296, 1296][1679616, 1679616, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 1), kwargs = {})
#   %full_default_3 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 1296, 1296][1679616, 1679616, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_3, %full_default_2), kwargs = {})
#   %expand : Tensor "bf16[1, 16, 1296, 1296][1679616, 0, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [1, 16, 1296, 1296]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_12, %clone_13, %clone_14, %expand, False), kwargs = {scale: 0.11785113019775792})
#   return %buf7
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8957952}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1492992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 72)
    x1 = ((xindex // 72) % 1296)
    x2 = xindex // 93312
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 72*x2 + 1152*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/sd/csdmypazzj7tdouduepewpihbb6sfbs3mgge2pnodkhjn7sxqnxi.py
# Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attention_mask => unsqueeze_5, unsqueeze_6
#   attn_output => _scaled_dot_product_efficient_attention, expand, full_default_2, full_default_3, where
#   eq => eq
#   getitem_13 => unsqueeze_3
#   getitem_14 => unsqueeze_4
#   key => clone_13
#   keys => view_18
#   keys_1 => view_22
#   keys_2 => permute_5
#   owners => lift_fresh_copy
#   queries => view_16
#   queries_1 => view_21
#   queries_2 => permute_4
#   query => clone_12
#   value => clone_14
#   values => view_20
#   values_1 => permute_3
#   view_2 => view_23
# Graph fragment:
#   %_tensor_constant0 : Tensor "i64[1296][1]cuda:0" = PlaceHolder[target=_tensor_constant0]
#   %view_16 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 1296, 1152]), kwargs = {})
#   %view_21 : Tensor "bf16[1, 1296, 16, 72][1492992, 1152, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_16, [1, 1296, 16, 72]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 16, 1296, 72][1492992, 72, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %clone_12 : Tensor "bf16[1, 16, 1296, 72][1492992, 93312, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_18 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 1296, 1152]), kwargs = {})
#   %view_22 : Tensor "bf16[1, 1296, 16, 72][1492992, 1152, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_18, [1, 1296, 16, 72]), kwargs = {})
#   %permute_5 : Tensor "bf16[1, 16, 1296, 72][1492992, 72, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %clone_13 : Tensor "bf16[1, 16, 1296, 72][1492992, 93312, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_20 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 1296, 1152]), kwargs = {})
#   %view_23 : Tensor "bf16[1, 1296, 16, 72][1492992, 1152, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [1, 1296, 16, 72]), kwargs = {})
#   %permute_3 : Tensor "bf16[1, 16, 1296, 72][1492992, 72, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %clone_14 : Tensor "bf16[1, 16, 1296, 72][1492992, 93312, 72, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %lift_fresh_copy : Tensor "i64[1296][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1296, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%lift_fresh_copy, 1), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, 1296][1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%lift_fresh_copy, 0), kwargs = {})
#   %eq : Tensor "b8[1296, 1296][1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_3, %unsqueeze_4), kwargs = {})
#   %unsqueeze_5 : Tensor "b8[1, 1296, 1296][1679616, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq, 0), kwargs = {})
#   %unsqueeze_6 : Tensor "b8[1, 1, 1296, 1296][1679616, 1679616, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 1), kwargs = {})
#   %full_default_3 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 1296, 1296][1679616, 1679616, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_3, %full_default_2), kwargs = {})
#   %expand : Tensor "bf16[1, 16, 1296, 1296][1679616, 0, 1296, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [1, 16, 1296, 1296]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%clone_12, %clone_13, %clone_14, %expand, False), kwargs = {scale: 0.11785113019775792})
#   return %buf10
triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6728832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1679616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1296
    x0 = (xindex % 1296)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0.0, tl.float32)
    tmp4 = tl.full([1], float("-inf"), tl.float32)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5.to(tl.bfloat16)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/2a/c2agylmbhcacqi4qu5pkh4bjxmsvdy2mrtfssyrmmmxtdxut4yk3.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_26
#   hidden_states_1 => add_2
#   hidden_states_2 => add_3, add_4, convert_element_type_18, convert_element_type_19, mul_5, mul_6, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %addmm_3 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_7 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf18 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=buf18]
#   %arg11_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %arg12_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %view_26 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1296, 1152]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %view_26), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_18, %getitem_7), kwargs = {})
#   %add_3 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_5 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_6 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %arg11_1), kwargs = {})
#   %add_4 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %arg12_1), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   return %getitem_7,%buf18,%convert_element_type_19
triton_red_fused_add_native_layer_norm_view_3 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 11948544}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.bfloat16)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(r0_mask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(r0_mask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(r0_mask & xmask, tmp11_weight_next, tmp11_weight)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp11_mean, tmp11_m2, tmp11_weight, 1)
    tmp11 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.bfloat16)
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp20.to(tl.bfloat16)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp19 + tmp22
        tmp24 = tmp23.to(tl.bfloat16)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 - tmp11
        tmp28 = tl.full([1, 1], 1152.0, tl.float32)
        tmp29 = (tmp15 / tmp28)
        tmp30 = tl.full([1, 1], 1e-06, tl.float32)
        tmp31 = tmp29 + tmp30
        tmp32 = libdevice.rsqrt(tmp31)
        tmp33 = tmp27 * tmp32
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 * tmp35
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp36 + tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp40.to(tl.bfloat16)
        tmp42 = tmp41.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp42, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/x7/cx7ls7mp634cwuc3r2tojrn34gi66n34ala5jj3gzwc6s7ey57q5.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => view_28
#   hidden_states_4 => add_5, add_6, convert_element_type_23, convert_element_type_24, mul_10, mul_11, mul_12, mul_7, mul_8, mul_9, tanh
# Graph fragment:
#   %addmm_4 : Tensor "bf16[1296, 4304][4304, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %view_28 : Tensor "bf16[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [1, 1296, 4304]), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_28, torch.float32), kwargs = {})
#   %mul_11 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, 0.5), kwargs = {})
#   %mul_7 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %convert_element_type_23), kwargs = {})
#   %mul_8 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %convert_element_type_23), kwargs = {})
#   %mul_9 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 0.044715), kwargs = {})
#   %add_5 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_23, %mul_9), kwargs = {})
#   %mul_10 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_10,), kwargs = {})
#   %add_6 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_12 : Tensor "f32[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %add_6), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[1, 1296, 4304][5577984, 4304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   return %convert_element_type_24
triton_poi_fused_gelu_view_4 = async_compile.triton('triton_poi_fused_gelu_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33467904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5577984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4304)
    x1 = xindex // 4304
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = tl.full([1], 0.044715, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.bfloat16)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 4352*x1), tmp17, xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/up/cup7pkbh2haacvrota5msxgd57nzupa26gsygxzmuqfzbug3nrje.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_26
#   hidden_states_1 => add_2
#   hidden_states_5 => view_30
#   hidden_states_6 => add_7
#   hidden_states_7 => add_8, add_9, convert_element_type_28, convert_element_type_29, mul_13, mul_14, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %addmm_3 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_5 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %getitem_9 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf25 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=buf25]
#   %arg17_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %mul_14 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=mul_14]
#   %arg18_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %view_26 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1296, 1152]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %view_26), kwargs = {})
#   %view_30 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 1296, 1152]), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_30), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_28, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_28, %getitem_9), kwargs = {})
#   %add_8 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_13 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_14 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %arg17_1), kwargs = {})
#   %add_9 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg18_1), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   return %getitem_9,%buf25,%mul_14,%convert_element_type_29
triton_red_fused_add_native_layer_norm_view_5 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 14934528}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp19_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.bfloat16)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8.to(tl.bfloat16)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.bfloat16)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight, roffset == 0
        )
        tmp19_mean = tl.where(r0_mask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(r0_mask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(r0_mask & xmask, tmp19_weight_next, tmp19_weight)
    tmp20, tmp21, tmp22 = triton_helpers.welford(tmp19_mean, tmp19_m2, tmp19_weight, 1)
    tmp19 = tmp20[:, None]
    tmp23 = tmp21[:, None]
    tmp24 = tmp22[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp25 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.bfloat16)
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tmp28.to(tl.bfloat16)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp27 + tmp30
        tmp32 = tmp31.to(tl.bfloat16)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33.to(tl.bfloat16)
        tmp35 = tmp34.to(tl.float32)
        tmp37 = tmp36.to(tl.bfloat16)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp35 + tmp38
        tmp40 = tmp39.to(tl.bfloat16)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42 - tmp19
        tmp44 = tl.full([1, 1], 1152.0, tl.float32)
        tmp45 = (tmp23 / tmp44)
        tmp46 = tl.full([1, 1], 1e-06, tl.float32)
        tmp47 = tmp45 + tmp46
        tmp48 = libdevice.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp49 * tmp51
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp52 + tmp54
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56.to(tl.bfloat16)
        tmp58 = tmp57.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 1152*x0), tmp58, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/tb/ctbfxlnf7omyzvqfyqjpp222pdxsn56tjmedk3xzxs62si46jrc6.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_26
#   attn_output_7 => view_42
#   hidden_states_1 => add_2
#   hidden_states_5 => view_30
#   hidden_states_6 => add_7
#   hidden_states_8 => add_10
#   hidden_states_9 => add_11, add_12, convert_element_type_42, convert_element_type_43, mul_15, mul_16, rsqrt_3, sub_3, var_mean_3
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %addmm_3 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_5 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %getitem_15 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf43 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=buf43]
#   %mul_15 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=mul_15]
#   %arg27_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %view_26 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1296, 1152]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %view_26), kwargs = {})
#   %view_30 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 1296, 1152]), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_30), kwargs = {})
#   %view_42 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 1296, 1152]), kwargs = {})
#   %add_10 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_42), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_42, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_42, %getitem_15), kwargs = {})
#   %add_11 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_15 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_16 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %arg27_1), kwargs = {})
#   %add_12 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %arg28_1), kwargs = {})
#   %convert_element_type_43 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   return %getitem_15,%buf43,%mul_15,%convert_element_type_43
triton_red_fused_add_native_layer_norm_view_6 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 2, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 17920512}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp27_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp27_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.bfloat16)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8.to(tl.bfloat16)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.bfloat16)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.bfloat16)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp18 + tmp21
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp27_mean_next, tmp27_m2_next, tmp27_weight_next = triton_helpers.welford_reduce(
            tmp26, tmp27_mean, tmp27_m2, tmp27_weight, roffset == 0
        )
        tmp27_mean = tl.where(r0_mask & xmask, tmp27_mean_next, tmp27_mean)
        tmp27_m2 = tl.where(r0_mask & xmask, tmp27_m2_next, tmp27_m2)
        tmp27_weight = tl.where(r0_mask & xmask, tmp27_weight_next, tmp27_weight)
    tmp28, tmp29, tmp30 = triton_helpers.welford(tmp27_mean, tmp27_m2, tmp27_weight, 1)
    tmp27 = tmp28[:, None]
    tmp31 = tmp29[:, None]
    tmp32 = tmp30[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp33 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp44 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp66 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp69 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tmp33.to(tl.bfloat16)
        tmp35 = tmp34.to(tl.float32)
        tmp37 = tmp36.to(tl.bfloat16)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp35 + tmp38
        tmp40 = tmp39.to(tl.bfloat16)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp41.to(tl.bfloat16)
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp44.to(tl.bfloat16)
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp43 + tmp46
        tmp48 = tmp47.to(tl.bfloat16)
        tmp49 = tmp48.to(tl.float32)
        tmp50 = tmp49.to(tl.bfloat16)
        tmp51 = tmp50.to(tl.float32)
        tmp53 = tmp52.to(tl.bfloat16)
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp51 + tmp54
        tmp56 = tmp55.to(tl.bfloat16)
        tmp57 = tmp56.to(tl.float32)
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58 - tmp27
        tmp60 = tl.full([1, 1], 1152.0, tl.float32)
        tmp61 = (tmp31 / tmp60)
        tmp62 = tl.full([1, 1], 1e-06, tl.float32)
        tmp63 = tmp61 + tmp62
        tmp64 = libdevice.rsqrt(tmp63)
        tmp65 = tmp59 * tmp64
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp65 * tmp67
        tmp70 = tmp69.to(tl.float32)
        tmp71 = tmp68 + tmp70
        tmp72 = tmp71.to(tl.float32)
        tmp73 = tmp72.to(tl.bfloat16)
        tmp74 = tmp73.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 1152*x0), tmp74, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/gr/cgrci5aaa5yq2nzqynwohc4xfud5ppmqhrjpdzbtw4osfwgxhbcx.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_3 => view_26
#   attn_output_7 => view_42
#   hidden_states_1 => add_2
#   hidden_states_12 => view_46
#   hidden_states_13 => add_15
#   hidden_states_14 => add_16, add_17, convert_element_type_52, convert_element_type_53, mul_23, mul_24, rsqrt_4, sub_4, var_mean_4
#   hidden_states_5 => view_30
#   hidden_states_6 => add_7
#   hidden_states_8 => add_10
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %addmm_3 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %addmm_5 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_5]
#   %addmm_9 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_9]
#   %addmm_11 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_11]
#   %add_15 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=add_15]
#   %getitem_17 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=getitem_17]
#   %buf52 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=buf52]
#   %arg33_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg33_1]
#   %arg34_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %view_26 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [1, 1296, 1152]), kwargs = {})
#   %add_2 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %view_26), kwargs = {})
#   %view_30 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_5, [1, 1296, 1152]), kwargs = {})
#   %add_7 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_30), kwargs = {})
#   %view_42 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_9, [1, 1296, 1152]), kwargs = {})
#   %add_10 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_42), kwargs = {})
#   %view_46 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_11, [1, 1296, 1152]), kwargs = {})
#   %add_15 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_46), kwargs = {})
#   %convert_element_type_52 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_52, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_52, %getitem_17), kwargs = {})
#   %add_16 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_23 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_24 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %arg33_1), kwargs = {})
#   %add_17 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %arg34_1), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.bfloat16), kwargs = {})
#   return %add_15,%getitem_17,%buf52,%convert_element_type_53
triton_red_fused_add_native_layer_norm_view_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 26878464}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp35_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp35_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp35_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.bfloat16)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8.to(tl.bfloat16)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.bfloat16)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.bfloat16)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp18 + tmp21
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24.to(tl.bfloat16)
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp27.to(tl.bfloat16)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp26 + tmp29
        tmp31 = tmp30.to(tl.bfloat16)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp35_mean_next, tmp35_m2_next, tmp35_weight_next = triton_helpers.welford_reduce(
            tmp34, tmp35_mean, tmp35_m2, tmp35_weight, roffset == 0
        )
        tmp35_mean = tl.where(r0_mask & xmask, tmp35_mean_next, tmp35_mean)
        tmp35_m2 = tl.where(r0_mask & xmask, tmp35_m2_next, tmp35_m2)
        tmp35_weight = tl.where(r0_mask & xmask, tmp35_weight_next, tmp35_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1152*x0), tmp32, r0_mask & xmask)
    tmp36, tmp37, tmp38 = triton_helpers.welford(tmp35_mean, tmp35_m2, tmp35_weight, 1)
    tmp35 = tmp36[:, None]
    tmp39 = tmp37[:, None]
    tmp40 = tmp38[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp41 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42 - tmp35
        tmp44 = tl.full([1, 1], 1152.0, tl.float32)
        tmp45 = (tmp39 / tmp44)
        tmp46 = tl.full([1, 1], 1e-06, tl.float32)
        tmp47 = tmp45 + tmp46
        tmp48 = libdevice.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp49 * tmp51
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp52 + tmp54
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56.to(tl.bfloat16)
        tmp58 = tmp57.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp58, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/nv/cnvzkguh2h4nztbi3m4xlhgiyxne4noi7f6uwyq43eie2ci5qrid.py
# Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_output_11 => view_58
#   attn_output_15 => view_74
#   hidden_states_15 => add_18
#   hidden_states_19 => view_62
#   hidden_states_20 => add_23
#   hidden_states_22 => add_26
#   hidden_states_26 => view_78
#   hidden_states_27 => add_31
#   hidden_states_28 => add_32, add_33, convert_element_type_100, convert_element_type_101, mul_43, mul_44, rsqrt_8, sub_8, var_mean_8
# Graph fragment:
#   %add_15 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=add_15]
#   %addmm_15 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_15]
#   %addmm_17 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_17]
#   %addmm_21 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_21]
#   %addmm_23 : Tensor "bf16[1296, 1152][1152, 1]cuda:0" = PlaceHolder[target=addmm_23]
#   %add_31 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0" = PlaceHolder[target=add_31]
#   %getitem_33 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=getitem_33]
#   %buf103 : Tensor "f32[1, 1296, 1][1312, 1, 1312]cuda:0" = PlaceHolder[target=buf103]
#   %arg65_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %arg66_1 : Tensor "bf16[1152][1]cuda:0" = PlaceHolder[target=arg66_1]
#   %view_58 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_15, [1, 1296, 1152]), kwargs = {})
#   %add_18 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_58), kwargs = {})
#   %view_62 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_17, [1, 1296, 1152]), kwargs = {})
#   %add_23 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_62), kwargs = {})
#   %view_74 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_21, [1, 1296, 1152]), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %view_74), kwargs = {})
#   %view_78 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_23, [1, 1296, 1152]), kwargs = {})
#   %add_31 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_78), kwargs = {})
#   %convert_element_type_100 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_100, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_100, %getitem_33), kwargs = {})
#   %add_32 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 1296, 1][1296, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_43 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %mul_44 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %arg65_1), kwargs = {})
#   %add_33 : Tensor "f32[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %arg66_1), kwargs = {})
#   %convert_element_type_101 : Tensor "bf16[1, 1296, 1152][1492992, 1152, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.bfloat16), kwargs = {})
#   return %add_31,%getitem_33,%buf103,%convert_element_type_101
triton_red_fused_add_native_layer_norm_view_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 26878464}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 1152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp35_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp35_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp35_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.bfloat16)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6.to(tl.bfloat16)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8.to(tl.bfloat16)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.bfloat16)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16.to(tl.bfloat16)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.bfloat16)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp18 + tmp21
        tmp23 = tmp22.to(tl.bfloat16)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp24.to(tl.bfloat16)
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp27.to(tl.bfloat16)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp26 + tmp29
        tmp31 = tmp30.to(tl.bfloat16)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp35_mean_next, tmp35_m2_next, tmp35_weight_next = triton_helpers.welford_reduce(
            tmp34, tmp35_mean, tmp35_m2, tmp35_weight, roffset == 0
        )
        tmp35_mean = tl.where(r0_mask & xmask, tmp35_mean_next, tmp35_mean)
        tmp35_m2 = tl.where(r0_mask & xmask, tmp35_m2_next, tmp35_m2)
        tmp35_weight = tl.where(r0_mask & xmask, tmp35_weight_next, tmp35_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1152*x0), tmp32, r0_mask & xmask)
    tmp36, tmp37, tmp38 = triton_helpers.welford(tmp35_mean, tmp35_m2, tmp35_weight, 1)
    tmp35 = tmp36[:, None]
    tmp39 = tmp37[:, None]
    tmp40 = tmp38[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp41 = tl.load(in_out_ptr0 + (r0_1 + 1152*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42 - tmp35
        tmp44 = tl.full([1, 1], 1152.0, tl.float32)
        tmp45 = (tmp39 / tmp44)
        tmp46 = tl.full([1, 1], 1e-06, tl.float32)
        tmp47 = tmp45 + tmp46
        tmp48 = libdevice.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp49 * tmp51
        tmp54 = tmp53.to(tl.float32)
        tmp55 = tmp52 + tmp54
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56.to(tl.bfloat16)
        tmp58 = tmp57.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1152*x0), tmp58, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/j3/cj3uofnqeiclaqcteje2pciaaytcopluqotvf2kh7twc5xxchyvf.py
# Topologically Sorted Source Nodes: [ones_like], Original ATen: [aten.ones_like]
# Source node to ATen node mapping:
#   ones_like => full_default
# Graph fragment:
#   %full_default : Tensor "f32[262144, 18][18, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([262144, 18], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_ones_like_9 = async_compile.triton('triton_poi_fused_ones_like_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_like_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_like_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 1.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/wy/cwy6lgriiemrrirq5mloecy4wf25fnscnkhgvtbanx4inefnfuv5.py
# Topologically Sorted Source Nodes: [arange, float_1, flat_pos, x_pos, outer, arange_1, float_2, dim_range, truediv, pow_1, freqs], Original ATen: [aten.arange, aten._to_copy, aten.remainder, aten.view, aten.div, aten.pow, aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   arange => iota
#   arange_1 => iota_1
#   dim_range => device_put_1
#   flat_pos => device_put
#   float_1 => convert_element_type
#   float_2 => convert_element_type_2
#   freqs => mul, reciprocal
#   outer => mul_1, view
#   pow_1 => pow_1
#   truediv => div_1
#   x_pos => remainder
# Graph fragment:
#   %iota : Tensor "i64[262144][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (262144,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cpu, requires_grad: False})
#   %convert_element_type : Tensor "f32[262144][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %device_put : Tensor "f32[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.device_put.default](args = (%convert_element_type, cuda:0), kwargs = {})
#   %remainder : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%device_put, 512), kwargs = {})
#   %view : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%remainder, [262144, 1]), kwargs = {})
#   %iota_1 : Tensor "i64[18][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (18,), kwargs = {start: 0, step: 4, dtype: torch.int64, device: cpu, requires_grad: False})
#   %convert_element_type_2 : Tensor "f32[18][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %device_put_1 : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.device_put.default](args = (%convert_element_type_2, cuda:0), kwargs = {})
#   %div_1 : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%device_put_1, 72), kwargs = {})
#   %pow_1 : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Scalar](args = (14, %div_1), kwargs = {})
#   %reciprocal : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%pow_1,), kwargs = {})
#   %mul : Tensor "f32[18][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_1 : Tensor "f32[262144, 18][18, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %mul), kwargs = {})
#   return %mul_1
triton_poi_fused__to_copy_arange_div_mul_pow_reciprocal_remainder_view_10 = async_compile.triton('triton_poi_fused__to_copy_arange_div_mul_pow_reciprocal_remainder_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_div_mul_pow_reciprocal_remainder_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_div_mul_pow_reciprocal_remainder_view_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 18
    x0 = (xindex % 18)
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 512.0, tl.float32)
    tmp3 = (tmp1 % tmp2)
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 != tmp4
    tmp6 = (libdevice.signbit(tmp3) != 0) if (tmp3).dtype is tl.float32 else tmp3 < 0
    tmp7 = (libdevice.signbit(tmp2) != 0) if (tmp2).dtype is tl.float32 else tmp2 < 0
    tmp8 = tmp6 != tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = tmp3 + tmp2
    tmp11 = tl.where(tmp9, tmp10, tmp3)
    tmp12 = 4*x0
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.full([1], 0.013888888888888888, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1], 14.0, tl.float32)
    tmp17 = libdevice.pow(tmp16, tmp15)
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = (tmp18 / tmp17)
    tmp20 = tl.full([1], 1.0, tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp11 * tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /var/tmp/anonymous-gr00t16-inductor/w4/cw4vzigefxtbapbs46dqtz7sodhb3xwira4yalinuijaeaowuizu.py
# Topologically Sorted Source Nodes: [arange, float_1, flat_pos, arange_1, float_2, dim_range, truediv, pow_1, freqs, y_pos, outer_1], Original ATen: [aten.arange, aten._to_copy, aten.div, aten.pow, aten.reciprocal, aten.mul, aten.floor_divide, aten.view]
# Source node to ATen node mapping:
#   arange => iota
#   arange_1 => iota_1
#   dim_range => device_put_1
#   flat_pos => device_put
#   float_1 => convert_element_type
#   float_2 => convert_element_type_2
#   freqs => mul, reciprocal
#   outer_1 => mul_2, view_1
#   pow_1 => pow_1
#   truediv => div_1
#   y_pos => div
# Graph fragment:
#   %iota : Tensor "i64[262144][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (262144,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cpu, requires_grad: False})
#   %convert_element_type : Tensor "f32[262144][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %device_put : Tensor "f32[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.device_put.default](args = (%convert_element_type, cuda:0), kwargs = {})
#   %iota_1 : Tensor "i64[18][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (18,), kwargs = {start: 0, step: 4, dtype: torch.int64, device: cpu, requires_grad: False})
#   %convert_element_type_2 : Tensor "f32[18][1]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %device_put_1 : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.device_put.default](args = (%convert_element_type_2, cuda:0), kwargs = {})
#   %div_1 : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%device_put_1, 72), kwargs = {})
#   %pow_1 : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Scalar](args = (14, %div_1), kwargs = {})
#   %reciprocal : Tensor "f32[18][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%pow_1,), kwargs = {})
#   %mul : Tensor "f32[18][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %div : Tensor "f32[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%device_put, 512), kwargs = {rounding_mode: floor})
#   %view_1 : Tensor "f32[262144, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [262144, 1]), kwargs = {})
#   %mul_2 : Tensor "f32[262144, 18][18, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %mul), kwargs = {})
#   return %mul_2
triton_poi_fused__to_copy_arange_div_floor_divide_mul_pow_reciprocal_view_11 = async_compile.triton('triton_poi_fused__to_copy_arange_div_floor_divide_mul_pow_reciprocal_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=110, major=11, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_div_floor_divide_mul_pow_reciprocal_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3BC24E6C01BAD5A0C85232FF07E436ED0A17249ACE5F334A4CA22C009AABC794', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 37748736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_div_floor_divide_mul_pow_reciprocal_view_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 18
    x0 = (xindex % 18)
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.001953125, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = 4*x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full([1], 0.013888888888888888, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 14.0, tl.float32)
    tmp10 = libdevice.pow(tmp9, tmp8)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = (tmp11 / tmp10)
    tmp13 = tl.full([1], 1.0, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp4 * tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1296, 1152), (1492992, 1152, 1))
        assert_size_stride(arg1_1, (1152, ), (1, ))
        assert_size_stride(arg2_1, (1152, ), (1, ))
        assert_size_stride(arg3_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg4_1, (1152, ), (1, ))
        assert_size_stride(arg5_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg6_1, (1152, ), (1, ))
        assert_size_stride(arg7_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg8_1, (1152, ), (1, ))
        assert_size_stride(arg9_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg10_1, (1152, ), (1, ))
        assert_size_stride(arg11_1, (1152, ), (1, ))
        assert_size_stride(arg12_1, (1152, ), (1, ))
        assert_size_stride(arg13_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg14_1, (4304, ), (1, ))
        assert_size_stride(arg15_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg16_1, (1152, ), (1, ))
        assert_size_stride(arg17_1, (1152, ), (1, ))
        assert_size_stride(arg18_1, (1152, ), (1, ))
        assert_size_stride(arg19_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg20_1, (1152, ), (1, ))
        assert_size_stride(arg21_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg22_1, (1152, ), (1, ))
        assert_size_stride(arg23_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg24_1, (1152, ), (1, ))
        assert_size_stride(arg25_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg26_1, (1152, ), (1, ))
        assert_size_stride(arg27_1, (1152, ), (1, ))
        assert_size_stride(arg28_1, (1152, ), (1, ))
        assert_size_stride(arg29_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg30_1, (4304, ), (1, ))
        assert_size_stride(arg31_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg32_1, (1152, ), (1, ))
        assert_size_stride(arg33_1, (1152, ), (1, ))
        assert_size_stride(arg34_1, (1152, ), (1, ))
        assert_size_stride(arg35_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg36_1, (1152, ), (1, ))
        assert_size_stride(arg37_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg38_1, (1152, ), (1, ))
        assert_size_stride(arg39_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg40_1, (1152, ), (1, ))
        assert_size_stride(arg41_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg42_1, (1152, ), (1, ))
        assert_size_stride(arg43_1, (1152, ), (1, ))
        assert_size_stride(arg44_1, (1152, ), (1, ))
        assert_size_stride(arg45_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg46_1, (4304, ), (1, ))
        assert_size_stride(arg47_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg48_1, (1152, ), (1, ))
        assert_size_stride(arg49_1, (1152, ), (1, ))
        assert_size_stride(arg50_1, (1152, ), (1, ))
        assert_size_stride(arg51_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg52_1, (1152, ), (1, ))
        assert_size_stride(arg53_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg54_1, (1152, ), (1, ))
        assert_size_stride(arg55_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg56_1, (1152, ), (1, ))
        assert_size_stride(arg57_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg58_1, (1152, ), (1, ))
        assert_size_stride(arg59_1, (1152, ), (1, ))
        assert_size_stride(arg60_1, (1152, ), (1, ))
        assert_size_stride(arg61_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg62_1, (4304, ), (1, ))
        assert_size_stride(arg63_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg64_1, (1152, ), (1, ))
        assert_size_stride(arg65_1, (1152, ), (1, ))
        assert_size_stride(arg66_1, (1152, ), (1, ))
        assert_size_stride(arg67_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg68_1, (1152, ), (1, ))
        assert_size_stride(arg69_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg70_1, (1152, ), (1, ))
        assert_size_stride(arg71_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg72_1, (1152, ), (1, ))
        assert_size_stride(arg73_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg74_1, (1152, ), (1, ))
        assert_size_stride(arg75_1, (1152, ), (1, ))
        assert_size_stride(arg76_1, (1152, ), (1, ))
        assert_size_stride(arg77_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg78_1, (4304, ), (1, ))
        assert_size_stride(arg79_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg80_1, (1152, ), (1, ))
        assert_size_stride(arg81_1, (1152, ), (1, ))
        assert_size_stride(arg82_1, (1152, ), (1, ))
        assert_size_stride(arg83_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg84_1, (1152, ), (1, ))
        assert_size_stride(arg85_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg86_1, (1152, ), (1, ))
        assert_size_stride(arg87_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg88_1, (1152, ), (1, ))
        assert_size_stride(arg89_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg90_1, (1152, ), (1, ))
        assert_size_stride(arg91_1, (1152, ), (1, ))
        assert_size_stride(arg92_1, (1152, ), (1, ))
        assert_size_stride(arg93_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg94_1, (4304, ), (1, ))
        assert_size_stride(arg95_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg96_1, (1152, ), (1, ))
        assert_size_stride(arg97_1, (1152, ), (1, ))
        assert_size_stride(arg98_1, (1152, ), (1, ))
        assert_size_stride(arg99_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg100_1, (1152, ), (1, ))
        assert_size_stride(arg101_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg102_1, (1152, ), (1, ))
        assert_size_stride(arg103_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg104_1, (1152, ), (1, ))
        assert_size_stride(arg105_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg106_1, (1152, ), (1, ))
        assert_size_stride(arg107_1, (1152, ), (1, ))
        assert_size_stride(arg108_1, (1152, ), (1, ))
        assert_size_stride(arg109_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg110_1, (4304, ), (1, ))
        assert_size_stride(arg111_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg112_1, (1152, ), (1, ))
        assert_size_stride(arg113_1, (1152, ), (1, ))
        assert_size_stride(arg114_1, (1152, ), (1, ))
        assert_size_stride(arg115_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg116_1, (1152, ), (1, ))
        assert_size_stride(arg117_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg118_1, (1152, ), (1, ))
        assert_size_stride(arg119_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg120_1, (1152, ), (1, ))
        assert_size_stride(arg121_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg122_1, (1152, ), (1, ))
        assert_size_stride(arg123_1, (1152, ), (1, ))
        assert_size_stride(arg124_1, (1152, ), (1, ))
        assert_size_stride(arg125_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg126_1, (4304, ), (1, ))
        assert_size_stride(arg127_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg128_1, (1152, ), (1, ))
        assert_size_stride(arg129_1, (1152, ), (1, ))
        assert_size_stride(arg130_1, (1152, ), (1, ))
        assert_size_stride(arg131_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg132_1, (1152, ), (1, ))
        assert_size_stride(arg133_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg134_1, (1152, ), (1, ))
        assert_size_stride(arg135_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg136_1, (1152, ), (1, ))
        assert_size_stride(arg137_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg138_1, (1152, ), (1, ))
        assert_size_stride(arg139_1, (1152, ), (1, ))
        assert_size_stride(arg140_1, (1152, ), (1, ))
        assert_size_stride(arg141_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg142_1, (4304, ), (1, ))
        assert_size_stride(arg143_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg144_1, (1152, ), (1, ))
        assert_size_stride(arg145_1, (1152, ), (1, ))
        assert_size_stride(arg146_1, (1152, ), (1, ))
        assert_size_stride(arg147_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg148_1, (1152, ), (1, ))
        assert_size_stride(arg149_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg150_1, (1152, ), (1, ))
        assert_size_stride(arg151_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg152_1, (1152, ), (1, ))
        assert_size_stride(arg153_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg154_1, (1152, ), (1, ))
        assert_size_stride(arg155_1, (1152, ), (1, ))
        assert_size_stride(arg156_1, (1152, ), (1, ))
        assert_size_stride(arg157_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg158_1, (4304, ), (1, ))
        assert_size_stride(arg159_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg160_1, (1152, ), (1, ))
        assert_size_stride(arg161_1, (1152, ), (1, ))
        assert_size_stride(arg162_1, (1152, ), (1, ))
        assert_size_stride(arg163_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg164_1, (1152, ), (1, ))
        assert_size_stride(arg165_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg166_1, (1152, ), (1, ))
        assert_size_stride(arg167_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg168_1, (1152, ), (1, ))
        assert_size_stride(arg169_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg170_1, (1152, ), (1, ))
        assert_size_stride(arg171_1, (1152, ), (1, ))
        assert_size_stride(arg172_1, (1152, ), (1, ))
        assert_size_stride(arg173_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg174_1, (4304, ), (1, ))
        assert_size_stride(arg175_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg176_1, (1152, ), (1, ))
        assert_size_stride(arg177_1, (1152, ), (1, ))
        assert_size_stride(arg178_1, (1152, ), (1, ))
        assert_size_stride(arg179_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg180_1, (1152, ), (1, ))
        assert_size_stride(arg181_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg182_1, (1152, ), (1, ))
        assert_size_stride(arg183_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg184_1, (1152, ), (1, ))
        assert_size_stride(arg185_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg186_1, (1152, ), (1, ))
        assert_size_stride(arg187_1, (1152, ), (1, ))
        assert_size_stride(arg188_1, (1152, ), (1, ))
        assert_size_stride(arg189_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg190_1, (4304, ), (1, ))
        assert_size_stride(arg191_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg192_1, (1152, ), (1, ))
        assert_size_stride(arg193_1, (1152, ), (1, ))
        assert_size_stride(arg194_1, (1152, ), (1, ))
        assert_size_stride(arg195_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg196_1, (1152, ), (1, ))
        assert_size_stride(arg197_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg198_1, (1152, ), (1, ))
        assert_size_stride(arg199_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg200_1, (1152, ), (1, ))
        assert_size_stride(arg201_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg202_1, (1152, ), (1, ))
        assert_size_stride(arg203_1, (1152, ), (1, ))
        assert_size_stride(arg204_1, (1152, ), (1, ))
        assert_size_stride(arg205_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg206_1, (4304, ), (1, ))
        assert_size_stride(arg207_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg208_1, (1152, ), (1, ))
        assert_size_stride(arg209_1, (1152, ), (1, ))
        assert_size_stride(arg210_1, (1152, ), (1, ))
        assert_size_stride(arg211_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg212_1, (1152, ), (1, ))
        assert_size_stride(arg213_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg214_1, (1152, ), (1, ))
        assert_size_stride(arg215_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg216_1, (1152, ), (1, ))
        assert_size_stride(arg217_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg218_1, (1152, ), (1, ))
        assert_size_stride(arg219_1, (1152, ), (1, ))
        assert_size_stride(arg220_1, (1152, ), (1, ))
        assert_size_stride(arg221_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg222_1, (4304, ), (1, ))
        assert_size_stride(arg223_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg224_1, (1152, ), (1, ))
        assert_size_stride(arg225_1, (1152, ), (1, ))
        assert_size_stride(arg226_1, (1152, ), (1, ))
        assert_size_stride(arg227_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg228_1, (1152, ), (1, ))
        assert_size_stride(arg229_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg230_1, (1152, ), (1, ))
        assert_size_stride(arg231_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg232_1, (1152, ), (1, ))
        assert_size_stride(arg233_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg234_1, (1152, ), (1, ))
        assert_size_stride(arg235_1, (1152, ), (1, ))
        assert_size_stride(arg236_1, (1152, ), (1, ))
        assert_size_stride(arg237_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg238_1, (4304, ), (1, ))
        assert_size_stride(arg239_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg240_1, (1152, ), (1, ))
        assert_size_stride(arg241_1, (1152, ), (1, ))
        assert_size_stride(arg242_1, (1152, ), (1, ))
        assert_size_stride(arg243_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg244_1, (1152, ), (1, ))
        assert_size_stride(arg245_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg246_1, (1152, ), (1, ))
        assert_size_stride(arg247_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg248_1, (1152, ), (1, ))
        assert_size_stride(arg249_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg250_1, (1152, ), (1, ))
        assert_size_stride(arg251_1, (1152, ), (1, ))
        assert_size_stride(arg252_1, (1152, ), (1, ))
        assert_size_stride(arg253_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg254_1, (4304, ), (1, ))
        assert_size_stride(arg255_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg256_1, (1152, ), (1, ))
        assert_size_stride(arg257_1, (1152, ), (1, ))
        assert_size_stride(arg258_1, (1152, ), (1, ))
        assert_size_stride(arg259_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg260_1, (1152, ), (1, ))
        assert_size_stride(arg261_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg262_1, (1152, ), (1, ))
        assert_size_stride(arg263_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg264_1, (1152, ), (1, ))
        assert_size_stride(arg265_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg266_1, (1152, ), (1, ))
        assert_size_stride(arg267_1, (1152, ), (1, ))
        assert_size_stride(arg268_1, (1152, ), (1, ))
        assert_size_stride(arg269_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg270_1, (4304, ), (1, ))
        assert_size_stride(arg271_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg272_1, (1152, ), (1, ))
        assert_size_stride(arg273_1, (1152, ), (1, ))
        assert_size_stride(arg274_1, (1152, ), (1, ))
        assert_size_stride(arg275_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg276_1, (1152, ), (1, ))
        assert_size_stride(arg277_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg278_1, (1152, ), (1, ))
        assert_size_stride(arg279_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg280_1, (1152, ), (1, ))
        assert_size_stride(arg281_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg282_1, (1152, ), (1, ))
        assert_size_stride(arg283_1, (1152, ), (1, ))
        assert_size_stride(arg284_1, (1152, ), (1, ))
        assert_size_stride(arg285_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg286_1, (4304, ), (1, ))
        assert_size_stride(arg287_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg288_1, (1152, ), (1, ))
        assert_size_stride(arg289_1, (1152, ), (1, ))
        assert_size_stride(arg290_1, (1152, ), (1, ))
        assert_size_stride(arg291_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg292_1, (1152, ), (1, ))
        assert_size_stride(arg293_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg294_1, (1152, ), (1, ))
        assert_size_stride(arg295_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg296_1, (1152, ), (1, ))
        assert_size_stride(arg297_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg298_1, (1152, ), (1, ))
        assert_size_stride(arg299_1, (1152, ), (1, ))
        assert_size_stride(arg300_1, (1152, ), (1, ))
        assert_size_stride(arg301_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg302_1, (4304, ), (1, ))
        assert_size_stride(arg303_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg304_1, (1152, ), (1, ))
        assert_size_stride(arg305_1, (1152, ), (1, ))
        assert_size_stride(arg306_1, (1152, ), (1, ))
        assert_size_stride(arg307_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg308_1, (1152, ), (1, ))
        assert_size_stride(arg309_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg310_1, (1152, ), (1, ))
        assert_size_stride(arg311_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg312_1, (1152, ), (1, ))
        assert_size_stride(arg313_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg314_1, (1152, ), (1, ))
        assert_size_stride(arg315_1, (1152, ), (1, ))
        assert_size_stride(arg316_1, (1152, ), (1, ))
        assert_size_stride(arg317_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg318_1, (4304, ), (1, ))
        assert_size_stride(arg319_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg320_1, (1152, ), (1, ))
        assert_size_stride(arg321_1, (1152, ), (1, ))
        assert_size_stride(arg322_1, (1152, ), (1, ))
        assert_size_stride(arg323_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg324_1, (1152, ), (1, ))
        assert_size_stride(arg325_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg326_1, (1152, ), (1, ))
        assert_size_stride(arg327_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg328_1, (1152, ), (1, ))
        assert_size_stride(arg329_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg330_1, (1152, ), (1, ))
        assert_size_stride(arg331_1, (1152, ), (1, ))
        assert_size_stride(arg332_1, (1152, ), (1, ))
        assert_size_stride(arg333_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg334_1, (4304, ), (1, ))
        assert_size_stride(arg335_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg336_1, (1152, ), (1, ))
        assert_size_stride(arg337_1, (1152, ), (1, ))
        assert_size_stride(arg338_1, (1152, ), (1, ))
        assert_size_stride(arg339_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg340_1, (1152, ), (1, ))
        assert_size_stride(arg341_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg342_1, (1152, ), (1, ))
        assert_size_stride(arg343_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg344_1, (1152, ), (1, ))
        assert_size_stride(arg345_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg346_1, (1152, ), (1, ))
        assert_size_stride(arg347_1, (1152, ), (1, ))
        assert_size_stride(arg348_1, (1152, ), (1, ))
        assert_size_stride(arg349_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg350_1, (4304, ), (1, ))
        assert_size_stride(arg351_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg352_1, (1152, ), (1, ))
        assert_size_stride(arg353_1, (1152, ), (1, ))
        assert_size_stride(arg354_1, (1152, ), (1, ))
        assert_size_stride(arg355_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg356_1, (1152, ), (1, ))
        assert_size_stride(arg357_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg358_1, (1152, ), (1, ))
        assert_size_stride(arg359_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg360_1, (1152, ), (1, ))
        assert_size_stride(arg361_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg362_1, (1152, ), (1, ))
        assert_size_stride(arg363_1, (1152, ), (1, ))
        assert_size_stride(arg364_1, (1152, ), (1, ))
        assert_size_stride(arg365_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg366_1, (4304, ), (1, ))
        assert_size_stride(arg367_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg368_1, (1152, ), (1, ))
        assert_size_stride(arg369_1, (1152, ), (1, ))
        assert_size_stride(arg370_1, (1152, ), (1, ))
        assert_size_stride(arg371_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg372_1, (1152, ), (1, ))
        assert_size_stride(arg373_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg374_1, (1152, ), (1, ))
        assert_size_stride(arg375_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg376_1, (1152, ), (1, ))
        assert_size_stride(arg377_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg378_1, (1152, ), (1, ))
        assert_size_stride(arg379_1, (1152, ), (1, ))
        assert_size_stride(arg380_1, (1152, ), (1, ))
        assert_size_stride(arg381_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg382_1, (4304, ), (1, ))
        assert_size_stride(arg383_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg384_1, (1152, ), (1, ))
        assert_size_stride(arg385_1, (1152, ), (1, ))
        assert_size_stride(arg386_1, (1152, ), (1, ))
        assert_size_stride(arg387_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg388_1, (1152, ), (1, ))
        assert_size_stride(arg389_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg390_1, (1152, ), (1, ))
        assert_size_stride(arg391_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg392_1, (1152, ), (1, ))
        assert_size_stride(arg393_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg394_1, (1152, ), (1, ))
        assert_size_stride(arg395_1, (1152, ), (1, ))
        assert_size_stride(arg396_1, (1152, ), (1, ))
        assert_size_stride(arg397_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg398_1, (4304, ), (1, ))
        assert_size_stride(arg399_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg400_1, (1152, ), (1, ))
        assert_size_stride(arg401_1, (1152, ), (1, ))
        assert_size_stride(arg402_1, (1152, ), (1, ))
        assert_size_stride(arg403_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg404_1, (1152, ), (1, ))
        assert_size_stride(arg405_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg406_1, (1152, ), (1, ))
        assert_size_stride(arg407_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg408_1, (1152, ), (1, ))
        assert_size_stride(arg409_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg410_1, (1152, ), (1, ))
        assert_size_stride(arg411_1, (1152, ), (1, ))
        assert_size_stride(arg412_1, (1152, ), (1, ))
        assert_size_stride(arg413_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg414_1, (4304, ), (1, ))
        assert_size_stride(arg415_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg416_1, (1152, ), (1, ))
        assert_size_stride(arg417_1, (1152, ), (1, ))
        assert_size_stride(arg418_1, (1152, ), (1, ))
        assert_size_stride(arg419_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg420_1, (1152, ), (1, ))
        assert_size_stride(arg421_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg422_1, (1152, ), (1, ))
        assert_size_stride(arg423_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg424_1, (1152, ), (1, ))
        assert_size_stride(arg425_1, (1152, 1152), (1152, 1))
        assert_size_stride(arg426_1, (1152, ), (1, ))
        assert_size_stride(arg427_1, (1152, ), (1, ))
        assert_size_stride(arg428_1, (1152, ), (1, ))
        assert_size_stride(arg429_1, (4304, 1152), (1152, 1))
        assert_size_stride(arg430_1, (4304, ), (1, ))
        assert_size_stride(arg431_1, (1152, 4304), (4304, 1))
        assert_size_stride(arg432_1, (1152, ), (1, ))
        assert_size_stride(arg433_1, (1152, ), (1, ))
        assert_size_stride(arg434_1, (1152, ), (1, ))
        assert_size_stride(arg435_1, (1296, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((1, 1296, 1152), (1492992, 1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, buf3, 1296, 1152, stream=stream0)
            del arg1_1
            del arg2_1
            buf4 = empty_strided_cuda((1296, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [queries], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg4_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf3, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg3_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf4)
            del arg3_1
            del arg4_1
            buf5 = empty_strided_cuda((1296, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [keys], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg6_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf3, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg5_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf5)
            del arg5_1
            del arg6_1
            buf6 = empty_strided_cuda((1296, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [values], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg8_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf3, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg7_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf6)
            del arg7_1
            del arg8_1
            buf7 = reinterpret_tensor(buf3, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf4, buf7, 1492992, stream=stream0)
            buf8 = reinterpret_tensor(buf4, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf5, buf8, 1492992, stream=stream0)
            buf9 = reinterpret_tensor(buf5, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf6, buf9, 1492992, stream=stream0)
            buf10 = empty_strided_cuda((1, 1, 1296, 1296), (1679616, 0, 1296, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant0, buf10, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries, queries_1, queries_2, query, keys, keys_1, keys_2, key, values, view_2, values_1, value, owners, getitem_13, getitem_14, eq, attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf7, buf8, buf9, reinterpret_tensor(buf10, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf12 = buf11[0]
            assert_size_stride(buf12, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf12, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf11
            buf16 = reinterpret_tensor(buf9, (1296, 1152), (1152, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape_13, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg10_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf12, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg9_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf16)
            del arg10_1
            del arg9_1
            buf20 = reinterpret_tensor(buf12, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(arg0_1, buf16, arg11_1, arg12_1, buf20, 1296, 1152, stream=stream0)
            del arg11_1
            del arg12_1
            buf21 = empty_strided_cuda((1296, 4304), (4304, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg14_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf20, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg13_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf21)
            del arg13_1
            del arg14_1
            buf22 = empty_strided_cuda((1, 1296, 4304), (5640192, 4352, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf21, buf22, 5577984, stream=stream0)
            buf23 = reinterpret_tensor(buf20, (1296, 1152), (1152, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg16_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf22, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg15_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf23)
            del arg15_1
            del arg16_1
            buf28 = reinterpret_tensor(buf8, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(arg0_1, buf16, buf23, arg17_1, arg18_1, buf28, 1296, 1152, stream=stream0)
            del arg17_1
            del arg18_1
            buf29 = reinterpret_tensor(buf7, (1296, 1152), (1152, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_7, queries_3], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg20_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf28, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg19_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf29)
            del arg19_1
            del arg20_1
            buf30 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [keys_3], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg22_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf28, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg21_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf30)
            del arg21_1
            del arg22_1
            buf31 = empty_strided_cuda((1296, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [values_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg24_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf28, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg23_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf31)
            del arg23_1
            del arg24_1
            buf32 = reinterpret_tensor(buf28, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [queries_3, queries_4, queries_5, query_1, keys_3, keys_4, keys_5, key_1, values_2, view_5, values_3, value_1, owners_1, getitem_17, getitem_18, eq_1, attention_mask_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf29, buf32, 1492992, stream=stream0)
            buf33 = reinterpret_tensor(buf29, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [queries_3, queries_4, queries_5, query_1, keys_3, keys_4, keys_5, key_1, values_2, view_5, values_3, value_1, owners_1, getitem_17, getitem_18, eq_1, attention_mask_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf30, buf33, 1492992, stream=stream0)
            buf34 = reinterpret_tensor(buf30, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [queries_3, queries_4, queries_5, query_1, keys_3, keys_4, keys_5, key_1, values_2, view_5, values_3, value_1, owners_1, getitem_17, getitem_18, eq_1, attention_mask_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf31, buf34, 1492992, stream=stream0)
            del buf31
            buf35 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [queries_3, queries_4, queries_5, query_1, keys_3, keys_4, keys_5, key_1, values_2, view_5, values_3, value_1, owners_1, getitem_17, getitem_18, eq_1, attention_mask_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant1, buf35, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_3, queries_4, queries_5, query_1, keys_3, keys_4, keys_5, key_1, values_2, view_5, values_3, value_1, owners_1, getitem_17, getitem_18, eq_1, attention_mask_1, attn_output_4], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf36 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf32, buf33, buf34, reinterpret_tensor(buf35, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf37 = buf36[0]
            assert_size_stride(buf37, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf37, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf36
            buf41 = reinterpret_tensor(buf34, (1296, 1152), (1152, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, reshape_14, attn_output_7], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg26_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf37, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg25_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf41)
            del arg25_1
            del arg26_1
            buf46 = reinterpret_tensor(buf37, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(arg0_1, buf16, buf23, buf41, arg27_1, arg28_1, buf46, 1296, 1152, stream=stream0)
            del arg27_1
            del arg28_1
            buf47 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_9, hidden_states_10], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg30_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf46, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg29_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf47)
            del arg29_1
            del arg30_1
            buf48 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf47, buf48, 5577984, stream=stream0)
            buf49 = reinterpret_tensor(buf46, (1296, 1152), (1152, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, hidden_states_12], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg32_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf48, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg31_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf49)
            del arg31_1
            del arg32_1
            buf50 = reinterpret_tensor(buf16, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf16  # reuse
            buf54 = reinterpret_tensor(buf33, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_7.run(buf50, arg0_1, buf23, buf41, buf49, arg33_1, arg34_1, buf54, 1296, 1152, stream=stream0)
            del arg0_1
            del arg33_1
            del arg34_1
            buf55 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [queries_6], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg36_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf54, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg35_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf55)
            del arg35_1
            del arg36_1
            buf56 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [keys_6], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg38_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf54, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg37_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf56)
            del arg37_1
            del arg38_1
            buf57 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [values_4], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg40_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf54, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg39_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf57)
            del arg39_1
            del arg40_1
            buf58 = reinterpret_tensor(buf54, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [queries_6, queries_7, queries_8, query_2, keys_6, keys_7, keys_8, key_2, values_4, view_8, values_5, value_2, owners_2, getitem_21, getitem_22, eq_2, attention_mask_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf55, buf58, 1492992, stream=stream0)
            buf59 = reinterpret_tensor(buf55, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [queries_6, queries_7, queries_8, query_2, keys_6, keys_7, keys_8, key_2, values_4, view_8, values_5, value_2, owners_2, getitem_21, getitem_22, eq_2, attention_mask_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf56, buf59, 1492992, stream=stream0)
            buf60 = reinterpret_tensor(buf56, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [queries_6, queries_7, queries_8, query_2, keys_6, keys_7, keys_8, key_2, values_4, view_8, values_5, value_2, owners_2, getitem_21, getitem_22, eq_2, attention_mask_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf57, buf60, 1492992, stream=stream0)
            buf61 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [queries_6, queries_7, queries_8, query_2, keys_6, keys_7, keys_8, key_2, values_4, view_8, values_5, value_2, owners_2, getitem_21, getitem_22, eq_2, attention_mask_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant2, buf61, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_6, queries_7, queries_8, query_2, keys_6, keys_7, keys_8, key_2, values_4, view_8, values_5, value_2, owners_2, getitem_21, getitem_22, eq_2, attention_mask_2, attn_output_8], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf62 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf58, buf59, buf60, reinterpret_tensor(buf61, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf63 = buf62[0]
            assert_size_stride(buf63, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf63, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf62
            buf67 = reinterpret_tensor(buf60, (1296, 1152), (1152, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, reshape_15, attn_output_11], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg42_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf63, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg41_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf67)
            del arg41_1
            del arg42_1
            buf71 = reinterpret_tensor(buf63, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf50, buf67, arg43_1, arg44_1, buf71, 1296, 1152, stream=stream0)
            del arg43_1
            del arg44_1
            buf72 = buf47; del buf47  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg46_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf71, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg45_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf72)
            del arg45_1
            del arg46_1
            buf73 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf72, buf73, 5577984, stream=stream0)
            buf74 = reinterpret_tensor(buf71, (1296, 1152), (1152, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18, hidden_states_19], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg48_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf73, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg47_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf74)
            del arg47_1
            del arg48_1
            buf79 = reinterpret_tensor(buf59, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf50, buf67, buf74, arg49_1, arg50_1, buf79, 1296, 1152, stream=stream0)
            del arg49_1
            del arg50_1
            buf80 = reinterpret_tensor(buf58, (1296, 1152), (1152, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, queries_9], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg52_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf79, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg51_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf80)
            del arg51_1
            del arg52_1
            buf81 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [keys_9], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg54_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf79, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg53_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf81)
            del arg53_1
            del arg54_1
            buf82 = reinterpret_tensor(buf32, (1296, 1152), (1152, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [values_6], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg56_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf79, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg55_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf82)
            del arg55_1
            del arg56_1
            buf83 = reinterpret_tensor(buf79, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [queries_9, queries_10, queries_11, query_3, keys_9, keys_10, keys_11, key_3, values_6, view_11, values_7, value_3, owners_3, getitem_25, getitem_26, eq_3, attention_mask_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf80, buf83, 1492992, stream=stream0)
            buf84 = reinterpret_tensor(buf80, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [queries_9, queries_10, queries_11, query_3, keys_9, keys_10, keys_11, key_3, values_6, view_11, values_7, value_3, owners_3, getitem_25, getitem_26, eq_3, attention_mask_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf81, buf84, 1492992, stream=stream0)
            buf85 = reinterpret_tensor(buf81, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [queries_9, queries_10, queries_11, query_3, keys_9, keys_10, keys_11, key_3, values_6, view_11, values_7, value_3, owners_3, getitem_25, getitem_26, eq_3, attention_mask_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf82, buf85, 1492992, stream=stream0)
            del buf82
            buf86 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [queries_9, queries_10, queries_11, query_3, keys_9, keys_10, keys_11, key_3, values_6, view_11, values_7, value_3, owners_3, getitem_25, getitem_26, eq_3, attention_mask_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant3, buf86, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_9, queries_10, queries_11, query_3, keys_9, keys_10, keys_11, key_3, values_6, view_11, values_7, value_3, owners_3, getitem_25, getitem_26, eq_3, attention_mask_3, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf87 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf83, buf84, buf85, reinterpret_tensor(buf86, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf83
            buf88 = buf87[0]
            assert_size_stride(buf88, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf88, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf87
            buf92 = reinterpret_tensor(buf85, (1296, 1152), (1152, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, reshape_16, attn_output_15], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg58_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf88, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg57_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf92)
            del arg57_1
            del arg58_1
            buf97 = reinterpret_tensor(buf88, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_23], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf50, buf67, buf74, buf92, arg59_1, arg60_1, buf97, 1296, 1152, stream=stream0)
            del arg59_1
            del arg60_1
            buf98 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_23, hidden_states_24], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg62_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf97, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg61_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf98)
            del arg61_1
            del arg62_1
            buf99 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf98, buf99, 5577984, stream=stream0)
            buf100 = reinterpret_tensor(buf97, (1296, 1152), (1152, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_24, hidden_states_25, hidden_states_26], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg64_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf99, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg63_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf100)
            del arg63_1
            del arg64_1
            buf101 = buf50; del buf50  # reuse
            buf105 = reinterpret_tensor(buf84, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf101, buf67, buf74, buf92, buf100, arg65_1, arg66_1, buf105, 1296, 1152, stream=stream0)
            del arg65_1
            del arg66_1
            buf106 = buf92; del buf92  # reuse
            # Topologically Sorted Source Nodes: [queries_12], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg68_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf105, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg67_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf106)
            del arg67_1
            del arg68_1
            buf107 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [keys_12], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg70_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf105, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg69_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf107)
            del arg69_1
            del arg70_1
            buf108 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [values_8], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg72_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf105, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg71_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf108)
            del arg71_1
            del arg72_1
            buf109 = reinterpret_tensor(buf105, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [queries_12, queries_13, queries_14, query_4, keys_12, keys_13, keys_14, key_4, values_8, view_14, values_9, value_4, owners_4, getitem_29, getitem_30, eq_4, attention_mask_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf106, buf109, 1492992, stream=stream0)
            buf110 = reinterpret_tensor(buf106, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [queries_12, queries_13, queries_14, query_4, keys_12, keys_13, keys_14, key_4, values_8, view_14, values_9, value_4, owners_4, getitem_29, getitem_30, eq_4, attention_mask_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf107, buf110, 1492992, stream=stream0)
            buf111 = reinterpret_tensor(buf107, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [queries_12, queries_13, queries_14, query_4, keys_12, keys_13, keys_14, key_4, values_8, view_14, values_9, value_4, owners_4, getitem_29, getitem_30, eq_4, attention_mask_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf108, buf111, 1492992, stream=stream0)
            buf112 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [queries_12, queries_13, queries_14, query_4, keys_12, keys_13, keys_14, key_4, values_8, view_14, values_9, value_4, owners_4, getitem_29, getitem_30, eq_4, attention_mask_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant4, buf112, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_12, queries_13, queries_14, query_4, keys_12, keys_13, keys_14, key_4, values_8, view_14, values_9, value_4, owners_4, getitem_29, getitem_30, eq_4, attention_mask_4, attn_output_16], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf113 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf109, buf110, buf111, reinterpret_tensor(buf112, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf114 = buf113[0]
            assert_size_stride(buf114, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf114, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf113
            buf118 = reinterpret_tensor(buf111, (1296, 1152), (1152, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, reshape_17, attn_output_19], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg74_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf114, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg73_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf118)
            del arg73_1
            del arg74_1
            buf122 = reinterpret_tensor(buf114, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_30], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf101, buf118, arg75_1, arg76_1, buf122, 1296, 1152, stream=stream0)
            del arg75_1
            del arg76_1
            buf123 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg78_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf122, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg77_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf123)
            del arg77_1
            del arg78_1
            buf124 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf123, buf124, 5577984, stream=stream0)
            buf125 = reinterpret_tensor(buf122, (1296, 1152), (1152, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32, hidden_states_33], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg80_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf124, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg79_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf125)
            del arg79_1
            del arg80_1
            buf130 = reinterpret_tensor(buf110, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf101, buf118, buf125, arg81_1, arg82_1, buf130, 1296, 1152, stream=stream0)
            del arg81_1
            del arg82_1
            buf131 = reinterpret_tensor(buf109, (1296, 1152), (1152, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, queries_15], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg84_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf130, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg83_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf131)
            del arg83_1
            del arg84_1
            buf132 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [keys_15], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg86_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf130, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg85_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf132)
            del arg85_1
            del arg86_1
            buf133 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [values_10], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg88_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf130, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg87_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf133)
            del arg87_1
            del arg88_1
            buf134 = reinterpret_tensor(buf130, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [queries_15, queries_16, queries_17, query_5, keys_15, keys_16, keys_17, key_5, values_10, view_17, values_11, value_5, owners_5, getitem_33, getitem_34, eq_5, attention_mask_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf131, buf134, 1492992, stream=stream0)
            buf135 = reinterpret_tensor(buf131, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [queries_15, queries_16, queries_17, query_5, keys_15, keys_16, keys_17, key_5, values_10, view_17, values_11, value_5, owners_5, getitem_33, getitem_34, eq_5, attention_mask_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf132, buf135, 1492992, stream=stream0)
            buf136 = reinterpret_tensor(buf132, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf132  # reuse
            # Topologically Sorted Source Nodes: [queries_15, queries_16, queries_17, query_5, keys_15, keys_16, keys_17, key_5, values_10, view_17, values_11, value_5, owners_5, getitem_33, getitem_34, eq_5, attention_mask_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf133, buf136, 1492992, stream=stream0)
            del buf133
            buf137 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [queries_15, queries_16, queries_17, query_5, keys_15, keys_16, keys_17, key_5, values_10, view_17, values_11, value_5, owners_5, getitem_33, getitem_34, eq_5, attention_mask_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant5, buf137, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_15, queries_16, queries_17, query_5, keys_15, keys_16, keys_17, key_5, values_10, view_17, values_11, value_5, owners_5, getitem_33, getitem_34, eq_5, attention_mask_5, attn_output_20], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf138 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf134, buf135, buf136, reinterpret_tensor(buf137, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf134
            buf139 = buf138[0]
            assert_size_stride(buf139, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf139, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf138
            buf143 = reinterpret_tensor(buf136, (1296, 1152), (1152, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, reshape_18, attn_output_23], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg90_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf139, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg89_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf143)
            del arg89_1
            del arg90_1
            buf148 = reinterpret_tensor(buf139, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_37], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf101, buf118, buf125, buf143, arg91_1, arg92_1, buf148, 1296, 1152, stream=stream0)
            del arg91_1
            del arg92_1
            buf149 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg94_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf148, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg93_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf149)
            del arg93_1
            del arg94_1
            buf150 = buf124; del buf124  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf149, buf150, 5577984, stream=stream0)
            buf151 = reinterpret_tensor(buf148, (1296, 1152), (1152, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_38, hidden_states_39, hidden_states_40], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg96_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf150, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg95_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf151)
            del arg95_1
            del arg96_1
            buf152 = buf101; del buf101  # reuse
            buf156 = reinterpret_tensor(buf135, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, attn_output_23, hidden_states_36, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf152, buf118, buf125, buf143, buf151, arg97_1, arg98_1, buf156, 1296, 1152, stream=stream0)
            del arg97_1
            del arg98_1
            buf157 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [queries_18], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg100_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf156, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg99_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf157)
            del arg100_1
            del arg99_1
            buf158 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [keys_18], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg102_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf156, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg101_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf158)
            del arg101_1
            del arg102_1
            buf159 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [values_12], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg104_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf156, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg103_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf159)
            del arg103_1
            del arg104_1
            buf160 = reinterpret_tensor(buf156, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [queries_18, queries_19, queries_20, query_6, keys_18, keys_19, keys_20, key_6, values_12, view_20, values_13, value_6, owners_6, getitem_37, getitem_38, eq_6, attention_mask_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf157, buf160, 1492992, stream=stream0)
            buf161 = reinterpret_tensor(buf157, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [queries_18, queries_19, queries_20, query_6, keys_18, keys_19, keys_20, key_6, values_12, view_20, values_13, value_6, owners_6, getitem_37, getitem_38, eq_6, attention_mask_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf158, buf161, 1492992, stream=stream0)
            buf162 = reinterpret_tensor(buf158, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [queries_18, queries_19, queries_20, query_6, keys_18, keys_19, keys_20, key_6, values_12, view_20, values_13, value_6, owners_6, getitem_37, getitem_38, eq_6, attention_mask_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf159, buf162, 1492992, stream=stream0)
            buf163 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [queries_18, queries_19, queries_20, query_6, keys_18, keys_19, keys_20, key_6, values_12, view_20, values_13, value_6, owners_6, getitem_37, getitem_38, eq_6, attention_mask_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant6, buf163, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_18, queries_19, queries_20, query_6, keys_18, keys_19, keys_20, key_6, values_12, view_20, values_13, value_6, owners_6, getitem_37, getitem_38, eq_6, attention_mask_6, attn_output_24], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf164 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf160, buf161, buf162, reinterpret_tensor(buf163, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf165 = buf164[0]
            assert_size_stride(buf165, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf165, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf164
            buf169 = reinterpret_tensor(buf162, (1296, 1152), (1152, 1), 0); del buf162  # reuse
            # Topologically Sorted Source Nodes: [transpose_27, reshape_19, attn_output_27], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg106_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf165, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg105_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf169)
            del arg105_1
            del arg106_1
            buf173 = reinterpret_tensor(buf165, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_44], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf152, buf169, arg107_1, arg108_1, buf173, 1296, 1152, stream=stream0)
            del arg107_1
            del arg108_1
            buf174 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_45], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg110_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf173, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg109_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf174)
            del arg109_1
            del arg110_1
            buf175 = buf150; del buf150  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf174, buf175, 5577984, stream=stream0)
            buf176 = reinterpret_tensor(buf173, (1296, 1152), (1152, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg112_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf175, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg111_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf176)
            del arg111_1
            del arg112_1
            buf181 = reinterpret_tensor(buf161, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, hidden_states_49], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf152, buf169, buf176, arg113_1, arg114_1, buf181, 1296, 1152, stream=stream0)
            del arg113_1
            del arg114_1
            buf182 = reinterpret_tensor(buf160, (1296, 1152), (1152, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_49, queries_21], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg116_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf181, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg115_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf182)
            del arg115_1
            del arg116_1
            buf183 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [keys_21], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg118_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf181, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg117_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf183)
            del arg117_1
            del arg118_1
            buf184 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [values_14], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg120_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf181, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg119_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf184)
            del arg119_1
            del arg120_1
            buf185 = reinterpret_tensor(buf181, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [queries_21, queries_22, queries_23, query_7, keys_21, keys_22, keys_23, key_7, values_14, view_23, values_15, value_7, owners_7, getitem_41, getitem_42, eq_7, attention_mask_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf182, buf185, 1492992, stream=stream0)
            buf186 = reinterpret_tensor(buf182, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [queries_21, queries_22, queries_23, query_7, keys_21, keys_22, keys_23, key_7, values_14, view_23, values_15, value_7, owners_7, getitem_41, getitem_42, eq_7, attention_mask_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf183, buf186, 1492992, stream=stream0)
            buf187 = reinterpret_tensor(buf183, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [queries_21, queries_22, queries_23, query_7, keys_21, keys_22, keys_23, key_7, values_14, view_23, values_15, value_7, owners_7, getitem_41, getitem_42, eq_7, attention_mask_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf184, buf187, 1492992, stream=stream0)
            del buf184
            buf188 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [queries_21, queries_22, queries_23, query_7, keys_21, keys_22, keys_23, key_7, values_14, view_23, values_15, value_7, owners_7, getitem_41, getitem_42, eq_7, attention_mask_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant7, buf188, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_21, queries_22, queries_23, query_7, keys_21, keys_22, keys_23, key_7, values_14, view_23, values_15, value_7, owners_7, getitem_41, getitem_42, eq_7, attention_mask_7, attn_output_28], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf189 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf185, buf186, buf187, reinterpret_tensor(buf188, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf185
            buf190 = buf189[0]
            assert_size_stride(buf190, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf190, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf189
            buf194 = reinterpret_tensor(buf187, (1296, 1152), (1152, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, reshape_20, attn_output_31], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg122_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf190, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg121_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf194)
            del arg121_1
            del arg122_1
            buf199 = reinterpret_tensor(buf190, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_51], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf152, buf169, buf176, buf194, arg123_1, arg124_1, buf199, 1296, 1152, stream=stream0)
            del arg123_1
            del arg124_1
            buf200 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg126_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf199, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg125_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf200)
            del arg125_1
            del arg126_1
            buf201 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_53], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf200, buf201, 5577984, stream=stream0)
            buf202 = reinterpret_tensor(buf199, (1296, 1152), (1152, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_53, hidden_states_54], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg128_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf201, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg127_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf202)
            del arg127_1
            del arg128_1
            buf203 = buf152; del buf152  # reuse
            buf207 = reinterpret_tensor(buf186, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, attn_output_31, hidden_states_50, hidden_states_54, hidden_states_55, hidden_states_56], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf203, buf169, buf176, buf194, buf202, arg129_1, arg130_1, buf207, 1296, 1152, stream=stream0)
            del arg129_1
            del arg130_1
            buf208 = buf202; del buf202  # reuse
            # Topologically Sorted Source Nodes: [queries_24], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg132_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf207, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg131_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf208)
            del arg131_1
            del arg132_1
            buf209 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [keys_24], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg134_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf207, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg133_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf209)
            del arg133_1
            del arg134_1
            buf210 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [values_16], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg136_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf207, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg135_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf210)
            del arg135_1
            del arg136_1
            buf211 = reinterpret_tensor(buf207, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [queries_24, queries_25, queries_26, query_8, keys_24, keys_25, keys_26, key_8, values_16, view_26, values_17, value_8, owners_8, getitem_45, getitem_46, eq_8, attention_mask_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf208, buf211, 1492992, stream=stream0)
            buf212 = reinterpret_tensor(buf208, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [queries_24, queries_25, queries_26, query_8, keys_24, keys_25, keys_26, key_8, values_16, view_26, values_17, value_8, owners_8, getitem_45, getitem_46, eq_8, attention_mask_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf209, buf212, 1492992, stream=stream0)
            buf213 = reinterpret_tensor(buf209, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [queries_24, queries_25, queries_26, query_8, keys_24, keys_25, keys_26, key_8, values_16, view_26, values_17, value_8, owners_8, getitem_45, getitem_46, eq_8, attention_mask_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf210, buf213, 1492992, stream=stream0)
            buf214 = buf188; del buf188  # reuse
            # Topologically Sorted Source Nodes: [queries_24, queries_25, queries_26, query_8, keys_24, keys_25, keys_26, key_8, values_16, view_26, values_17, value_8, owners_8, getitem_45, getitem_46, eq_8, attention_mask_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant8, buf214, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_24, queries_25, queries_26, query_8, keys_24, keys_25, keys_26, key_8, values_16, view_26, values_17, value_8, owners_8, getitem_45, getitem_46, eq_8, attention_mask_8, attn_output_32], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf215 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf211, buf212, buf213, reinterpret_tensor(buf214, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf216 = buf215[0]
            assert_size_stride(buf216, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf216, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf215
            buf220 = reinterpret_tensor(buf213, (1296, 1152), (1152, 1), 0); del buf213  # reuse
            # Topologically Sorted Source Nodes: [transpose_35, reshape_21, attn_output_35], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg138_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf216, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg137_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf220)
            del arg137_1
            del arg138_1
            buf224 = reinterpret_tensor(buf216, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_58], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf203, buf220, arg139_1, arg140_1, buf224, 1296, 1152, stream=stream0)
            del arg139_1
            del arg140_1
            buf225 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg142_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf224, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg141_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf225)
            del arg141_1
            del arg142_1
            buf226 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf225, buf226, 5577984, stream=stream0)
            buf227 = reinterpret_tensor(buf224, (1296, 1152), (1152, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg144_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf226, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg143_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf227)
            del arg143_1
            del arg144_1
            buf232 = reinterpret_tensor(buf212, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, hidden_states_63], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf203, buf220, buf227, arg145_1, arg146_1, buf232, 1296, 1152, stream=stream0)
            del arg145_1
            del arg146_1
            buf233 = reinterpret_tensor(buf211, (1296, 1152), (1152, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_63, queries_27], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg148_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf232, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg147_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf233)
            del arg147_1
            del arg148_1
            buf234 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [keys_27], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg150_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf232, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg149_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf234)
            del arg149_1
            del arg150_1
            buf235 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [values_18], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg152_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf232, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg151_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf235)
            del arg151_1
            del arg152_1
            buf236 = reinterpret_tensor(buf232, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [queries_27, queries_28, queries_29, query_9, keys_27, keys_28, keys_29, key_9, values_18, view_29, values_19, value_9, owners_9, getitem_49, getitem_50, eq_9, attention_mask_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf233, buf236, 1492992, stream=stream0)
            buf237 = reinterpret_tensor(buf233, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [queries_27, queries_28, queries_29, query_9, keys_27, keys_28, keys_29, key_9, values_18, view_29, values_19, value_9, owners_9, getitem_49, getitem_50, eq_9, attention_mask_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf234, buf237, 1492992, stream=stream0)
            buf238 = reinterpret_tensor(buf234, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [queries_27, queries_28, queries_29, query_9, keys_27, keys_28, keys_29, key_9, values_18, view_29, values_19, value_9, owners_9, getitem_49, getitem_50, eq_9, attention_mask_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf235, buf238, 1492992, stream=stream0)
            del buf235
            buf239 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [queries_27, queries_28, queries_29, query_9, keys_27, keys_28, keys_29, key_9, values_18, view_29, values_19, value_9, owners_9, getitem_49, getitem_50, eq_9, attention_mask_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant9, buf239, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_27, queries_28, queries_29, query_9, keys_27, keys_28, keys_29, key_9, values_18, view_29, values_19, value_9, owners_9, getitem_49, getitem_50, eq_9, attention_mask_9, attn_output_36], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf240 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf236, buf237, buf238, reinterpret_tensor(buf239, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf236
            buf241 = buf240[0]
            assert_size_stride(buf241, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf241, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf240
            buf245 = reinterpret_tensor(buf238, (1296, 1152), (1152, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [transpose_39, reshape_22, attn_output_39], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg154_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf241, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg153_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf245)
            del arg153_1
            del arg154_1
            buf250 = reinterpret_tensor(buf241, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_65], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf203, buf220, buf227, buf245, arg155_1, arg156_1, buf250, 1296, 1152, stream=stream0)
            del arg155_1
            del arg156_1
            buf251 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_66], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg158_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf250, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg157_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf251)
            del arg157_1
            del arg158_1
            buf252 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf251, buf252, 5577984, stream=stream0)
            buf253 = reinterpret_tensor(buf250, (1296, 1152), (1152, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, hidden_states_68], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg160_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf252, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg159_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf253)
            del arg159_1
            del arg160_1
            buf254 = buf203; del buf203  # reuse
            buf258 = reinterpret_tensor(buf237, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, attn_output_39, hidden_states_64, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf254, buf220, buf227, buf245, buf253, arg161_1, arg162_1, buf258, 1296, 1152, stream=stream0)
            del arg161_1
            del arg162_1
            buf259 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [queries_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg164_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf258, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg163_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf259)
            del arg163_1
            del arg164_1
            buf260 = buf245; del buf245  # reuse
            # Topologically Sorted Source Nodes: [keys_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg166_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf258, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg165_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf260)
            del arg165_1
            del arg166_1
            buf261 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [values_20], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg168_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf258, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg167_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf261)
            del arg167_1
            del arg168_1
            buf262 = reinterpret_tensor(buf258, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [queries_30, queries_31, queries_32, query_10, keys_30, keys_31, keys_32, key_10, values_20, view_32, values_21, value_10, owners_10, getitem_53, getitem_54, eq_10, attention_mask_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf259, buf262, 1492992, stream=stream0)
            buf263 = reinterpret_tensor(buf259, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [queries_30, queries_31, queries_32, query_10, keys_30, keys_31, keys_32, key_10, values_20, view_32, values_21, value_10, owners_10, getitem_53, getitem_54, eq_10, attention_mask_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf260, buf263, 1492992, stream=stream0)
            buf264 = reinterpret_tensor(buf260, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [queries_30, queries_31, queries_32, query_10, keys_30, keys_31, keys_32, key_10, values_20, view_32, values_21, value_10, owners_10, getitem_53, getitem_54, eq_10, attention_mask_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf261, buf264, 1492992, stream=stream0)
            buf265 = buf239; del buf239  # reuse
            # Topologically Sorted Source Nodes: [queries_30, queries_31, queries_32, query_10, keys_30, keys_31, keys_32, key_10, values_20, view_32, values_21, value_10, owners_10, getitem_53, getitem_54, eq_10, attention_mask_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant10, buf265, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_30, queries_31, queries_32, query_10, keys_30, keys_31, keys_32, key_10, values_20, view_32, values_21, value_10, owners_10, getitem_53, getitem_54, eq_10, attention_mask_10, attn_output_40], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf266 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf262, buf263, buf264, reinterpret_tensor(buf265, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf267 = buf266[0]
            assert_size_stride(buf267, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf267, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf266
            buf271 = reinterpret_tensor(buf264, (1296, 1152), (1152, 1), 0); del buf264  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, reshape_23, attn_output_43], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg170_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf267, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg169_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf271)
            del arg169_1
            del arg170_1
            buf275 = reinterpret_tensor(buf267, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_72], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf254, buf271, arg171_1, arg172_1, buf275, 1296, 1152, stream=stream0)
            del arg171_1
            del arg172_1
            buf276 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg174_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf275, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg173_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf276)
            del arg173_1
            del arg174_1
            buf277 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf276, buf277, 5577984, stream=stream0)
            buf278 = reinterpret_tensor(buf275, (1296, 1152), (1152, 1), 0); del buf275  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74, hidden_states_75], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg176_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf277, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg175_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf278)
            del arg175_1
            del arg176_1
            buf283 = reinterpret_tensor(buf263, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf254, buf271, buf278, arg177_1, arg178_1, buf283, 1296, 1152, stream=stream0)
            del arg177_1
            del arg178_1
            buf284 = reinterpret_tensor(buf262, (1296, 1152), (1152, 1), 0); del buf262  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_77, queries_33], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg180_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf283, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg179_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf284)
            del arg179_1
            del arg180_1
            buf285 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [keys_33], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg182_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf283, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg181_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf285)
            del arg181_1
            del arg182_1
            buf286 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [values_22], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg184_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf283, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg183_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf286)
            del arg183_1
            del arg184_1
            buf287 = reinterpret_tensor(buf283, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [queries_33, queries_34, queries_35, query_11, keys_33, keys_34, keys_35, key_11, values_22, view_35, values_23, value_11, owners_11, getitem_57, getitem_58, eq_11, attention_mask_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf284, buf287, 1492992, stream=stream0)
            buf288 = reinterpret_tensor(buf284, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf284  # reuse
            # Topologically Sorted Source Nodes: [queries_33, queries_34, queries_35, query_11, keys_33, keys_34, keys_35, key_11, values_22, view_35, values_23, value_11, owners_11, getitem_57, getitem_58, eq_11, attention_mask_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf285, buf288, 1492992, stream=stream0)
            buf289 = reinterpret_tensor(buf285, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [queries_33, queries_34, queries_35, query_11, keys_33, keys_34, keys_35, key_11, values_22, view_35, values_23, value_11, owners_11, getitem_57, getitem_58, eq_11, attention_mask_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf286, buf289, 1492992, stream=stream0)
            del buf286
            buf290 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [queries_33, queries_34, queries_35, query_11, keys_33, keys_34, keys_35, key_11, values_22, view_35, values_23, value_11, owners_11, getitem_57, getitem_58, eq_11, attention_mask_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant11, buf290, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_33, queries_34, queries_35, query_11, keys_33, keys_34, keys_35, key_11, values_22, view_35, values_23, value_11, owners_11, getitem_57, getitem_58, eq_11, attention_mask_11, attn_output_44], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf291 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf287, buf288, buf289, reinterpret_tensor(buf290, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf287
            buf292 = buf291[0]
            assert_size_stride(buf292, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf292, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf291
            buf296 = reinterpret_tensor(buf289, (1296, 1152), (1152, 1), 0); del buf289  # reuse
            # Topologically Sorted Source Nodes: [transpose_47, reshape_24, attn_output_47], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg186_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf292, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg185_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf296)
            del arg185_1
            del arg186_1
            buf301 = reinterpret_tensor(buf292, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_79], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf254, buf271, buf278, buf296, arg187_1, arg188_1, buf301, 1296, 1152, stream=stream0)
            del arg187_1
            del arg188_1
            buf302 = buf276; del buf276  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_79, hidden_states_80], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg190_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf301, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg189_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf302)
            del arg189_1
            del arg190_1
            buf303 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf302, buf303, 5577984, stream=stream0)
            buf304 = reinterpret_tensor(buf301, (1296, 1152), (1152, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, hidden_states_82], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg192_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf303, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg191_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf304)
            del arg191_1
            del arg192_1
            buf305 = buf254; del buf254  # reuse
            buf309 = reinterpret_tensor(buf288, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, hidden_states_84], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf305, buf271, buf278, buf296, buf304, arg193_1, arg194_1, buf309, 1296, 1152, stream=stream0)
            del arg193_1
            del arg194_1
            buf310 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [queries_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg196_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf309, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg195_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf310)
            del arg195_1
            del arg196_1
            buf311 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [keys_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg198_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf309, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg197_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf311)
            del arg197_1
            del arg198_1
            buf312 = buf278; del buf278  # reuse
            # Topologically Sorted Source Nodes: [values_24], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg200_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf309, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg199_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf312)
            del arg199_1
            del arg200_1
            buf313 = reinterpret_tensor(buf309, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf309  # reuse
            # Topologically Sorted Source Nodes: [queries_36, queries_37, queries_38, query_12, keys_36, keys_37, keys_38, key_12, values_24, view_38, values_25, value_12, owners_12, getitem_61, getitem_62, eq_12, attention_mask_12, attn_output_48], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf310, buf313, 1492992, stream=stream0)
            buf314 = reinterpret_tensor(buf310, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [queries_36, queries_37, queries_38, query_12, keys_36, keys_37, keys_38, key_12, values_24, view_38, values_25, value_12, owners_12, getitem_61, getitem_62, eq_12, attention_mask_12, attn_output_48], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf311, buf314, 1492992, stream=stream0)
            buf315 = reinterpret_tensor(buf311, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf311  # reuse
            # Topologically Sorted Source Nodes: [queries_36, queries_37, queries_38, query_12, keys_36, keys_37, keys_38, key_12, values_24, view_38, values_25, value_12, owners_12, getitem_61, getitem_62, eq_12, attention_mask_12, attn_output_48], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf312, buf315, 1492992, stream=stream0)
            buf316 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [queries_36, queries_37, queries_38, query_12, keys_36, keys_37, keys_38, key_12, values_24, view_38, values_25, value_12, owners_12, getitem_61, getitem_62, eq_12, attention_mask_12, attn_output_48], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant12, buf316, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_36, queries_37, queries_38, query_12, keys_36, keys_37, keys_38, key_12, values_24, view_38, values_25, value_12, owners_12, getitem_61, getitem_62, eq_12, attention_mask_12, attn_output_48], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf317 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf313, buf314, buf315, reinterpret_tensor(buf316, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf318 = buf317[0]
            assert_size_stride(buf318, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf318, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf317
            buf322 = reinterpret_tensor(buf315, (1296, 1152), (1152, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [transpose_51, reshape_25, attn_output_51], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg202_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf318, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg201_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf322)
            del arg201_1
            del arg202_1
            buf326 = reinterpret_tensor(buf318, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [attn_output_51, hidden_states_85, hidden_states_86], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf305, buf322, arg203_1, arg204_1, buf326, 1296, 1152, stream=stream0)
            del arg203_1
            del arg204_1
            buf327 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg206_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf326, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg205_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf327)
            del arg205_1
            del arg206_1
            buf328 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_87, hidden_states_88], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf327, buf328, 5577984, stream=stream0)
            buf329 = reinterpret_tensor(buf326, (1296, 1152), (1152, 1), 0); del buf326  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_87, hidden_states_88, hidden_states_89], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg208_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf328, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg207_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf329)
            del arg207_1
            del arg208_1
            buf334 = reinterpret_tensor(buf314, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [attn_output_51, hidden_states_85, hidden_states_89, hidden_states_90, hidden_states_91], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf305, buf322, buf329, arg209_1, arg210_1, buf334, 1296, 1152, stream=stream0)
            del arg209_1
            del arg210_1
            buf335 = reinterpret_tensor(buf313, (1296, 1152), (1152, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_91, queries_39], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg212_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf334, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg211_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf335)
            del arg211_1
            del arg212_1
            buf336 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [keys_39], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg214_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf334, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg213_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf336)
            del arg213_1
            del arg214_1
            buf337 = buf271; del buf271  # reuse
            # Topologically Sorted Source Nodes: [values_26], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg216_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf334, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg215_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf337)
            del arg215_1
            del arg216_1
            buf338 = reinterpret_tensor(buf334, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf334  # reuse
            # Topologically Sorted Source Nodes: [queries_39, queries_40, queries_41, query_13, keys_39, keys_40, keys_41, key_13, values_26, view_41, values_27, value_13, owners_13, getitem_65, getitem_66, eq_13, attention_mask_13, attn_output_52], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf335, buf338, 1492992, stream=stream0)
            buf339 = reinterpret_tensor(buf335, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [queries_39, queries_40, queries_41, query_13, keys_39, keys_40, keys_41, key_13, values_26, view_41, values_27, value_13, owners_13, getitem_65, getitem_66, eq_13, attention_mask_13, attn_output_52], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf336, buf339, 1492992, stream=stream0)
            buf340 = reinterpret_tensor(buf336, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf336  # reuse
            # Topologically Sorted Source Nodes: [queries_39, queries_40, queries_41, query_13, keys_39, keys_40, keys_41, key_13, values_26, view_41, values_27, value_13, owners_13, getitem_65, getitem_66, eq_13, attention_mask_13, attn_output_52], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf337, buf340, 1492992, stream=stream0)
            del buf337
            buf341 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [queries_39, queries_40, queries_41, query_13, keys_39, keys_40, keys_41, key_13, values_26, view_41, values_27, value_13, owners_13, getitem_65, getitem_66, eq_13, attention_mask_13, attn_output_52], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant13, buf341, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_39, queries_40, queries_41, query_13, keys_39, keys_40, keys_41, key_13, values_26, view_41, values_27, value_13, owners_13, getitem_65, getitem_66, eq_13, attention_mask_13, attn_output_52], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf342 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf338, buf339, buf340, reinterpret_tensor(buf341, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf338
            buf343 = buf342[0]
            assert_size_stride(buf343, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf343, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf342
            buf347 = reinterpret_tensor(buf340, (1296, 1152), (1152, 1), 0); del buf340  # reuse
            # Topologically Sorted Source Nodes: [transpose_55, reshape_26, attn_output_55], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg218_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf343, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg217_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf347)
            del arg217_1
            del arg218_1
            buf352 = reinterpret_tensor(buf343, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf343  # reuse
            # Topologically Sorted Source Nodes: [attn_output_51, hidden_states_85, hidden_states_89, hidden_states_90, attn_output_55, hidden_states_92, hidden_states_93], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf305, buf322, buf329, buf347, arg219_1, arg220_1, buf352, 1296, 1152, stream=stream0)
            del arg219_1
            del arg220_1
            buf353 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg222_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf352, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg221_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf353)
            del arg221_1
            del arg222_1
            buf354 = buf328; del buf328  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf353, buf354, 5577984, stream=stream0)
            buf355 = reinterpret_tensor(buf352, (1296, 1152), (1152, 1), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg224_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf354, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg223_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf355)
            del arg223_1
            del arg224_1
            buf356 = buf305; del buf305  # reuse
            buf360 = reinterpret_tensor(buf339, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [attn_output_51, hidden_states_85, hidden_states_89, hidden_states_90, attn_output_55, hidden_states_92, hidden_states_96, hidden_states_97, hidden_states_98], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf356, buf322, buf329, buf347, buf355, arg225_1, arg226_1, buf360, 1296, 1152, stream=stream0)
            del arg225_1
            del arg226_1
            buf361 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [queries_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg228_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf360, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg227_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf361)
            del arg227_1
            del arg228_1
            buf362 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [keys_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg230_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf360, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg229_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf362)
            del arg229_1
            del arg230_1
            buf363 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [values_28], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg232_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf360, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg231_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf363)
            del arg231_1
            del arg232_1
            buf364 = reinterpret_tensor(buf360, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [queries_42, queries_43, queries_44, query_14, keys_42, keys_43, keys_44, key_14, values_28, view_44, values_29, value_14, owners_14, getitem_69, getitem_70, eq_14, attention_mask_14, attn_output_56], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf361, buf364, 1492992, stream=stream0)
            buf365 = reinterpret_tensor(buf361, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf361  # reuse
            # Topologically Sorted Source Nodes: [queries_42, queries_43, queries_44, query_14, keys_42, keys_43, keys_44, key_14, values_28, view_44, values_29, value_14, owners_14, getitem_69, getitem_70, eq_14, attention_mask_14, attn_output_56], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf362, buf365, 1492992, stream=stream0)
            buf366 = reinterpret_tensor(buf362, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [queries_42, queries_43, queries_44, query_14, keys_42, keys_43, keys_44, key_14, values_28, view_44, values_29, value_14, owners_14, getitem_69, getitem_70, eq_14, attention_mask_14, attn_output_56], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf363, buf366, 1492992, stream=stream0)
            buf367 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [queries_42, queries_43, queries_44, query_14, keys_42, keys_43, keys_44, key_14, values_28, view_44, values_29, value_14, owners_14, getitem_69, getitem_70, eq_14, attention_mask_14, attn_output_56], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant14, buf367, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_42, queries_43, queries_44, query_14, keys_42, keys_43, keys_44, key_14, values_28, view_44, values_29, value_14, owners_14, getitem_69, getitem_70, eq_14, attention_mask_14, attn_output_56], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf368 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf364, buf365, buf366, reinterpret_tensor(buf367, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf369 = buf368[0]
            assert_size_stride(buf369, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf369, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf368
            buf373 = reinterpret_tensor(buf366, (1296, 1152), (1152, 1), 0); del buf366  # reuse
            # Topologically Sorted Source Nodes: [transpose_59, reshape_27, attn_output_59], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg234_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf369, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg233_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf373)
            del arg233_1
            del arg234_1
            buf377 = reinterpret_tensor(buf369, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf369  # reuse
            # Topologically Sorted Source Nodes: [attn_output_59, hidden_states_99, hidden_states_100], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf356, buf373, arg235_1, arg236_1, buf377, 1296, 1152, stream=stream0)
            del arg235_1
            del arg236_1
            buf378 = buf353; del buf353  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg238_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf377, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg237_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf378)
            del arg237_1
            del arg238_1
            buf379 = buf354; del buf354  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf378, buf379, 5577984, stream=stream0)
            buf380 = reinterpret_tensor(buf377, (1296, 1152), (1152, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg240_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf379, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg239_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf380)
            del arg239_1
            del arg240_1
            buf385 = reinterpret_tensor(buf365, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf365  # reuse
            # Topologically Sorted Source Nodes: [attn_output_59, hidden_states_99, hidden_states_103, hidden_states_104, hidden_states_105], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf356, buf373, buf380, arg241_1, arg242_1, buf385, 1296, 1152, stream=stream0)
            del arg241_1
            del arg242_1
            buf386 = reinterpret_tensor(buf364, (1296, 1152), (1152, 1), 0); del buf364  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_105, queries_45], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg244_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf385, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg243_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf386)
            del arg243_1
            del arg244_1
            buf387 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [keys_45], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg246_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf385, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg245_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf387)
            del arg245_1
            del arg246_1
            buf388 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [values_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg248_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf385, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg247_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf388)
            del arg247_1
            del arg248_1
            buf389 = reinterpret_tensor(buf385, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf385  # reuse
            # Topologically Sorted Source Nodes: [queries_45, queries_46, queries_47, query_15, keys_45, keys_46, keys_47, key_15, values_30, view_47, values_31, value_15, owners_15, getitem_73, getitem_74, eq_15, attention_mask_15, attn_output_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf386, buf389, 1492992, stream=stream0)
            buf390 = reinterpret_tensor(buf386, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf386  # reuse
            # Topologically Sorted Source Nodes: [queries_45, queries_46, queries_47, query_15, keys_45, keys_46, keys_47, key_15, values_30, view_47, values_31, value_15, owners_15, getitem_73, getitem_74, eq_15, attention_mask_15, attn_output_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf387, buf390, 1492992, stream=stream0)
            buf391 = reinterpret_tensor(buf387, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [queries_45, queries_46, queries_47, query_15, keys_45, keys_46, keys_47, key_15, values_30, view_47, values_31, value_15, owners_15, getitem_73, getitem_74, eq_15, attention_mask_15, attn_output_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf388, buf391, 1492992, stream=stream0)
            del buf388
            buf392 = buf367; del buf367  # reuse
            # Topologically Sorted Source Nodes: [queries_45, queries_46, queries_47, query_15, keys_45, keys_46, keys_47, key_15, values_30, view_47, values_31, value_15, owners_15, getitem_73, getitem_74, eq_15, attention_mask_15, attn_output_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant15, buf392, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_45, queries_46, queries_47, query_15, keys_45, keys_46, keys_47, key_15, values_30, view_47, values_31, value_15, owners_15, getitem_73, getitem_74, eq_15, attention_mask_15, attn_output_60], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf393 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf389, buf390, buf391, reinterpret_tensor(buf392, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf389
            buf394 = buf393[0]
            assert_size_stride(buf394, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf394, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf393
            buf398 = reinterpret_tensor(buf391, (1296, 1152), (1152, 1), 0); del buf391  # reuse
            # Topologically Sorted Source Nodes: [transpose_63, reshape_28, attn_output_63], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg250_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf394, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg249_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf398)
            del arg249_1
            del arg250_1
            buf403 = reinterpret_tensor(buf394, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf394  # reuse
            # Topologically Sorted Source Nodes: [attn_output_59, hidden_states_99, hidden_states_103, hidden_states_104, attn_output_63, hidden_states_106, hidden_states_107], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf356, buf373, buf380, buf398, arg251_1, arg252_1, buf403, 1296, 1152, stream=stream0)
            del arg251_1
            del arg252_1
            buf404 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg254_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf403, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg253_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf404)
            del arg253_1
            del arg254_1
            buf405 = buf379; del buf379  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf404, buf405, 5577984, stream=stream0)
            buf406 = reinterpret_tensor(buf403, (1296, 1152), (1152, 1), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg256_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf405, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg255_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf406)
            del arg255_1
            del arg256_1
            buf407 = buf356; del buf356  # reuse
            buf411 = reinterpret_tensor(buf390, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf390  # reuse
            # Topologically Sorted Source Nodes: [attn_output_59, hidden_states_99, hidden_states_103, hidden_states_104, attn_output_63, hidden_states_106, hidden_states_110, hidden_states_111, hidden_states_112], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf407, buf373, buf380, buf398, buf406, arg257_1, arg258_1, buf411, 1296, 1152, stream=stream0)
            del arg257_1
            del arg258_1
            buf412 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [queries_48], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg260_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf411, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg259_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf412)
            del arg259_1
            del arg260_1
            buf413 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [keys_48], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg262_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf411, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg261_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf413)
            del arg261_1
            del arg262_1
            buf414 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [values_32], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg264_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf411, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg263_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf414)
            del arg263_1
            del arg264_1
            buf415 = reinterpret_tensor(buf411, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [queries_48, queries_49, queries_50, query_16, keys_48, keys_49, keys_50, key_16, values_32, view_50, values_33, value_16, owners_16, getitem_77, getitem_78, eq_16, attention_mask_16, attn_output_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf412, buf415, 1492992, stream=stream0)
            buf416 = reinterpret_tensor(buf412, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf412  # reuse
            # Topologically Sorted Source Nodes: [queries_48, queries_49, queries_50, query_16, keys_48, keys_49, keys_50, key_16, values_32, view_50, values_33, value_16, owners_16, getitem_77, getitem_78, eq_16, attention_mask_16, attn_output_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf413, buf416, 1492992, stream=stream0)
            buf417 = reinterpret_tensor(buf413, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf413  # reuse
            # Topologically Sorted Source Nodes: [queries_48, queries_49, queries_50, query_16, keys_48, keys_49, keys_50, key_16, values_32, view_50, values_33, value_16, owners_16, getitem_77, getitem_78, eq_16, attention_mask_16, attn_output_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf414, buf417, 1492992, stream=stream0)
            buf418 = buf392; del buf392  # reuse
            # Topologically Sorted Source Nodes: [queries_48, queries_49, queries_50, query_16, keys_48, keys_49, keys_50, key_16, values_32, view_50, values_33, value_16, owners_16, getitem_77, getitem_78, eq_16, attention_mask_16, attn_output_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant16, buf418, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_48, queries_49, queries_50, query_16, keys_48, keys_49, keys_50, key_16, values_32, view_50, values_33, value_16, owners_16, getitem_77, getitem_78, eq_16, attention_mask_16, attn_output_64], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf419 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf415, buf416, buf417, reinterpret_tensor(buf418, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf420 = buf419[0]
            assert_size_stride(buf420, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf420, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf419
            buf424 = reinterpret_tensor(buf417, (1296, 1152), (1152, 1), 0); del buf417  # reuse
            # Topologically Sorted Source Nodes: [transpose_67, reshape_29, attn_output_67], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg266_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf420, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg265_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf424)
            del arg265_1
            del arg266_1
            buf428 = reinterpret_tensor(buf420, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf420  # reuse
            # Topologically Sorted Source Nodes: [attn_output_67, hidden_states_113, hidden_states_114], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf407, buf424, arg267_1, arg268_1, buf428, 1296, 1152, stream=stream0)
            del arg267_1
            del arg268_1
            buf429 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_115], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg270_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf428, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg269_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf429)
            del arg269_1
            del arg270_1
            buf430 = buf405; del buf405  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf429, buf430, 5577984, stream=stream0)
            buf431 = reinterpret_tensor(buf428, (1296, 1152), (1152, 1), 0); del buf428  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116, hidden_states_117], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg272_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf430, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg271_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf431)
            del arg271_1
            del arg272_1
            buf436 = reinterpret_tensor(buf416, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf416  # reuse
            # Topologically Sorted Source Nodes: [attn_output_67, hidden_states_113, hidden_states_117, hidden_states_118, hidden_states_119], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf407, buf424, buf431, arg273_1, arg274_1, buf436, 1296, 1152, stream=stream0)
            del arg273_1
            del arg274_1
            buf437 = reinterpret_tensor(buf415, (1296, 1152), (1152, 1), 0); del buf415  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_119, queries_51], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg276_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf436, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg275_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf437)
            del arg275_1
            del arg276_1
            buf438 = buf414; del buf414  # reuse
            # Topologically Sorted Source Nodes: [keys_51], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg278_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf436, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg277_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf438)
            del arg277_1
            del arg278_1
            buf439 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [values_34], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg280_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf436, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg279_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf439)
            del arg279_1
            del arg280_1
            buf440 = reinterpret_tensor(buf436, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf436  # reuse
            # Topologically Sorted Source Nodes: [queries_51, queries_52, queries_53, query_17, keys_51, keys_52, keys_53, key_17, values_34, view_53, values_35, value_17, owners_17, getitem_81, getitem_82, eq_17, attention_mask_17, attn_output_68], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf437, buf440, 1492992, stream=stream0)
            buf441 = reinterpret_tensor(buf437, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf437  # reuse
            # Topologically Sorted Source Nodes: [queries_51, queries_52, queries_53, query_17, keys_51, keys_52, keys_53, key_17, values_34, view_53, values_35, value_17, owners_17, getitem_81, getitem_82, eq_17, attention_mask_17, attn_output_68], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf438, buf441, 1492992, stream=stream0)
            buf442 = reinterpret_tensor(buf438, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [queries_51, queries_52, queries_53, query_17, keys_51, keys_52, keys_53, key_17, values_34, view_53, values_35, value_17, owners_17, getitem_81, getitem_82, eq_17, attention_mask_17, attn_output_68], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf439, buf442, 1492992, stream=stream0)
            del buf439
            buf443 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [queries_51, queries_52, queries_53, query_17, keys_51, keys_52, keys_53, key_17, values_34, view_53, values_35, value_17, owners_17, getitem_81, getitem_82, eq_17, attention_mask_17, attn_output_68], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant17, buf443, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_51, queries_52, queries_53, query_17, keys_51, keys_52, keys_53, key_17, values_34, view_53, values_35, value_17, owners_17, getitem_81, getitem_82, eq_17, attention_mask_17, attn_output_68], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf444 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf440, buf441, buf442, reinterpret_tensor(buf443, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf440
            buf445 = buf444[0]
            assert_size_stride(buf445, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf445, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf444
            buf449 = reinterpret_tensor(buf442, (1296, 1152), (1152, 1), 0); del buf442  # reuse
            # Topologically Sorted Source Nodes: [transpose_71, reshape_30, attn_output_71], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg282_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf445, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg281_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf449)
            del arg281_1
            del arg282_1
            buf454 = reinterpret_tensor(buf445, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf445  # reuse
            # Topologically Sorted Source Nodes: [attn_output_67, hidden_states_113, hidden_states_117, hidden_states_118, attn_output_71, hidden_states_120, hidden_states_121], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf407, buf424, buf431, buf449, arg283_1, arg284_1, buf454, 1296, 1152, stream=stream0)
            del arg283_1
            del arg284_1
            buf455 = buf429; del buf429  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_121, hidden_states_122], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg286_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf454, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg285_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf455)
            del arg285_1
            del arg286_1
            buf456 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_122, hidden_states_123], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf455, buf456, 5577984, stream=stream0)
            buf457 = reinterpret_tensor(buf454, (1296, 1152), (1152, 1), 0); del buf454  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_122, hidden_states_123, hidden_states_124], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg288_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf456, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg287_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf457)
            del arg287_1
            del arg288_1
            buf458 = buf407; del buf407  # reuse
            buf462 = reinterpret_tensor(buf441, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf441  # reuse
            # Topologically Sorted Source Nodes: [attn_output_67, hidden_states_113, hidden_states_117, hidden_states_118, attn_output_71, hidden_states_120, hidden_states_124, hidden_states_125, hidden_states_126], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf458, buf424, buf431, buf449, buf457, arg289_1, arg290_1, buf462, 1296, 1152, stream=stream0)
            del arg289_1
            del arg290_1
            buf463 = buf457; del buf457  # reuse
            # Topologically Sorted Source Nodes: [queries_54], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg292_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf462, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg291_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf463)
            del arg291_1
            del arg292_1
            buf464 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [keys_54], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg294_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf462, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg293_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf464)
            del arg293_1
            del arg294_1
            buf465 = buf431; del buf431  # reuse
            # Topologically Sorted Source Nodes: [values_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg296_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf462, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg295_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf465)
            del arg295_1
            del arg296_1
            buf466 = reinterpret_tensor(buf462, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf462  # reuse
            # Topologically Sorted Source Nodes: [queries_54, queries_55, queries_56, query_18, keys_54, keys_55, keys_56, key_18, values_36, view_56, values_37, value_18, owners_18, getitem_85, getitem_86, eq_18, attention_mask_18, attn_output_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf463, buf466, 1492992, stream=stream0)
            buf467 = reinterpret_tensor(buf463, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf463  # reuse
            # Topologically Sorted Source Nodes: [queries_54, queries_55, queries_56, query_18, keys_54, keys_55, keys_56, key_18, values_36, view_56, values_37, value_18, owners_18, getitem_85, getitem_86, eq_18, attention_mask_18, attn_output_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf464, buf467, 1492992, stream=stream0)
            buf468 = reinterpret_tensor(buf464, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf464  # reuse
            # Topologically Sorted Source Nodes: [queries_54, queries_55, queries_56, query_18, keys_54, keys_55, keys_56, key_18, values_36, view_56, values_37, value_18, owners_18, getitem_85, getitem_86, eq_18, attention_mask_18, attn_output_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf465, buf468, 1492992, stream=stream0)
            buf469 = buf443; del buf443  # reuse
            # Topologically Sorted Source Nodes: [queries_54, queries_55, queries_56, query_18, keys_54, keys_55, keys_56, key_18, values_36, view_56, values_37, value_18, owners_18, getitem_85, getitem_86, eq_18, attention_mask_18, attn_output_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant18, buf469, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_54, queries_55, queries_56, query_18, keys_54, keys_55, keys_56, key_18, values_36, view_56, values_37, value_18, owners_18, getitem_85, getitem_86, eq_18, attention_mask_18, attn_output_72], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf470 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf466, buf467, buf468, reinterpret_tensor(buf469, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf471 = buf470[0]
            assert_size_stride(buf471, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf471, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf470
            buf475 = reinterpret_tensor(buf468, (1296, 1152), (1152, 1), 0); del buf468  # reuse
            # Topologically Sorted Source Nodes: [transpose_75, reshape_31, attn_output_75], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg298_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf471, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg297_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf475)
            del arg297_1
            del arg298_1
            buf479 = reinterpret_tensor(buf471, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf471  # reuse
            # Topologically Sorted Source Nodes: [attn_output_75, hidden_states_127, hidden_states_128], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf458, buf475, arg299_1, arg300_1, buf479, 1296, 1152, stream=stream0)
            del arg299_1
            del arg300_1
            buf480 = buf455; del buf455  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_129], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg302_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf479, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg301_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf480)
            del arg301_1
            del arg302_1
            buf481 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf480, buf481, 5577984, stream=stream0)
            buf482 = reinterpret_tensor(buf479, (1296, 1152), (1152, 1), 0); del buf479  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130, hidden_states_131], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg304_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf481, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg303_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf482)
            del arg303_1
            del arg304_1
            buf487 = reinterpret_tensor(buf467, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf467  # reuse
            # Topologically Sorted Source Nodes: [attn_output_75, hidden_states_127, hidden_states_131, hidden_states_132, hidden_states_133], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf458, buf475, buf482, arg305_1, arg306_1, buf487, 1296, 1152, stream=stream0)
            del arg305_1
            del arg306_1
            buf488 = reinterpret_tensor(buf466, (1296, 1152), (1152, 1), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_133, queries_57], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg308_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf487, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg307_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf488)
            del arg307_1
            del arg308_1
            buf489 = buf465; del buf465  # reuse
            # Topologically Sorted Source Nodes: [keys_57], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg310_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf487, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg309_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf489)
            del arg309_1
            del arg310_1
            buf490 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [values_38], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg312_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf487, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg311_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf490)
            del arg311_1
            del arg312_1
            buf491 = reinterpret_tensor(buf487, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [queries_57, queries_58, queries_59, query_19, keys_57, keys_58, keys_59, key_19, values_38, view_59, values_39, value_19, owners_19, getitem_89, getitem_90, eq_19, attention_mask_19, attn_output_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf488, buf491, 1492992, stream=stream0)
            buf492 = reinterpret_tensor(buf488, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf488  # reuse
            # Topologically Sorted Source Nodes: [queries_57, queries_58, queries_59, query_19, keys_57, keys_58, keys_59, key_19, values_38, view_59, values_39, value_19, owners_19, getitem_89, getitem_90, eq_19, attention_mask_19, attn_output_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf489, buf492, 1492992, stream=stream0)
            buf493 = reinterpret_tensor(buf489, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf489  # reuse
            # Topologically Sorted Source Nodes: [queries_57, queries_58, queries_59, query_19, keys_57, keys_58, keys_59, key_19, values_38, view_59, values_39, value_19, owners_19, getitem_89, getitem_90, eq_19, attention_mask_19, attn_output_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf490, buf493, 1492992, stream=stream0)
            del buf490
            buf494 = buf469; del buf469  # reuse
            # Topologically Sorted Source Nodes: [queries_57, queries_58, queries_59, query_19, keys_57, keys_58, keys_59, key_19, values_38, view_59, values_39, value_19, owners_19, getitem_89, getitem_90, eq_19, attention_mask_19, attn_output_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant19, buf494, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_57, queries_58, queries_59, query_19, keys_57, keys_58, keys_59, key_19, values_38, view_59, values_39, value_19, owners_19, getitem_89, getitem_90, eq_19, attention_mask_19, attn_output_76], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf495 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf491, buf492, buf493, reinterpret_tensor(buf494, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf491
            buf496 = buf495[0]
            assert_size_stride(buf496, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf496, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf495
            buf500 = reinterpret_tensor(buf493, (1296, 1152), (1152, 1), 0); del buf493  # reuse
            # Topologically Sorted Source Nodes: [transpose_79, reshape_32, attn_output_79], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg314_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf496, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg313_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf500)
            del arg313_1
            del arg314_1
            buf505 = reinterpret_tensor(buf496, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf496  # reuse
            # Topologically Sorted Source Nodes: [attn_output_75, hidden_states_127, hidden_states_131, hidden_states_132, attn_output_79, hidden_states_134, hidden_states_135], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf458, buf475, buf482, buf500, arg315_1, arg316_1, buf505, 1296, 1152, stream=stream0)
            del arg315_1
            del arg316_1
            buf506 = buf480; del buf480  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_135, hidden_states_136], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg318_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf505, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg317_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf506)
            del arg317_1
            del arg318_1
            buf507 = buf481; del buf481  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf506, buf507, 5577984, stream=stream0)
            buf508 = reinterpret_tensor(buf505, (1296, 1152), (1152, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, hidden_states_138], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg320_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf507, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg319_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf508)
            del arg319_1
            del arg320_1
            buf509 = buf458; del buf458  # reuse
            buf513 = reinterpret_tensor(buf492, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf492  # reuse
            # Topologically Sorted Source Nodes: [attn_output_75, hidden_states_127, hidden_states_131, hidden_states_132, attn_output_79, hidden_states_134, hidden_states_138, hidden_states_139, hidden_states_140], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf509, buf475, buf482, buf500, buf508, arg321_1, arg322_1, buf513, 1296, 1152, stream=stream0)
            del arg321_1
            del arg322_1
            buf514 = buf508; del buf508  # reuse
            # Topologically Sorted Source Nodes: [queries_60], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg324_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf513, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg323_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf514)
            del arg323_1
            del arg324_1
            buf515 = buf500; del buf500  # reuse
            # Topologically Sorted Source Nodes: [keys_60], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg326_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf513, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg325_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf515)
            del arg325_1
            del arg326_1
            buf516 = buf482; del buf482  # reuse
            # Topologically Sorted Source Nodes: [values_40], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg328_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf513, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg327_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf516)
            del arg327_1
            del arg328_1
            buf517 = reinterpret_tensor(buf513, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf513  # reuse
            # Topologically Sorted Source Nodes: [queries_60, queries_61, queries_62, query_20, keys_60, keys_61, keys_62, key_20, values_40, view_62, values_41, value_20, owners_20, getitem_93, getitem_94, eq_20, attention_mask_20, attn_output_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf514, buf517, 1492992, stream=stream0)
            buf518 = reinterpret_tensor(buf514, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf514  # reuse
            # Topologically Sorted Source Nodes: [queries_60, queries_61, queries_62, query_20, keys_60, keys_61, keys_62, key_20, values_40, view_62, values_41, value_20, owners_20, getitem_93, getitem_94, eq_20, attention_mask_20, attn_output_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf515, buf518, 1492992, stream=stream0)
            buf519 = reinterpret_tensor(buf515, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf515  # reuse
            # Topologically Sorted Source Nodes: [queries_60, queries_61, queries_62, query_20, keys_60, keys_61, keys_62, key_20, values_40, view_62, values_41, value_20, owners_20, getitem_93, getitem_94, eq_20, attention_mask_20, attn_output_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf516, buf519, 1492992, stream=stream0)
            buf520 = buf494; del buf494  # reuse
            # Topologically Sorted Source Nodes: [queries_60, queries_61, queries_62, query_20, keys_60, keys_61, keys_62, key_20, values_40, view_62, values_41, value_20, owners_20, getitem_93, getitem_94, eq_20, attention_mask_20, attn_output_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant20, buf520, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_60, queries_61, queries_62, query_20, keys_60, keys_61, keys_62, key_20, values_40, view_62, values_41, value_20, owners_20, getitem_93, getitem_94, eq_20, attention_mask_20, attn_output_80], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf521 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf517, buf518, buf519, reinterpret_tensor(buf520, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf522 = buf521[0]
            assert_size_stride(buf522, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf522, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf521
            buf526 = reinterpret_tensor(buf519, (1296, 1152), (1152, 1), 0); del buf519  # reuse
            # Topologically Sorted Source Nodes: [transpose_83, reshape_33, attn_output_83], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg330_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf522, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg329_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf526)
            del arg329_1
            del arg330_1
            buf530 = reinterpret_tensor(buf522, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf522  # reuse
            # Topologically Sorted Source Nodes: [attn_output_83, hidden_states_141, hidden_states_142], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf509, buf526, arg331_1, arg332_1, buf530, 1296, 1152, stream=stream0)
            del arg331_1
            del arg332_1
            buf531 = buf506; del buf506  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_143], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg334_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf530, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg333_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf531)
            del arg333_1
            del arg334_1
            buf532 = buf507; del buf507  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_143, hidden_states_144], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf531, buf532, 5577984, stream=stream0)
            buf533 = reinterpret_tensor(buf530, (1296, 1152), (1152, 1), 0); del buf530  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_143, hidden_states_144, hidden_states_145], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg336_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf532, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg335_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf533)
            del arg335_1
            del arg336_1
            buf538 = reinterpret_tensor(buf518, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf518  # reuse
            # Topologically Sorted Source Nodes: [attn_output_83, hidden_states_141, hidden_states_145, hidden_states_146, hidden_states_147], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf509, buf526, buf533, arg337_1, arg338_1, buf538, 1296, 1152, stream=stream0)
            del arg337_1
            del arg338_1
            buf539 = reinterpret_tensor(buf517, (1296, 1152), (1152, 1), 0); del buf517  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_147, queries_63], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg340_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf538, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg339_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf539)
            del arg339_1
            del arg340_1
            buf540 = buf516; del buf516  # reuse
            # Topologically Sorted Source Nodes: [keys_63], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg342_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf538, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg341_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf540)
            del arg341_1
            del arg342_1
            buf541 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [values_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg344_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf538, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg343_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf541)
            del arg343_1
            del arg344_1
            buf542 = reinterpret_tensor(buf538, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf538  # reuse
            # Topologically Sorted Source Nodes: [queries_63, queries_64, queries_65, query_21, keys_63, keys_64, keys_65, key_21, values_42, view_65, values_43, value_21, owners_21, getitem_97, getitem_98, eq_21, attention_mask_21, attn_output_84], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf539, buf542, 1492992, stream=stream0)
            buf543 = reinterpret_tensor(buf539, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf539  # reuse
            # Topologically Sorted Source Nodes: [queries_63, queries_64, queries_65, query_21, keys_63, keys_64, keys_65, key_21, values_42, view_65, values_43, value_21, owners_21, getitem_97, getitem_98, eq_21, attention_mask_21, attn_output_84], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf540, buf543, 1492992, stream=stream0)
            buf544 = reinterpret_tensor(buf540, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf540  # reuse
            # Topologically Sorted Source Nodes: [queries_63, queries_64, queries_65, query_21, keys_63, keys_64, keys_65, key_21, values_42, view_65, values_43, value_21, owners_21, getitem_97, getitem_98, eq_21, attention_mask_21, attn_output_84], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf541, buf544, 1492992, stream=stream0)
            del buf541
            buf545 = buf520; del buf520  # reuse
            # Topologically Sorted Source Nodes: [queries_63, queries_64, queries_65, query_21, keys_63, keys_64, keys_65, key_21, values_42, view_65, values_43, value_21, owners_21, getitem_97, getitem_98, eq_21, attention_mask_21, attn_output_84], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant21, buf545, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_63, queries_64, queries_65, query_21, keys_63, keys_64, keys_65, key_21, values_42, view_65, values_43, value_21, owners_21, getitem_97, getitem_98, eq_21, attention_mask_21, attn_output_84], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf546 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf542, buf543, buf544, reinterpret_tensor(buf545, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf542
            buf547 = buf546[0]
            assert_size_stride(buf547, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf547, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf546
            buf551 = reinterpret_tensor(buf544, (1296, 1152), (1152, 1), 0); del buf544  # reuse
            # Topologically Sorted Source Nodes: [transpose_87, reshape_34, attn_output_87], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg346_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf547, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg345_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf551)
            del arg345_1
            del arg346_1
            buf556 = reinterpret_tensor(buf547, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf547  # reuse
            # Topologically Sorted Source Nodes: [attn_output_83, hidden_states_141, hidden_states_145, hidden_states_146, attn_output_87, hidden_states_148, hidden_states_149], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf509, buf526, buf533, buf551, arg347_1, arg348_1, buf556, 1296, 1152, stream=stream0)
            del arg347_1
            del arg348_1
            buf557 = buf531; del buf531  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_149, hidden_states_150], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg350_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf556, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg349_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf557)
            del arg349_1
            del arg350_1
            buf558 = buf532; del buf532  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf557, buf558, 5577984, stream=stream0)
            buf559 = reinterpret_tensor(buf556, (1296, 1152), (1152, 1), 0); del buf556  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, hidden_states_152], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg352_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf558, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg351_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf559)
            del arg351_1
            del arg352_1
            buf560 = buf509; del buf509  # reuse
            buf564 = reinterpret_tensor(buf543, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf543  # reuse
            # Topologically Sorted Source Nodes: [attn_output_83, hidden_states_141, hidden_states_145, hidden_states_146, attn_output_87, hidden_states_148, hidden_states_152, hidden_states_153, hidden_states_154], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf560, buf526, buf533, buf551, buf559, arg353_1, arg354_1, buf564, 1296, 1152, stream=stream0)
            del arg353_1
            del arg354_1
            buf565 = buf559; del buf559  # reuse
            # Topologically Sorted Source Nodes: [queries_66], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg356_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf564, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg355_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf565)
            del arg355_1
            del arg356_1
            buf566 = buf551; del buf551  # reuse
            # Topologically Sorted Source Nodes: [keys_66], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg358_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf564, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg357_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf566)
            del arg357_1
            del arg358_1
            buf567 = buf533; del buf533  # reuse
            # Topologically Sorted Source Nodes: [values_44], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg360_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf564, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg359_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf567)
            del arg359_1
            del arg360_1
            buf568 = reinterpret_tensor(buf564, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf564  # reuse
            # Topologically Sorted Source Nodes: [queries_66, queries_67, queries_68, query_22, keys_66, keys_67, keys_68, key_22, values_44, view_68, values_45, value_22, owners_22, getitem_101, getitem_102, eq_22, attention_mask_22, attn_output_88], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf565, buf568, 1492992, stream=stream0)
            buf569 = reinterpret_tensor(buf565, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf565  # reuse
            # Topologically Sorted Source Nodes: [queries_66, queries_67, queries_68, query_22, keys_66, keys_67, keys_68, key_22, values_44, view_68, values_45, value_22, owners_22, getitem_101, getitem_102, eq_22, attention_mask_22, attn_output_88], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf566, buf569, 1492992, stream=stream0)
            buf570 = reinterpret_tensor(buf566, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf566  # reuse
            # Topologically Sorted Source Nodes: [queries_66, queries_67, queries_68, query_22, keys_66, keys_67, keys_68, key_22, values_44, view_68, values_45, value_22, owners_22, getitem_101, getitem_102, eq_22, attention_mask_22, attn_output_88], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf567, buf570, 1492992, stream=stream0)
            buf571 = buf545; del buf545  # reuse
            # Topologically Sorted Source Nodes: [queries_66, queries_67, queries_68, query_22, keys_66, keys_67, keys_68, key_22, values_44, view_68, values_45, value_22, owners_22, getitem_101, getitem_102, eq_22, attention_mask_22, attn_output_88], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant22, buf571, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_66, queries_67, queries_68, query_22, keys_66, keys_67, keys_68, key_22, values_44, view_68, values_45, value_22, owners_22, getitem_101, getitem_102, eq_22, attention_mask_22, attn_output_88], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf572 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf568, buf569, buf570, reinterpret_tensor(buf571, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf573 = buf572[0]
            assert_size_stride(buf573, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf573, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf572
            buf577 = reinterpret_tensor(buf570, (1296, 1152), (1152, 1), 0); del buf570  # reuse
            # Topologically Sorted Source Nodes: [transpose_91, reshape_35, attn_output_91], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg362_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf573, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg361_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf577)
            del arg361_1
            del arg362_1
            buf581 = reinterpret_tensor(buf573, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf573  # reuse
            # Topologically Sorted Source Nodes: [attn_output_91, hidden_states_155, hidden_states_156], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf560, buf577, arg363_1, arg364_1, buf581, 1296, 1152, stream=stream0)
            del arg363_1
            del arg364_1
            buf582 = buf557; del buf557  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_157], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg366_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf581, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg365_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf582)
            del arg365_1
            del arg366_1
            buf583 = buf558; del buf558  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_157, hidden_states_158], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf582, buf583, 5577984, stream=stream0)
            buf584 = reinterpret_tensor(buf581, (1296, 1152), (1152, 1), 0); del buf581  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_157, hidden_states_158, hidden_states_159], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg368_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf583, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg367_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf584)
            del arg367_1
            del arg368_1
            buf589 = reinterpret_tensor(buf569, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf569  # reuse
            # Topologically Sorted Source Nodes: [attn_output_91, hidden_states_155, hidden_states_159, hidden_states_160, hidden_states_161], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf560, buf577, buf584, arg369_1, arg370_1, buf589, 1296, 1152, stream=stream0)
            del arg369_1
            del arg370_1
            buf590 = reinterpret_tensor(buf568, (1296, 1152), (1152, 1), 0); del buf568  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_161, queries_69], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg372_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf589, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg371_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf590)
            del arg371_1
            del arg372_1
            buf591 = buf567; del buf567  # reuse
            # Topologically Sorted Source Nodes: [keys_69], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg374_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf589, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg373_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf591)
            del arg373_1
            del arg374_1
            buf592 = buf526; del buf526  # reuse
            # Topologically Sorted Source Nodes: [values_46], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg376_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf589, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg375_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf592)
            del arg375_1
            del arg376_1
            buf593 = reinterpret_tensor(buf589, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf589  # reuse
            # Topologically Sorted Source Nodes: [queries_69, queries_70, queries_71, query_23, keys_69, keys_70, keys_71, key_23, values_46, view_71, values_47, value_23, owners_23, getitem_105, getitem_106, eq_23, attention_mask_23, attn_output_92], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf590, buf593, 1492992, stream=stream0)
            buf594 = reinterpret_tensor(buf590, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf590  # reuse
            # Topologically Sorted Source Nodes: [queries_69, queries_70, queries_71, query_23, keys_69, keys_70, keys_71, key_23, values_46, view_71, values_47, value_23, owners_23, getitem_105, getitem_106, eq_23, attention_mask_23, attn_output_92], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf591, buf594, 1492992, stream=stream0)
            buf595 = reinterpret_tensor(buf591, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf591  # reuse
            # Topologically Sorted Source Nodes: [queries_69, queries_70, queries_71, query_23, keys_69, keys_70, keys_71, key_23, values_46, view_71, values_47, value_23, owners_23, getitem_105, getitem_106, eq_23, attention_mask_23, attn_output_92], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf592, buf595, 1492992, stream=stream0)
            del buf592
            buf596 = buf571; del buf571  # reuse
            # Topologically Sorted Source Nodes: [queries_69, queries_70, queries_71, query_23, keys_69, keys_70, keys_71, key_23, values_46, view_71, values_47, value_23, owners_23, getitem_105, getitem_106, eq_23, attention_mask_23, attn_output_92], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant23, buf596, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_69, queries_70, queries_71, query_23, keys_69, keys_70, keys_71, key_23, values_46, view_71, values_47, value_23, owners_23, getitem_105, getitem_106, eq_23, attention_mask_23, attn_output_92], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf597 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf593, buf594, buf595, reinterpret_tensor(buf596, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf593
            buf598 = buf597[0]
            assert_size_stride(buf598, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf598, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf597
            buf602 = reinterpret_tensor(buf595, (1296, 1152), (1152, 1), 0); del buf595  # reuse
            # Topologically Sorted Source Nodes: [transpose_95, reshape_36, attn_output_95], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg378_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf598, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg377_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf602)
            del arg377_1
            del arg378_1
            buf607 = reinterpret_tensor(buf598, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf598  # reuse
            # Topologically Sorted Source Nodes: [attn_output_91, hidden_states_155, hidden_states_159, hidden_states_160, attn_output_95, hidden_states_162, hidden_states_163], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf560, buf577, buf584, buf602, arg379_1, arg380_1, buf607, 1296, 1152, stream=stream0)
            del arg379_1
            del arg380_1
            buf608 = buf582; del buf582  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_163, hidden_states_164], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg382_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf607, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg381_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf608)
            del arg381_1
            del arg382_1
            buf609 = buf583; del buf583  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_164, hidden_states_165], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf608, buf609, 5577984, stream=stream0)
            buf610 = reinterpret_tensor(buf607, (1296, 1152), (1152, 1), 0); del buf607  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_164, hidden_states_165, hidden_states_166], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg384_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf609, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg383_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf610)
            del arg383_1
            del arg384_1
            buf611 = buf560; del buf560  # reuse
            buf615 = reinterpret_tensor(buf594, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf594  # reuse
            # Topologically Sorted Source Nodes: [attn_output_91, hidden_states_155, hidden_states_159, hidden_states_160, attn_output_95, hidden_states_162, hidden_states_166, hidden_states_167, hidden_states_168], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf611, buf577, buf584, buf602, buf610, arg385_1, arg386_1, buf615, 1296, 1152, stream=stream0)
            del arg385_1
            del arg386_1
            buf616 = buf610; del buf610  # reuse
            # Topologically Sorted Source Nodes: [queries_72], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg388_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf615, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg387_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf616)
            del arg387_1
            del arg388_1
            buf617 = buf602; del buf602  # reuse
            # Topologically Sorted Source Nodes: [keys_72], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg390_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf615, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg389_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf617)
            del arg389_1
            del arg390_1
            buf618 = buf584; del buf584  # reuse
            # Topologically Sorted Source Nodes: [values_48], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg392_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf615, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg391_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf618)
            del arg391_1
            del arg392_1
            buf619 = reinterpret_tensor(buf615, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf615  # reuse
            # Topologically Sorted Source Nodes: [queries_72, queries_73, queries_74, query_24, keys_72, keys_73, keys_74, key_24, values_48, view_74, values_49, value_24, owners_24, getitem_109, getitem_110, eq_24, attention_mask_24, attn_output_96], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf616, buf619, 1492992, stream=stream0)
            buf620 = reinterpret_tensor(buf616, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf616  # reuse
            # Topologically Sorted Source Nodes: [queries_72, queries_73, queries_74, query_24, keys_72, keys_73, keys_74, key_24, values_48, view_74, values_49, value_24, owners_24, getitem_109, getitem_110, eq_24, attention_mask_24, attn_output_96], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf617, buf620, 1492992, stream=stream0)
            buf621 = reinterpret_tensor(buf617, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf617  # reuse
            # Topologically Sorted Source Nodes: [queries_72, queries_73, queries_74, query_24, keys_72, keys_73, keys_74, key_24, values_48, view_74, values_49, value_24, owners_24, getitem_109, getitem_110, eq_24, attention_mask_24, attn_output_96], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf618, buf621, 1492992, stream=stream0)
            buf622 = buf596; del buf596  # reuse
            # Topologically Sorted Source Nodes: [queries_72, queries_73, queries_74, query_24, keys_72, keys_73, keys_74, key_24, values_48, view_74, values_49, value_24, owners_24, getitem_109, getitem_110, eq_24, attention_mask_24, attn_output_96], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant24, buf622, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_72, queries_73, queries_74, query_24, keys_72, keys_73, keys_74, key_24, values_48, view_74, values_49, value_24, owners_24, getitem_109, getitem_110, eq_24, attention_mask_24, attn_output_96], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf623 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf619, buf620, buf621, reinterpret_tensor(buf622, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            buf624 = buf623[0]
            assert_size_stride(buf624, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf624, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf623
            buf628 = reinterpret_tensor(buf621, (1296, 1152), (1152, 1), 0); del buf621  # reuse
            # Topologically Sorted Source Nodes: [transpose_99, reshape_37, attn_output_99], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg394_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf624, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg393_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf628)
            del arg393_1
            del arg394_1
            buf632 = reinterpret_tensor(buf624, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf624  # reuse
            # Topologically Sorted Source Nodes: [attn_output_99, hidden_states_169, hidden_states_170], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf611, buf628, arg395_1, arg396_1, buf632, 1296, 1152, stream=stream0)
            del arg395_1
            del arg396_1
            buf633 = buf608; del buf608  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_171], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg398_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf632, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg397_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf633)
            del arg397_1
            del arg398_1
            buf634 = buf609; del buf609  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_171, hidden_states_172], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf633, buf634, 5577984, stream=stream0)
            buf635 = reinterpret_tensor(buf632, (1296, 1152), (1152, 1), 0); del buf632  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_171, hidden_states_172, hidden_states_173], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg400_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf634, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg399_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf635)
            del arg399_1
            del arg400_1
            buf640 = reinterpret_tensor(buf620, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf620  # reuse
            # Topologically Sorted Source Nodes: [attn_output_99, hidden_states_169, hidden_states_173, hidden_states_174, hidden_states_175], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf611, buf628, buf635, arg401_1, arg402_1, buf640, 1296, 1152, stream=stream0)
            del arg401_1
            del arg402_1
            buf641 = reinterpret_tensor(buf619, (1296, 1152), (1152, 1), 0); del buf619  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_175, queries_75], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg404_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf640, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg403_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf641)
            del arg403_1
            del arg404_1
            buf642 = buf618; del buf618  # reuse
            # Topologically Sorted Source Nodes: [keys_75], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg406_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf640, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg405_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf642)
            del arg405_1
            del arg406_1
            buf643 = buf577; del buf577  # reuse
            # Topologically Sorted Source Nodes: [values_50], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg408_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf640, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg407_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf643)
            del arg407_1
            del arg408_1
            buf644 = reinterpret_tensor(buf640, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf640  # reuse
            # Topologically Sorted Source Nodes: [queries_75, queries_76, queries_77, query_25, keys_75, keys_76, keys_77, key_25, values_50, view_77, values_51, value_25, owners_25, getitem_113, getitem_114, eq_25, attention_mask_25, attn_output_100], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf641, buf644, 1492992, stream=stream0)
            buf645 = reinterpret_tensor(buf641, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf641  # reuse
            # Topologically Sorted Source Nodes: [queries_75, queries_76, queries_77, query_25, keys_75, keys_76, keys_77, key_25, values_50, view_77, values_51, value_25, owners_25, getitem_113, getitem_114, eq_25, attention_mask_25, attn_output_100], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf642, buf645, 1492992, stream=stream0)
            buf646 = reinterpret_tensor(buf642, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf642  # reuse
            # Topologically Sorted Source Nodes: [queries_75, queries_76, queries_77, query_25, keys_75, keys_76, keys_77, key_25, values_50, view_77, values_51, value_25, owners_25, getitem_113, getitem_114, eq_25, attention_mask_25, attn_output_100], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf643, buf646, 1492992, stream=stream0)
            del buf643
            buf647 = buf622; del buf622  # reuse
            # Topologically Sorted Source Nodes: [queries_75, queries_76, queries_77, query_25, keys_75, keys_76, keys_77, key_25, values_50, view_77, values_51, value_25, owners_25, getitem_113, getitem_114, eq_25, attention_mask_25, attn_output_100], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant25, buf647, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_75, queries_76, queries_77, query_25, keys_75, keys_76, keys_77, key_25, values_50, view_77, values_51, value_25, owners_25, getitem_113, getitem_114, eq_25, attention_mask_25, attn_output_100], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf648 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf644, buf645, buf646, reinterpret_tensor(buf647, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf644
            buf649 = buf648[0]
            assert_size_stride(buf649, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf649, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf648
            buf653 = reinterpret_tensor(buf646, (1296, 1152), (1152, 1), 0); del buf646  # reuse
            # Topologically Sorted Source Nodes: [transpose_103, reshape_38, attn_output_103], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg410_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf649, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg409_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf653)
            del arg409_1
            del arg410_1
            buf658 = reinterpret_tensor(buf649, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf649  # reuse
            # Topologically Sorted Source Nodes: [attn_output_99, hidden_states_169, hidden_states_173, hidden_states_174, attn_output_103, hidden_states_176, hidden_states_177], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_6.run(buf611, buf628, buf635, buf653, arg411_1, arg412_1, buf658, 1296, 1152, stream=stream0)
            del arg411_1
            del arg412_1
            buf659 = buf633; del buf633  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_177, hidden_states_178], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg414_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf658, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg413_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf659)
            del arg413_1
            del arg414_1
            buf660 = buf634; del buf634  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_179], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf659, buf660, 5577984, stream=stream0)
            buf661 = reinterpret_tensor(buf658, (1296, 1152), (1152, 1), 0); del buf658  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_178, hidden_states_179, hidden_states_180], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg416_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf660, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg415_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf661)
            del arg415_1
            del arg416_1
            buf662 = buf611; del buf611  # reuse
            buf666 = reinterpret_tensor(buf645, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf645  # reuse
            # Topologically Sorted Source Nodes: [attn_output_99, hidden_states_169, hidden_states_173, hidden_states_174, attn_output_103, hidden_states_176, hidden_states_180, hidden_states_181, hidden_states_182], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_8.run(buf662, buf628, buf635, buf653, buf661, arg417_1, arg418_1, buf666, 1296, 1152, stream=stream0)
            del arg417_1
            del arg418_1
            del buf628
            buf667 = buf661; del buf661  # reuse
            # Topologically Sorted Source Nodes: [queries_78], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg420_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf666, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg419_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf667)
            del arg419_1
            del arg420_1
            buf668 = buf653; del buf653  # reuse
            # Topologically Sorted Source Nodes: [keys_78], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg422_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf666, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg421_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf668)
            del arg421_1
            del arg422_1
            buf669 = buf635; del buf635  # reuse
            # Topologically Sorted Source Nodes: [values_52], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg424_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf666, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg423_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf669)
            del arg423_1
            del arg424_1
            buf670 = reinterpret_tensor(buf666, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf666  # reuse
            # Topologically Sorted Source Nodes: [queries_78, queries_79, queries_80, query_26, keys_78, keys_79, keys_80, key_26, values_52, view_80, values_53, value_26, owners_26, getitem_117, getitem_118, eq_26, attention_mask_26, attn_output_104], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf667, buf670, 1492992, stream=stream0)
            buf671 = reinterpret_tensor(buf667, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf667  # reuse
            # Topologically Sorted Source Nodes: [queries_78, queries_79, queries_80, query_26, keys_78, keys_79, keys_80, key_26, values_52, view_80, values_53, value_26, owners_26, getitem_117, getitem_118, eq_26, attention_mask_26, attn_output_104], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf668, buf671, 1492992, stream=stream0)
            buf672 = reinterpret_tensor(buf668, (1, 16, 1296, 72), (1492992, 93312, 72, 1), 0); del buf668  # reuse
            # Topologically Sorted Source Nodes: [queries_78, queries_79, queries_80, query_26, keys_78, keys_79, keys_80, key_26, values_52, view_80, values_53, value_26, owners_26, getitem_117, getitem_118, eq_26, attention_mask_26, attn_output_104], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_1.run(buf669, buf672, 1492992, stream=stream0)
            del buf669
            buf673 = buf647; del buf647  # reuse
            # Topologically Sorted Source Nodes: [queries_78, queries_79, queries_80, query_26, keys_78, keys_79, keys_80, key_26, values_52, view_80, values_53, value_26, owners_26, getitem_117, getitem_118, eq_26, attention_mask_26, attn_output_104], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_clone_eq_expand_lift_fresh_scalar_tensor_transpose_unsqueeze_view_where_2.run(_tensor_constant26, buf673, 1679616, stream=stream0)
            # Topologically Sorted Source Nodes: [queries_78, queries_79, queries_80, query_26, keys_78, keys_79, keys_80, key_26, values_52, view_80, values_53, value_26, owners_26, getitem_117, getitem_118, eq_26, attention_mask_26, attn_output_104], Original ATen: [aten.view, aten.transpose, aten.clone, aten.lift_fresh, aten.unsqueeze, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf674 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf670, buf671, buf672, reinterpret_tensor(buf673, (1, 16, 1296, 1296), (1679616, 0, 1296, 1), 0), False, scale=0.11785113019775792)
            del buf670
            del buf673
            buf675 = buf674[0]
            assert_size_stride(buf675, (1, 16, 1296, 72), (1492992, 72, 1152, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf675, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf674
            buf679 = reinterpret_tensor(buf672, (1296, 1152), (1152, 1), 0); del buf672  # reuse
            # Topologically Sorted Source Nodes: [transpose_107, reshape_39, attn_output_107], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(reinterpret_tensor(arg426_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf675, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg425_1, (1152, 1152), (1, 1152), 0), alpha=1, beta=1, out=buf679)
            del arg425_1
            del arg426_1
            buf683 = reinterpret_tensor(buf675, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf675  # reuse
            # Topologically Sorted Source Nodes: [attn_output_107, hidden_states_183, hidden_states_184], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_3.run(buf662, buf679, arg427_1, arg428_1, buf683, 1296, 1152, stream=stream0)
            del arg427_1
            del arg428_1
            buf684 = buf659; del buf659  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_185], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg430_1, (1296, 4304), (0, 1), 0), reinterpret_tensor(buf683, (1296, 1152), (1152, 1), 0), reinterpret_tensor(arg429_1, (1152, 4304), (1, 1152), 0), alpha=1, beta=1, out=buf684)
            del arg429_1
            del arg430_1
            buf685 = buf660; del buf660  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_185, hidden_states_186], Original ATen: [aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_4.run(buf684, buf685, 5577984, stream=stream0)
            del buf684
            buf686 = reinterpret_tensor(buf683, (1296, 1152), (1152, 1), 0); del buf683  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_185, hidden_states_186, hidden_states_187], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            extern_kernels.bias_addmm(reinterpret_tensor(arg432_1, (1296, 1152), (0, 1), 0), reinterpret_tensor(buf685, (1296, 4304), (4352, 1), 0), reinterpret_tensor(arg431_1, (4304, 1152), (1, 4304), 0), alpha=1, beta=1, out=buf686)
            del arg431_1
            del arg432_1
            del buf685
            buf707 = reinterpret_tensor(buf671, (1, 1296, 1152), (1492992, 1152, 1), 0); del buf671  # reuse
            # Topologically Sorted Source Nodes: [attn_output_107, hidden_states_183, hidden_states_187, hidden_states_188, last_hidden_state], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_view_5.run(buf662, buf679, buf686, arg433_1, arg434_1, buf707, 1296, 1152, stream=stream0)
            del arg433_1
            del arg434_1
            del buf662
            del buf679
            del buf686
            buf690 = empty_strided_cuda((262144, 18), (18, 1), torch.float32)
            # Topologically Sorted Source Nodes: [ones_like], Original ATen: [aten.ones_like]
            stream0 = get_raw_stream(0)
            triton_poi_fused_ones_like_9.run(buf690, 4718592, stream=stream0)
            buf691 = empty_strided_cuda((262144, 18), (18, 1), torch.float32)
            # Topologically Sorted Source Nodes: [arange, float_1, flat_pos, x_pos, outer, arange_1, float_2, dim_range, truediv, pow_1, freqs], Original ATen: [aten.arange, aten._to_copy, aten.remainder, aten.view, aten.div, aten.pow, aten.reciprocal, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_div_mul_pow_reciprocal_remainder_view_10.run(buf691, 4718592, stream=stream0)
            # Topologically Sorted Source Nodes: [ones_like, arange, float_1, flat_pos, x_pos, outer, arange_1, float_2, dim_range, truediv, pow_1, freqs, x_cis], Original ATen: [aten.ones_like, aten.arange, aten._to_copy, aten.remainder, aten.view, aten.div, aten.pow, aten.reciprocal, aten.mul, aten.polar]
            buf692 = torch.ops.aten.polar.default(buf690, buf691)
            buf693 = buf692
            assert_size_stride(buf693, (262144, 18), (18, 1), 'torch.ops.aten.polar.default')
            assert_alignment(buf693, 16, 'torch.ops.aten.polar.default')
            del buf692
            # Topologically Sorted Source Nodes: [unsqueeze], Original ATen: [aten.unsqueeze]
            buf694 = torch.ops.aten.unsqueeze.default(buf693, -1)
            buf695 = buf694
            assert_size_stride(buf695, (262144, 18, 1), (18, 1, 1), 'torch.ops.aten.unsqueeze.default')
            assert_alignment(buf695, 16, 'torch.ops.aten.unsqueeze.default')
            buf696 = buf691; del buf691  # reuse
            # Topologically Sorted Source Nodes: [ones_like_1], Original ATen: [aten.ones_like]
            stream0 = get_raw_stream(0)
            triton_poi_fused_ones_like_9.run(buf696, 4718592, stream=stream0)
            buf697 = buf690; del buf690  # reuse
            # Topologically Sorted Source Nodes: [arange, float_1, flat_pos, arange_1, float_2, dim_range, truediv, pow_1, freqs, y_pos, outer_1], Original ATen: [aten.arange, aten._to_copy, aten.div, aten.pow, aten.reciprocal, aten.mul, aten.floor_divide, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_div_floor_divide_mul_pow_reciprocal_view_11.run(buf697, 4718592, stream=stream0)
            # Topologically Sorted Source Nodes: [arange, float_1, flat_pos, arange_1, float_2, dim_range, truediv, pow_1, freqs, ones_like_1, y_pos, outer_1, y_cis], Original ATen: [aten.arange, aten._to_copy, aten.div, aten.pow, aten.reciprocal, aten.mul, aten.ones_like, aten.floor_divide, aten.view, aten.polar]
            buf698 = torch.ops.aten.polar.default(buf696, buf697)
            del buf696
            del buf697
            buf699 = buf698
            assert_size_stride(buf699, (262144, 18), (18, 1), 'torch.ops.aten.polar.default')
            assert_alignment(buf699, 16, 'torch.ops.aten.polar.default')
            del buf698
            # Topologically Sorted Source Nodes: [unsqueeze_1], Original ATen: [aten.unsqueeze]
            buf700 = torch.ops.aten.unsqueeze.default(buf699, -1)
            buf701 = buf700
            assert_size_stride(buf701, (262144, 18, 1), (18, 1, 1), 'torch.ops.aten.unsqueeze.default')
            assert_alignment(buf701, 16, 'torch.ops.aten.unsqueeze.default')
            # Topologically Sorted Source Nodes: [freqs_cis], Original ATen: [aten.cat]
            buf702 = torch.ops.aten.cat.default([buf695, buf701], -1)
            del buf693
            del buf694
            del buf695
            del buf699
            del buf700
            del buf701
            buf703 = buf702
            assert_size_stride(buf703, (262144, 18, 2), (36, 2, 1), 'torch.ops.aten.cat.default')
            assert_alignment(buf703, 16, 'torch.ops.aten.cat.default')
            del buf702
            # Topologically Sorted Source Nodes: [freqs_cis_1], Original ATen: [aten.view]
            buf704 = torch.ops.aten.reshape.default(buf703, [512, 512, -1])
            buf705 = buf704
            assert_size_stride(buf705, (512, 512, 36), (18432, 36, 1), 'torch.ops.aten.reshape.default')
            assert_alignment(buf705, 16, 'torch.ops.aten.reshape.default')
            # Topologically Sorted Source Nodes: [last_hidden_state, last_hidden_state_1], Original ATen: [aten.native_layer_norm, aten.index]
            buf708 = torch.ops.aten.index.Tensor(buf707, [None, arg435_1])
            del arg435_1
            del buf707
            buf709 = buf708
            assert_size_stride(buf709, (1, 1296, 1152), (1492992, 1152, 1), 'torch.ops.aten.index.Tensor')
            # buffer buf709 (op: torch.ops.aten.index.Tensor) is assumed to be not aligned
            del buf708
        return (buf705, buf709, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    global _tensor_constant0
    _tensor_constant0 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant1
    _tensor_constant1 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant2
    _tensor_constant2 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant3
    _tensor_constant3 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant4
    _tensor_constant4 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant5
    _tensor_constant5 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant6
    _tensor_constant6 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant7
    _tensor_constant7 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant8
    _tensor_constant8 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant9
    _tensor_constant9 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant10
    _tensor_constant10 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant11
    _tensor_constant11 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant12
    _tensor_constant12 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant13
    _tensor_constant13 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant14
    _tensor_constant14 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant15
    _tensor_constant15 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant16
    _tensor_constant16 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant17
    _tensor_constant17 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant18
    _tensor_constant18 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant19
    _tensor_constant19 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant20
    _tensor_constant20 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant21
    _tensor_constant21 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant22
    _tensor_constant22 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant23
    _tensor_constant23 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant24
    _tensor_constant24 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant25
    _tensor_constant25 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    global _tensor_constant26
    _tensor_constant26 = rand_strided((1296, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg0_1 = rand_strided((1, 1296, 1152), (1492992, 1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg198_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg199_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg200_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg201_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg202_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg203_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg204_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg205_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg206_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg207_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg208_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg209_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg210_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg211_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg212_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg213_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg214_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg215_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg216_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg217_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg218_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg219_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg220_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg221_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg222_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg223_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg224_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg225_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg226_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg227_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg228_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg229_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg230_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg231_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg232_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg233_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg234_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg235_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg236_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg237_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg238_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg239_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg240_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg241_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg242_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg243_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg244_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg245_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg246_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg247_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg248_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg249_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg250_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg251_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg252_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg253_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg254_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg255_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg256_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg257_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg258_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg259_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg260_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg261_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg262_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg263_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg264_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg265_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg266_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg267_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg268_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg269_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg270_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg271_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg272_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg273_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg274_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg275_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg276_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg277_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg278_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg279_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg280_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg281_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg282_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg283_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg284_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg285_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg286_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg287_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg288_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg289_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg290_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg291_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg292_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg293_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg294_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg295_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg296_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg297_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg298_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg299_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg300_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg301_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg302_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg303_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg304_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg305_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg306_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg307_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg308_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg309_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg310_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg311_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg312_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg313_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg314_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg315_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg316_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg317_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg318_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg319_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg320_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg321_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg322_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg323_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg324_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg325_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg326_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg327_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg328_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg329_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg330_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg331_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg332_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg333_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg334_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg335_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg336_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg337_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg338_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg339_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg340_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg341_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg342_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg343_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg344_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg345_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg346_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg347_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg348_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg349_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg350_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg351_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg352_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg353_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg354_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg355_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg356_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg357_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg358_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg359_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg360_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg361_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg362_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg363_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg364_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg365_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg366_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg367_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg368_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg369_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg370_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg371_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg372_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg373_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg374_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg375_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg376_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg377_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg378_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg379_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg380_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg381_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg382_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg383_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg384_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg385_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg386_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg387_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg388_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg389_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg390_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg391_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg392_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg393_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg394_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg395_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg396_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg397_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg398_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg399_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg400_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg401_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg402_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg403_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg404_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg405_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg406_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg407_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg408_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg409_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg410_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg411_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg412_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg413_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg414_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg415_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg416_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg417_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg418_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg419_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg420_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg421_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg422_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg423_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg424_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg425_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg426_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg427_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg428_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg429_1 = rand_strided((4304, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg430_1 = rand_strided((4304, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg431_1 = rand_strided((1152, 4304), (4304, 1), device='cuda:0', dtype=torch.bfloat16)
    arg432_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg433_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg434_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg435_1 = rand_strided((1296, ), (1, ), device='cpu', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
