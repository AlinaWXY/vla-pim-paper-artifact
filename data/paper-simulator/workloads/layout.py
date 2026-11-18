"""Physical tensor ownership; no DRAM timing or old simulator imports.

Addresses use [channel, rank, bank_group, bank, row, 32-byte-column].
The original branch's ownership formulas are retained. Down uses the requested
co-resident weight/input rows; other operator address formulas are unchanged.
"""
CHANNELS = 16
BANKS = 16
ROW_BYTES = 2048
BURST_BYTES = 32
HEADS, HEAD_DIM, HIDDEN, QUERIES, PREFIX = 8, 256, 1024, 16, 968
PADDED_KV = 992
SRF_BYTES, OUTPUT_BYTES = 1024, 2048
LAYER_STRIDE_BYTES = 1 << 20
LAYERS, STEPS = 18, 10
QKV_BASE, V_BASE = 0x60000, 0x28000
K_PREFIX_BASE, WORK_BASE = 0x10000, 0x14000
Q_BASE, K_SUFFIX_BASE, COEF_BASE = 0x14100, 0x14200, 0x14000
Q_IMPORT_BASE = 0x15000
STAGING_BASE, SCORE_BASE, O_BASE = 0x80000, 0xA0000, 0xC0000
FUSED_RESULT_BASE, FUSED_RESULT_BYTES_PER_BANK = 0x90000, 2048
GU_BASE, DOWN_BASE = 0x40000, 0x50000
DOWN_FRAGMENTS = 256
DOWN_WEIGHT_COLUMNS = 4
DOWN_COLUMNS_PER_FRAGMENT = DOWN_WEIGHT_COLUMNS + 1
DOWN_FRAGMENTS_PER_ROW = 12
DOWN_ROWS = (DOWN_FRAGMENTS + DOWN_FRAGMENTS_PER_ROW - 1) // DOWN_FRAGMENTS_PER_ROW
DOWN_REGION_BYTES = DOWN_ROWS * ROW_BYTES


def address(channel, offset, bank=0, rank=1):
    if not 0 <= channel < CHANNELS or rank not in (0, 1):
        raise ValueError("invalid channel/rank")
    if offset < 0 or offset % BURST_BYTES or bank not in range(-1, BANKS):
        raise ValueError("unaligned offset or invalid bank")
    row, within = divmod(offset, ROW_BYTES)
    bg, ba = (-1, -1) if bank == -1 else divmod(bank, 4)
    return [channel, rank, bg, ba, row, within // BURST_BYTES]


def layer_address(relative, layer):
    if not 0 <= layer < LAYERS or len(relative) != 6 or relative[1] != 1:
        raise ValueError("invalid PIM layer address")
    result = list(relative)
    result[4] += layer * (LAYER_STRIDE_BYTES // ROW_BYTES)
    return result


def feature_owner(feature):
    if not 0 <= feature < HEAD_DIM:
        raise ValueError("invalid feature")
    pair = feature % 128
    return pair // 8, 2 * (pair % 8) + feature // 128


def feature_from_lane(bank, lane):
    if not 0 <= bank < 16 or not 0 <= lane < 16:
        raise ValueError("invalid feature lane")
    return bank * 8 + lane // 2 + 128 * (lane % 2)


def feature_shard_from_lane(channel, bank, lane):
    """1108 feature baseline: natural feature order within each 128-wide shard."""
    if not 0 <= channel < 16 or not 0 <= bank < 16 or not 0 <= lane < 8:
        raise ValueError("invalid feature-sharded lane")
    return bank * 8 + lane + 64 * (channel % 2) + (64 if bank >= 8 else 0)


def token_owner(head, token):
    if not 0 <= head < 8 or not 0 <= token < PADDED_KV:
        raise ValueError("invalid token")
    return 2 * head + token % 2, token // 2


def staging(fragment, token, rows):
    return 4 * (fragment % 2), STAGING_BASE + (fragment // 2 * rows + token) * 32


def key_offset(local_token, feature=False):
    token = local_token // 2 if feature else local_token
    if token < 484:
        return K_PREFIX_BASE + token * 32
    if token < 492:
        return K_SUFFIX_BASE + (token - 484) * 32
    return WORK_BASE + (24 + token - 492) * 32


def score_location(token, packet):
    # Two FP32 score bursts per token, on different bank groups.
    if packet not in (0, 1):
        raise ValueError("a score column has two 32-byte packets")
    return packet * 4, SCORE_BASE + token * 32


def fused_result_location(packet):
    """Stripe one operator's result packets across existing DRAM banks."""
    capacity = BANKS * FUSED_RESULT_BYTES_PER_BANK // BURST_BYTES
    if type(packet) is not int or not 0 <= packet < capacity:
        raise ValueError("fused result packet exceeds reserved DRAM capacity")
    return packet % BANKS, FUSED_RESULT_BASE + packet // BANKS * BURST_BYTES


def validate_regions(regions, layer_stride_bytes=LAYER_STRIDE_BYTES):
    """Reserved per-bank DRAM ranges must fit one layer without overlap."""
    intervals = []
    for name, bounds in regions.items():
        if len(bounds) != 2:
            raise ValueError(f"invalid DRAM region {name}")
        begin, end = bounds
        if not (type(begin) is int and type(end) is int and
                0 <= begin < end <= layer_stride_bytes and
                begin % BURST_BYTES == 0 and end % BURST_BYTES == 0):
            raise ValueError(f"invalid or out-of-capacity DRAM region {name}")
        intervals.append((begin, end, name))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping DRAM regions {left[2]} and {right[2]}")


def gu_weight_location(reduction, output, projection=0):
    channel, local = divmod(output, 256)
    output_slot, bank = divmod(local, 16)
    fragment, lane = divmod(reduction, 16)
    return channel, bank, GU_BASE + ((fragment * 2 + projection) * 16 + output_slot) * 32 + lane * 2


def down_weight_location(reduction, output):
    if not 0 <= reduction < 4096 or not 0 <= output < 1024:
        raise ValueError("invalid Down reduction/output coordinate")
    channel, local = divmod(output, 64)
    bank, output_slot = divmod(local, 4)
    fragment, lane = divmod(reduction, 16)
    return channel, bank, down_weight_offset(fragment, output_slot) + lane * 2


def down_weight_offset(fragment, output_slot):
    """32B weight column in a 12-fragment [W0,W1,W2,W3,input] row."""
    if not 0 <= fragment < DOWN_FRAGMENTS or not 0 <= output_slot < DOWN_WEIGHT_COLUMNS:
        raise ValueError("invalid Down fragment/output slot")
    row, within = divmod(fragment, DOWN_FRAGMENTS_PER_ROW)
    return DOWN_BASE + row * ROW_BYTES + (within * DOWN_COLUMNS_PER_FRAGMENT + output_slot) * BURST_BYTES


def down_input_location(fragment, query):
    """Query q's input is in bank q, adjacent to the fragment's four weights.

    One input burst per bank is read into that bank's SRF and broadcast to the
    corresponding query slot in every bank. No input write addresses a weight.
    """
    if not 0 <= fragment < DOWN_FRAGMENTS or not 0 <= query < QUERIES:
        raise ValueError("invalid Down fragment/query")
    offset = down_weight_offset(fragment, 0) + DOWN_WEIGHT_COLUMNS * BURST_BYTES
    return query, offset


def down_staging_location(fragment, query):
    """Place two fragments in one broadcast row and stripe batches over banks 0/4."""
    if not 0 <= fragment < DOWN_FRAGMENTS or not 0 <= query < QUERIES:
        raise ValueError("invalid Down fragment/query")
    batch, view = divmod(fragment, 2)
    bank = 4 * (batch % 2)
    column = (batch // 2 * 2 + view) * QUERIES + query
    return bank, STAGING_BASE + column * BURST_BYTES


def contract(layout, *, fused_kernel_windows=False, down_staging=False):
    if layout not in ("feature", "token"):
        raise ValueError(layout)
    result = {
        "shape": {"heads": 8, "head_dim": 256, "hidden": 1024, "ffn": 4096,
                  "queries": 16, "prefix": 968, "valid_kv": 984, "padded_kv": 992},
        "topology": {"channels": 16, "ranks": 2, "gpu_rank": 0, "pim_rank": 1,
                     "bank_groups": 4, "banks_per_group": 4, "row_bytes": 2048},
        "buffers": {"srf_bytes_per_bank": 1024, "output_bytes_per_bank": 2048,
                    "rank_score_cache_bytes": 0, "rope_units_per_rank": 1,
                    "rope_vector_views": "coefficient and two 32B inputs reuse SRF; two 32B outputs reuse output pool"},
        "layout": layout,
        "layer_stride_bytes_per_bank": LAYER_STRIDE_BYTES,
        "layers": LAYERS,
        "denoising_steps": STEPS,
        "address_template": "row is relative to the current layer; physical row adds layer*512",
        "ownership": "channel=2*head+token%2; local_token=token//2" if layout == "token" else
                     "rank feature subsets [0:64,128:192] and [64:128,192:256]; all tokens",
        "feature_lanes_by_channel": [
            [[feature_from_lane(b, x) for x in range(16)] if layout == "token" else
             [feature_shard_from_lane(c, b, x) for x in range(8)] for b in range(16)]
            for c in range(16)],
        "formats": {"inputs": "BF16", "weights": "BF16", "accumulators": "FP32",
                    "scores": "FP32", "pv_o": "BF16", "ffn_exports": "FP32"},
        "regions_per_bank": {
            "k_prefix": [K_PREFIX_BASE, WORK_BASE], "q_k_coefficient_work": [WORK_BASE, WORK_BASE + 2048],
            "qkv": [QKV_BASE, QKV_BASE + 96 * 1024], "v_cache": [V_BASE, V_BASE + 16384],
            "gate_up_weights": [GU_BASE, DOWN_BASE],
            "down_weights_and_inputs": [DOWN_BASE, DOWN_BASE + DOWN_REGION_BYTES],
            "q_import": [Q_IMPORT_BASE, Q_IMPORT_BASE + 2048],
            "input_staging_bank0_bank4": [STAGING_BASE, STAGING_BASE + 65536],
            "score_bank0_bank4": [SCORE_BASE, SCORE_BASE + 16384],
            "wo_full_resident": [O_BASE, O_BASE + 256 * 1024]},
        "down_row_layout": {
            "base_byte_offset": DOWN_BASE, "rows_per_bank": DOWN_ROWS,
            "reserved_bytes_per_bank": DOWN_REGION_BYTES, "columns_per_row": ROW_BYTES // BURST_BYTES,
            "fragments": DOWN_FRAGMENTS, "fragments_per_row": DOWN_FRAGMENTS_PER_ROW,
            "columns_per_fragment": DOWN_COLUMNS_PER_FRAGMENT,
            "weight_columns_within_fragment": list(range(DOWN_WEIGHT_COLUMNS)),
            "input_column_within_fragment": DOWN_WEIGHT_COLUMNS,
            "input_bank": "query (0..15)", "fused_fragments_per_srf_batch": 2,
            "weight_bytes_per_bank": DOWN_FRAGMENTS * DOWN_WEIGHT_COLUMNS * BURST_BYTES,
            "input_bytes_per_bank": DOWN_FRAGMENTS * BURST_BYTES,
            "unused_columns_full_row": 4, "fragments_last_row": DOWN_FRAGMENTS % DOWN_FRAGMENTS_PER_ROW,
            "streaming": "same weight mapping; WRGB supplies SRF directly and does not write array input columns",
            "fused": "ordinary WR scatters inputs by query bank; LOCALRD plus BCAST reads the co-resident row",
            "fused_row_activation": "one explicit ACTAB per row after prior-row MAC completion; first-batch LOCALRD waits for issue, then native per-bank RCD"},
        "coefficient_source": "BF16 sin/cos generated offline from actual model position IDs",
        "operator_shapes": operator_shapes(layout),
        "numeric_execution": False,
    }
    if down_staging:
        result["down_row_layout"].update(
            co_resident_input_columns_used=False, input_bank="4 * (fragment // 2 % 2)",
            fused="32 consecutive input columns per SRF batch in the existing bank 0/4 staging region",
            fused_row_activation="single-bank ACT and 32 broadcast reads, PRE, then ACTAB for weights")
        result["down_input_staging"] = dict(base_byte_offset=STAGING_BASE,
                                           bytes_per_source_bank=65536, source_banks=[0, 4],
                                           fragments_per_batch=2, srf_slot="column % 32")
    if fused_kernel_windows:
        result["regions_per_bank"]["fused_result_spill"] = [
            FUSED_RESULT_BASE, FUSED_RESULT_BASE + FUSED_RESULT_BYTES_PER_BANK]
        result["fused_result_spill"] = {
            "storage": "DRAM; no additional SRAM",
            "bytes_per_bank": FUSED_RESULT_BYTES_PER_BANK,
            "packet_mapping": "bank=packet%16; byte_offset=base+floor(packet/16)*32",
            "operator_bytes_per_bank": {"o": 2048, "gate_up": 2048, "down": 256},
            "reuse": "after the preceding operator's complete external export",
            "formats": {"o": "BF16", "gate_up": "FP32", "down": "FP32"},
        }
    validate_regions(result["regions_per_bank"])
    return result


def operator_shapes(layout):
    token = layout == "token"
    queries, features, kv = (8, 256, 496) if token else (16, 128, 992)
    return {
        "qkv": {"input_per_rank": [queries, 1024], "weight_per_rank": [1024, 3 * features],
                "output_per_rank": [queries, 3 * features], "active_weight_bytes_per_bank": 98304 if token else 49152,
                "scalar_macs_all_channels": 100663296},
        "qkt": {"q_per_rank": [16, features], "k_per_rank": [kv, features], "score_per_rank": [16, kv],
                "score_semantics": "complete score; disjoint KV tokens" if token else "partial score; GPU sums the feature ranks",
                "scalar_macs_all_channels": 32505856},
        "pv": {"p_per_rank": [16, kv], "v_per_rank": [kv, features], "output_per_rank": [16, features],
               "output_semantics": "partial over KV parity" if token else "complete for owned features",
               "scalar_macs_all_channels": 32505856},
        "o": {"input_per_rank": [16, features], "active_weight_per_rank": [features, 1024],
              "resident_weight_per_rank": [2048, 1024], "output_per_rank": [16, 1024],
              "resident_order": "[rotated_head_slot8][tile2][fragment16][column32][lane16]" if token else
                                "[rotated_feature_shard_slot16][tile2][fragment8][column32][lane16]",
              "active_weight_bytes_per_bank": 32768 if token else 16384,
              "scalar_macs_all_channels": 67108864 if token else 33554432,
              "gpu_reduction_operands": 16},
        "gate_up": {"input_per_rank": [16, 1024], "weight_per_rank": [1024, 512],
                    "output_per_rank": [16, 512], "scalar_macs_all_channels": 134217728,
                    "output_feature_formula": "projection=column//16; feature=channel*256+(column%16)*16+bank"},
        "down": {"input_per_rank": [16, 4096], "weight_per_rank": [4096, 64],
                 "output_per_rank": [16, 64], "scalar_macs_all_channels": 67108864,
                 "output_feature_formula": "feature=channel*64+bank*4+column"},
    }
