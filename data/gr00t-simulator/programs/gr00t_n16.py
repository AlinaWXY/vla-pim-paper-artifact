"""GR00T DiT PIM request templates with a measured, explicit VLM sequence length."""
from collections import Counter
from math import ceil

from experiments.config import GR00T_CASES, variant_name
from programs import descriptors as D
from workloads import layout as L
from workloads.models import GR00T_N16_H15 as SPEC

VARIANTS = {variant_name(c): (c["layout"], c["dataflow"]) for c in GR00T_CASES.values()}


def regions(spec, layout, tokens):
    """Conservative per-bank bounds shared by alternating layer templates."""
    width = spec.head_dim if layout == "token" else 32
    slots = spec.heads // L.CHANNELS if layout == "token" else 2 * spec.heads // L.CHANNELS
    shapes = {
        "qkv": (spec.hidden // 16, 3 * spec.hidden // 256),
        "k": (width // 16, tokens // 16 * slots),
        "v": (tokens // 16, width // 16 * slots),
        "o": (width * slots // 16, spec.hidden // L.BANKS),
        "up": (spec.hidden // 16, spec.ffn // 256),
        "down": (spec.ffn // 16, spec.hidden // 256),
        "upload": (max(spec.ffn, tokens) // 16, spec.queries),
        "pv_o": (width * slots // 16, spec.queries),
        "export": (1, max(32 * spec.queries * 4 // L.BURST_BYTES, tokens * spec.queries * 4 // (L.BANKS * L.BURST_BYTES))),
    }
    result = {}
    cursor = 0
    for name, (fragments, columns) in shapes.items():
        size = ceil(fragments * columns * L.BURST_BYTES / L.ROW_BYTES) * L.ROW_BYTES
        result[name] = [cursor, cursor + size]
        cursor += size
    L.validate_regions(result)
    return result


def generate(variant, channels, vl_tokens, helpers):
    # Imported here so that the CLI and library share the same native builders.
    Builder, staged_input, gemm, walk, attach, validate = helpers
    layout, flow = VARIANTS[variant]
    fused = flow == "fused"
    self_kv = SPEC.kv_shape("self")
    cross_kv = SPEC.kv_shape("cross", vl_tokens)
    area = regions(SPEC, layout, cross_kv[1])
    width = SPEC.head_dim if layout == "token" else 32
    slots = SPEC.heads // L.CHANNELS if layout == "token" else 2 * SPEC.heads // L.CHANNELS
    queries = SPEC.queries
    counts = Counter()

    def input_location(base):
        def location(fragment, query):
            return 4 * (fragment % 2), base + (fragment // 2 * queries + query) * 32
        return location

    def cache(channel, kind, tokens):
        b = Builder(channel, f"{kind}_kv_store")
        for region, fragments, columns in (("k", width // 16, tokens // 16),
                                           ("v", tokens // 16, width // 16)):
            for slot in range(slots):
                for fragment in range(fragments):
                    for col in range(columns):
                        at = area[region][0] + ((slot * fragments + fragment) * columns + col) * 32
                        for bank in range(L.BANKS):
                            b.emit("WR", at, bank, epoch=b.group, head_slot=slot,
                                   tensor_fragment=fragment, tensor_column=col)
        return b.ops

    def matrix(channel, name, fragments, columns, base, *, head_slot=None,
               capture_local=False, resident_input=False, bf16=False):
        """One bank-parallel matrix; tile the accumulator at 32 outputs per bank."""
        outer = Builder(channel, name)
        location = input_location(area["pv_o" if resident_input else "upload"][0])
        previous = []
        for tile, first in enumerate(range(0, columns, 32)):
            n = min(32, columns - first)
            b = Builder(channel, name)
            initial = ()
            if fused and not resident_input:
                initial = (staged_input(outer, queries, fragments, input_location=location),)
            last = gemm(b, rows=queries, fragments=fragments, columns=n, base=base,
                        fused=fused, views=2, input_location=location, output_tile=tile,
                        weight_offset=lambda f, col: base + (f * columns + first + col) * 32)
            stores = []
            stripe = []
            pack = 16 if bf16 else 8
            count = queries * L.BANKS * n // pack
            for packet in range(count):
                scalar = packet * pack
                query, rem = divmod(scalar, L.BANKS * n)
                regions = []
                for start in range(rem, rem + pack):
                    bank, col = divmod(start, n)
                    if regions and regions[-1]["bank"] == bank:
                        regions[-1]["bytes"] += 4
                    else:
                        regions.append(dict(bank=bank, byte_offset=(query * n + col) * 4, bytes=4))
                source_bank = regions[0]["bank"]
                gather = b.emit("GATHER", bank=source_bank, deps=[last], bytes=pack * 4,
                                elements=pack, source="output_pool", source_regions=regions, output_tile=tile)
                ready = gather
                if bf16:
                    ready = b.emit("CAST", bank=source_bank, deps=[gather], elements=pack,
                                   output_bytes=32, source_regions=regions, output_tile=tile)
                if capture_local:
                    bank, offset = input_location(area["pv_o"][0])(head_slot * (width // 16) +
                                                                      rem // 16, query)
                else:
                    bank, offset = packet % L.BANKS, area["export"][0] + packet // L.BANKS * 32
                stripe.append((bank, offset, dict(deps=[ready],
                               source="pv_bf16" if capture_local else "fused_result",
                               source_bank=source_bank, source_regions=regions, output_tile=tile,
                               output_bytes=32, release_result=True)))
                if capture_local or len(stripe) == L.BANKS:
                    stores.extend((bk, at, b.emit("LOCALWR", at, bk, **fields)) for bk, at, fields in stripe)
                    stripe.clear()
            seal = b.fence([i for _, _, i in stores])
            if fused:
                words = D.mac(m=queries, n=n, tile=n, fragments=fragments, pack=2,
                              a_base=location(0, 0)[1], a_stride=queries * 32,
                              b_base=base + first * 32, b_stride=columns * 32)
                done = outer.emit("MAC_RUN", bank=-1, deps=[*previous, *initial],
                                  operator=name, descriptor=words, crc16=D.crc16(words), children=b.ops)
            else:
                # Merge the tile in FIFO order, retaining the producer dependencies.
                begin = len(outer.ops)
                for op in b.ops:
                    op["id"] += begin
                    op["deps"] = [i + begin for i in op["deps"]]
                    if op["id"] == begin:
                        op["deps"].extend(previous)
                    outer.ops.append(op)
                done = begin + seal
            if capture_local:
                previous = [done]
            else:
                reads = [outer.emit("RD", offset, bank, deps=[done], epoch=name + "_export")
                         for bank, offset, _ in stores]
                previous = [outer.fence(reads)]
        return outer.ops

    def stages_for(kind, kv):
        yield f"{kind}_kv_store", lambda c: cache(c, kind, kv)
        q_columns = SPEC.hidden // (L.CHANNELS * L.BANKS) * (3 if kind == "self" else 1)
        yield f"{kind}_qkv", lambda c: matrix(c, f"{kind}_qkv", SPEC.hidden // 16,
                                               q_columns, area["qkv"][0], bf16=True)
        for slot in range(slots):
            name = f"{kind}_qkt_{slot}"
            fragments, columns = width // 16, kv // L.BANKS
            yield name, lambda c, n=name, f=fragments, b=area["k"][0] + slot * fragments * columns * 32: matrix(
                c, n, f, kv // L.BANKS, b)
        for slot in range(slots):
            name = f"{kind}_pv_{slot}"
            fragments, columns = kv // 16, width // L.BANKS
            yield name, lambda c, n=name, s=slot, b=area["v"][0] + slot * fragments * columns * 32: matrix(
                c, n, kv // 16, width // L.BANKS, b, capture_local=fused, head_slot=s, bf16=True)
        yield f"{kind}_o", lambda c: matrix(c, f"{kind}_o", width * slots // 16,
                                             SPEC.hidden // L.BANKS, area["o"][0], resident_input=fused)

    stages = []
    factories = list(stages_for("cross", cross_kv[1])) + list(stages_for("self", self_kv[1]))
    for name, dim_in, dim_out in (("up", SPEC.hidden, SPEC.ffn), ("down", SPEC.ffn, SPEC.hidden)):
        factories.append((name, lambda c, n=name, i=dim_in, o=dim_out: matrix(
            c, n, i // 16, o // (L.CHANNELS * L.BANKS), area[n][0])))
    for name, generate_channel in factories:
        ops = [generate_channel(c) for c in range(channels)]
        stage_counts = Counter()
        for channel, commands in enumerate(ops):
            attach(commands)
            validate(commands, channel)
            stage_counts.update(op["op"] for op in walk(commands))
        dq = sum(stage_counts[x] for x in ("RD", "WR", "WRGB", "RDRES")) * L.BURST_BYTES
        counts.update(stage_counts)
        stages.append(dict(name=name, gpu_before_us=0, expected_dq_bytes=dq, channels=ops))

    contract = dict(shape=dict(heads=SPEC.heads, head_dim=SPEC.head_dim, hidden=SPEC.hidden,
                               ffn=SPEC.ffn, queries=queries, horizon=SPEC.horizon,
                               state_tokens=SPEC.state_tokens, backbone_embedding_dim=SPEC.backbone_width,
                               valid_kv=cross_kv[0], padded_kv=cross_kv[1]),
                    kv={"self": self_kv, "cross": cross_kv}, layout=layout,
                    layers=SPEC.layers, denoising_steps=SPEC.steps, layer_kinds=SPEC.layer_kinds,
                    layer_stride_bytes_per_bank=L.LAYER_STRIDE_BYTES, regions_per_bank=area,
                    ownership=("two complete heads per channel" if layout == "token" else
                               "four head shards per channel; adjacent channels own the other shard"),
                    head_features_per_slot=width, heads_or_shards_per_channel=slots,
                    numeric_execution=False)
    return dict(schema_version=1, architecture_version=2, model=SPEC.name,
                execution_model="native_dram_requests", variant=variant, channels=channels,
                layout=layout, rope="none", dataflow=flow, layer_kinds=SPEC.layer_kinds,
                repetitions=dict(layers=SPEC.layers, steps=SPEC.steps), tensor_contract=contract,
                counts=dict(counts), stages=stages)
