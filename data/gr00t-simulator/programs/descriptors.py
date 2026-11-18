"""Version-2 256-bit descriptors; no scheduling/timing code.

Wire words follow anonymous_1108 COMMANDS.zh-CN.md. All addresses are burst units.
Word 0 bit 29 extends the format for Down's co-resident rows: input bank=query,
fragment offset=f*stride+floor(f/split)*jump. Both A and B use this row padding.
Word 0 bit 30 describes Down's two-bank staging under top-level MC control.
In that form the row padding applies only to the weights.
The native simulator executes the timing-free expanded children, not a binary
descriptor decoder; the words remain a checked description of that expansion.
"""
import binascii
import struct


def crc16(words):
    return binascii.crc_hqx(struct.pack("<8I", *words), 0xFFFF)


def mac(*, m, n, tile, fragments, pack, a_base, a_stride, b_base, b_stride,
        resident=False, reduce=False, tile_major=False, split=0, jump=0, wait=0, done=1,
        row_colocated=False, down_staging=False):
    if not (1 <= m <= 16 and 1 <= tile <= min(n, 64) and n <= 1024 and
            1 <= fragments <= 4096 and pack in (1, 2, 4, 8) and
            pack * m * 32 <= 1024 and m * tile * 4 <= 2048 and done > 0):
        raise ValueError("invalid descriptor geometry")
    if any(x % 32 for x in (a_base, a_stride, b_base, b_stride, jump)):
        raise ValueError("descriptor addresses must be burst aligned")
    if (row_colocated or down_staging) and (resident or reduce or tile_major or not 0 < split < 1024):
        raise ValueError("invalid Down descriptor addressing")
    if row_colocated and down_staging:
        raise ValueError("Down input must use one layout")
    return [1 | 2 << 4 | (3 if resident else 1) << 8 | int(reduce) << 10 | 0xFFFF << 12
            | int(tile_major) << 28 | int(row_colocated) << 29 | int(down_staging) << 30,
            m - 1 | (n - 1) << 4 | (fragments - 1) << 14 | (pack.bit_length() - 1) << 26 | 15 << 28,
            a_base // 32 | (0 if resident else 4) << 28,
            1 | a_stride // 32 << 16,
            b_base // 32 | (split & 255) << 24,
            1 | b_stride // 32 << 16,
            (jump // 32 & 0xFFFFFF) | (tile - 1) << 24 | (split >> 8) << 30,
            wait << 11 | done << 19]


def rope(wait=0, done=1):
    return [2 | 2 << 4 | 3 << 8 | 0xFFFF << 12, 7 | wait << 16 | done << 24,
            0xA08 | 1 << 24, 0xA10 | 1 << 24, 0xA00 | 1 << 24,
            0xA08 | 1 << 24, 0xA10 | 1 << 24, 0]


def for_operator(name, *, down_staging=False):
    common = dict(a_base=0x80000, a_stride=512)
    if name == "qkv":
        return mac(m=8, n=48, tile=48, fragments=64, pack=4, a_base=0x80000,
                   a_stride=256, b_base=0x60000, b_stride=1536)
    if name == "qkt":
        return mac(m=16, n=496, tile=16, fragments=1, pack=1, a_base=0, a_stride=512,
                   b_base=0x10000, b_stride=0, resident=True, reduce=True, split=484, jump=1408)
    if name == "pv":
        return mac(m=16, n=16, tile=16, fragments=31, pack=2, b_base=0x28000, b_stride=512, **common)
    if name == "gate_up":
        return mac(m=16, n=32, tile=32, fragments=64, pack=2, b_base=0x40000, b_stride=1024, **common)
    if name == "down":
        inputs = common if down_staging else dict(a_base=0x50080, a_stride=160)
        # Weight rows retain the original four weight columns and one input
        # column per fragment, followed by four padding columns per row.
        return mac(m=16, n=4, tile=4, fragments=256, pack=2, b_base=0x50000, b_stride=160,
                   split=12, jump=128, row_colocated=not down_staging,
                   down_staging=down_staging, **inputs)
    if name == "o":
        return mac(m=16, n=64, tile=32, fragments=16, pack=2, b_base=0xC0000, b_stride=1024,
                   tile_major=True, **common)
    if name == "rope":
        return rope()
    raise ValueError(name)
