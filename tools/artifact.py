#!/usr/bin/env python3
"""Verify, unpack and replay the self-contained anonymous paper artifact."""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('spec_path', 'program_path', 'ae_schedule_path', 'vlm_path', 'power_spec_path')


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def digest(path, compressed=False):
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest():
    return read(ROOT / 'ARTIFACT-MANIFEST.json')['files']


def stored(path):
    return path if path.is_file() else Path(str(path) + '.gz')


def verify():
    rows = manifest()
    for row in rows:
        p = ROOT / row['path']
        assert p.is_file(), f'Missing {p.relative_to(ROOT)}'
        assert p.stat().st_size == row['bytes'], f'Size mismatch: {p}'
        assert digest(p) == row['sha256'], f'Checksum mismatch: {p}'
        if row['path'].endswith('.gz'):
            assert digest(p, compressed=True) == row['uncompressed_sha256'], p
    configs = []
    for family in ('pi05', 'gr00t', 'wck'):
        configs.extend((ROOT / 'data' / family / 'configs').glob('*.json'))
    configs.extend((ROOT / 'data/timeline').glob('config-*.json'))
    for config in configs:
        cfg = read(config)
        for key in PATH_KEYS:
            if key not in cfg:
                continue
            value = Path(cfg[key])
            assert not value.is_absolute(), f'Nonportable {config}: {key}={value}'
            assert stored(ROOT / value).is_file(), f'Missing input: {config.name}: {key}={value}'
    print(f'Verified {len(rows)} files and {len(configs)} portable replay configurations.')


def unpack():
    count = 0
    for row in manifest():
        if not row['path'].endswith('.gz'):
            continue
        target = ROOT / row['uncompressed_path']
        if target.is_file() and digest(target) == row['uncompressed_sha256']:
            continue
        with gzip.open(ROOT / row['path'], 'rb') as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        assert digest(target) == row['uncompressed_sha256'], target
        count += 1
    print(f'Unpacked {count} losslessly compressed data files.')


def get_input(relative):
    path = ROOT / relative
    if not path.is_file():
        gz = Path(str(path) + '.gz')
        if not gz.is_file():
            raise FileNotFoundError(path)
        with gzip.open(gz, 'rb') as src, path.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    return path


def replay(args):
    config = ROOT / args.config
    cfg = read(config)
    for key in PATH_KEYS:
        if key in cfg:
            cfg[key] = str(get_input(cfg[key]))
    if args.latest:
        binary = ROOT / 'build/pim_vla_sim'
    elif args.config.startswith('data/gr00t/'):
        binary = ROOT / 'data/gr00t-simulator/build/pim_vla_sim'
    elif args.config.startswith(('data/wck/', 'data/power/')):
        binary = ROOT / 'build/pim_vla_sim'
    else:
        binary = ROOT / 'data/paper-simulator/build/pim_vla_sim'
    if args.binary:
        binary = Path(args.binary).resolve()
    if not binary.is_file():
        raise SystemExit(f'Build the simulator first: {binary}')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runconfig = out / 'config.json'
    runconfig.write_text(json.dumps(cfg, indent=2) + '\n')
    subprocess.run([str(binary), '--config', str(runconfig), '--output', str(out)], check=True)
    if args.expected:
        expected = read(stored(ROOT / args.expected))
        actual = read(out / 'raw.json')
        assert actual == expected, 'Native raw.json differs from the archived result.'
        print('Native raw.json matches the archived result exactly.')


def paper_checks():
    """Check the rounded performance numbers against native archived results."""
    chart = read(ROOT / 'paper/figures/data/async-vla-performance.json')
    pi = next(m for m in chart['models'] if m['key'] == 'pi05')
    for item in pi['configurations']:
        case = item['original_label']
        if case in ('a-serial', 'a-dual'):
            continue
        if case == 'b':
            raw = read(stored(ROOT / 'data/pi05/evidence/case-b-single-read.bauEXp/10/raw.json'))
            calibration = read(ROOT / 'reference/2026-11-15/simulation/compiled-vlm-calibration.json')
            cycle = raw['metrics']['total_us'] / 1000 + calibration['result']['solo_ms']
        else:
            raw = read(stored(ROOT / f'data/pi05/evidence/compiled-single-read-matrix.683RuH/compiled/{case}/raw.json'))
            cycle = raw['metrics']['total_us'] / 1000
        assert abs(cycle - item['cycle_ms']) < 0.0051, (case, cycle, item['cycle_ms'])
    gr = next(m for m in chart['models'] if m['key'] == 'gr00t_n16')
    for item in gr['configurations']:
        case = item['original_label']
        if not case.startswith('n16-'):
            continue
        raw = read(stored(ROOT / f'data/gr00t/native/{case}/repeat-0/raw.json'))
        expected = item['cycle_ms']
        assert abs(raw['metrics']['total_us'] / 1000 - expected) < 1e-7, (case, expected)
        repeat = read(stored(ROOT / f'data/gr00t/native/{case}/repeat-1/raw.json'))
        assert raw == repeat, case
    timeline = read(ROOT / 'paper/figures/data/timeline-vlm-priority.json')
    assert [c['completed_vlm_kernels'] for c in timeline['timeline_cases']] == [15, 19]
    for case in timeline['timeline_cases']:
        kernels = case['vlm_kernels']
        gpu = [e for e in case['events'] if e['kind'] == 'gpu']
        assert all(not k.get('paused_us', 0) for k in kernels)
        assert all(min(k['end_ck'], e['end_ck']) <= max(k['start_ck'], e['start_ck']) for k in kernels for e in gpu)
    print('Both models\' paper values, GR00T repeated runs and the 15/19-kernel timeline checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    sub.add_parser('unpack')
    sub.add_parser('paper-checks')
    run = sub.add_parser('replay')
    run.add_argument('--config', required=True, help='Repository-relative archived config')
    run.add_argument('--output', required=True)
    run.add_argument('--expected', help='Repository-relative archived raw.json[.gz]')
    run.add_argument('--latest', action='store_true', help='Use the latest simulator for a historical config')
    run.add_argument('--binary', help='Override simulator binary')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    elif args.command == 'unpack':
        unpack()
    elif args.command == 'paper-checks':
        paper_checks()
    else:
        replay(args)


if __name__ == '__main__':
    main()
