"""Reversed-order fresh-process CPU/CUDA/RT S/M/L timing with complete outputs.

The caller owns one exclusive GPU window. No GPU warm-up is performed before
each child's first backend call. OS/driver/compiler caches are not cleared.
CPU load snapshots accompany the raw times; no statistical speed claim is made.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[2]
_probe_sys.path[:0] = [str(_probe_root / 'scripts'), str(_probe_root / 'src'), str(_probe_root / 'src/field_engine'), str(_probe_root / 'probes/field_engine')]
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def snapshot():
    values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return dict(cpu_ticks=values, load_average=list(os.getloadavg()), cores=os.cpu_count(),
                ticks_per_second=os.sysconf('SC_CLK_TCK'))


def child(backend, pitch, leg, output):
    import trimesh
    import faltkarna_v1_mesh_to_sdf as baseline
    from mesh_field_cuda_columns_v1 import PreparedMeshField as CUDA
    from mesh_field_native_mask_v1 import PreparedMeshField as RT
    mesh = trimesh.load(os.environ['FIELD_PLATE_STL'], process=False)
    mesh.merge_vertices()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    origin = vertices.min(0) - 3 * pitch
    cls = CUDA if backend == 'cuda' else RT
    kwargs = dict(edt_library=os.environ['NATIVE_EDT_LARGE_LIBRARY'], column_library=os.environ['COLUMN_MASK_LIBRARY'])
    if backend == 'cuda':
        kwargs['columns_library'] = os.environ['CUDA_COLUMNS_LIBRARY']
    else:
        kwargs.update(winding_library=os.environ['RT_COLUMNS_LIBRARY'], ptx=os.environ['RT_COLUMNS_PTX'])
    arrays, observations = {}, []
    started = snapshot()

    def timed(call):
        start = time.perf_counter_ns()
        result = call()
        return result, (time.perf_counter_ns() - start) * 1e-6

    expected, cpu_first = timed(lambda: baseline.surface_raster_and_flood(vertices, faces, pitch, origin))

    def record(name, result):
        hashes, differences = {}, {}
        for i, key in ((0, 'gmin'), (2, 'surface'), (3, 'solid'), (4, 'distance')):
            value = np.ascontiguousarray(result[i])
            reference = np.ascontiguousarray(expected[i])
            hashes[key] = hashlib.sha256(value.tobytes()).hexdigest()
            differences[key] = (0 if value.shape == reference.shape and value.dtype == reference.dtype
                                and value.tobytes() == reference.tobytes() else 1)
            arrays[name + '_' + key] = value
        observations.append(dict(name=name, hashes=hashes, exact=not any(differences.values()),
                                 shape_equal=result[1] == expected[1]))

    def one_shot():
        field = cls(vertices, faces, pitch, origin, **kwargs)
        try:
            return field.evaluate()
        finally:
            field.close()

    first, cold_ms = timed(one_shot)
    record('cpu_first', expected)
    record('cold', first)
    warm, warm_ms = timed(one_shot)
    record('warm_one_shot', warm)
    prepared, setup_ms = timed(lambda: cls(vertices, faces, pitch, origin, **kwargs))
    cpu_times, prepared_times = [], []
    try:
        for repeat in range(4):
            # Reverse CPU/evaluate order in alternate rounds within each child.
            if (repeat + leg) % 2:
                actual, elapsed = timed(prepared.evaluate)
                reference, cpu_elapsed = timed(lambda: baseline.surface_raster_and_flood(vertices, faces, pitch, origin))
            else:
                reference, cpu_elapsed = timed(lambda: baseline.surface_raster_and_flood(vertices, faces, pitch, origin))
                actual, elapsed = timed(prepared.evaluate)
            cpu_times.append(cpu_elapsed)
            prepared_times.append(elapsed)
            record(f'cpu_{repeat}', reference)
            record(f'prepared_{repeat}', actual)
    finally:
        _, close_ms = timed(prepared.close)
    row = dict(backend=backend, pitch=pitch, leg=leg, cpu_first_ms=cpu_first, cold_ms=cold_ms,
               warm_one_shot_ms=warm_ms, prepared_setup_ms=setup_ms, prepared_close_ms=close_ms,
               cpu_ms=cpu_times, prepared_ms=prepared_times, observations=observations,
               cpu_before=started, cpu_after=snapshot())
    name = f'backend_timing_{backend}_{pitch}_{leg}'
    np.savez_compressed(output / (name + '_arrays.npz'), **arrays)
    (output / (name + '.json')).write_text(json.dumps(row, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'reports')
    parser.add_argument('--child', choices=('cuda', 'rt'))
    parser.add_argument('--pitch', type=float)
    parser.add_argument('--leg', type=int)
    parser.add_argument('--journal-guard', action='store_true')
    parser.add_argument('--allow-desktop-background', action='store_true')
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.child:
        child(args.child, args.pitch, args.leg, output)
        return 0

    def guard():
        if args.journal_guard:
            result = subprocess.run(['journalctl', '-k', '-b', '--no-pager'], capture_output=True, text=True, timeout=10)
            if result.returncode or re.search(r'NVRM:.*Xid', result.stdout):
                raise RuntimeError('Kernel fault guard refused timing')

    guard()
    meta = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,utilization.gpu,memory.used',
                           '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=10)
    no_compute = meta.returncode == apps.returncode == 0 and not apps.stdout.strip()
    idle = no_compute and meta.stdout.split(',')[2].strip() == '0 %'
    graphics = []
    graphics_only = False
    if args.allow_desktop_background and no_compute:
        observed = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u'],
                                  capture_output=True, text=True, timeout=10)
        graphics_only = observed.returncode == 0
        for line in observed.stdout.splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 3 or not fields[1].isdigit() or fields[2] != 'G':
                graphics_only = False
                continue
            try:
                executable = Path(os.readlink(f'/proc/{int(fields[1])}/exe')).name
            except OSError:
                graphics_only = False
                continue
            graphics.append(executable)
            graphics_only &= executable in {'Xorg', 'gnome-shell', 'gnome-control-center', 'chrome', 'chromium', 'firefox'}
    eligible = idle or (args.allow_desktop_background and no_compute and graphics_only)
    if not eligible:
        print(json.dumps(dict(samples=0, idle=False, hardware=meta.stdout, contexts=apps.stdout)))
        return 75
    rows = []
    order = [(backend, pitch) for pitch in (2., 1., .5) for backend in ('cuda', 'rt')]
    for leg in range(2):
        for backend, pitch in (order if leg == 0 else list(reversed(order))):
            guard()
            begin = time.perf_counter_ns()
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            result = subprocess.run([sys.executable, __file__, '--child', backend, '--pitch', str(pitch),
                                     '--leg', str(leg), '--output-directory', str(output)],
                                    capture_output=True, text=True, timeout=60)
            guard()
            if result.returncode:
                print(result.stderr, file=sys.stderr)
                return result.returncode
            row = json.loads((output / f'backend_timing_{backend}_{pitch}_{leg}.json').read_text())
            elapsed = resource.getrusage(resource.RUSAGE_CHILDREN)
            row.update(process_ms=(time.perf_counter_ns() - begin) * 1e-6,
                       child_cpu_seconds=elapsed.ru_utime + elapsed.ru_stime - usage.ru_utime - usage.ru_stime)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('backend', 'pitch', 'leg', 'cold_ms', 'warm_one_shot_ms', 'prepared_ms')}), flush=True)
    by_pitch = {pitch: [q for r in rows if r['pitch'] == pitch for q in r['observations']] for pitch in (2., 1., .5)}
    gates = dict(complete=len(rows) == 12 and all(len(r['prepared_ms']) == 4 for r in rows),
                 exact=all(q['exact'] and q['shape_equal'] for r in rows for q in r['observations']),
                 full_repeat=all(all(q['hashes'] == values[0]['hashes'] for q in values) for values in by_pitch.values()),
                 no_competing_compute=no_compute, background_eligible=eligible)
    report = dict(status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  hardware=meta.stdout.strip(), initial_zero_utilization=idle,
                  allowed_graphics=sorted(set(graphics)), rows=rows, gates=gates,
                  scope='Twelve reversed-order fresh children; observed timings with CPU load snapshots; no statistical guarantee.')
    (output / 'backend_timing.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(gates=gates, hardware=report['hardware'])))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
