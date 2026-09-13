"""Observe cold RT setup without changing native arithmetic or adding GPU fences.

Run --prepare with OPTIX_INCLUDE and CUDA_ROOT set to build the external native
libraries. The hardware-coordinating caller supplies an exclusive GPU window
for the two fresh child processes. Timings are host intervals, not GPU events.
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
import subprocess
import sys
import time
import numpy as np

HERE = _probe_root / 'src/field_engine'
ROOT = _probe_root


def instrument(source):
    anchors = [
        ('check(cudaFree(nullptr));', 'cuda_primary_context'),
        ('check(optixDeviceContextCreate(nullptr, &options, &context));', 'optix_context'),
        ('sizes.tempSizeInBytes, storage, sizes.outputSizeInBytes, &gas, nullptr, 0));', 'geometry_and_acceleration'),
        ('ptx.data(), ptx.size(), nullptr, nullptr, &module));', 'module_compile'),
        ('check(optixProgramGroupCreate(context, descriptions, 3, &group_options, nullptr, nullptr, groups));', 'program_groups'),
        ('check(optixPipelineCreate(context, &compile, &link, groups, 3, nullptr, nullptr, &pipeline));', 'pipeline_link'),
        ('params_data = allocate(sizeof(parameters), &parameters);', 'stack_sbt_io_buffers'),
    ]
    prologue = '''
        auto phase_start = std::chrono::steady_clock::now();
        auto emit_phase = [&](const char* name) {
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - phase_start).count();
            std::cerr << "RT_PHASE " << name << " " << std::setprecision(17) << ms << "\\n";
            phase_start = std::chrono::steady_clock::now();
        };
'''
    first = 'check(cudaFree(nullptr));'
    if source.count(first) != 1:
        raise ValueError("Unexpected native source")
    source = source.replace(first, prologue + first)
    for anchor, label in anchors:
        if source.count(anchor) != 1:
            raise ValueError("Native phase anchor changed")
        source = source.replace(anchor, anchor + f'\n        emit_phase("{label}");')
    return '#include <chrono>\n#include <iomanip>\n' + source


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    cuda = Path(os.environ["CUDA_ROOT"])
    optix = Path(os.environ["OPTIX_INCLUDE"])
    base = (HERE / "rt_columns_v1/api.cpp").read_text()
    compact = (HERE / "rt_columns_compact_v1/api.cpp").read_text()
    source = compact.replace('#include "../rt_columns_v1/api.cpp"', instrument(base))
    cpp = folder / "columns_phases.cpp"
    cpp.write_text(source)
    commands = [
        ['c++', '-shared', '-fPIC', '-pthread', '-O2', '-std=c++17',
         '-I' + str(optix), '-I' + str(cuda / 'include'), '-I' + str(HERE / 'rt_columns_v1'),
         str(cpp), '-L' + str(cuda / 'lib64'), '-Wl,-rpath,' + str(cuda / 'lib64'),
         '-lcudart', '-ldl', '-o', str(folder / 'columns_phases.so')],
        [str(cuda / 'bin/nvcc'), '-ptx', '--fmad=false', '-std=c++17', '-I' + str(optix),
         str(HERE / 'rt_columns_v1/program.cu'), '-o', str(folder / 'columns.ptx')],
    ]
    architecture = os.environ.get('CUDA_ARCH', 'sm_120')
    for name, target in (('edt_native_large_v1/edt.cu', 'edt_large.so'),
                         ('column_mask_native_v1/mask.cu', 'column_mask.so')):
        commands.append([str(cuda / 'bin/nvcc'), '-shared', '-Xcompiler=-fPIC', '--fmad=false',
                         '-O3', '-arch=' + architecture, str(HERE / name), '-o', str(folder / target)])
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
              for name in ('columns_phases.cpp', 'columns_phases.so', 'columns.ptx', 'edt_large.so', 'column_mask.so')}
    hashes['baseline_cpp'] = hashlib.sha256(base.encode()).hexdigest()
    (folder / 'build_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(hashes))


def child(folder, output, leg):
    import trimesh
    import faltkarna_v1_mesh_to_sdf as baseline
    from mesh_field_native_mask_v1 import PreparedMeshField
    mesh = trimesh.load(os.environ['FIELD_PLATE_STL'], process=False)
    mesh.merge_vertices()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    pitch = .5
    origin = vertices.min(0) - 3 * pitch
    begin = time.perf_counter_ns()
    expected = baseline.surface_raster_and_flood(vertices, faces, pitch, origin)
    cpu_ms = (time.perf_counter_ns() - begin) * 1e-6
    begin = time.perf_counter_ns()
    field = PreparedMeshField(vertices, faces, pitch, origin,
                              winding_library=folder / 'columns_phases.so', ptx=folder / 'columns.ptx',
                              edt_library=folder / 'edt_large.so', column_library=folder / 'column_mask.so')
    constructed = time.perf_counter_ns()
    try:
        actual = field.evaluate()
        evaluated = time.perf_counter_ns()
    finally:
        field.close()
    ended = time.perf_counter_ns()
    arrays, differences, hashes = {}, {}, {}
    for i, name in ((0, 'gmin'), (2, 'surface'), (3, 'solid'), (4, 'distance')):
        arrays[name] = actual[i]
        differences[name] = int(np.count_nonzero(actual[i] != expected[i]))
        hashes[name] = hashlib.sha256(actual[i].tobytes()).hexdigest()
    np.savez_compressed(output / f'cold_phases_{leg}_arrays.npz', **arrays)
    row = dict(cpu_ms=cpu_ms, constructor_ms=(constructed - begin) * 1e-6,
               evaluate_ms=(evaluated - constructed) * 1e-6, close_ms=(ended - evaluated) * 1e-6,
               total_ms=(ended - begin) * 1e-6, differences=differences, hashes=hashes,
               shape_equal=actual[1] == expected[1])
    (output / f'cold_phases_{leg}.json').write_text(json.dumps(row, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'reports')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--child', type=int)
    parser.add_argument('--journal-guard', action='store_true')
    args = parser.parse_args()
    folder, output = args.build_directory.resolve(), args.output_directory.resolve()
    if args.prepare:
        prepare(folder)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if args.child is not None:
        child(folder, output, args.child)
        return 0
    rows = []
    def journal_guard():
        if args.journal_guard:
            result = subprocess.run(['journalctl', '-k', '-b', '--no-pager'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode or re.search(r'NVRM:.*Xid', result.stdout):
                raise RuntimeError('Kernel fault guard refused the measurement')
    for leg in range(2):
        journal_guard()
        command = [sys.executable, __file__, '--build-directory', str(folder),
                   '--output-directory', str(output), '--child', str(leg)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        journal_guard()
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        row = json.loads((output / f'cold_phases_{leg}.json').read_text())
        row['native_host_phases_ms'] = {parts[1]: float(parts[2]) for line in result.stderr.splitlines()
                                        if (parts := line.split()) and parts[0] == 'RT_PHASE'}
        rows.append(row)
    gates = dict(exact=all(r['shape_equal'] and not any(r['differences'].values()) for r in rows),
                 repeat=rows[0]['hashes'] == rows[1]['hashes'],
                 phase_coverage=all(len(r['native_host_phases_ms']) == 7 for r in rows))
    report = dict(rows=rows, gates=gates, build_hashes=json.loads((folder / 'build_hashes.json').read_text()),
                  scope='Instrumented host phase observation; caller owns exclusive hardware window; no speed gate promotion.')
    (output / 'cold_phases.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
