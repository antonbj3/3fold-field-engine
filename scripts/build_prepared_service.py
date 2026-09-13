"""Build the prepared field service without opening a GPU context."""
import argparse
from field_paths import field_path
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/field_engine'
ARTIFACTS = {
    'RT_COLUMNS_LIBRARY': 'columns_compact.so', 'RT_COLUMNS_PTX': 'columns.ptx',
    'NATIVE_EDT_LARGE_LIBRARY': 'edt_large.so', 'COLUMN_MASK_LIBRARY': 'column_mask.so',
    'RT_PREPARED_PROBE_LIBRARY': 'prepared_probe.so', 'RT_PREPARED_LIBRARY': 'prepared_columns.so',
    'RT_PREPARED_PTX': 'prepared.ptx', 'MESH_FUSED_PREPARED_LIBRARY': 'mesh_fused_prepared.so',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure(paths):
    """Include every repository-local quoted native include in the build receipt."""
    result = {}
    pending = list(paths)
    while pending:
        path = pending.pop().resolve()
        name = str(path.relative_to(ROOT))
        if name in result:
            continue
        result[name] = digest(path)
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
            pending.append(path.parent / include)
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root', type=Path, default=os.environ.get('CUDA_ROOT'))
    parser.add_argument('--optix-include', type=Path, default=os.environ.get('OPTIX_INCLUDE'))
    parser.add_argument('--arch', default=os.environ.get('CUDA_ARCH'))
    parser.add_argument('--output', type=Path, required=True, help='New, empty build directory')
    args = parser.parse_args()
    if not args.cuda_root or not args.optix_include or not args.arch:
        parser.error('Specify CUDA root, OptiX include directory and explicit CUDA architecture')
    if not re.fullmatch(r'sm_[0-9]+[a-z]?', args.arch):
        parser.error('Invalid CUDA architecture')
    cuda, optix = args.cuda_root.resolve(), args.optix_include.resolve()
    if not (cuda/'bin/nvcc').is_file() or not (optix/'optix.h').is_file():
        parser.error('CUDA compiler or OptiX header missing')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Build directory must be empty; previous evidence is never overwritten')
    nvcc = [str(cuda/'bin/nvcc')]
    host = ['c++', '-shared', '-fPIC', '-pthread', '-O2', '-ffp-contract=off', '-std=c++17',
            '-I'+str(optix), '-I'+str(cuda/'include')]
    link = ['-L'+str(cuda/'lib64'), '-Wl,-rpath,'+str(cuda/'lib64'), '-lcudart', '-ldl']
    device = nvcc + ['-shared', '-Xcompiler=-fPIC', '--fmad=false', '--ftz=false', '-O3', '-arch='+args.arch]
    ptx = nvcc + ['-ptx', '--fmad=false', '--ftz=false', '-std=c++17', '-I'+str(optix)]
    specs = [
        (host, 'rt_columns_compact_v1/api.cpp', link, 'columns_compact.so'),
        (ptx, 'rt_columns_v1/program.cu', [], 'columns.ptx'),
        (device, 'edt_native_large_v1/edt.cu', [], 'edt_large.so'),
        (device, 'column_mask_native_v1/mask.cu', [], 'column_mask.so'),
        (device, 'rt_columns_prepared_v1/probe.cu', [], 'prepared_probe.so'),
        (host, 'rt_columns_prepared_compact_v1/api.cpp', link, 'prepared_columns.so'),
        (ptx, 'rt_columns_prepared_v1/program.cu', [], 'prepared.ptx'),
        (device+['-std=c++17', '-I'+str(optix)], 'mesh_fused_prepared_graph_v1/field.cu', ['-ldl'], 'mesh_fused_prepared.so'),
    ]
    sources = source_closure([field_path(ROOT,s[1]) for s in specs])
    commands = []
    def portable(value):
        for prefix, replacement in ((str(output), '<build>'), (str(ROOT), '<repo>'),
                                    (str(optix), '<optix>'), (str(cuda), '<cuda>')):
            value = value.replace(prefix, replacement)
        return value
    for prefix, source, suffix, target in specs:
        command = prefix + [str(field_path(ROOT,source))] + suffix + ['-o', str(output/target)]
        print('Building '+target, flush=True)
        subprocess.run(command, check=True, timeout=120)
        commands.append([portable(x) for x in command])
    if sources != source_closure([field_path(ROOT,s[1]) for s in specs]):
        raise RuntimeError('Source changed during compilation')
    manifest = dict(version=1, architecture=args.arch, artifacts=ARTIFACTS,
                    artifact_sha256={name: digest(output/name) for name in ARTIFACTS.values()},
                    source_sha256=sources, commands=commands,
                    cuda_version=subprocess.check_output(nvcc+['--version'], text=True),
                    host_version=subprocess.check_output(['c++', '--version'], text=True),
                    optix_header_sha256={str(p.relative_to(optix)): digest(p) for p in sorted(optix.rglob('*.h'))},
                    scope='Compilation only; no GPU context or numerical/performance acceptance claim.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Build complete; manifest.json records sources, SDK, commands and artifact hashes.')


if __name__ == '__main__':
    main()
