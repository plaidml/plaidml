import hashlib
import json
import os
import pathlib
import shutil
import sys
import tarfile
from distutils.dir_util import copy_tree

import yaml

import analysis
import util

DEFAULT_BUILD_URL = 'https://buildkite.com/plaidml'


def buildkite_metadata(key, default=None):
    return os.getenv('BUILDKITE_AGENT_META_DATA_' + key, os.getenv(key, default))


def run(args, remainder):
    util.verbose = True
    util.printf('args:', args)
    if args.shard_count:
        util.printf('running shard:', args.shard)

    with open('ci/plan.yml') as fp:
        plan = yaml.safe_load(fp)

    gpu_flops = plan['CONST']['gpu_flops']
    baseline_name = plan['CONST']['efficiency_baseline']
    platform = plan['PLATFORMS'][args.platform]
    variant_name = platform['variant']
    variant = plan['VARIANTS'][variant_name]
    arch = variant['arch']

    suites = plan['SUITES']
    suite = suites.get(args.suite)
    if suite is None:
        sys.exit('Invalid suite. Available suites: {}'.format(list(suites)))
    platform_cfg = suite['platforms'][args.platform]

    workload = suite['workloads'].get(args.workload)
    if workload is None:
        sys.exit('Invalid workload. Available workloads: {}'.format(list(suite['workloads'])))

    popt = util.PlanOption(suite, workload, args.platform)
    test_info = util.TestInfo(
        suite=(args.suite, suite),
        workload=(args.workload, workload),
        platform=(args.platform, util.Platform(args.platform, gpu_flops)),
        batch_size=args.batch_size,
        variant=variant,
        popt=popt,
        shard_id=args.shard,
        shards=args.shard_count,
    )

    root = pathlib.Path('tmp').resolve()
    input = root / 'input'
    output_root = root / 'test'
    output = test_info.path(output_root)

    shutil.rmtree(input, ignore_errors=True)
    if args.local:
        pkg_path = pathlib.Path('bazel-bin/pkg.tar.gz')
        version = '0.0.0.dev0'
    else:
        archive_path = os.path.join('tmp', 'build', variant_name, 'pkg.tar.gz')
        util.buildkite_download(archive_path, '.')
        pkg_path = root / 'build' / variant_name / 'pkg.tar.gz'
        version = args.version

    util.printf('--- Extracting {} -> {}'.format(pkg_path, input))
    with tarfile.open(str(pkg_path), 'r') as tar:
        tar.extractall(input)

    shutil.rmtree(output_root, ignore_errors=True)
    output.mkdir(parents=True)

    cwd = os.path.abspath(popt.get('cwd', '.'))
    spec = pathlib.Path(popt.get('conda_env'))

    util.printf('--- :snake: Creating conda env from {}'.format(spec))
    instance_name = os.getenv('BUILDKITE_AGENT_NAME', 'harness')
    sig = hashlib.md5()
    sig.update(spec.read_bytes())
    base_path = pathlib.Path('~', '.t2', instance_name, sig.hexdigest()).expanduser()

    base_env = util.CondaEnv(base_path)
    base_env.create(spec)
    conda_env = base_env.clone(root / pathlib.Path('cenv'))
    env = os.environ.copy()
    env.update(conda_env.env())

    for whl in popt.get('wheels', []):
        whl_filename = whl.format(arch=arch, version=version)
        whl_path = input / whl_filename
        conda_env.install(whl_path)

    if 'cuda' in args.platform:
        env['CUDA_DEVICE_ORDER'] = buildkite_metadata('CUDA_CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        env['CUDA_VISIBLE_DEVICES'] = buildkite_metadata('CUDA_VISIBLE_DEVICES', '0')
    device = buildkite_metadata('PLAIDML_DEVICE')
    target = buildkite_metadata('PLAIDML_TARGET')
    if device != None:
        env['PLAIDML_DEVICE'] = device
    if target != None:
        env['PLAIDML_TARGET'] = target

    util.printf('--- :bazel: Running test {suite}/{workload} on {platform}'.format(
        suite=args.suite,
        workload=args.workload,
        platform=args.platform,
    ))

    cmd_args = platform_cfg.get('prepend_args', []) + popt.get('prepend_args', [])
    cmd_args += platform_cfg.get('args', []) + popt.get('args', [])
    cmd_args += platform_cfg.get('append_args', []) + popt.get('append_args', [])

    if args.shard_count:
        cmd_args += ['--shard', str(args.shard)]
        cmd_args += ['--shard-count', str(args.shard_count)]

    ctx = dict(
        results=output,
        batch_size=args.batch_size,
        workload=args.workload,
    )
    cmd_args = [str(x).format(**ctx) for x in cmd_args]
    if 'pml' in args.platform:
        try:
            cmd_args.remove('--no-kernel-timing')
        except ValueError:
            pass

    runner = shutil.which(popt.get('runner'), path=os.pathsep.join([cwd, env['PATH']]))
    cmd = [runner] + cmd_args
    retcode = util.call(cmd, cwd=cwd, env=env)

    build_url = os.getenv('BUILDKITE_BUILD_URL')
    if build_url:
        build_url = '{}#{}'.format(build_url, os.getenv('BUILDKITE_JOB_ID'))
    else:
        build_url = DEFAULT_BUILD_URL

    golden_info = util.TestInfo(
        suite=(args.suite, suite),
        workload=(args.workload, workload),
        platform=(baseline_name, util.Platform(baseline_name, gpu_flops)),
        batch_size=args.batch_size,
        variant=variant,
        popt=popt,
    )

    result = analysis.Result(output_root, test_info, golden_info)
    report = {
        'build_url': build_url,
        'compare': result.test_result.compare,
        'efficiency': result.efficiency,
        'errors': result.test_result.errors,
        'failures': result.test_result.failures,
        'ratio': result.ratio,
        'reason': result.test_result.reason(),
        'status': result.test_result.status(),
        'compile_duration': result.cur.compile_duration,
        'cur.execution_duration': result.cur.execution_duration,
        'ref.execution_duration': result.ref.execution_duration,
    }

    with (output / 'report.json').open('w') as fp:
        json.dump(report, fp)

    if retcode:
        sys.exit(retcode)
    if not result.test_result.is_ok():
        sys.exit(1)
